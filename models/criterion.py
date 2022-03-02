import torch
from torch import nn
from torch.nn import functional as F

from utils.index import (
    xywh_to_xyxy,
    box_giou,
    box_iou,
    get_world_size,
    is_distributed_available_and_initialized,
)
from .matcher import build_matcher


class DeTRCriterion(nn.Module):
    """ 
    Compute the loss for DETR.
    """

    def __init__(self, num_classes, matcher, no_obj_weight=.1):
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher

        loss_weight = torch.ones(self.num_classes + 1)
        loss_weight[-1] = no_obj_weight
        self.register_buffer('loss_weight', loss_weight)

    def loss_classes(self, c_preds, c_tgts_aligned, idx):
        target_classes = torch.full(c_preds.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64)
        target_classes[idx] = c_tgts_aligned

        loss_classes = F.cross_entropy(
            c_preds.transpose(1, 2),
            target_classes,
            self.loss_weight,
        )

        return {"loss_classes": loss_classes}

    def loss_boxes(self, b_preds_aligned, b_tgts_aligned, tgts):
        num_boxes = torch.as_tensor([sum(len(t["labels"]) for t in tgts)],
                                    dtype=torch.float)
        if is_distributed_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        loss_bbox = F.l1_loss(b_preds_aligned,
                              b_tgts_aligned,
                              reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_giou(xywh_to_xyxy(b_preds_aligned),
                     xywh_to_xyxy(b_tgts_aligned)))
        loss_giou = loss_giou.sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def get_pred_permuted_idx(self, matches):
        batch_idx = torch.cat(
            [torch.full_like(pred, i) for i, (pred, _) in enumerate(matches)])
        pred_idx = torch.cat([pred for (pred, _) in matches])
        return batch_idx, pred_idx

    @torch.no_grad()
    def get_mAP(self,
                c_preds_aligned,
                c_tgts_aligned,
                b_preds_aligned,
                b_tgts_aligned,
                iou_threshold=0.5):
        APs = []
        num_class = c_preds_aligned.size(1) - 1
        _num_class = 0
        for c in range(num_class):
            # 得到标签为 i 的样本的数量, 即 tp+fn
            tp_add_fn = (c_tgts_aligned == c).float().sum()

            if tp_add_fn == 0:
                continue
            _num_class += 1

            # 得到预测为 i 的样本
            _c_idx = c_preds_aligned.max(dim=1)[1] == c
            _c_preds = c_preds_aligned[_c_idx][:, c]
            _c_tgts = c_tgts_aligned[_c_idx]
            _b_preds = b_preds_aligned[_c_idx]
            _b_tgts = b_tgts_aligned[_c_idx]

            sorted_idx = _c_preds.sort(0, True)[1]
            _c_preds = _c_preds[sorted_idx]
            _c_tgts = _c_tgts[sorted_idx]
            _b_preds = _b_preds[sorted_idx]
            _b_tgts = _b_tgts[sorted_idx]

            num_preds = _c_preds.size(0)
            tp = torch.zeros_like(_c_preds)
            fp = torch.zeros_like(_c_preds)
            for i in range(num_preds):
                if _c_tgts[i] == c:
                    _b_pred = _b_preds[i].unsqueeze(0)
                    _b_tgt = _b_tgts[i].unsqueeze(0)
                    iou = box_iou(xywh_to_xyxy(_b_pred),
                                  xywh_to_xyxy(_b_tgt))[0][0].item()
                    if iou > iou_threshold:
                        tp[i] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            tp = tp.cumsum(dim=0)
            fp = fp.cumsum(dim=0)
            rec = tp / tp_add_fn
            prec = tp / torch.maximum(tp + fp, torch.tensor(1e-7))
            APs.append(self.get_AP(rec, prec))

        return torch.as_tensor(APs).sum() / _num_class

    def get_AP(self, rec, prec, is_use_07_metric=False):
        if is_use_07_metric:  # 使用 VOC2007 的评价方法
            ap = 0.
            for i in range(0, 11):
                idx = rec >= (i * .1)
                if torch.sum(idx) == 0:  # 不存在大于 (i*.1) 的 recall
                    p = 0
                else:
                    p = torch.max(prec[idx])  # 取满足 recall 阈值的最大的 precision
                ap = ap + p / 11.  # 将 11 个 precision 加和平均
        else:  # 使用 VOC2010 之后的评价方法
            mrec = torch.cat((torch.tensor([0.]), rec, torch.tensor([1.])))
            mprec = torch.cat((torch.tensor([0.]), prec, torch.tensor([0.])))

            for i in range(mprec.size(0) - 1, 0, -1):
                mprec[i - 1] = torch.maximum(mprec[i - 1], mprec[i])

            i = torch.where(mrec[1:] != mrec[:-1])[0]

            ap = torch.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1])

        return ap

    @torch.no_grad()
    def get_accuracy(self, c_preds_aligned, c_tgts_aligned, topk=(1, )):
        if c_tgts_aligned.numel() == 0:
            return [torch.zeros([])]

        _, _pred = c_preds_aligned.topk(max(topk), 1, True, True)
        _pred = _pred.t()
        correct = _pred.eq(c_tgts_aligned.view(1, -1).expand_as(_pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / correct[:k].numel()))

        return res

    def forward(self, preds, tgts, is_train=True, iou_thresholds=(.5, .75)):
        matches = self.matcher(preds, tgts)
        idx = self.get_pred_permuted_idx(matches)

        c_preds = preds['pred_classes']
        b_preds = preds['pred_boxes']

        c_preds_aligned = c_preds[idx]
        c_tgts_aligned = torch.cat(
            [t["labels"][i] for t, (_, i) in zip(tgts, matches)])
        b_preds_aligned = b_preds[idx]
        b_tgts_aligned = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(tgts, matches)], dim=0)

        losses = {}

        if is_train:
            losses.update(self.loss_classes(c_preds, c_tgts_aligned, idx))
            losses.update(
                self.loss_boxes(b_preds_aligned, b_tgts_aligned, tgts))
        else:
            losses.update({
                'class_accuracy':
                self.get_accuracy(c_preds_aligned, c_tgts_aligned)[0]
            })
            mAP = []
            for iou_threshold in iou_thresholds:
                mAP.append(
                    self.get_mAP(
                        c_preds_aligned,
                        c_tgts_aligned,
                        b_preds_aligned,
                        b_tgts_aligned,
                        iou_threshold,
                    ))
            losses.update({'mAP': mAP})

        return losses


def build_criterion(num_classes, no_obj_weight=.1):
    return DeTRCriterion(num_classes, build_matcher(), no_obj_weight)


if __name__ == "__main__":
    preds = {
        "pred_classes":
        torch.tensor([
            [[0.1, 0.6, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.5, 0.3, 0.1, 0.1]],
            [[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]],
        ]),
        "pred_boxes":
        torch.tensor([
            [[0.8, 0.7, 0.2, 0.1], [0.6, 0.6, 0.1, 0.1], [0.7, 0.3, 0.2, 0.1]],
            [[0.1, 0.2, 0.1, 0.3], [0.2, 0.4, 0.4, 0.3], [0.8, 0.7, 0.2, 0.1]],
        ]),
    }
    tgts = [{
        'labels':
        torch.tensor([1, 0]),
        'boxes':
        torch.tensor([[0.8, 0.7, 0.2, 0.1], [0.6, 0.7, 0.1, 0.2]]),
    }, {
        'labels':
        torch.tensor([0, 0]),
        'boxes':
        torch.tensor([[0.1, 0.25, 0.1, 0.4], [0.2, 0.4, 0.4, 0.3]]),
    }]
    criterion = DeTRCriterion(3, build_matcher())
    losses = criterion(preds, tgts, is_train=True)
    evaluations = criterion(preds, tgts, is_train=False)

    print(losses["loss_classes"])  # tensor(1.3728)
    print(losses["loss_bbox"])  # tensor(0.5500)
    print(losses["loss_giou"])  # tensor(1.2541)

    print(evaluations["class_accuracy"])  # tensor(75.)
    print(evaluations["mAP"])  # [tensor(0.5556), tensor(0.4444)]
