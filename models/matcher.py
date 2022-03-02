from numpy import dtype
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from utils.index import xywh_to_xyxy, box_giou


class HungarianMatcher(nn.Module):
    """ 
    Compute an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. 
    Because of this, in general, there are more predictions than targets. 
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        class_weight: float = 1,
        bbox_weight: float = 1,
        giou_weight: float = 1,
    ):
        super().__init__()

        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

    @torch.no_grad()
    def forward(self, preds, tgts):
        """
        Params:
            preds: A dict contains two entries:
                "pred_classes": Tensor of dim [batch_size, num_queries, num_classes] 
                                with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] 
                                with the predicted box coordinates

            tgts: A list of targets (len(targets) = batch_size), where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of 
                        ground-truth objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = preds["pred_classes"].shape[:2]

        # 把一批预测数据都聚合在一起
        pred_classes = preds["pred_classes"].flatten(
            0, 1)  # [batch_size * num_queries, num_classes]
        pred_boxes = preds["pred_boxes"].flatten(
            0, 1)  # [batch_size * num_queries, 4]

        # 把一批标签数据都聚合在一起
        tgt_lbs = torch.cat([t["labels"] for t in tgts]).long()
        tgt_bbox = torch.cat([t["boxes"] for t in tgts])

        # compute the classification cost
        cost_class = -pred_classes[:, tgt_lbs]

        # compute the L1 cost between boxes
        cost_bbox = torch.cdist(pred_boxes, tgt_bbox, p=1)

        # compute the giou cost betwen boxes
        cost_giou = -box_giou(xywh_to_xyxy(pred_boxes), xywh_to_xyxy(tgt_bbox))

        # compute the final cost matrix
        total_cost = self.bbox_weight * cost_bbox + self.class_weight * cost_class + self.giou_weight * cost_giou

        # 把数据重新分开并计算匹配
        total_cost = total_cost.view(batch_size, num_queries, -1)
        size_array = [len(t["boxes"]) for t in tgts]
        matches = [
            linear_sum_assignment(c[i].cpu())
            for i, c in enumerate(total_cost.split(size_array, -1))
        ]
        # 转化成 Tensor 再返回
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in matches]


def build_matcher(class_weight=1, bbox_weight=1, giou_weight=1):
    return HungarianMatcher(
        class_weight=class_weight,
        bbox_weight=bbox_weight,
        giou_weight=giou_weight,
    )


if __name__ == "__main__":
    preds = {
        "pred_classes":
        torch.tensor([
            [[0.1, 0.4, 0.4, 0.1], [0.1, 0.6, 0.2, 0.1], [0.5, 0.3, 0.1, 0.1]],
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
        torch.tensor([[0.1, 0.2, 0.1, 0.2], [0.6, 0.7, 0.1, 0.2]]),
    }, {
        'labels':
        torch.tensor([2, 1]),
        'boxes':
        torch.tensor([[0.5, 0.5, 0.1, 0.2], [0.7, 0.7, 0.1, 0.2]]),
    }]
    matcher = HungarianMatcher()
    matches = matcher(preds, tgts)
    print(matches)
    # [(tensor([1, 2]), tensor([1, 0])), (tensor([1, 2]), tensor([0, 1]))]
