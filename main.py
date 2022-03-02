import time
import datetime
import torch
import os

from models.detr import build_detr
from models.criterion import build_criterion
from data.index import build_dataloader
from config import CONFIG

if CONFIG.is_use_cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(CONFIG.gpu)


def prepare():
    detr = build_detr(CONFIG.embed_dim, CONFIG.num_classes, CONFIG.num_queries,
                      CONFIG.backbone_name, CONFIG.is_pretrained_backbone,
                      CONFIG.is_frozen_backbone_pre, CONFIG.is_fpn)
    train_dataloader = build_dataloader(CONFIG.trainset_type,
                                        CONFIG.trainset_root, CONFIG.img_size,
                                        CONFIG.train_batch_size,
                                        CONFIG.shuffle_trainset)
    test_dataloader = build_dataloader(CONFIG.evalset_type,
                                       CONFIG.evalset_root, CONFIG.img_size,
                                       CONFIG.eval_batch_size)
    criterion = build_criterion(CONFIG.num_classes, CONFIG.no_obj_weight)

    _params = [{
        "params": [
            p for n, p in detr.named_parameters()
            if "backbone" not in n and p.requires_grad
        ]
    }, {
        "params": [
            p for n, p in detr.named_parameters()
            if "backbone" in n and p.requires_grad
        ],
        "lr":
        CONFIG.lr_backbone
    }]
    optimizer = torch.optim.AdamW(
        _params,
        lr=CONFIG.lr,
        weight_decay=CONFIG.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   CONFIG.lr_drop_step)

    return detr, train_dataloader, test_dataloader, criterion, optimizer, lr_scheduler


def train(model: torch.nn.Module, dataloader, criterion,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.StepLR):
    losses = []

    if os.path.exists(CONFIG.model_save_path):
        if CONFIG.is_use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state_dict = torch.load(CONFIG.model_save_path, map_location=device)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        losses = state_dict["losses"]

    print('Training start.')
    start_time = time.time()

    epoch = 1
    while epoch <= CONFIG.epochs:
        print('-' * 20 + ' EPOCH %s ' % epoch + '-' * 20)

        batch_index = 0
        for samples, targets in dataloader:
            if CONFIG.is_use_cuda and torch.cuda.is_available():
                samples = samples.cuda()

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * CONFIG.loss_weight_dict[k]
                       for k in loss_dict.keys()
                       if k in CONFIG.loss_weight_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 600 == 0:
                losses.append(loss.item())
                print("loss: %.3f" % loss.item())
            batch_index += 1

        lr_scheduler.step()

        losses.append(loss.item())
        print("loss: %.3f" % loss.item())
        torch.save(
            {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'model': model.state_dict(),
                'losses': losses
            }, CONFIG.model_save_path)
        epoch += 1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training finish.')
    print('Training time: {}'.format(total_time_str))
    torch.save(
        {
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'model': model.state_dict(),
            'losses': losses
        }, CONFIG.model_save_path)


def eval(model: torch.nn.Module, test_dataloader, criterion):
    if os.path.exists(CONFIG.model_save_path):
        if CONFIG.is_use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state_dict = torch.load(CONFIG.model_save_path, map_location=device)
        model.load_state_dict(state_dict["model"])

    print('-' * 20 + ' Evaluating ' + '-' * 20)
    evals = torch.FloatTensor([0, 0, 0])
    num_batch = 0
    for samples, targets in test_dataloader:
        if CONFIG.is_use_cuda and torch.cuda.is_available():
            samples = samples.cuda()

        outputs = model(samples)
        eval = criterion(
            outputs,
            targets,
            is_train=False,
            iou_thresholds=CONFIG.iou_thresholds,
        )

        evals[0] += eval['class_accuracy']
        evals[1] += eval['mAP'][0]
        evals[2] += eval['mAP'][1]

        num_batch += 1

    evals /= num_batch
    print("class_accuracy: %.3f" % evals[0].item())
    print("mAP50: %.3f" % evals[1].item())
    print("mAP75: %.3f" % evals[2].item())
    # torch.save(evals, CONFIG.log_save_path)


if __name__ == "__main__":
    detr, train_dataloader, test_dataloader, \
    criterion, optimizer, lr_scheduler = prepare()
    # train(detr, train_dataloader, criterion, optimizer, lr_scheduler)
    eval(detr, test_dataloader, criterion)
