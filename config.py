class CONFIG:
    name = 'debug'
    is_use_cuda = False
    gpu = 0

    trainset_root = './data/assets/VOC2012_trainval'
    trainset_type = 'VOC2012'
    epochs = 10
    train_batch_size = 32
    shuffle_trainset = False
    model_save_path = f'./checkpoint/{name}_model.pth'

    evalset_root = './data/assets/VOC2006_test'
    evalset_type = 'VOC2006'
    eval_batch_size = 64
    log_save_path = f'./checkpoint/{name}_evallog.pth'

    img_size = 10
    embed_dim = 512
    num_classes = 20
    num_queries = 56
    lr = .0001
    lr_backbone = .00001
    lr_drop_step = 1
    weight_decay = .0001
    no_obj_weight = .1
    loss_weight_dict = {
        "loss_classes": 1,
        "loss_bbox": 5,
        "loss_giou": 2,
    }

    backbone_name = 'resnet50'  # 18, 34, 50, 101, 152
    is_fpn = False
    is_frozen_backbone_pre = True
    is_pretrained_backbone = True
    iou_thresholds = (.5, .75)
