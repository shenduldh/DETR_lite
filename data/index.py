import torch
from torch.utils.data import DataLoader

from .voc2006 import VOC2006
from .voc2012 import VOC2012

datasets = {'VOC2006': VOC2006, 'VOC2012': VOC2012}


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.cat([d.unsqueeze(0) for d in batch[0]])
    return tuple(batch)


def build_dataloader(dataset_name,
                     root,
                     img_size=224,
                     batch_size=32,
                     shuffle=False):
    return DataLoader(
        dataset=datasets[dataset_name](root, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    root = './data/assets/VOC2012_trainval'
    dataloader = build_dataloader('VOC2012', root, 5, 2, True)
    dataiter = iter(dataloader)
    batch = next(dataiter)
    print(len(batch[0]))  # 2
    print(batch[1])
