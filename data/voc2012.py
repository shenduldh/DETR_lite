import os
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.ElementTree as ET

ANNOTATION_DIR = 'Annotations'
IMAGE_DIR = 'JPEGImages'
CLASSES = {
    'person': 0,
    'bird': 1,
    'cat': 2,
    'cow': 3,
    'dog': 4,
    'horse': 5,
    'sheep': 6,
    'aeroplane': 7,
    'bicycle': 8,
    'boat': 9,
    'bus': 10,
    'car': 11,
    'motorbike': 12,
    'train': 13,
    'bottle': 14,
    'chair': 15,
    'diningtable': 16,
    'pottedplant': 17,
    'sofa': 18,
    'tvmonitor': 19
}


class VOC2012(Dataset):

    def __init__(self, root, img_size=224):
        self.root = root
        imgs = os.listdir(root + '/' + IMAGE_DIR)
        self.imgs = [root + '/' + IMAGE_DIR + '/' + img for img in imgs]
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img_name = img_path.split('/')[-1][:-4]

        data = self.transforms(PIL.Image.open(img_path))
        target = self.get_labels_and_boxes(img_name)

        return data, target

    def __len__(self):
        return len(self.imgs)

    def get_labels_and_boxes(self, img_name):
        annotation_path = self.root + '/' + ANNOTATION_DIR + '/' + img_name + '.xml'

        root = ET.parse(annotation_path).getroot()

        size = root.find('size')
        w = float(size.find('width').text)
        h = float(size.find('height').text)

        objs = root.findall('object')
        labels = []
        boxes = []
        for obj in objs:
            label = CLASSES[obj.find('name').text]

            box = obj.find('bndbox')
            x1 = float(box.find('xmin').text) / w
            y1 = float(box.find('ymin').text) / h
            x2 = float(box.find('xmax').text) / w
            y2 = float(box.find('ymax').text) / h
            box = [(x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1]

            labels.append(label)
            boxes.append(box)

        return {
            'labels': torch.as_tensor(labels),
            'boxes': torch.as_tensor(boxes)
        }


if __name__ == "__main__":
    root = './data/assets/VOC2012_trainval'

    # max obj count: 56 
    dataset = VOC2012(root, img_size=500)
    print(len(dataset))  # 17125
    data0 = dataset[0]
    print(data0[0].size())  # torch.Size([3, 5, 5])
    print(data0[1])
