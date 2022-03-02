import os
import PIL
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ANNOTATION_DIR = 'Annotations'
IMAGE_DIR = 'PNGImages'
CLASSES = {
    'bicycle': 0,
    'bus': 1,
    'car': 2,
    'cat': 3,
    'cow': 4,
    'dog': 5,
    'horse': 6,
    'motorbike': 7,
    'person': 8,
    'sheep': 9,
}
REGEX_IMGSIZE = re.compile(r'Image size \(X x Y x C\) : (\d+) x (\d+) x 3')
REGEX_LB = re.compile(
    r'Bounding box for object \d+ "PAS(bicycle|bus|car|cat|cow|dog|horse|motorbike|person|sheep).*" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)'
)


class VOC2006(Dataset):

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
        annotation_path = self.root + '/' + ANNOTATION_DIR + '/' + img_name + '.txt'
        with open(annotation_path, 'r') as f:
            data = f.read()

        w, h = REGEX_IMGSIZE.findall(data)[0]
        w = float(w)
        h = float(h)

        labels = []
        boxes = []

        for lb in REGEX_LB.findall(data):
            label = CLASSES[lb[0]]

            x1 = float(lb[1]) / w
            y1 = float(lb[2]) / h
            x2 = float(lb[3]) / w
            y2 = float(lb[4]) / h
            box = [(x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1]

            labels.append(label)
            boxes.append(box)

        return {
            'labels': torch.as_tensor(labels),
            'boxes': torch.as_tensor(boxes)
        }


if __name__ == "__main__":
    root = './data/assets/VOC2006_trainval'

    # max obj count: 19 
    dataset = VOC2006(root, img_size=5)
    print(len(dataset))  # 2618
    data0 = dataset[0]
    print(data0[0].size())  # torch.Size([3, 5, 5])
    print(data0[1])
