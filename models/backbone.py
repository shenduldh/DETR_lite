import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

##########
## 定义一些基础网络层
##########


class FrozenBatchNorm2d(nn.Module):

    def __init__(self, num_features: int):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        return (x - rm) * w / (rv + eps).sqrt() - b


def conv3x3_p1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1_p0(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


##########
## 定义 ResNet 的 Block
##########


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        is_short_block=True,
        stride=1,
        downsample=None,
    ):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        if is_short_block:
            # resnet18, resnet34
            # final out_channels = out_channels
            self.conv1 = conv3x3_p1(in_channels, out_channels, stride)
            self.bn1 = FrozenBatchNorm2d(out_channels)
            self.conv2 = conv3x3_p1(out_channels, out_channels, 1)
            self.bn2 = FrozenBatchNorm2d(out_channels)

            self.left = [self.conv1, self.bn1, self.relu, self.conv2, self.bn2]
        else:
            # resnet50, resnet101, resnet152
            # final out_channels = out_channels × 4
            self.conv1 = conv1x1_p0(in_channels, out_channels, 1)
            self.bn1 = FrozenBatchNorm2d(out_channels)
            self.conv2 = conv3x3_p1(out_channels, out_channels, stride)
            self.bn2 = FrozenBatchNorm2d(out_channels)
            self.conv3 = conv1x1_p0(out_channels, out_channels * 4, 1)
            self.bn3 = FrozenBatchNorm2d(out_channels * 4)

            self.left = [
                self.conv1, self.bn1, self.relu, self.conv2, self.bn2,
                self.relu, self.conv3, self.bn3
            ]
        self.downsample = downsample

    def forward(self, x):
        # left
        out = x
        for l in self.left:
            out = l(out)

        # right
        residual = x if self.downsample is None else self.downsample(x)

        out += residual
        return self.relu(out)


##########
## 定义 ResNet
##########


class ResNet(nn.Module):

    def __init__(self, is_short_block, block_counts):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 64
        self.layer1 = self._make_layer(is_short_block, 64, block_counts[0], 1)
        self.layer2 = self._make_layer(is_short_block, 128, block_counts[1], 2)
        self.layer3 = self._make_layer(is_short_block, 256, block_counts[2], 2)
        self.layer4 = self._make_layer(is_short_block, 512, block_counts[3], 2)

        self.init_conv_weights()

    def init_conv_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _make_layer(self, is_short_block, out_channels, block_count, stride=1):
        channel_expansion = 1 if is_short_block else 4

        downsample = None
        if stride != 1 or self.in_channels != out_channels * channel_expansion:
            downsample = nn.Sequential(
                conv1x1_p0(
                    self.in_channels,
                    out_channels * channel_expansion,
                    stride,
                ),
                FrozenBatchNorm2d(out_channels * channel_expansion),
            )

        layers = [
            ResidualBlock(
                self.in_channels,
                out_channels,
                is_short_block,
                stride,
                downsample,
            )
        ]
        self.in_channels = out_channels * channel_expansion
        for _ in range(1, block_count):
            layers.append(
                ResidualBlock(self.in_channels, out_channels, is_short_block))

        return nn.Sequential(*layers)

    def freeze(self):
        for p in self.conv1.parameters():
            p.requires_grad_(False)
        for p in self.layer1.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


##########
## 定义 FPN
##########


class FPN(ResNet):

    def __init__(self, is_short_block, block_counts):
        super().__init__(is_short_block, block_counts)
        c_expansion = 1 if is_short_block else 4
        o_channels = 64 * c_expansion

        # lateral layers
        self.latlayer1 = nn.Conv2d(512 * c_expansion, o_channels, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(256 * c_expansion, o_channels, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(128 * c_expansion, o_channels, 1, 1, 0)
        self.latlayer4 = nn.Conv2d(64 * c_expansion, o_channels, 1, 1, 0)
        # smooth layer
        self.smooth = nn.Conv2d(o_channels, o_channels, 3, 1, 1)

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        return F.interpolate(
            x, size=(h, w), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # top-down
        o4 = self.latlayer1(l4)
        o3 = self._upsample_add(o4, self.latlayer2(l3))
        o2 = self._upsample_add(o3, self.latlayer3(l2))
        o1 = self._upsample_add(o2, self.latlayer4(l1))

        # smooth
        o = self.smooth(o1)

        return o


models = {
    'resnet18': {
        'pretrained_url':
        'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'is_short_block': True,
        'block_counts': [2, 2, 2, 2],
        'out_channels': 512,
    },
    'resnet34': {
        'pretrained_url':
        'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'is_short_block': True,
        'block_counts': [3, 4, 6, 3],
        'out_channels': 512,
    },
    'resnet50': {
        'pretrained_url':
        'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'is_short_block': False,
        'block_counts': [3, 4, 6, 3],
        'out_channels': 2048,
    },
    'resnet101': {
        'pretrained_url':
        'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'is_short_block': False,
        'block_counts': [3, 4, 23, 3],
        'out_channels': 2048,
    },
    'resnet152': {
        'pretrained_url':
        'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'is_short_block': False,
        'block_counts': [3, 8, 36, 3],
        'out_channels': 2048,
    },
}


def build_backbone(
    model_name="resnet34",
    is_pretrained=True,
    is_frozen=False,
    is_fpn=False,
):
    model_args = models[model_name]
    if is_fpn:
        model = FPN(model_args['is_short_block'], model_args['block_counts'])
        model.out_channels = int(model_args['out_channels'] / 8)
    else:
        model = ResNet(model_args['is_short_block'],
                       model_args['block_counts'])
        model.out_channels = model_args['out_channels']

    if is_pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_args['pretrained_url']),
            strict=False,
        )

    if is_frozen:
        model.freeze()

    return model


if __name__ == '__main__':
    model = build_backbone('resnet50', True, False)
    print(model.out_channels)  # 2048
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    print(output.size())  # torch.Size([1, 2048, 8, 8])

    model = build_backbone('resnet50', True, False, True)
    print(model.out_channels)  # 256
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    print(output.size())  # torch.Size([1, 256, 64, 64])
