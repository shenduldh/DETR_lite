import torch
from torch import nn
import torch.utils.model_zoo as model_zoo


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


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(  # conv3x3
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = FrozenBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(  # conv3x3
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = FrozenBatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        # left
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # right
        residual = x if self.downsample is None else self.downsample(x)

        out += residual
        return self.relu(out)


class ResNet34(nn.Module):

    def __init__(self):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)

        for m in self.modules():  # init
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                FrozenBatchNorm2d(out_channels),
            )

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]

        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def freeze(self):  # freeze pre and layer1
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


def build_backbone(pretrained=False, is_frozen=False):
    model = ResNet34()
    model.out_channels = 512

    if pretrained:
        model_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        model.load_state_dict(model_zoo.load_url(model_url), strict=False)
    if is_frozen:
        model.freeze()

    return model, 512


if __name__ == "__main__":
    model = build_backbone(pretrained=True, is_frozen=True)

    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    print(output.size())  # torch.Size([1, 512, 8, 8])
