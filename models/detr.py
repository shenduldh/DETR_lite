import torch
from torch import nn

from .backbone import build_backbone
from .transformer import build_transformer
from .position_encoding import build_position_encoding
from .ffn import build_ffn


class DeTR(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 num_classes=1000,
                 num_queries=100,
                 backbone_name='resnet34',
                 is_pretrained_backbone=True,
                 is_frozen_backbone_pre=True,
                 is_fpn=False):
        super().__init__()

        self.backbone = build_backbone(backbone_name, is_pretrained_backbone,
                                       is_frozen_backbone_pre, is_fpn)
        self.input_proj = nn.Conv2d(self.backbone.out_channels,
                                    embed_dim,
                                    kernel_size=1)
        self.pos_encoding = build_position_encoding(embed_dim // 2)
        self.transformer = build_transformer(embed_dim)
        self.class_proj, self.bbox_proj = build_ffn(embed_dim, num_classes)

        self.query_pos = nn.Embedding(num_queries, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        memory_pos = self.pos_encoding(x)

        x = self.input_proj(x)
        queries = self.transformer(x, memory_pos, self.query_pos.weight)

        pred_classes = self.class_proj(queries)
        pred_boxes = self.bbox_proj(queries)

        return {"pred_classes": pred_classes, "pred_boxes": pred_boxes}


def build_detr(
    embed_dim=512,
    num_classes=1000,
    num_queries=100,
    backbone_name='resnet34',
    is_pretrained_backbone=True,
    is_frozen_backbone_pre=True,
    is_fpn=False,
):
    return DeTR(embed_dim, num_classes, num_queries, backbone_name,
                is_pretrained_backbone, is_frozen_backbone_pre,is_fpn)


if __name__ == "__main__":
    detr = DeTR(embed_dim=512, num_classes=5, num_queries=10)
    input = torch.randn((1, 3, 228, 228))
    output = detr(input)
    print(output["pred_classes"].size())  # torch.Size([1, 10, 6])
    print(output["pred_boxes"].size())  # torch.Size([1, 10, 4])
