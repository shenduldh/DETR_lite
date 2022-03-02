import torch
from torch import nn


class PositionEmbeddingLearned(nn.Module):

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embedding = nn.Embedding(50, num_pos_feats)
        self.col_embedding = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embedding.weight)
        nn.init.uniform_(self.col_embedding.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w)
        j = torch.arange(h)
        x_embedding = self.col_embedding(i)
        y_embedding = self.row_embedding(j)
        x_embedding = x_embedding.unsqueeze(0).repeat(h, 1, 1)
        y_embedding = y_embedding.unsqueeze(1).repeat(1, w, 1)
        pos = torch.cat([x_embedding, y_embedding], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(num_pos_feats=256):
    return PositionEmbeddingLearned(num_pos_feats)


if __name__ == "__main__":
    input = torch.randn((10, 512, 16, 16))
    position_encoding = build_position_encoding(512 // 2)
    pos = position_encoding(input)
    print(pos.size())  # torch.Size([10, 512, 16, 16])
