import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        dim_array = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(i, o)
            for i, o in zip([input_dim] + dim_array, dim_array + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid()


def build_ffn(embed_dim, num_classes):
    return (
        nn.Sequential(
            nn.Linear(embed_dim, num_classes + 1),
            nn.Softmax(dim=2),
        ),
        MLP(embed_dim, embed_dim, 4, 3),
    )
