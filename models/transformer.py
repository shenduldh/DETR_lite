import torch
from torch import nn, Tensor
from copy import deepcopy
from typing import Optional


class Transformer(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        ffn_hidden_dim=1024,
        dropout=0.1,
        activation="relu",
        is_before_norm=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads,
                                                ffn_hidden_dim, dropout,
                                                activation, is_before_norm)
        encoder_norm = nn.LayerNorm(embed_dim) if not is_before_norm else None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
        )

        decoder_layer = TransformerDecoderLayer(
            embed_dim,
            num_heads,
            ffn_hidden_dim,
            dropout,
            activation,
            is_before_norm,
        )
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, memory_pos, query_pos):
        # flatten N*C*H*W to HW*N*C
        n, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        memory_pos = memory_pos.flatten(2).permute(2, 0, 1)
        query_pos = query_pos.unsqueeze(1).repeat(1, n, 1)

        memory = self.encoder(x, pos_embed=memory_pos)

        query = torch.zeros_like(query_pos)
        query = self.decoder(
            query,
            memory,
            query_pos=query_pos,
            memory_pos=memory_pos,
        )

        return query.permute(1, 0, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = clone_layer(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        x,
        pos_embed: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        for layer in self.layers:
            x = layer(
                x,
                pos_embed=pos_embed,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        norm=None,
    ):
        super().__init__()
        self.layers = clone_layer(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        query_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        self_key_padding_mask: Optional[Tensor] = None,
        cross_key_padding_mask: Optional[Tensor] = None,
    ):
        for layer in self.layers:
            query = layer(
                query,
                memory,
                query_pos=query_pos,
                memory_pos=memory_pos,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                cross_key_padding_mask=cross_key_padding_mask,
            )

        if self.norm is not None:
            query = self.norm(query)

        return query


class ResidualFFN(nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout=0.1, activation='relu'):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x) + x


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        ffn_hidden_dim=1024,
        dropout=0.1,
        activation="relu",
        is_before_norm=False,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
        )
        self.ffn = ResidualFFN(embed_dim, ffn_hidden_dim, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.is_before_norm = is_before_norm

    def with_pos_embed(self, x: Tensor, pos: Optional[Tensor]):
        return x if pos is None else x + pos

    def forward(
        self,
        x: Tensor,
        pos_embed: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        # norm
        if not self.is_before_norm:
            x = self.layer_norm1(x)

        q = k = self.with_pos_embed(x, pos_embed)
        _x = self.self_attn(
            query=q,
            key=k,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        x = x + self.dropout(_x)

        # norm
        if self.is_before_norm:
            x = self.layer_norm1(x)
        else:
            x = self.layer_norm2(x)

        x = self.ffn(x)

        # norm
        if self.is_before_norm:
            x = self.layer_norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_hidden_dim=1024,
        dropout=0.1,
        activation="relu",
        is_before_norm=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
        )

        self.ffn = ResidualFFN(embed_dim, ffn_hidden_dim, dropout, activation)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.is_before_norm = is_before_norm

    def with_pos_embed(self, x: Tensor, pos: Optional[Tensor]):
        return x if pos is None else x + pos

    def forward(
        self,
        query,
        memory,
        query_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        self_key_padding_mask: Optional[Tensor] = None,
        cross_key_padding_mask: Optional[Tensor] = None,
    ):
        # norm
        if not self.is_before_norm:
            query = self.layer_norm1(query)

        q = k = self.with_pos_embed(query, query_pos)
        _query = self.self_attn(
            query=q,
            key=k,
            value=query,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
        )[0]
        query = query + self.dropout1(_query)

        # norm
        if self.is_before_norm:
            query = self.layer_norm1(query)
        else:
            query = self.layer_norm2(query)

        _query = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=cross_attn_mask,
            key_padding_mask=cross_key_padding_mask,
        )[0]
        query = query + self.dropout2(_query)

        # norm
        if self.is_before_norm:
            query = self.layer_norm2(query)
        else:
            query = self.layer_norm3(query)

        query = self.ffn(query)

        # norm
        if self.is_before_norm:
            query = self.layer_norm3(query)

        return query


def clone_layer(layer, num_layers):
    return nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])


def build_transformer(embed_dim: 512):
    return Transformer(
        embed_dim=embed_dim,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        ffn_hidden_dim=1024,
        dropout=0.1,
        activation="relu",
        is_before_norm=False,
    )


def get_activation(activation_name: str):
    if activation_name == "relu":
        return nn.ReLU(inplace=True)
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "glu":
        return nn.GLU()


if __name__ == "__main__":
    encoder_layer = TransformerEncoderLayer(512, 8)
    input = torch.randn((64, 1, 512))
    output = encoder_layer(input)
    print(output.size())  # torch.Size([64, 1, 512])

    decoder_layer = TransformerDecoderLayer(512, 8)
    input = torch.randn((64, 1, 512))
    memory = torch.randn((64, 1, 512))
    output = decoder_layer(input, memory)
    print(output.size())  # torch.Size([64, 1, 512])

    encoder = TransformerEncoder(encoder_layer, 6)
    input = torch.randn((64, 1, 512))
    output = encoder(input)
    print(output.size())  # torch.Size([64, 1, 512])

    decoder = TransformerDecoder(decoder_layer, 6)
    input = torch.randn((64, 1, 512))
    memory = torch.randn((64, 1, 512))
    output = decoder(input, memory)
    print(output.size())  # torch.Size([64, 1, 512])

    transformer = Transformer(512)
    batch_size = 6
    num_queries = 50
    input = torch.randn((batch_size, 512, 8, 8))
    memory_pos = torch.randn((batch_size, 512, 8, 8))
    query_pos = torch.randn((num_queries, 512))
    output = transformer(input, memory_pos, query_pos)
    print(output.size())  # torch.Size([6, 50, 512])
