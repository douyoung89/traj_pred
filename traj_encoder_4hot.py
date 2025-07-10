from typing import Optional
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        self.pe = pe 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class FourHotTrajectoryEncoder(nn.Module):
    def __init__(
        self,
        lat_size: int,
        lon_size: int,
        sog_size: int,
        cog_size: int,
        n_lat_embd: int,
        n_lon_embd: int,
        n_sog_embd: int,
        n_cog_embd: int,
        n_head: int,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        # register bin sizes for bucketization
        self.register_buffer(
            'att_sizes', torch.tensor([lat_size, lon_size, sog_size, cog_size])
        )
        # embedding layers per attribute
        self.lat_emb = nn.Embedding(lat_size, n_lat_embd)
        self.lon_emb = nn.Embedding(lon_size, n_lon_embd)
        self.sog_emb = nn.Embedding(sog_size, n_sog_embd)
        self.cog_emb = nn.Embedding(cog_size, n_cog_embd)

        # combined model dimension
        d_model = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
        self.d_model = d_model

        # positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        # transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, 4) 연속값으로 정규화된 [lat, lon, sog, cog]
            src_key_padding_mask: (B, L) boolean mask, True는 패딩
        Returns:
            out: (B, L, d_model)
        """
        # bucketize -> (B, L, 4) long indices
        idxs = (x * self.att_sizes).long()
        # embedding lookup
        lat_e = self.lat_emb(idxs[..., 0])  # (B, L, n_lat_embd)
        lon_e = self.lon_emb(idxs[..., 1])  # (B, L, n_lon_embd)
        sog_e = self.sog_emb(idxs[..., 2])  # (B, L, n_sog_embd)
        cog_e = self.cog_emb(idxs[..., 3])  # (B, L, n_cog_embd)
        # concat into four-hot embedding
        token_e = torch.cat([lat_e, lon_e, sog_e, cog_e], dim=-1)  # (B, L, d_model)

        # prepare for transformer: (L, B, d_model)
        feat = token_e.transpose(0, 1)
        # add positional encoding
        feat = self.pos_enc(feat)
        # apply transformer encoder
        out = self.transformer(feat, src_key_padding_mask=src_key_padding_mask)
        # back to (B, L, d_model)
        return out.transpose(0, 1)
