from typing import Optional
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        self.pe = pe 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (L, B, d_model)
        Returns:
            x + positional encoding, same shape
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TrajectoryEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 n_head: int,
                 num_layers: int = 3,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        """
        Args:
            input_dim:       per-step feature 수 (F)
            d_model:         Transformer 임베딩 차원
            n_head:          multi-head attention 헤드 수
            num_layers:      TransformerEncoderLayer 개수
            dim_feedforward: FFN 내부 차원
            dropout:         드롭아웃 확률
            max_len:         위치 인코딩 최대 시퀀스 길이
        """
        super().__init__()
        # 1) Input projection: F -> d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        # 2) Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        # 3) Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, 
                x: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_dim)
            src_key_padding_mask: (B, L) 의 boolean mask,
                                  True인 위치는 패딩으로 처리
        Returns:
            out: (B, L, d_model)
        """
        # 1) 선형 투영
        x = self.input_proj(x)           # (B, L, d_model)
        # 2) Transformer expects (L, B, E)
        x = x.transpose(0, 1)            # (L, B, d_model)
        # 3) 위치 인코딩
        x = self.pos_enc(x)              # (L, B, d_model)
        # 4) 인코더 통과
        out = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )                                 # (L, B, d_model)
        # 5) 원래 형태로 복귀
        return out.transpose(0, 1)       # (B, L, d_model)
    
    