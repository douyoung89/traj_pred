import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        """
        하나의 Fusion block:
          1) Cross-Attention (Query=traj, Key/Value=context)
          2) Position-wise FFN
        Args:
            d_model: embedding 차원
            n_head: multi-head attention 헤드 개수
            dim_feedforward: FFN 내부 hidden 차원
            dropout: dropout 확률
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self,
                traj: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj:    (B, L, d_model) 과거 궤적 임베딩 시퀀스
            context: (B, Nc, d_model) 컨텍스트 임베딩 토큰들
                     (e.g. Nc=1 이면 shape (B,1,d_model))
        Returns:
            (B, L, d_model) Cross‐Attention 후 Residual+FFN 처리된 traj
        """
        # 1) Cross-Attn
        # PyTorch MHA는 (S, B, E) 포맷 사용
        q = self.norm1(traj).transpose(0, 1)      # → (L, B, E)
        k = context.transpose(0, 1)               # → (Nc, B, E)
        v = context.transpose(0, 1)               # → (Nc, B, E)
        attn_out, _ = self.attn(q, k, v)          # → (L, B, E)
        attn_out = attn_out.transpose(0, 1)       # → (B, L, E)
        x = traj + attn_out                       # residual

        # 2) FFN
        y = self.ffn(self.norm2(x))
        return x + y                               # residual


class FusionEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_layers: int = 2):
        """
        여러 겹의 FusionLayer를 쌓아 컨텍스트와 궤적을 융합
        Args:
            d_model: embedding 차원
            n_head: multi-head 헤드 수
            dim_feedforward: FFN 내부 차원
            dropout: dropout 확률
            num_layers: FusionLayer 블록 개수
        """
        super().__init__()
        self.layers = nn.ModuleList([
            FusionLayer(d_model, n_head, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self,
                traj_emb: torch.Tensor,
                context_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj_emb:    (B, L, d_model)
            context_emb: (B, d_model) 또는 (B, Nc, d_model)
        Returns:
            (B, L, d_model) 융합된 궤적 표현
        """
        # context_emb이 2D이면 sequence dim 추가
        if context_emb.dim() == 2: # (B, out_dim)
            context = context_emb.unsqueeze(1)  # → (B,1,d_model)
        else:
            context = context_emb               # → (B,Nc,d_model)

        output = traj_emb
        for layer in self.layers:
            output = layer(output, context)
        return output
        