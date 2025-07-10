import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        """
        Args:
            in_channels: Dataloader가 반환하는 컨텍스트 맵의 채널 수
            out_dim: 최종 컨텍스트 임베딩 차원
        """
        super().__init__()
        self.cnn = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),   # H/2 x W/2

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),   # H/4 x W/4

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 1×1 feature map
        )
        # 마지막에 128채널 → out_dim 임베딩
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) 형태의 텐서
        Returns:
            (B, out_dim) 형태의 컨텍스트 임베딩
        """
        h = self.cnn(x)                # (B, 128, 1, 1)
        h = h.view(h.size(0), -1)      # (B, 128)
        emb = self.fc(h)               # (B, out_dim)
        return emb