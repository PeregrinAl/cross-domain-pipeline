import torch
import torch.nn as nn


class RawEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, L]
        returns: [B, embedding_dim]
        """
        x = self.features(x)
        x = self.head(x)
        return x