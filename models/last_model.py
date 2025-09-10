import torch
import torch.nn as nn
import torch.nn.functional as F


class FCRegressor(nn.Module):
    """
    输入:  (batch_size, 793)
    输出:  (batch_size, 1)
    """
    def __init__(self, dropout=0.2, use_sigmoid=False):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(819, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
