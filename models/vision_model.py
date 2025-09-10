import torch
import torch.nn as nn


class VisionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 20,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 50,
    ):
        """
        input_dim:    视觉特征维度（这里为20）
        n_heads:      Transformer 多头注意力头数
        n_layers:     Transformer Encoder 层数
        dim_feedforward: Feed-Forward 中间维度
        dropout:      dropout 概率
        seq_len:      序列长度（帧数，默认50）
        """
        super().__init__()
        # 将原始视觉特征投射到 models 维度
        self.input_proj = nn.Linear(input_dim, input_dim)
        # 可学习的位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, input_dim))
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.last = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1)
        )

    def forward(self, vision_inputs: torch.Tensor) -> torch.Tensor:
        """
        vision_inputs: Tensor of shape [batch_size, seq_len=50, input_dim=20]
        return: Tensor of shape [batch_size, 1]
        """
        x = self.input_proj(vision_inputs) + self.pos_emb  # [B, T, D]
        x = x.transpose(0, 1)                              # [T, B, D]
        x = self.transformer(x)                            # [T, B, D]
        x = x.transpose(0, 1)                              # [B, T, D]
        pooled = x.mean(dim=1)                             # [B, D]
        out = self.last(pooled)                            # [B, 1]
        return out

