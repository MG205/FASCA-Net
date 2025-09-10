import torch
import torch.nn as nn


class TextSentimentModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        input_dim:  (768)
        n_heads: Transformer
        n_layers: Transformer Encoder
        dim_feedforward: Feed-Forward
        dropout: dropout
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim)

        self.pos_emb = nn.Parameter(torch.randn(1, 50, input_dim))

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

    def forward(self, text_inputs: torch.Tensor) -> torch.Tensor:
        """
        text_inputs: Tensor of shape [batch_size, seq_len=50, input_dim=768]
        return: logits of shape [batch_size, num_classes]
        """
        x = self.input_proj(text_inputs) + self.pos_emb  # [B, T, D]

        x = x.transpose(0, 1)  # -> [T, B, D]
        x = self.transformer(x)  # -> [T, B, D]
        x = x.transpose(0, 1)     # -> [B, T, D]

        pooled = x.mean(dim=1)  # -> [B, D]
        logits = self.last(pooled)  # -> [B, num_classes]
        return logits
