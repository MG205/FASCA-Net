import torch
import torch.nn as nn


class AudioSentimentModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        seq_len: int = 50
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.last = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        """
        audio_inputs: [batch_size, seq_len=50, input_dim=5]
        return:       [batch_size, 1]
        """
        x = self.input_proj(audio_inputs) + self.pos_emb   # [B, T, hidden_dim]
        x = x.transpose(0, 1)                              # -> [T, B, hidden_dim]
        x = self.transformer(x)                            # -> [T, B, hidden_dim]
        x = x.transpose(0, 1)                              # -> [B, T, hidden_dim]
        pooled = x.mean(dim=1)                             # -> [B, hidden_dim]
        out = self.last(pooled)                           # -> [B, 1]
        return out

