import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalBlock(nn.Module):
    """
    Apply a single cross-modal attention where the current modality provides the Query,
    and the concatenation of the remaining two modalities provides Key / Value.

    Internal hidden_dim can be different from each modality's original dimension.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln= nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, query, kv, query_mask=None, kv_mask=None):
        # query / kv: (B, L, hidden_dim)
        attn_out, _ = self.attn(query, kv, kv, key_padding_mask=kv_mask)
        x = self.ln(query + attn_out)          # Residual + LN
        x = x + self.ffn(x)                    # Feed-Forward + Residual
        return x


class SelfAttnBlock(nn.Module):
    """Standard Transformer encoder block (without positional encoding)."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class TriModalFusion(nn.Module):
    """
    End-to-end tri-modal fusion encoder consisting of:
    1. Modality-specific projection → shared hidden_dim
    2. One round of cross-modal attention per modality
    3. One round of self-attention per modality
    4. Modality-specific *inverse* projection that maps the hidden representation back to the
       original feature dimension, followed by a residual addition with the raw input.

    Thanks to the inverse projection, the final output for each modality has **exactly** the
    same shape as its corresponding input tensor.
    """
    def __init__(self,
                 txt_dim: int = 768,
                 aud_dim: int = 5,
                 vis_dim: int = 20,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        # A. Projections to shared space
        self.proj_txt = nn.Linear(txt_dim, hidden_dim)
        self.proj_aud = nn.Linear(aud_dim, hidden_dim)
        self.proj_vis = nn.Linear(vis_dim, hidden_dim)

        # B. Cross-modal attention blocks
        self.txt_cross = CrossModalBlock(hidden_dim, num_heads, dropout)
        self.aud_cross = CrossModalBlock(hidden_dim, num_heads, dropout)
        self.vis_cross = CrossModalBlock(hidden_dim, num_heads, dropout)

        # C. Self-attention blocks
        self.txt_self = SelfAttnBlock(hidden_dim, num_heads, dropout)
        self.aud_self = SelfAttnBlock(hidden_dim, num_heads, dropout)
        self.vis_self = SelfAttnBlock(hidden_dim, num_heads, dropout)

        # D. Inverse projections back to original dimensions
        self.deproj_txt = nn.Linear(hidden_dim, txt_dim)
        self.deproj_aud = nn.Linear(hidden_dim, aud_dim)
        self.deproj_vis = nn.Linear(hidden_dim, vis_dim)

    def forward(self,
                txt_input, txt_mask,
                aud_input, aud_mask,
                vis_input, vis_mask):
        """
        Parameters
        ----------
        txt_input : (B, L, 768)
        aud_input : (B, L, 5)
        vis_input : (B, L, 20)
        *_mask    : (B, L)   # True for *padding* positions

        Returns
        -------
        txt_out  : (B, L, 768)
        aud_out  : (B, L, 5)
        vis_out  : (B, L, 20)
        Shapes are identical to the corresponding inputs.
        """
        # 1. Shared-space projection
        t = self.proj_txt(txt_input)
        a = self.proj_aud(aud_input)
        v = self.proj_vis(vis_input)

        # 2. Build KV banks (concatenate the other two modalities)
        kv_for_t = torch.cat([a, v], dim=1)
        kv_for_a = torch.cat([t, v], dim=1)
        kv_for_v = torch.cat([t, a], dim=1)
        mask_for_t = torch.cat([aud_mask, vis_mask], dim=1)
        mask_for_a = torch.cat([txt_mask, vis_mask], dim=1)
        mask_for_v = torch.cat([txt_mask, aud_mask], dim=1)

        # 3. Cross-modal attention
        t = self.txt_cross(t, kv_for_t, txt_mask, mask_for_t)
        a = self.aud_cross(a, kv_for_a, aud_mask, mask_for_a)
        v = self.vis_cross(v, kv_for_v, vis_mask, mask_for_v)

        # 4. Self-attention
        t = self.txt_self(t, txt_mask)
        a = self.aud_self(a, aud_mask)
        v = self.vis_self(v, vis_mask)

        # 5. Inverse projection + residual (guarantees shape preservation)
        t_out = txt_input + self.deproj_txt(t)
        a_out = aud_input + self.deproj_aud(a)
        v_out = vis_input + self.deproj_vis(v)

        return t_out, a_out, v_out


if __name__ == "__main__":
    B, L = 32, 51
    txt = torch.randn(B, L, 768)
    aud = torch.randn(B, L, 5)
    vis = torch.randn(B, L, 20)
    txt_mask = aud_mask = vis_mask = torch.zeros(B, L, dtype=torch.bool)  # no padding for demo

    model = TriModalFusion()
    out_txt, out_aud, out_vis = model(txt, txt_mask, aud, aud_mask, vis, vis_mask)
    print("Output shapes:", out_txt.shape, out_aud.shape, out_vis.shape)
