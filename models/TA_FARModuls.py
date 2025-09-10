import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from einops import rearrange
import numbers


def to_3d(x):
    # [B, C, H, W] -> [B, H*W, C]
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    # [B, H*W, C] -> [B, C, H, W]
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    def forward(self, x):
        # x: [..., C]
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu    = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, type='WithBias'):
        super().__init__()
        if type == 'BiasFree':
            self.norm = BiasFree_LayerNorm(dim)
        else:
            self.norm = WithBias_LayerNorm(dim)

    def forward(self, x):
        # apply to last (H*W) dimension
        b, c, h, w = x.shape
        y = to_3d(x)         # [B, H*W, C]
        y = self.norm(y)     # normalized
        return to_4d(y, h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,
                                     padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.project_in(x)
        a, b = self.dwconv(x1).chunk(2, dim=1)
        x2 = F.gelu(a) * b
        return self.project_out(x2)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, stride=1, bias=False):
        super().__init__()
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.stride = stride
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                               stride=stride, padding=1,
                               groups=dim * 2, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dw = nn.Conv2d(dim, dim, kernel_size=3,
                              padding=1, groups=dim, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qk = self.qk_dw(self.qk(x))
        q, k = qk.chunk(2, dim=1)
        v = self.v_dw(self.v(x))

        # reshape for multi-head
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(-1)

        out = attn @ v
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.num_heads, x=h, y=w)
        return self.proj_out(out)


class BFA(nn.Module):
    def __init__(self, dim, heads=4, stride=1, ffn_factor=2.66, bias=False, ln_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, ln_type)
        self.attn = Attention(dim, heads, stride, bias)
        self.norm2 = LayerNorm(dim, ln_type)
        self.ffn = FeedForward(dim, ffn_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class RBFM(nn.Module):
    def __init__(self, in_ch, stride=1):
        super().__init__()
        bias = False
        self.encoder1 = nn.Sequential(*[
            BFA(in_ch, heads=1, stride=stride, ffn_factor=2.66, bias=bias, ln_type='WithBias')
            for _ in range(2)
        ])
        self.feat_fuse = nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1, bias=bias)
        self.feat_expand = nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, padding=1, bias=bias)
        self.diff_fuse = nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        B, f, H, W = x.shape
        feat = self.encoder1(x)
        ref = feat[0:1].expand(B, -1, -1, -1)
        feat = torch.cat([ref, feat], dim=1)
        fused = F.gelu(self.feat_fuse(feat))
        expd = F.gelu(self.feat_expand(fused))
        resid = expd - feat
        resid = F.gelu(self.diff_fuse(resid))
        return fused + resid


class FARModuls(nn.Module):
    def __init__(self, dim, memory=True, stride=1):
        super().__init__()
        bias = False
        kp, pad, dilation = 3, 1, 2
        dg = 1  # ensure in_channels % groups == 0
        out_ch = dg * 3 * (kp ** 2)

        self.offset_conv = nn.Conv2d(dim, out_ch, kernel_size=kp, padding=pad, bias=bias)
        self.deform = DeformConv2d(dim, dim, kernel_size=kp,
                                        padding=dilation,
                                        groups=dg,
                                        dilation=dilation)
        self.back_proj = RBFM(dim, stride=stride)
        self.bottleneck = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=bias)
        if memory:
            self.bottleneck_o = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=bias)

    def offset_gen(self, x):
        o1, o2, m = x.chunk(3, dim=1)
        return torch.cat([o1, o2], dim=1), torch.sigmoid(m)

    def forward(self, x, prev=None):
        B, f, H, W = x.shape
        ref = x[0:1].expand(B, -1, -1, -1)
        feat = self.bottleneck(torch.cat([ref, x], dim=1))
        if prev is not None:
            feat = self.bottleneck_o(torch.cat([prev, feat], dim=1))

        off, mask = self.offset_gen(self.offset_conv(feat))
        aligned  = self.deform(x, off, mask)
        aligned[0] = x[0]
        return self.back_proj(aligned)

# Fine-Grained Feature Alignment Mechanism
class FGFA_TA(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        # project input2 -> match dim1
        self.proj2  = nn.Conv2d(dim2, dim1, kernel_size=1, bias=False)
        self.fuse   = nn.Conv2d(dim1 * 2, dim1, kernel_size=1, bias=False)
        self.farmod = FARModuls(dim1, memory=False, stride=1)

    def forward(self, x1, x2):
        # x1: [B, dim1, 1, W], x2: [B, dim2, 1, w2]
        p2 = self.proj2(x2)  # -> [B, dim1, 1, w2]
        # upsample to match x1 width
        p2 = F.interpolate(p2, size=(1, x1.size(3)),
                           mode='bilinear', align_corners=False)
        cat = torch.cat([x1, p2], dim=1)   # [B, 2*dim1, 1, W]
        ftd = self.fuse(cat)              # [B, dim1, 1, W]
        return self.farmod(ftd)



