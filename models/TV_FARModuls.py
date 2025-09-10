import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
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
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden*2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden*2, hidden*2, 3, padding=1, groups=hidden*2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = torch.nn.functional.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, stride=1, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1))
        self.stride = stride
        self.qk = nn.Conv2d(dim, dim*2, 1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim*2, dim*2, 3, stride=stride, padding=1, groups=dim*2, bias=bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qk = self.qk_dwconv(self.qk(x))
        q, k = qk.chunk(2, dim=1)
        v = self.v_dwconv(self.v(x))
        # reshape for attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2,-1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class BFA(nn.Module):
    def __init__(self, dim, num_heads=4, stride=1, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim, num_heads, stride, bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RBFM(nn.Module):
    def __init__(self, dim, stride=1):
        super().__init__()
        self.encoder = nn.Sequential(*[BFA(dim, num_heads=1, stride=stride) for _ in range(2)])
        self.fuse = nn.Conv2d(dim*2, dim, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.size()
        feat = self.encoder(x)
        ref = feat[0:1].expand(B, -1, -1, -1)
        cat = torch.cat([ref, feat], dim=1)
        return self.fuse(cat)


class FARModuls(nn.Module):
    def __init__(self, dim=64, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # Bottleneck for fused features
        self.bottleneck = nn.Conv2d(dim*2, dim, 3, padding=1)
        # Deformable convolution with modulated offsets
        self.deform = DeformConv2d(dim, dim, kernel_size, padding=kernel_size//2, dilation=1)
        self.backproj = RBFM(dim)
        # Offset conv: generate offsets and modulation masks
        out_channels = 3 * (kernel_size * kernel_size)
        self.offset_conv = nn.Conv2d(dim*2, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        B, C, H, W = x.size()
        # concatenate reference and current feature along channel dim
        ref = x[0:1].expand(B, -1, -1, -1)
        concat = torch.cat([ref, x], dim=1)  # channels: 2*dim
        # generate offsets and mask
        o = self.offset_conv(concat)   # [B, 3*k*k, H, W]
        # split into offset x, offset y, and mask (each k*k)
        k2 = self.kernel_size * self.kernel_size
        o1, o2, mask = torch.split(o, k2, dim=1)
        offset = torch.cat([o1, o2], dim=1)  # [B, 2*k*k, H, W]
        mask = mask.sigmoid()             # [B,   k*k, H, W]
        # apply modulated deformable convolution
        aligned= self.deform(x, offset, mask)
        # ensure first frame remains unchanged
        aligned[0]= x[0]
        # project back
        aligned = self.backproj(aligned)
        return aligned


class FGFA_TV(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # project input1 (768-dim) and input2 (20-dim) to 'dim' channels
        self.proj1 = nn.Conv2d(768, dim, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(20,  dim, kernel_size=1, bias=False)
        self.fuse = nn.Conv2d(dim*2, dim, kernel_size=1, bias=False)
        self.align = FARModuls(dim, kernel_size=3)  # using 3x3 deform kernel

    def forward(self, input1, input2):

        x1 = input1.permute(0, 3, 2, 1)  # -> (B, 768, 1, 51)
        x2 = input2.permute(0, 3, 2, 1)  # -> (B, 20, 1, 51)
        p1 = self.proj1(x1)             # -> (B, dim, 1, 51)
        p2 = self.proj2(x2)             # -> (B, dim, 1, 51)
        fused = torch.cat([p1, p2], dim=1)  # -> (B, 2*dim, 1, 51)
        fused = self.fuse(fused)
        aligned = self.align(fused)
        return aligned
