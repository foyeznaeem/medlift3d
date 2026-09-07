"""2-D axial-slice UNet used as the anatomical prior.

A 2-D prior rather than a 3-D one is the decision that makes this project fit on
a single 12 GB GPU: ~7 GB and a day of training instead of 16-40 GB and a week,
and it removes a 3-D autoencoder from the critical path entirely. 3-D coherence
comes from the physics -- rays cross slices, so the projector couples them --
plus an explicit z-direction TV term in the solver.

Conditioning is by **channel concatenation of a spatially aligned volume** (a
CGLS or FBP initialisation), not by cross-attention to a pooled descriptor. A
globally average-pooled projection encoding cannot localise a nodule: it says
roughly "a chest with about this much total attenuation" and nothing about
where anything is. An aligned conditioning channel says exactly where.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_ch: int = 1                # noisy slice
    cond_ch: int = 1              # aligned conditioning slice(s); 0 disables
    out_ch: int = 1
    base_dim: int = 64
    dim_mults: tuple[int, ...] = (1, 2, 4, 8)
    attn_at: tuple[int, ...] = (2, 3)   # levels (0-indexed) that get attention
    groups: int = 8
    dropout: float = 0.0

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(d):
        d = dict(d)
        for k in ("dim_mults", "attn_at"):
            if k in d:
                d[k] = tuple(d[k])
        return UNetConfig(**d)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, groups=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, dim_in), dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.time = nn.Linear(time_dim, dim_out * 2)   # scale-shift conditioning
        self.norm2 = nn.GroupNorm(min(groups, dim_out), dim_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.skip = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(t))[..., None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Applied only at coarse levels -- attention is O(N^2) in pixels."""

    def __init__(self, dim, heads=4, groups=8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(groups, dim), dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        shape = (b, self.heads, c // self.heads, h * w)
        q, k, v = (t.reshape(*shape).transpose(-2, -1) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(o)


class UNet2D(nn.Module):
    """Noise estimator. Returns the same spatial shape it is given."""

    def __init__(self, cfg: UNetConfig | None = None, **kw):
        super().__init__()
        self.cfg = cfg or UNetConfig(**kw)
        c = self.cfg
        time_dim = c.base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(c.base_dim),
            nn.Linear(c.base_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        dims = [c.base_dim * m for m in c.dim_mults]
        self.stem = nn.Conv2d(c.in_ch + c.cond_ch, dims[0], 3, padding=1)

        self.downs = nn.ModuleList()
        chans = [dims[0]]
        cur = dims[0]
        for lvl, d in enumerate(dims):
            blocks = nn.ModuleList([
                ResBlock(cur, d, time_dim, c.groups, c.dropout),
                ResBlock(d, d, time_dim, c.groups, c.dropout),
                SelfAttention2d(d, groups=c.groups) if lvl in c.attn_at else nn.Identity(),
            ])
            down = nn.Conv2d(d, d, 3, stride=2, padding=1) if lvl < len(dims) - 1 else nn.Identity()
            self.downs.append(nn.ModuleList([blocks, down]))
            chans.append(d)
            cur = d

        self.mid1 = ResBlock(cur, cur, time_dim, c.groups, c.dropout)
        self.mid_attn = SelfAttention2d(cur, groups=c.groups)
        self.mid2 = ResBlock(cur, cur, time_dim, c.groups, c.dropout)

        self.ups = nn.ModuleList()
        for lvl, d in reversed(list(enumerate(dims))):
            blocks = nn.ModuleList([
                ResBlock(cur + chans.pop(), d, time_dim, c.groups, c.dropout),
                ResBlock(d, d, time_dim, c.groups, c.dropout),
                SelfAttention2d(d, groups=c.groups) if lvl in c.attn_at else nn.Identity(),
            ])
            up = nn.Upsample(scale_factor=2, mode="nearest") if lvl > 0 else nn.Identity()
            post = nn.Conv2d(d, d, 3, padding=1) if lvl > 0 else nn.Identity()
            self.ups.append(nn.ModuleList([blocks, up, post]))
            cur = d

        # The stem skip is still unconsumed at this point; a final ResBlock uses it.
        self.final_res = ResBlock(cur + chans.pop(), cur, time_dim, c.groups, c.dropout)
        self.out_norm = nn.GroupNorm(min(c.groups, cur), cur)
        self.out_conv = nn.Conv2d(cur, c.out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, t, cond=None):
        """x: [B, in_ch, H, W]; t: [B]; cond: [B, cond_ch, H, W] or None.

        `cond=None` with `cond_ch > 0` feeds zeros, which is the unconditional
        branch used for classifier-free guidance.
        """
        if self.cfg.cond_ch:
            if cond is None:
                cond = torch.zeros(x.shape[0], self.cfg.cond_ch, *x.shape[-2:],
                                   device=x.device, dtype=x.dtype)
            x = torch.cat([x, cond], dim=1)
        temb = self.time_mlp(t)

        h = self.stem(x)
        skips = [h]
        for (blocks, down) in self.downs:
            res1, res2, attn = blocks
            h = attn(res2(res1(h, temb), temb))
            skips.append(h)
            h = down(h)

        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)

        for (blocks, up, post) in self.ups:
            res1, res2, attn = blocks
            h = torch.cat([h, skips.pop()], dim=1)
            h = attn(res2(res1(h, temb), temb))
            h = post(up(h))

        h = self.final_res(torch.cat([h, skips.pop()], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))
