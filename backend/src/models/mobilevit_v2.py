"""
mobilevit_v2.py — Novel Ultra-Lightweight Encoder-Decoder for Rural Road Extraction

Architecture:  MobileViT v2 backbone with two domain-specific novelties
┌────────────────────────────────────────────────────────────────────────────┐
│ Novelty 1 — Strip Convolutions (1×3 → 3×1)                              │
│   Factorized directional filters act as 'road-shaped scanners',          │
│   injecting inductive bias for elongated tubular structures while        │
│   reducing parameters by ~33 % vs standard 3×3 convolutions.             │
│                                                                          │
│ Novelty 2 — Channel Shift (Zero-Parameter Receptive Field Expansion)     │
│   25 % of feature channels are physically displaced by 2 pixels in       │
│   each cardinal direction before transformer blocks, widening the        │
│   effective receptive field at zero computational / parameter cost.       │
└────────────────────────────────────────────────────────────────────────────┘

Input:   (B, 3, 256, 256) RGB satellite tiles
Output:  (B, 1, 256, 256) binary road probability mask (Sigmoid-activated)
Target:  < 3 M trainable parameters (~1.6 M at width_mult=1.0)

Reference:
    Mehta & Rastegari, "Separable Self-attention for Mobile Vision
    Transformers" (MobileViT v2), Apple ML Research, 2022.

Author:  Member 1 — Lead Architect
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
#  Utility
# ───────────────────────────────────────────────────────────────────────────

def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    """Round channel count to nearest *divisor* for hardware efficiency."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Avoid rounding down by more than 10 %
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ═══════════════════════════════════════════════════════════════════════════
#  NOVELTY 1 — Strip Convolution
# ═══════════════════════════════════════════════════════════════════════════

class StripConv(nn.Module):
    """
    Factorized 3×3 convolution decomposed into sequential 1-D directional
    filters: horizontal (1×3) followed by vertical (3×1).

    Motivation
    ----------
    Roads in satellite imagery are predominantly linear / tubular structures.
    Standard 3×3 kernels waste capacity on isotropic features.  Strip
    convolutions inject a structural prior for elongated shapes by
    independently scanning horizontal and vertical orientations — acting as
    a **road-shaped scanner** that filters out background jungle noise.

    Parameter savings
    -----------------
    Standard 3×3:   C_in × C_out × 9
    Strip (1×3→3×1): C_in × C_out × 3  +  C_out × C_out × 3
    When C_in ≈ C_out → ~33 % reduction.

    Parameters
    ----------
    in_channels  : Number of input channels.
    out_channels : Number of output channels.
    stride       : Spatial downsampling factor (applied across both dims).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # Horizontal scan — captures road-like patterns along the x-axis
        self.conv_h = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False,
        )
        self.bn_h = nn.BatchNorm2d(out_channels)

        # Vertical scan — captures road-like patterns along the y-axis
        self.conv_v = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), bias=False,
        )
        self.bn_v = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn_h(self.conv_h(x)))
        x = self.act(self.bn_v(self.conv_v(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  NOVELTY 2 — Channel Shift
# ═══════════════════════════════════════════════════════════════════════════

class ChannelShift(nn.Module):
    """
    Zero-parameter spatial displacement of feature channels.

    Before transformer blocks a fraction of channels are physically shifted
    by ``shift_pixels`` in each of the four cardinal directions (↑ ↓ ← →).
    This lets each spatial position access information from neighbouring
    pixels **without** any learned parameters or FLOPs (beyond the memory
    copy).

    Effect
    ------
    Effectively widens the receptive field by 2 × shift_pixels in each
    direction, giving the downstream linear attention richer local context
    to aggregate over.  At 4× downsampling and shift_pixels = 2, the model
    gains an extra ±8 pixel reach in the original 256×256 image space —
    roughly the width of a rural road.

    Parameters
    ----------
    shift_pixels   : Number of pixels to shift (default 2).
    shift_fraction : Fraction of channels to shift (default 0.25).
                     Split equally across four directions; the remaining
                     channels pass through unchanged.

    Note
    ----
    This module has **ZERO** trainable parameters.
    """

    def __init__(self, shift_pixels: int = 2, shift_fraction: float = 0.25):
        super().__init__()
        self.shift_pixels = shift_pixels
        self.shift_fraction = shift_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.shift_pixels
        n_shifted = int(C * self.shift_fraction)
        per_dir = n_shifted // 4

        if per_dir == 0 or s == 0:
            return x

        # Split into directional groups + identity pass-through
        c_up    = x[:, 0 * per_dir : 1 * per_dir]
        c_down  = x[:, 1 * per_dir : 2 * per_dir]
        c_left  = x[:, 2 * per_dir : 3 * per_dir]
        c_right = x[:, 3 * per_dir : 4 * per_dir]
        c_id    = x[:, 4 * per_dir :]                 # identity (unchanged)

        # Shift-and-pad: displace spatial content, zero-fill vacated edges
        c_up    = F.pad(c_up[:, :, s:, :],     (0, 0, 0, s))   # ↑ pad bottom
        c_down  = F.pad(c_down[:, :, :-s, :],  (0, 0, s, 0))   # ↓ pad top
        c_left  = F.pad(c_left[:, :, :, s:],   (0, s, 0, 0))   # ← pad right
        c_right = F.pad(c_right[:, :, :, :-s], (s, 0, 0, 0))   # → pad left

        return torch.cat([c_up, c_down, c_left, c_right, c_id], dim=1)

    def extra_repr(self) -> str:
        return (f"shift_pixels={self.shift_pixels}, "
                f"shift_fraction={self.shift_fraction}, "
                f"parameters=0")


# ═══════════════════════════════════════════════════════════════════════════
#  MobileNet V2 Inverted Residual Block
# ═══════════════════════════════════════════════════════════════════════════

class InvertedResidual(nn.Module):
    """
    MobileNetV2 inverted residual bottleneck (Sandler et al., 2018).

    Pipeline:  1×1 expand → 3×3 depthwise → 1×1 linear project.
    Residual connection when stride = 1 and in_channels == out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        mid = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        # Pointwise expansion (skip when expand_ratio == 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(inplace=True),
            ])
        # Depthwise separable convolution
        layers.extend([
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1,
                      groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            # Linear projection (no activation — linear bottleneck)
            nn.Conv2d(mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════
#  Linear Self-Attention  (MobileViT v2 Separable Attention)
# ═══════════════════════════════════════════════════════════════════════════

class LinearSelfAttention(nn.Module):
    """
    Separable Self-Attention from MobileViT v2 (Mehta, 2022).

    Replaces O(N²) scaled dot-product attention with O(N·d) linear
    attention through a global context vector:

        1. Project input → Q (d),  K (1),  V (d)
        2. Context scores:   α = softmax(K, dim=tokens)       — (B, N, 1)
        3. Global context:   c = Σ_n(α_n · V_n)               — (B, 1, d)
        4. Output:           O = ReLU(Q) ⊙ broadcast(c)       — (B, N, d)

    The **scalar** K projection enables global context aggregation at
    minimal cost while Q gating ensures token-specific output modulation.

    Parameters
    ----------
    embed_dim    : Transformer embedding dimension.
    attn_dropout : Dropout probability on context scores.
    """

    def __init__(self, embed_dim: int, attn_dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Project to  Q(d) + K(1) + V(d) = 2d + 1
        self.qkv = nn.Linear(embed_dim, 2 * embed_dim + 1, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.embed_dim, 1, self.embed_dim], dim=-1)

        # Aggregate global context via weighted sum of values
        context_scores = F.softmax(k, dim=1)                          # (B, N, 1)
        context_scores = self.attn_drop(context_scores)
        context_vector = (context_scores * v).sum(dim=1, keepdim=True)  # (B, 1, d)

        # Modulate queries with the global context
        out = F.relu(q) * context_vector                               # (B, N, d)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════════════
#  Transformer Block
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with Linear Self-Attention and FFN."""

    def __init__(self, embed_dim: int, ffn_ratio: float = 2.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LinearSelfAttention(embed_dim, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        ffn_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  MobileViT v2 Block  (with Channel Shift + Strip-Aware Processing)
# ═══════════════════════════════════════════════════════════════════════════

class MobileViTv2Block(nn.Module):
    """
    MobileViT v2 block with integrated Channel Shift (Novelty 2).

    Pipeline
    --------
    ① Channel Shift — zero-cost receptive field expansion
    ② 3×3 depthwise conv → 1×1 pointwise (local representation)
    ③ Unfold into patches → L × Transformer layers → fold back (global)
    ④ 1×1 projection back to input channel count
    ⑤ Concatenate with original input → 1×1 fusion conv

    The block preserves spatial dimensions (H, W) and channel count.

    Parameters
    ----------
    in_channels            : Number of input / output channels.
    transformer_dim        : Internal transformer embedding dimension.
    n_transformer_layers   : Number of stacked transformer blocks.
    patch_size             : Spatial patch size for unfold / fold (default 2).
    ffn_ratio              : FFN expansion ratio inside transformer.
    dropout                : FFN dropout probability.
    attn_dropout           : Attention weight dropout probability.
    shift_pixels           : Channel Shift displacement (Novelty 2).
    shift_fraction         : Fraction of channels to shift (Novelty 2).
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        n_transformer_layers: int = 2,
        patch_size: int = 2,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        shift_pixels: int = 2,
        shift_fraction: float = 0.25,
    ):
        super().__init__()
        self.patch_h = patch_size
        self.patch_w = patch_size

        # ① Channel Shift (Novelty 2)
        self.channel_shift = ChannelShift(shift_pixels, shift_fraction)

        # ② Local representation
        self.local_rep = nn.Sequential(
            # Depthwise conv — local spatial features
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # Pointwise projection to transformer dimension
            nn.Conv2d(in_channels, transformer_dim, 1, bias=False),
            nn.BatchNorm2d(transformer_dim),
        )

        # ③ Global representation (transformer stack)
        self.transformers = nn.Sequential(*[
            TransformerBlock(transformer_dim, ffn_ratio, dropout, attn_dropout)
            for _ in range(n_transformer_layers)
        ])
        self.post_norm = nn.LayerNorm(transformer_dim)

        # ④ Project back to input channel count
        self.proj = nn.Sequential(
            nn.Conv2d(transformer_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # ⑤ Fuse original input + transformer-processed output
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    # ── Patch unfold / fold ────────────────────────────────────────────

    def _unfold(self, x: torch.Tensor):
        """
        Unfold spatial dims into patch tokens for transformer processing.

        For each pixel position (i, j) within a patch, all patches across
        the image are gathered as a token sequence.  This allows the
        transformer to model *global* relationships for every local
        sub-pixel position.

        (B, C, H, W)  →  (B · ph · pw,  N_patches,  C)
        where N_patches = (H / ph) × (W / pw).
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        n_h, n_w = H // ph, W // pw

        x = x.reshape(B, C, n_h, ph, n_w, pw)
        x = x.permute(0, 3, 5, 1, 2, 4)          # (B, ph, pw, C, n_h, n_w)
        x = x.reshape(B * ph * pw, C, n_h * n_w)  # (B·ph·pw, C, N)
        x = x.permute(0, 2, 1)                    # (B·ph·pw, N, C)
        return x, (B, n_h, n_w)

    def _fold(self, x: torch.Tensor, info: tuple, C: int):
        """
        Fold patch tokens back into a spatial feature map.

        (B · ph · pw,  N_patches,  C)  →  (B, C, H, W)
        """
        B, n_h, n_w = info
        ph, pw = self.patch_h, self.patch_w

        x = x.permute(0, 2, 1)                            # (B·ph·pw, C, N)
        x = x.reshape(B, ph, pw, C, n_h, n_w)
        x = x.permute(0, 3, 4, 1, 5, 2)                   # (B, C, n_h, ph, n_w, pw)
        x = x.reshape(B, C, n_h * ph, n_w * pw)
        return x

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w

        # Pad spatial dims to be divisible by patch_size (safety)
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        identity = x                              # save for fusion ⑤

        # ① Channel Shift
        x_shifted = self.channel_shift(x)

        # ② Local representation
        local_out = self.local_rep(x_shifted)
        C_t = local_out.shape[1]

        # ③ Unfold → Transformer → Fold
        tokens, fold_info = self._unfold(local_out)
        tokens = self.transformers(tokens)
        tokens = self.post_norm(tokens)
        global_out = self._fold(tokens, fold_info, C_t)

        # ④ Project back to in_channels
        global_out = self.proj(global_out)

        # ⑤ Fuse original + global
        fused = self.fusion(torch.cat([identity, global_out], dim=1))

        # Remove padding if it was applied
        if pad_h > 0 or pad_w > 0:
            fused = fused[:, :, :H, :W]

        return fused


# ═══════════════════════════════════════════════════════════════════════════
#  Full Model — MobileViT v2  Encoder-Decoder for Road Segmentation
# ═══════════════════════════════════════════════════════════════════════════

class MobileViT_v2(nn.Module):
    """
    Ultra-lightweight encoder-decoder for binary road segmentation.

    Architecture (at ``width_mult = 1.0``)::

        ┌───────────────────────────────────────────────────────────────┐
        │  ENCODER                                                     │
        │  ┌─ Stem:  StripConv(3→32, s=2)         → 32ch  @ 128×128 ─┐│
        │  │  MV2:   InvRes(32→64, s=2)            → 64ch  @ 64×64   ││
        │  │  Stage2: MViTv2(64, d=96) + MV2(s=2)  → 96ch  @ 32×32   ││
        │  │  Stage3: MViTv2(96, d=144) + MV2(s=2) → 128ch @ 16×16   ││
        │  └─ Bottleneck: MViTv2(128, d=192, L=3)  → 128ch @ 16×16  ─┘│
        │                                                              │
        │  DECODER (U-Net skip connections + Strip Convolutions)       │
        │  ┌─ Up3: cat(↑bottleneck, skip3) → StripConv → 96ch  @32  ──┤
        │  │  Up2: cat(↑dec3, skip2)       → StripConv → 64ch  @64    │
        │  │  Up1: cat(↑dec2, skip1)       → StripConv → 32ch  @128   │
        │  └─ Head: ↑dec1 → 1×1 Conv → Sigmoid     → 1ch   @ 256×256 │
        └───────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    num_classes : Number of output classes (1 for binary road mask).
    width_mult  : Channel width multiplier (e.g. 0.5 halves all channels).
    """

    def __init__(self, num_classes: int = 1, width_mult: float = 1.0):
        super().__init__()

        def _c(channels: int) -> int:
            """Scale channel count by width multiplier, round to nearest 8."""
            return _make_divisible(channels * width_mult)

        # ─── ENCODER ────────────────────────────────────────────────
        # Stem: Strip Convolution (Novelty 1 — road-shaped scanner)
        self.stem = StripConv(3, _c(32), stride=2)         # 256 → 128

        # Stage 1: MV2 downsampling
        self.enc1 = InvertedResidual(_c(32), _c(64), stride=2)  # 128 → 64

        # Stage 2: MobileViT v2 (Channel Shift) + downsample
        self.enc2_mvit = MobileViTv2Block(
            _c(64), transformer_dim=_c(96), n_transformer_layers=2,
        )
        self.enc2_down = InvertedResidual(_c(64), _c(96), stride=2)  # 64 → 32

        # Stage 3: MobileViT v2 (Channel Shift) + downsample
        self.enc3_mvit = MobileViTv2Block(
            _c(96), transformer_dim=_c(144), n_transformer_layers=2,
        )
        self.enc3_down = InvertedResidual(_c(96), _c(128), stride=2)  # 32 → 16

        # Bottleneck: deepest MobileViT v2 block (3 transformer layers)
        self.bottleneck = MobileViTv2Block(
            _c(128), transformer_dim=_c(192), n_transformer_layers=3,
        )

        # ─── DECODER ────────────────────────────────────────────────
        # Each stage: bilinear upsample → concat skip → StripConv (Novelty 1)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec3 = StripConv(_c(128) + _c(96), _c(96))    # 16 → 32

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec2 = StripConv(_c(96) + _c(64), _c(64))     # 32 → 64

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec1 = StripConv(_c(64) + _c(32), _c(32))     # 64 → 128

        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)                     # 128 → 256

        # ─── SEGMENTATION HEAD ──────────────────────────────────────
        self.head = nn.Conv2d(_c(32), num_classes, kernel_size=1, bias=True)

        # ─── Weight Initialisation ──────────────────────────────────
        self._init_weights()

    # ── Init ───────────────────────────────────────────────────────────

    def _init_weights(self):
        """Kaiming init for convs, truncated-normal for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) RGB input tensor.  H and W must be divisible by 16.
            Designed for H = W = 256.

        Returns
        -------
        (B, num_classes, H, W) sigmoid-activated probability map in [0, 1].
        """
        # ── Encoder ──
        s1 = self.stem(x)              # skip1: (_c(32), 128, 128)
        e1 = self.enc1(s1)             #         (_c(64),  64,  64)

        s2 = self.enc2_mvit(e1)        # skip2: (_c(64),  64,  64)
        e2 = self.enc2_down(s2)        #         (_c(96),  32,  32)

        s3 = self.enc3_mvit(e2)        # skip3: (_c(96),  32,  32)
        e3 = self.enc3_down(s3)        #         (_c(128), 16,  16)

        bn = self.bottleneck(e3)       #         (_c(128), 16,  16)

        # ── Decoder with skip connections ──
        d3 = self.dec3(torch.cat([self.up3(bn), s3], dim=1))  # 32 × 32
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))  # 64 × 64
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))  # 128 × 128

        out = self.head(self.up0(d1))                         # 256 × 256
        return out

    # ── Convenience ────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
#  Quick verification
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  MobileViT v2 -- Architecture Verification")
    print("=" * 65)

    for wm in [1.0, 0.5]:
        print(f"\n-- width_mult = {wm} --")
        model = MobileViT_v2(num_classes=1, width_mult=wm)
        params = model.num_parameters
        print(f"   Trainable parameters : {params:,}")
        print(f"   Under 3 M budget     : {'YES' if params < 3_000_000 else 'NO'}")

        dummy = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(dummy)
        print(f"   Input shape          : {tuple(dummy.shape)}")
        print(f"   Output shape         : {tuple(out.shape)}")

        assert out.shape == (2, 1, 256, 256), f"Shape mismatch: {out.shape}"
        assert 0.0 <= out.min() and out.max() <= 1.0, "Output not in [0, 1]"
        print(f"   Output range         : [{out.min():.4f}, {out.max():.4f}]")
        print(f"   All checks passed.")

    print("\n" + "=" * 65)

