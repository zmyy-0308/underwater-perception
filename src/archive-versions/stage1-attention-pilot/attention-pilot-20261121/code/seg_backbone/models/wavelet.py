r"""
小波域模块：Haar DWT / IDWT + Wavelet Attention

对应你的图中的分支:
    Local Feature ──► DWT ──► LL / LH,HL / HH ──► Wavelet Attention ──► (IDWT 回到原分辨率)

设计要点（针对蓝光弱光 + 细长管道）:
  1. LL 子带承载"结构 + 照度"，HF 子带承载"边缘 + 噪声"；
     因此用 **LL 上下文门控 HF**（而非对四个子带做无差别注意力）。
  2. HF 做 **噪声自适应软阈值**：阈值 λ 由 HF 自身能量预测（弱光下图块间噪声差异大，
     固定阈值不合适）。这等价于可学习的"小波去噪"，对海雪/弱光噪声有效。
  3. 逆变换保持几何一致（Haar 正交，IDWT 是 DWT 的转置），不做上采样插值。

提供两种模式以支持你的消融表:
  · 'plain'     : 只做子带投影 + 逆变换（对应消融表里的"小波"）
  · 'attention' : 上述 + LL 门控 + 自适应软阈值（对应"小波+Attention"）
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Haar 小波正/逆变换（纯张量运算，支持奇数尺寸：先反射填充到偶数）
# ---------------------------------------------------------------------------
def _pad_even(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    H, W = x.shape[-2:]
    ph, pw = H % 2, W % 2
    if ph or pw:
        # Reflection requires at least two pixels along a padded dimension.
        mode = 'reflect' if H > 1 and W > 1 else 'replicate'
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, ph, pw


def haar_dwt2d(x: torch.Tensor):
    """二维 Haar 一级正变换。返回 LL, LH, HL, HH，尺寸各为 (H//2, W//2)。"""
    x, _, _ = _pad_even(x)
    x_even, x_odd = x[:, :, :, 0::2], x[:, :, :, 1::2]
    col_l, col_h = (x_even + x_odd) * 0.5, (x_even - x_odd) * 0.5
    LL = (col_l[:, :, 0::2, :] + col_l[:, :, 1::2, :]) * 0.5
    LH = (col_l[:, :, 0::2, :] - col_l[:, :, 1::2, :]) * 0.5
    HL = (col_h[:, :, 0::2, :] + col_h[:, :, 1::2, :]) * 0.5
    HH = (col_h[:, :, 0::2, :] - col_h[:, :, 1::2, :]) * 0.5
    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH):
    """二维 Haar 一级逆变换，返回 (2H, 2W)。"""
    rL, rH = LL + LH, HL + HH
    rL2, rH2 = LL - LH, HL - HH
    B, C, Hh, Wh = LL.shape
    out = torch.empty(B, C, Hh * 2, Wh * 2, device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = rL + rH
    out[:, :, 0::2, 1::2] = rL - rH
    out[:, :, 1::2, 0::2] = rL2 + rH2
    out[:, :, 1::2, 1::2] = rL2 - rH2
    return out


def soft_threshold(x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """软阈值: sign(x) * max(|x| - lam, 0)，lam 可为逐通道/逐图张量。"""
    return torch.sign(x) * F.relu(x.abs() - lam)


def _cbr(cin, cout, k=1, s=1, g=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class PlainWaveletBlock(nn.Module):
    """'plain' 模式：子带投影 + 逆变换（作为"小波"消融项，不含注意力）。

    extra_blocks: 追加若干 3x3 DW + 1x1 卷积块，用于与 WaveletAttention **参数对齐**
                  （消融公平性：否则 M3 vs M4 的增益可能只是"参数更多"）。
                  每个额外块约增加 0.069M（C=256），5 个块 ≈ +0.34M。
    """

    def __init__(self, dim: int, extra_blocks: int = 0):
        super().__init__()
        self.proj_ll = _cbr(dim, dim, 1)
        self.proj_lh = _cbr(dim, dim, 1)
        self.proj_hl = _cbr(dim, dim, 1)
        self.proj_hh = _cbr(dim, dim, 1)
        blocks = [
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim),
        ]
        for _ in range(extra_blocks):
            blocks += [
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
                nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim),
            ]
        self.post = nn.Sequential(*blocks)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        LL, LH, HL, HH = haar_dwt2d(feat)
        out = haar_idwt2d(self.proj_ll(LL), self.proj_lh(LH), self.proj_hl(HL), self.proj_hh(HH))
        out = out[..., :feat.shape[-2], :feat.shape[-1]]
        return self.post(out) + feat


class WaveletAttention(nn.Module):
    """'attention' 模式：LL 门控 HF + 噪声自适应软阈值（对应"小波+Attention"）。

    参数开销很小（C=128 时约 0.1M），远低于同分辨率的 Swin。
    """

    def __init__(self, dim: int, reduction: int = 4, init_lambda: float = 0.02):
        super().__init__()
        self.proj_ll = _cbr(dim, dim, 1)
        self.proj_lh = _cbr(dim, dim, 1)
        self.proj_hl = _cbr(dim, dim, 1)
        self.proj_hh = _cbr(dim, dim, 1)
        # LL 上下文 + HF 统计 -> 每个高频子带一个门控
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, max(8, dim // reduction), 3, 1, 1, bias=False),
            nn.BatchNorm2d(max(8, dim // reduction)), nn.ReLU(inplace=True),
            nn.Conv2d(max(8, dim // reduction), dim * 3, 1),
            nn.Sigmoid(),
        )
        # 由 HF 能量预测阈值系数（逐通道），实现"噪声自适应"
        self.lam_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(8, dim // reduction), 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(max(8, dim // reduction), dim, 1), nn.Sigmoid(),
        )
        self.lam_init = init_lambda
        self.post = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        LL, LH, HL, HH = haar_dwt2d(feat)
        ll = self.proj_ll(LL)
        lh, hl, hh = self.proj_lh(LH), self.proj_hl(HL), self.proj_hh(HH)

        hf_sum = lh + hl + hh
        g = self.gate(torch.cat([ll, hf_sum], dim=1))
        g_lh, g_hl, g_hh = g.chunk(3, dim=1)

        # 能量自适应的软阈值：lambda = lam_scale * 通道平均 |HF|
        scale = 2.0 * self.lam_init * self.lam_fc(hf_sum)
        lam = scale * hf_sum.abs().mean(dim=(2, 3), keepdim=True)
        lh = soft_threshold(lh, lam) * g_lh
        hl = soft_threshold(hl, lam) * g_hl
        hh = soft_threshold(hh, lam) * g_hh

        out = haar_idwt2d(ll, lh, hl, hh)
        out = out[..., :feat.shape[-2], :feat.shape[-1]]
        return self.post(out) + feat


def build_wavelet(mode: str, dim: int, plain_extra_blocks: int = 0) -> nn.Module:
    """mode: 'none' | 'plain' | 'attention'。'none' 返回恒等（不做小波分支）。

    plain_extra_blocks 仅对 'plain' 生效，用于与 'attention' 做参数对齐。
    """
    if mode == 'none':
        return nn.Identity()
    if mode == 'plain':
        return PlainWaveletBlock(dim, extra_blocks=plain_extra_blocks)
    if mode == 'attention':
        return WaveletAttention(dim)
    raise ValueError(f"未知 wavelet mode: {mode}")
