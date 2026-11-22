r"""
轻量解码器（对应你图中的 "Lightweight Decoder → Pipeline Segmentation"）

参考论文：融合特征 → 3x3 Conv(128) → 上采样 x4 → 3x3 Conv(64) → 上采样至输入分辨率 → 1x1 输出。
本实现增加:
  · 多尺度融合（s16 全局增强特征 + s8 + s4），逐级上采样后拼接再卷积
  · 可选 stride-4 高分辨率旁路（use_skip），提升管道边界的定位精度
  · 输出 (logits, aux)：aux 供深监督或后续中心线/边界头扩展

分辨率记账（以输入 /1 为基准）:
  · use_skip=True : 融合后位于 /4，一次 x4 上采样到 /1
  · use_skip=False: 融合后位于 /16，x4 -> /4，再 x4 -> /1
  最后统一按 out_size 做一次精确对齐（应对不能被 16 整除的输入尺寸）
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def _cbr_dw(cin, cout, k=3):
    """深度可分离版：DW kxk + 1x1。在 /4 高分辨率处可大幅降低算力。"""
    return nn.Sequential(
        nn.Conv2d(cin, cin, k, 1, k // 2, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def _up_to(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class LightweightDecoder(nn.Module):
    def __init__(self, dim_s4: int, dim_s8: int, dim_s16: int, mid: int = 128,
                 num_classes: int = 2, use_skip: bool = True, dw: bool = False):
        super().__init__()
        self.use_skip = use_skip
        conv = _cbr_dw if dw else _cbr
        self.lat8 = _cbr(dim_s8, mid, 1)
        self.lat4 = _cbr(dim_s4, mid, 1)
        self.fuse16 = _cbr(dim_s16, mid, 1)
        self.fuse8 = conv(mid * 2, mid, 3)
        self.fuse4 = conv(mid * 2, mid, 3)
        self.mid_conv = conv(mid, 64, 3)          # 论文口径的 3x3 Conv(64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, feats: Dict[str, torch.Tensor], fused16: torch.Tensor, out_size=None):
        f16 = self.fuse16(fused16)
        if self.use_skip:
            f8 = self.fuse8(torch.cat([_up_to(f16, feats['s8'].shape[-2:]), self.lat8(feats['s8'])], 1))
            f4 = self.fuse4(torch.cat([_up_to(f8, feats['s4'].shape[-2:]), self.lat4(feats['s4'])], 1))
            x = self.mid_conv(f4)                              # 位于 /4
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)   # -> /1
        else:
            x = self.mid_conv(f16)                             # 位于 /16
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)   # -> /4
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)   # -> /1
        if out_size is not None:
            x = _up_to(x, out_size)
        return self.head(x), {'s16': fused16}
