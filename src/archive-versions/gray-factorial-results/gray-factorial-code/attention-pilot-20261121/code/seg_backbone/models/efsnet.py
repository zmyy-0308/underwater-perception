r"""
EFSNet：轻量局部特征编码器（对应你图中的 "EFSNet → Local Feature"）

依据论文设计：以 EfficientNet-B0 为骨架，**只保留前 4 个 MBConv 阶段**
（刻意强化低层边缘/纹理，这对细长管道的边界至关重要），
再用 Strided-Conv 做通道对齐与额外下采样。

输出多尺度局部特征:
    s4  : stride 4   (通道 64)
    s8  : stride 8   (通道 128)
    s16 : stride 16  (通道 256)  ← 主特征（Local Feature）
    s32 : stride 32  (通道 256)  ← 供"深层全局模块"使用（轻量化 P0 选项）

代码依赖 torchvision 的 EfficientNet-B0 结构（可加载 ImageNet 预训练权重）。
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

try:  # 允许离线环境
    from torchvision.models import EfficientNet_B0_Weights
    _W = EfficientNet_B0_Weights.IMAGENET1K_V1
except Exception:  # pragma: no cover
    _W = 'IMAGENET1K_V1'


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=1, s=1, g=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class EFSNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        pretrained: bool = True,
        out_ch: Tuple[int, int, int] = (64, 128, 256),
        use_down32: bool = False,
    ):
        """use_down32=False（默认）时不构建 /32 分支，避免 1.18M 死重参数。

        说明：全局模块放在 /32 时由 GlobalContext 内部的 avg_pool 完成下采样；
        若希望使用"可学习的 stride-2 下采样"以获得更高质量，可置 True 并让
        GlobalContext 接收 encoder 输出的 's32'（接口已预留）。
        """
        super().__init__()
        self.use_down32 = use_down32
        if pretrained:
            try:
                net = efficientnet_b0(weights=_W)
                self.pretrained_loaded = True
            except Exception as e:  # 无网络时退化为随机初始化
                print(f'[EFSNet] 预训练权重加载失败，改为随机初始化: {e}')
                net = efficientnet_b0(weights=None)
                self.pretrained_loaded = False
        else:
            net = efficientnet_b0(weights=None)
            self.pretrained_loaded = False

        feats = list(net.features.children())
        # B0 结构: [0]=stem(/2,32) [1]=MBConv1(/2,16) [2]=MBConv6(/4,24)
        #          [3]=MBConv6x2(/8,40) [4]=MBConv6x2(/16,80) [5]=x3(/16,112)
        #          [6]=MBConv6x4(/32,192) [7]=Conv1x1(320) [8]=pool
        self.stem = feats[0]                     # /2
        self.mb1 = feats[1]                      # /2
        self.mb2 = feats[2]                      # /4
        self.mb3 = feats[3]                      # /8
        self.mb4 = feats[4]                      # /16  ← 只保留前 4 个 MBConv
        c2, c4, c8, c16 = 32, 24, 40, 80

        if in_ch != 3:                            # 支持单通道/多通道输入
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, c2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            ) if in_ch != 3 else self.stem

        # Strided-Conv 通道对齐（论文的 down-sampling 模块）
        self.adapt4 = ConvBNReLU(c4, out_ch[0], 1)
        self.adapt8 = ConvBNReLU(c8, out_ch[1], 1)
        self.adapt16 = ConvBNReLU(c16, out_ch[2], 1)
        # 额外的 stride-2 Strided-Conv -> /32（默认不构建，避免 1.18M 死重参数）
        if self.use_down32:
            self.down32 = nn.Sequential(
                ConvBNReLU(out_ch[2], out_ch[2], 3, s=2),
                ConvBNReLU(out_ch[2], out_ch[2], 3, s=1),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.mb1(x)
        f4 = self.mb2(x)
        f8 = self.mb3(f4)
        f16 = self.mb4(f8)
        out = {'s4': self.adapt4(f4), 's8': self.adapt8(f8), 's16': self.adapt16(f16)}
        if self.use_down32:
            out['s32'] = self.down32(out['s16'])
        return out


if __name__ == '__main__':
    net = EFSNet(pretrained=False).eval()
    n = sum(p.numel() for p in net.parameters()) / 1e6
    with torch.no_grad():
        o = net(torch.randn(1, 3, 512, 768))
    print(f'EFSNet 参数 {n:.3f} M')
    for k, v in o.items():
        print(f'  {k}: {tuple(v.shape)}')
