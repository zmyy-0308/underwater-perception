import torch
from torch import nn
from torchvision.models.swin_transformer import SwinTransformerBlock


class EfficientChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_filter = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, feature):
        descriptor = self.pool(feature).flatten(1).unsqueeze(1)
        scale = torch.sigmoid(self.channel_filter(descriptor)).squeeze(1)
        return feature * scale[:, :, None, None]


class LightweightSwin(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = 64
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.blocks = nn.Sequential(
            SwinTransformerBlock(hidden, 4, [6, 8], [0, 0], mlp_ratio=2.0),
            SwinTransformerBlock(hidden, 4, [6, 8], [3, 4], mlp_ratio=2.0),
        )
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, feature):
        reduced = self.reduce(feature).permute(0, 2, 3, 1)
        transformed = self.blocks(reduced).permute(0, 3, 1, 2)
        return feature + self.expand(transformed)


class CoordinateAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 16)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(),
        )
        self.height_gate = nn.Conv2d(hidden, channels, 1)
        self.width_gate = nn.Conv2d(hidden, channels, 1)

    def forward(self, feature):
        height, width = feature.shape[-2:]
        along_height = feature.mean(dim=3, keepdim=True)
        along_width = feature.mean(dim=2, keepdim=True).transpose(2, 3)
        joined = self.reduce(torch.cat((along_height, along_width), dim=2))
        height_code, width_code = joined.split((height, width), dim=2)
        height_scale = torch.sigmoid(self.height_gate(height_code))
        width_scale = torch.sigmoid(self.width_gate(width_code.transpose(2, 3)))
        return feature * height_scale * width_scale


class StripDepthwiseContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.horizontal = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False)
        self.vertical = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
            nn.Conv2d(channels, channels, 1, groups=8, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, feature):
        context = self.horizontal(feature) + self.vertical(feature)
        return feature + self.mix(context)
