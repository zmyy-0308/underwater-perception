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
