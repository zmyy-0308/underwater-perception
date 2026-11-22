"""Lightweight shared-feature horizontal stereo matching."""
import torch
from torch import nn
from torch.nn import functional as F


class StereoNet(nn.Module):
    def __init__(self, semantic, spec):
        super().__init__()
        self.semantic = semantic
        self.spec = spec
        c, d = 32, spec.shifts
        self.feature = nn.Conv2d(semantic.stereo_channels, c, 1, bias=False)
        self.regularize = nn.Sequential(nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False),
                                        nn.Conv2d(d, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(),
                                        nn.Conv2d(d, d, 1))

    def segment(self, rgb):
        return self.semantic(rgb)

    def forward(self, left, right, fb):
        if left.shape != right.shape or left.shape[-2:] != tuple(self.spec.size):
            raise ValueError("Stereo inputs must share the configured stereo raster")
        features = self.feature(self.semantic.stereo_features(torch.cat((left, right), 0)))
        ref, other = F.normalize(features, dim=1, eps=1e-6).chunk(2, dim=0)
        width = ref.shape[-1]
        if width < self.spec.shifts:
            raise ValueError("Stereo feature width is smaller than the disparity search")
        other = F.pad(other, (self.spec.shifts-1, 0))
        offset = self.spec.shifts-1
        volume = torch.stack([(ref*other[..., offset-d:offset-d+width]).sum(1)
                              for d in range(self.spec.shifts)], dim=1)
        scores = volume+self.regularize(volume)
        shifts = torch.arange(self.spec.shifts, device=left.device).view(1, -1, 1, 1)
        coords = torch.arange(width, device=left.device).view(1, 1, 1, -1)
        scores = scores.masked_fill(shifts > coords, torch.finfo(scores.dtype).min)
        disparity = (scores.float().softmax(1)*shifts).sum(1, keepdim=True)*8
        disparity = F.interpolate(disparity, left.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        fb = fb.to(device=left.device, dtype=disparity.dtype).reshape(-1, 1, 1)
        low = fb/self.spec.max_depth
        high = fb/self.spec.min_depth
        return fb/disparity.clamp(min=low, max=high)


def supervised_depth_loss(estimate, truth, valid):
    if not valid.any():
        raise ValueError("Stereo batch has no valid supervised pixels")
    predicted, measured = estimate[valid].float(), truth[valid].float()
    return F.smooth_l1_loss(predicted, measured, beta=.25)
