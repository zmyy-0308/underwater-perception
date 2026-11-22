"""Architecture registry with an aligned stride-eight interface for stereo."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, resnet18
from torchvision.models.segmentation import deeplabv3_resnet50, lraspp_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models.segmentation.lraspp import LRASPPHead

from multitask.attention import EfficientChannelAttention, LightweightSwin
from seg_backbone.models.decoder import LightweightDecoder
from seg_backbone.models.efsnet import ConvBNReLU
from seg_backbone.models.wavelet import PlainWaveletBlock, build_wavelet

WIDTHS = (64, 96, 128)
CHECKPOINT_WIDTHS = tuple(range(32, 257, 32))
SIZES = {"pipe": (288, 512), "bio": (384, 384)}
FAMILIES = {"E2": "deeplab", "E3": "lraspp", "E4": "yolo26n-sem", "E5": "yolo26s-sem"}
PIPE_ATTENTION = {"P11": EfficientChannelAttention, "P12": LightweightSwin, "P13": LightweightSwin}


@dataclass(frozen=True)
class Group:
    task: str
    name: str
    width: int

    def __post_init__(self):
        valid_name = self.name in group_names(self.task) or (
            self.task == "pipe" and self.name in PIPE_ATTENTION
        )
        if self.task not in SIZES or not valid_name or self.width not in CHECKPOINT_WIDTHS:
            raise ValueError(f"Unknown experiment group {self.task}/{self.name}/W{self.width}")

    @property
    def classes(self):
        return 2 if self.task == "pipe" else 8

    @property
    def weight_key(self):
        if self.name.startswith("U"):
            return "resnet18"
        return FAMILIES.get(self.name, "efficientnet_b0")


def group_names(task):
    n = 10 if task == "pipe" else 6
    prefix = "P" if task == "pipe" else "B"
    return [*(f"{prefix}{i}" for i in range(1, n+1)), *(f"U{i}" for i in range(1, 9)), *FAMILIES]


def all_groups(task):
    main = "P8" if task == "pipe" else "B4"
    groups = []
    for name in group_names(task):
        default = 64 if task == "bio" and name.startswith("B") else 128
        widths = WIDTHS if name == main else (default,)
        groups.extend(Group(task, name, width) for width in widths)
    return groups


def shared_seed(seed, offset):
    torch.random.default_generator.manual_seed((seed+offset) % 2**63)


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class Normalized(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)

    def normalize(self, x):
        return (x-self.mean)/self.std


class EFS(nn.Module):
    def __init__(self, full, weights, seed):
        super().__init__()
        shared_seed(seed, 0)
        backbone = efficientnet_b0(weights=None)
        if weights:
            backbone.load_state_dict(checkpoint(weights), strict=True)
        self.features = backbone.features if full else backbone.features[:5]
        shared_seed(seed, 1)
        self.adapt4 = ConvBNReLU(24, 64)
        self.adapt8 = ConvBNReLU(40, 128)
        shared_seed(seed, 2)
        self.adapt16 = ConvBNReLU(1280 if full else 80, 256)

    def forward(self, x, stereo=False):
        x = self.features[1](self.features[0](x))
        p4 = self.features[2](x)
        p8 = self.features[3](p4)
        if stereo:
            return self.adapt8(p8)
        p16 = self.features[4](p8)
        deep = p16
        for stage in self.features[5:]:
            deep = stage(deep)
        deep = self.adapt16(deep)
        if deep.shape[-2:] != p16.shape[-2:]:
            deep = F.interpolate(deep, p16.shape[-2:], mode="bilinear", align_corners=False)
        return {"s4": self.adapt4(p4), "s8": self.adapt8(p8), "s16": deep}


class ConcatFusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(width*2, width),
                                  nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                                  nn.BatchNorm2d(width))

    def forward(self, x, branch):
        return x+self.conv(torch.cat((x, branch), dim=1))


class SpatialResidual(PlainWaveletBlock):
    """Use the same trainable layers as ordinary Haar with a spatial pooling path."""
    def forward(self, feat):
        h, w = feat.shape[-2:]
        padded = F.pad(feat, (0, w % 2, 0, h % 2), mode="replicate")
        x = F.avg_pool2d(padded, 2)
        combined = (self.proj_ll(x)+self.proj_lh(x)+self.proj_hl(x)+self.proj_hh(x))/4
        return feat+self.post(F.interpolate(combined, scale_factor=2, mode="nearest")[..., :h, :w])


class InternalSeg(Normalized):
    stereo_channels = 128

    def __init__(self, group, weights, seed):
        super().__init__()
        n = int(group.name[1:])
        if group.task == "pipe":
            options = [*product((True, False), (False, True), ("none", "plain")),
                       (False, True, "attention"), (False, True, "spatial")]
        else:
            options = [(full, dw, "none") for full, dw in product((True, False), (False, True))]
            options += [(False, True, "plain"), (False, True, "attention")]
        attention = PIPE_ATTENTION.get(group.name)
        full, dw, wave = (False, group.name != "P13", "none") if attention else options[n-1]
        self.encoder = EFS(full, weights, seed)
        self.pipe = group.task == "pipe"
        dim = 256 if self.pipe else 128
        shared_seed(seed, 3)
        self.decoder = LightweightDecoder(64, 128, 256, group.width, group.classes, True, dw)
        shared_seed(seed, 4)
        self.wave = SpatialResidual(dim) if wave == "spatial" else build_wavelet(wave, dim)
        shared_seed(seed, 5)
        self.merge = ConcatFusion(dim) if self.pipe and wave != "none" else None
        shared_seed(seed, 6)
        if attention is LightweightSwin:
            self.attention = LightweightSwin(dim)
        elif attention is EfficientChannelAttention:
            self.attention = EfficientChannelAttention()
        else:
            self.attention = nn.Identity()

    def stereo_features(self, x):
        return self.encoder(self.normalize(x), stereo=True)

    def forward(self, x):
        feat = self.encoder(self.normalize(x))
        if self.merge is not None:
            feat["s16"] = self.merge(feat["s16"], self.wave(feat["s16"]))
        feat["s16"] = self.attention(feat["s16"])
        if not self.pipe:
            feat["s8"] = self.wave(feat["s8"])
        return self.decoder(feat, feat["s16"], x.shape[-2:])[0]


class SCSE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c//16, 1), nn.ReLU(),
                                     nn.Conv2d(c//16, c, 1), nn.Sigmoid())
        self.space = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x*self.channel(x)+x*self.space(x)


class UnetBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(ConvBNReLU(cin, cout, 3), ConvBNReLU(cout, cout, 3))


class UNet(Normalized):
    stereo_channels = 128

    def __init__(self, group, weights, seed):
        super().__init__()
        full, haar, attention = list(product((True, False), (False, True), (False, True)))[int(group.name[1:])-1]
        # U2 adds only Haar, while U3 adds only attention.
        haar, attention = attention, haar
        shared_seed(seed, 0)
        res = resnet18(weights=None)
        if weights:
            res.load_state_dict(checkpoint(weights), strict=True)
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu)
        self.low = nn.Sequential(res.maxpool, res.layer1, res.layer2)
        self.stage16 = res.layer3
        shared_seed(seed, 1)
        self.stage32 = res.layer4 if full else nn.Sequential(nn.AvgPool2d(2, ceil_mode=True),
                                                             ConvBNReLU(256, 512))
        shared_seed(seed, 2)
        self.wave = build_wavelet("plain" if haar else "none", 128)
        self.attention = SCSE(128) if attention else nn.Identity()
        shared_seed(seed, 3)
        widths = [group.width//(2**i) for i in range(5)]
        skips = (256, 128, 64, 64, 0)
        self.decode = nn.ModuleList(UnetBlock(c+skip, w) for c, skip, w in zip(
            (512, *widths), skips, widths))
        self.head = nn.Conv2d(widths[-1], group.classes, 3, padding=1)

    def stereo_features(self, x):
        return self.low(self.stem(self.normalize(x)))

    def forward(self, x):
        p2 = self.stem(self.normalize(x))
        p4 = self.low[1](self.low[0](p2))
        p8 = self.low[2](p4)
        p16 = self.stage16(p8)
        out = self.stage32(p16)
        for skip, block in zip((p16, self.attention(self.wave(p8)), p4, p2, None), self.decode):
            shape = x.shape[-2:] if skip is None else skip.shape[-2:]
            out = F.interpolate(out, shape, mode="nearest")
            out = block(out if skip is None else torch.cat((out, skip), dim=1))
        return self.head(out)


class VisionSeg(Normalized):
    def __init__(self, group, weights, seed):
        super().__init__()
        self.deeplab = group.name == "E2"
        shared_seed(seed, 0)
        builder = deeplabv3_resnet50 if self.deeplab else lraspp_mobilenet_v3_large
        base = builder(weights=None, weights_backbone=None)
        if weights:
            state = checkpoint(weights)
            encoder = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
            base.backbone.load_state_dict(encoder, strict=True)
        self.backbone = base.backbone
        shared_seed(seed, 3)
        w = group.width
        if self.deeplab:
            self.stereo_channels = 512
            self.head = nn.Sequential(ASPP(2048, (12, 24, 36), w), nn.Conv2d(w, w, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(w), nn.ReLU(), nn.Conv2d(w, group.classes, 1))
        else:
            self.stereo_channels = base.classifier.low_classifier.in_channels
            self.head = LRASPPHead(self.stereo_channels, base.classifier.cbr[0].in_channels, group.classes, w)

    def stereo_features(self, x):
        x = self.normalize(x)
        goal = "layer2" if self.deeplab else next(k for k, v in self.backbone.return_layers.items() if v == "low")
        for name, layer in self.backbone.items():
            x = layer(x)
            if name == goal:
                return x
        raise RuntimeError(f"Backbone feature tap {goal} is missing")

    def forward(self, x):
        features = self.backbone(self.normalize(x))
        logits = self.head(features["out"] if self.deeplab else features)
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


class YoloSem(nn.Module):
    def __init__(self, group, weights, seed):
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.nn.tasks import SemanticSegmentationModel, yaml_model_load

        shared_seed(seed, 0)
        mode = group.weight_key
        cfg = yaml_model_load(mode+".yaml")
        scale = mode.removeprefix("yolo26").split("-")[0]
        depth, multiplier, cap = cfg["scales"][scale]
        cfg["scales"][scale] = [depth, multiplier, max(cap, int(2*group.width/multiplier))]
        fusion = [entry for entry in cfg["head"] if entry[2] == "C3k2"]
        if len(fusion) != 2 or cfg["head"][-1][2] != "SemanticSegment":
            raise RuntimeError("YOLO26 semantic YAML topology differs from the pinned adapter")
        for entry, w in zip(fusion, (2*group.width, group.width)):
            entry[3][0] = int(w/multiplier)
        self.net = SemanticSegmentationModel(cfg, ch=3, nc=group.classes, verbose=False)
        # Common training uses only the main dense output in all model families.
        self.net.model[-1].aux_head = None
        self.stereo_channels = int(512*multiplier)
        if weights:
            official = YOLO(str(weights), task="semantic").model
            n = len(cfg["backbone"])
            self.net.model[:n].load_state_dict(official.model[:n].state_dict(), strict=True)

    def stereo_features(self, x):
        return self.net.model[:5](x)

    def forward(self, x):
        logits = self.net(x)
        if isinstance(logits, dict):
            logits = logits["semantic"]
        return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)


def build(group, weights=None, seed=42):
    builder = UNet if group.name.startswith("U") else VisionSeg if group.name in ("E2", "E3") else (
        YoloSem if group.name in ("E4", "E5") else InternalSeg)
    with torch.random.fork_rng(devices=[]):
        return builder(group, Path(weights) if weights else None, seed)


class DepthEncoder(Normalized):
    stereo_channels = 128

    def __init__(self, weights, seed):
        super().__init__()
        encoder = EFS(False, weights, seed)
        self.features = encoder.features[:4]
        self.adapter = encoder.adapt8

    def stereo_features(self, x):
        return self.adapter(self.features(self.normalize(x)))


def build_depth_encoder(weights=None, seed=42):
    with torch.random.fork_rng(devices=[]):
        return DepthEncoder(weights, seed)
