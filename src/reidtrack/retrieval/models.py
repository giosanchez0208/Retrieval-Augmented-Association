"""Re-identification models: a backbone mapping crops to a feature vector, and the
BNNeck head of Luo et al. (Bag of Tricks, 2019) used during training."""

from __future__ import annotations

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from reidtrack.retrieval.osnet import osnet

BACKBONES = ("osnet_x1_0", "osnet_x0_5", "osnet_x0_25", "resnet18", "resnet34")


class OSNetBackbone(nn.Module):
    def __init__(self, width: str = "x1_0") -> None:
        super().__init__()
        self.net = osnet(width, num_classes=1)
        del self.net.classifier
        self.feature_dim = self.net.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.fc(self.net.global_avgpool(self.net.featuremaps(x)).flatten(1))


class ResNetBackbone(nn.Module):
    """torchvision ResNet without its classifier. ``last_stride=1`` keeps the last
    stage at twice the resolution, a standard ReID choice (16x8 maps for 256x128)."""

    def __init__(self, depth: int = 18, last_stride: int = 1, weights: str | None = None) -> None:
        super().__init__()
        base = getattr(torchvision.models, f"resnet{depth}")(weights=weights)
        if last_stride == 1:
            base.layer4[0].conv1.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
        self.body = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.feature_dim = base.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.body(x), 1).flatten(1)


def build_backbone(name: str) -> nn.Module:
    if name.startswith("osnet_"):
        return OSNetBackbone(name.removeprefix("osnet_"))
    if name.startswith("resnet"):
        return ResNetBackbone(int(name.removeprefix("resnet")))
    raise ValueError(f"unknown backbone {name!r}; expected one of {BACKBONES}")


class ReIDModel(nn.Module):
    """Backbone + BNNeck. Training returns (feature, logits): the triplet loss uses the
    feature before the BN, the ID loss the logits after it. Eval returns the
    post-BN feature, which is what gets compared by cosine similarity."""

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        dim = backbone.feature_dim
        self.bnneck = nn.BatchNorm1d(dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    def forward(self, x: torch.Tensor):
        feature = self.backbone(x)
        normed = self.bnneck(feature)
        return (feature, self.classifier(normed)) if self.training else normed
