# Adapted from deep-person-reid (torchreid/models/osnet.py), reduced to the
# layers needed for inference. Module names are unchanged so published
# checkpoints load as-is.
#
# MIT License
#
# Copyright (c) 2018 Kaiyang Zhou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""OSNet (Zhou et al., Omni-Scale Feature Learning for Person Re-Identification, ICCV 2019)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    """Convolution + batch norm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 convolution + batch norm + ReLU."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution + batch norm, no non-linearity."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 (linear) followed by a depthwise 3x3 (non-linear)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Channel-wise sigmoid gates conditioned on the input (shared across streams)."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        gates = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * gates


class OSBlock(nn.Module):
    """Omni-scale block: four streams with 1-4 stacked light 3x3 convolutions, fused by a shared gate."""

    def __init__(self, in_channels, out_channels, bottleneck_reduction=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(*(LightConv3x3(mid_channels, mid_channels) for _ in range(2)))
        self.conv2c = nn.Sequential(*(LightConv3x3(mid_channels, mid_channels) for _ in range(3)))
        self.conv2d = nn.Sequential(*(LightConv3x3(mid_channels, mid_channels) for _ in range(4)))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSNet(nn.Module):
    """Omni-Scale Network. In eval mode ``forward`` returns the feature vector."""

    def __init__(self, num_classes, channels, layers=(2, 2, 2), feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_layer(layer, in_channels, out_channels, reduce_spatial_size):
        blocks = [OSBlock(in_channels, out_channels)]
        blocks += [OSBlock(out_channels, out_channels) for _ in range(1, layer)]
        if reduce_spatial_size:
            blocks.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*blocks)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.maxpool(self.conv1(x))
        return self.conv5(self.conv4(self.conv3(self.conv2(x))))

    def forward(self, x):
        v = self.fc(self.global_avgpool(self.featuremaps(x)).flatten(1))
        return v if not self.training else self.classifier(v)


WIDTHS = {
    "x1_0": (64, 256, 384, 512),
    "x0_75": (48, 192, 288, 384),
    "x0_5": (32, 128, 192, 256),
    "x0_25": (16, 64, 96, 128),
}


def osnet(width: str = "x1_0", num_classes: int = 1000) -> OSNet:
    return OSNet(num_classes, WIDTHS[width])


def load_osnet(path, width: str = "x1_0") -> OSNet:
    """Build OSNet and load a torchreid checkpoint (plain or wrapped state dict)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = osnet(width, num_classes=state["classifier.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    return model
