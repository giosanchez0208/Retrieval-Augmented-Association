"""Batch augmentations on the GPU for retriever training.

Photometric changes cover the lighting drift a camera sees between sightings:
global brightness, contrast, saturation and white balance ("warmth"), and a
partial change across the crop. Random erasing stands in for partial occlusion.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from reidtrack.retrieval.embedder import IMAGENET_MEAN, IMAGENET_STD


class Augment:
    def __init__(
        self,
        flip: float = 0.5,
        pad: int = 10,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        warmth: float = 0.1,
        gradient_p: float = 0.3,
        gradient: float = 0.4,
        erase_p: float = 0.5,
        erase_area: tuple[float, float] = (0.02, 0.4),
    ) -> None:
        self.flip, self.pad = flip, pad
        self.brightness, self.contrast, self.saturation, self.warmth = brightness, contrast, saturation, warmth
        self.gradient_p, self.gradient = gradient_p, gradient
        self.erase_p, self.erase_area = erase_p, erase_area

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 (B, 3, H, W) on the GPU -> normalised float (B, 3, H, W)."""
        x = images.float() / 255
        b, _, h, w = x.shape
        dev = x.device

        def uniform(lo: float, hi: float, *shape: int) -> torch.Tensor:
            return torch.empty(b, *shape, device=dev).uniform_(lo, hi)

        flip = torch.rand(b, device=dev) < self.flip
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        if self.pad:
            padded = F.pad(x, (self.pad,) * 4)
            dx = torch.randint(0, 2 * self.pad + 1, (b,)).tolist()
            dy = torch.randint(0, 2 * self.pad + 1, (b,)).tolist()
            x = torch.stack([padded[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w] for i in range(b)])

        x = x * uniform(1 - self.brightness, 1 + self.brightness, 1, 1, 1)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * uniform(1 - self.contrast, 1 + self.contrast, 1, 1, 1) + mean
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = (x - gray) * uniform(1 - self.saturation, 1 + self.saturation, 1, 1, 1) + gray
        warm = uniform(-self.warmth, self.warmth, 1, 1)
        x = x * torch.cat([1 + warm, torch.ones_like(warm), 1 - warm], dim=1)[..., None]

        # partial lighting: a linear ramp across the crop in a random direction
        use = (torch.rand(b, device=dev) < self.gradient_p).float()
        angle = uniform(0, 2 * math.pi)
        ys = torch.linspace(-0.5, 0.5, h, device=dev)[None, :, None]
        xs = torch.linspace(-0.5, 0.5, w, device=dev)[None, None, :]
        ramp = torch.cos(angle)[:, None, None] * xs + torch.sin(angle)[:, None, None] * ys
        strength = uniform(-self.gradient, self.gradient) * use
        x = x * (1 + strength[:, None, None] * 2 * ramp)[:, None]

        x = x.clamp(0, 1)
        mean_t = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
        std_t = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
        x = (x - mean_t) / std_t
        return self._erase(x)

    def _erase(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        for i in torch.nonzero(torch.rand(b) < self.erase_p).flatten().tolist():
            for _ in range(10):
                area = h * w * float(torch.empty(1).uniform_(*self.erase_area))
                aspect = math.exp(float(torch.empty(1).uniform_(math.log(0.3), math.log(3.3))))
                eh, ew = int(round(math.sqrt(area * aspect))), int(round(math.sqrt(area / aspect)))
                if 0 < eh < h and 0 < ew < w:
                    top, left = int(torch.randint(0, h - eh + 1, (1,))), int(torch.randint(0, w - ew + 1, (1,)))
                    x[i, :, top : top + eh, left : left + ew] = torch.randn(c, eh, ew, device=x.device)
                    break
        return x


def normalize(images: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> normalised float, no augmentation."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images.float() / 255 - mean) / std
