"""Appearance embeddings for detection boxes, computed on the GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.io import decode_jpeg
from torchvision.ops import roi_align

from reidtrack.retrieval.osnet import load_osnet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Embedder:
    """Crops boxes out of a frame and embeds them; returns L2-normalised vectors.

    Crops are stretched to ``input_size`` (height, width) with ``roi_align``, whose
    adaptive sampling averages each output pixel over its source area.
    """

    def __init__(
        self,
        weights: str | Path,
        width: str = "x1_0",
        input_size: tuple[int, int] = (256, 128),
        device: str = "cuda",
        half: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"
        self.input_size = input_size
        self.model = load_osnet(weights, width).eval().to(self.device)
        self.dim = self.model.feature_dim
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def decode(self, jpeg: bytes) -> torch.Tensor:
        """JPEG bytes to a (3, H, W) uint8 RGB tensor on the device."""
        data = torch.frombuffer(bytearray(jpeg), dtype=torch.uint8)
        return decode_jpeg(data, device=self.device) if self.device.type == "cuda" else decode_jpeg(data)

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor, xyxy: np.ndarray) -> np.ndarray:
        if len(xyxy) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        h, w = image.shape[1:]
        boxes = torch.as_tensor(np.asarray(xyxy, dtype=np.float32), device=self.device)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        rois = torch.cat([boxes.new_zeros(len(boxes), 1), boxes], dim=1)
        crops = roi_align(image[None].float(), rois, self.input_size, aligned=True)
        crops = (crops / 255 - self.mean) / self.std
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
            features = self.model(crops)
        return F.normalize(features.float(), dim=1).cpu().numpy()
