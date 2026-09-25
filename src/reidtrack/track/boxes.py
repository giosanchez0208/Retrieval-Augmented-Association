"""Box geometry. Boxes are float arrays of 0-based ``x1, y1, x2, y2``."""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray, plus_one: bool = False) -> np.ndarray:
    """Pairwise IoU between boxes ``a`` (N, 4) and ``b`` (M, 4).

    ``plus_one`` counts pixels inclusively (width = x2 - x1 + 1), the convention of
    the ``cython_bbox`` routine ByteTrack uses.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    o = 1.0 if plus_one else 0.0
    area_a = (a[:, 2] - a[:, 0] + o) * (a[:, 3] - a[:, 1] + o)
    area_b = (b[:, 2] - b[:, 0] + o) * (b[:, 3] - b[:, 1] + o)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + o
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + o
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    """Centre x, centre y, aspect ratio (w / h), height."""
    w, h = box[..., 2] - box[..., 0], box[..., 3] - box[..., 1]
    return np.stack([box[..., 0] + w / 2, box[..., 1] + h / 2, w / h, h], axis=-1)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    h = xyah[..., 3]
    w = xyah[..., 2] * h
    return np.stack([xyah[..., 0] - w / 2, xyah[..., 1] - h / 2, xyah[..., 0] + w / 2, xyah[..., 1] + h / 2], axis=-1)


def xyxy_to_xysr(box: np.ndarray) -> np.ndarray:
    """Centre x, centre y, area, aspect ratio (w / h)."""
    w, h = box[..., 2] - box[..., 0], box[..., 3] - box[..., 1]
    return np.stack([box[..., 0] + w / 2, box[..., 1] + h / 2, w * h, w / h], axis=-1)


def xysr_to_xyxy(xysr: np.ndarray) -> np.ndarray:
    w = np.sqrt(xysr[..., 2] * xysr[..., 3])
    h = xysr[..., 2] / w
    return np.stack([xysr[..., 0] - w / 2, xysr[..., 1] - h / 2, xysr[..., 0] + w / 2, xysr[..., 1] + h / 2], axis=-1)
