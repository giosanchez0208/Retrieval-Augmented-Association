"""The identity bank: one entry per known person.

Each entry keeps a motion state, a running-average appearance vector, a few
diverse views (prototypes), and a regime:

- ACTIVE: matched in the last frame
- OCCLUDED: unmatched, last seen inside the frame, so probably hidden
- EXITED: unmatched, last seen at the image border moving outward
- TENTATIVE: new and not yet confirmed by a second match
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Regime(IntEnum):
    TENTATIVE = 0
    ACTIVE = 1
    OCCLUDED = 2
    EXITED = 3


@dataclass
class Entry:
    track_id: int
    mean: np.ndarray  # Kalman state (cx, cy, a, h, vcx, vcy, va, vh)
    cov: np.ndarray
    first_seen: float  # seconds
    last_seen: float
    score: float
    regime: Regime = Regime.TENTATIVE
    appearance: np.ndarray | None = None  # running average, unit length
    prototypes: list[np.ndarray] = field(default_factory=list)
    hits: int = 1
    last_box: np.ndarray | None = None  # xyxy of the last matched detection
    recalls: int = 0  # times re-matched after being occluded or exited

    def views(self) -> list[np.ndarray]:
        return ([self.appearance] if self.appearance is not None else []) + self.prototypes


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


class AppearanceMemory:
    """Write policy for an entry's appearance: a running average plus diverse views."""

    def __init__(self, momentum: float = 0.9, max_prototypes: int = 4, min_novelty: float = 0.1) -> None:
        self.momentum = momentum
        self.max_prototypes = max_prototypes
        self.min_novelty = min_novelty  # cosine distance a view needs to count as new

    def write(self, entry: Entry, feature: np.ndarray) -> None:
        feature = _unit(np.asarray(feature, dtype=np.float32))
        if entry.appearance is None:
            entry.appearance = feature
        else:
            entry.appearance = _unit(self.momentum * entry.appearance + (1 - self.momentum) * feature)
        if not entry.prototypes:
            entry.prototypes.append(feature)
            return
        protos = np.stack(entry.prototypes)
        closest = float((protos @ feature).max())
        if 1 - closest < self.min_novelty:
            return  # nothing new about this view
        if len(entry.prototypes) < self.max_prototypes:
            entry.prototypes.append(feature)
            return
        # replace the most redundant view, if the new one is more distinct than it
        sims = protos @ protos.T
        np.fill_diagonal(sims, -1)
        redundancy = sims.max(axis=1)
        worst = int(np.argmax(redundancy))
        if closest < redundancy[worst]:
            entry.prototypes[worst] = feature


def similarity(entries: list[Entry], features: np.ndarray) -> np.ndarray:
    """Best cosine similarity between each entry's stored views and each feature, (N, M)."""
    if not entries or len(features) == 0:
        return np.zeros((len(entries), len(features)), dtype=np.float32)
    views, owner = [], []
    for i, entry in enumerate(entries):
        for v in entry.views():
            views.append(v)
            owner.append(i)
    out = np.full((len(entries), len(features)), -1.0, dtype=np.float32)
    if views:
        sims = np.stack(views) @ np.asarray(features, dtype=np.float32).T
        np.maximum.at(out, np.array(owner), sims)
    return out


def exiting(box: np.ndarray, velocity: np.ndarray, width: int, height: int, margin: float = 0.02) -> bool:
    """Whether a box touches the image border and is moving out through it."""
    x1, y1, x2, y2 = box
    vx, vy = velocity
    mx, my = margin * width, margin * height
    return bool(
        (x1 <= mx and vx < 0) or (x2 >= width - mx and vx > 0) or (y1 <= my and vy < 0) or (y2 >= height - my and vy > 0)
    )
