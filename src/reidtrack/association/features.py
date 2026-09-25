"""Cues describing one (bank entry, detection) pair, for learned reranking.

The spatial cues are the ones the v1 project fed its pair classifier (offsets,
overlap, scale, corner offsets, time gap), now measured against the entry's
predicted box and combined with appearance and memory state.
"""

from __future__ import annotations

import numpy as np

from reidtrack.memory.bank import Entry, Regime, similarity
from reidtrack.track.boxes import iou_matrix, xyah_to_xyxy, xyxy_to_xyah
from reidtrack.track.kalman import XYAHKalman

NAMES = (
    "sim_best", "sim_average", "iou", "dx", "dy", "log_h", "log_w",
    "dx1", "dy1", "dx2", "dy2", "mahalanobis", "gap_s", "det_score", "crowding",
    "active", "occluded", "exited", "log_hits", "rank_best", "rank_average",
)


class NegativeCalibration:
    """Running distribution of appearance distances between detections in the same
    frame, which are always different people. ``rank`` turns a distance into the share
    of those different-people pairs that were farther apart, a scale that means the
    same for any appearance model and needs no labels. A small uniform prior keeps it
    defined from the first frame."""

    def __init__(self, bins: int = 200, prior: float = 20.0) -> None:
        self.edges = np.linspace(0.0, 2.0, bins + 1)
        self.counts = np.full(bins, prior / bins)

    def observe(self, feats: np.ndarray) -> None:
        if len(feats) < 2:
            return
        dist = 1 - feats @ feats.T
        self.counts += np.histogram(dist[np.triu_indices(len(feats), 1)], bins=self.edges)[0]

    def rank(self, distance: np.ndarray) -> np.ndarray:
        cdf = np.concatenate([[0.0], np.cumsum(self.counts) / self.counts.sum()])
        return 1 - np.interp(distance, self.edges, cdf)


def pair_features(
    entries: list[Entry],
    boxes: np.ndarray,
    scores: np.ndarray,
    feats: np.ndarray,
    crowding: np.ndarray,
    kf: XYAHKalman,
    now: float,
    calibration: NegativeCalibration | None = None,
) -> np.ndarray:
    """(N entries, M detections, len(NAMES)) float32 cue tensor."""
    n, m = len(entries), len(boxes)
    out = np.zeros((n, m, len(NAMES)), dtype=np.float32)
    if n == 0 or m == 0:
        return out
    pred = np.array([xyah_to_xyxy(e.mean[:4]) for e in entries])
    ph = np.maximum(pred[:, 3] - pred[:, 1], 1)[:, None]
    pw = np.maximum(pred[:, 2] - pred[:, 0], 1)[:, None]
    dh = np.maximum(boxes[:, 3] - boxes[:, 1], 1)[None, :]
    dw = np.maximum(boxes[:, 2] - boxes[:, 0], 1)[None, :]
    pc = (pred[:, :2] + pred[:, 2:]) / 2
    dc = (boxes[:, :2] + boxes[:, 2:]) / 2

    out[..., 0] = similarity(entries, feats)
    average = np.stack([e.appearance if e.appearance is not None else np.zeros(feats.shape[1]) for e in entries])
    out[..., 1] = average @ feats.T
    out[..., 2] = iou_matrix(pred, boxes)
    out[..., 3] = (dc[None, :, 0] - pc[:, None, 0]) / ph
    out[..., 4] = (dc[None, :, 1] - pc[:, None, 1]) / ph
    out[..., 5] = np.log(dh / ph)
    out[..., 6] = np.log(dw / pw)
    for k in range(4):
        out[..., 7 + k] = (boxes[None, :, k] - pred[:, None, k]) / ph
    maha = kf.gating_distances(np.stack([e.mean for e in entries]), np.stack([e.cov for e in entries]), xyxy_to_xyah(boxes))
    out[..., 11] = np.minimum(maha, 100) / 100
    out[..., 12] = np.array([min(now - e.last_seen, 10.0) for e in entries])[:, None] / 10
    out[..., 13] = scores[None, :]
    out[..., 14] = crowding[None, :]
    for k, regime in enumerate((Regime.ACTIVE, Regime.OCCLUDED, Regime.EXITED)):
        out[..., 15 + k] = np.array([e.regime == regime for e in entries], dtype=np.float32)[:, None]
    out[..., 18] = np.log1p([e.hits for e in entries])[:, None] / 5
    calibration = calibration or NegativeCalibration()
    out[..., 19] = calibration.rank(1 - out[..., 0])
    out[..., 20] = calibration.rank(1 - out[..., 1])
    np.clip(out[..., 3:11], -10, 10, out=out[..., 3:11])
    return out
