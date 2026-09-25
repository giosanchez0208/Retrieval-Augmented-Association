"""DeepSORT (Wojke et al., 2017): SORT plus an appearance gallery per track.

Written from the paper with the reference defaults. Each confirmed track keeps its
last ``budget`` appearance vectors. Detections are matched to tracks by the
smallest cosine distance to that gallery, allowed only where the Kalman
prediction makes the position plausible (Mahalanobis gate at the 95% chi-square
bound). A matching cascade lets recently seen tracks choose first. Leftover
detections are matched by IoU to tracks that are unconfirmed or were seen in the
previous frame. In retrieval terms, this is the naive version: nearest-neighbour
lookup, no reranking, every match written to memory.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from reidtrack.track.boxes import iou_matrix, xyah_to_xyxy, xyxy_to_xyah
from reidtrack.track.camera import warp_state
from reidtrack.track.kalman import XYAHKalman

CHI2_95_4DOF = 9.4877  # 95% quantile of the chi-square distribution, 4 degrees of freedom


class _Track:
    __slots__ = ("track_id", "mean", "cov", "hits", "misses", "confirmed", "gallery", "score")

    def __init__(self, track_id: int, mean: np.ndarray, cov: np.ndarray, feature: np.ndarray, score: float, budget: int):
        self.track_id = track_id
        self.mean, self.cov = mean, cov
        self.hits = 1
        self.misses = 0  # frames since the last matched detection
        self.confirmed = False
        self.gallery: deque[np.ndarray] = deque([feature], maxlen=budget)
        self.score = score


class DeepSort:
    def __init__(
        self,
        frame_rate: float = 30,
        min_score: float = 0.3,
        max_cosine_distance: float = 0.2,
        budget: int = 100,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
    ) -> None:
        self.min_score = min_score
        self.max_cosine_distance = max_cosine_distance
        self.budget = budget
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = XYAHKalman()
        self.tracks: list[_Track] = []
        self._next_id = 1

    def update(
        self, xyxy: np.ndarray, scores: np.ndarray, features: np.ndarray | None = None, warp: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``warp`` is the optional camera motion from the previous frame, applied to every
        track's prediction as in BoT-SORT (not part of the original DeepSORT)."""
        if features is None:
            raise ValueError("DeepSort needs appearance features for every detection")
        keep = scores >= self.min_score
        dets, det_scores, feats = xyxy[keep], scores[keep], features[keep]
        measurements = xyxy_to_xyah(dets) if len(dets) else np.zeros((0, 4))

        if self.tracks:
            mean, cov = self.kf.predict(np.stack([t.mean for t in self.tracks]), np.stack([t.cov for t in self.tracks]))
            if warp is not None:
                mean, cov = warp_state(mean, cov, warp)
            for t, m, c in zip(self.tracks, mean, cov):
                t.mean, t.cov = m, c
                t.misses += 1

        confirmed = [i for i, t in enumerate(self.tracks) if t.confirmed]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.confirmed]
        matches, unmatched_dets = self._cascade(confirmed, measurements, feats)
        matched_tracks = {t for t, _ in matches}
        leftover = [i for i in confirmed if i not in matched_tracks]
        candidates = unconfirmed + [i for i in leftover if self.tracks[i].misses == 1]
        iou_matches, unmatched_dets = self._iou_match(candidates, dets, unmatched_dets)
        matches += iou_matches

        for t, d in matches:
            track = self.tracks[t]
            track.mean, track.cov = self.kf.update(track.mean, track.cov, measurements[d])
            track.gallery.append(feats[d])
            track.hits += 1
            track.misses = 0
            track.score = float(det_scores[d])
            if not track.confirmed and track.hits >= self.n_init:
                track.confirmed = True
        matched_tracks = {t for t, _ in matches}
        self.tracks = [
            t for i, t in enumerate(self.tracks)
            if i in matched_tracks or (t.confirmed and t.misses <= self.max_age)
        ]
        for d in unmatched_dets:
            mean, cov = self.kf.initiate(measurements[d])
            self.tracks.append(_Track(self._next_id, mean, cov, feats[d], float(det_scores[d]), self.budget))
            self._next_id += 1

        out = [t for t in self.tracks if t.confirmed and t.misses <= 1]
        return (
            np.array([t.track_id for t in out], dtype=np.int32),
            np.array([xyah_to_xyxy(t.mean[:4]) for t in out], dtype=np.float32).reshape(-1, 4),
            np.array([t.score for t in out], dtype=np.float32),
        )

    def _cascade(self, track_ids: list[int], measurements: np.ndarray, feats: np.ndarray) -> tuple[list, list[int]]:
        unmatched = list(range(len(measurements)))
        matches: list[tuple[int, int]] = []
        for level in range(1, self.max_age + 1):
            if not unmatched:
                break
            level_tracks = [i for i in track_ids if self.tracks[i].misses == level]
            if not level_tracks:
                continue
            cost = np.empty((len(level_tracks), len(unmatched)))
            for r, i in enumerate(level_tracks):
                track = self.tracks[i]
                gallery = np.stack(track.gallery)
                cost[r] = (1 - feats[unmatched] @ gallery.T).min(axis=1)
                gated = self.kf.gating_distance(track.mean, track.cov, measurements[unmatched]) > CHI2_95_4DOF
                cost[r, gated] = np.inf
            pairs, unmatched_cols = _min_cost_matching(cost, self.max_cosine_distance)
            matches += [(level_tracks[r], unmatched[c]) for r, c in pairs]
            unmatched = [unmatched[c] for c in unmatched_cols]
        return matches, unmatched

    def _iou_match(self, track_ids: list[int], dets: np.ndarray, det_ids: list[int]) -> tuple[list, list[int]]:
        if not track_ids or not det_ids:
            return [], det_ids
        boxes = np.array([xyah_to_xyxy(self.tracks[i].mean[:4]) for i in track_ids])
        cost = 1 - iou_matrix(boxes, dets[det_ids])
        pairs, unmatched_cols = _min_cost_matching(cost, self.max_iou_distance)
        return [(track_ids[r], det_ids[c]) for r, c in pairs], [det_ids[c] for c in unmatched_cols]


def _min_cost_matching(cost: np.ndarray, max_distance: float) -> tuple[list[tuple[int, int]], list[int]]:
    """Hungarian matching on a cost matrix clipped at ``max_distance``; pairs above it are rejected."""
    cost = np.minimum(cost, max_distance + 1e-5)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= max_distance]
    matched_cols = {c for _, c in pairs}
    return pairs, [c for c in range(cost.shape[1]) if c not in matched_cols]
