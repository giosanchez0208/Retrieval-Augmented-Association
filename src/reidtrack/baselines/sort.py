"""SORT (Bewley et al., 2016): Kalman prediction, IoU cost, Hungarian matching.

Written from the paper using the reference implementation's defaults: a track
survives ``max_age`` frames without a detection and is reported once it has
``min_hits`` consecutive matches (or during the first ``min_hits`` frames).
There is no appearance model and no memory beyond ``max_age``.
"""

from __future__ import annotations

import numpy as np

from reidtrack.track.assignment import linear_assignment
from reidtrack.track.boxes import iou_matrix, xysr_to_xyxy, xyxy_to_xysr
from reidtrack.track.kalman import XYSRKalman


class _Track:
    __slots__ = ("track_id", "x", "P", "score", "misses", "streak")

    def __init__(self, track_id: int, x: np.ndarray, P: np.ndarray, score: float) -> None:
        self.track_id = track_id
        self.x, self.P = x, P
        self.score = score
        self.misses = 0  # frames since the last matched detection
        self.streak = 0  # consecutive matched frames


class Sort:
    def __init__(
        self,
        frame_rate: float = 30,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        min_score: float = 0.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.min_score = min_score
        self.kf = XYSRKalman()
        self.tracks: list[_Track] = []
        self.frame = 0
        self._next_id = 1

    def update(self, xyxy: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.frame += 1
        keep = scores >= self.min_score
        dets, det_scores = xyxy[keep], scores[keep]

        predicted = []
        for track in self.tracks:
            track.x, track.P = self.kf.predict(track.x, track.P)
            if track.misses > 0:
                track.streak = 0
            track.misses += 1
            predicted.append(xysr_to_xyxy(track.x[:4]))
        alive = [i for i, box in enumerate(predicted) if np.all(np.isfinite(box))]
        self.tracks = [self.tracks[i] for i in alive]
        predicted = np.array([predicted[i] for i in alive]).reshape(-1, 4)

        matches, unmatched_dets = self._associate(dets, predicted)
        for d, t in matches:
            track = self.tracks[t]
            track.x, track.P = self.kf.update(track.x, track.P, xyxy_to_xysr(dets[d]))
            track.score = float(det_scores[d])
            track.misses = 0
            track.streak += 1
        for d in unmatched_dets:
            x, P = self.kf.initiate(xyxy_to_xysr(dets[d]))
            self.tracks.append(_Track(self._next_id, x, P, float(det_scores[d])))
            self._next_id += 1

        ids, boxes, out_scores = [], [], []
        for track in self.tracks:
            if track.misses == 0 and (track.streak >= self.min_hits or self.frame <= self.min_hits):
                ids.append(track.track_id)
                boxes.append(xysr_to_xyxy(track.x[:4]))
                out_scores.append(track.score)
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return np.array(ids, dtype=np.int32), np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(out_scores, dtype=np.float32)

    def _associate(self, dets: np.ndarray, predicted: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
        iou = iou_matrix(dets, predicted)
        if iou.size == 0:
            return [], list(range(len(dets)))
        above = iou > self.iou_threshold
        if above.sum(1).max() == 1 and above.sum(0).max() == 1:
            pairs = np.argwhere(above)  # unambiguous: every overlap is exclusive
        else:
            pairs, _, _ = linear_assignment(-iou)
        matches = [(int(d), int(t)) for d, t in pairs if iou[d, t] >= self.iou_threshold]
        matched = {d for d, _ in matches}
        return matches, [d for d in range(len(dets)) if d not in matched]
