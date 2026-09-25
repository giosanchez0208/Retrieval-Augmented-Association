"""ByteTrack (Zhang et al., 2022): associate every detection box.

High-score detections are matched first against active and lost tracks. Low-score
detections, often partly occluded people, get a second pass against the active
tracks the first pass missed. Follows the reference implementation (MIT) and its
MOT17 evaluation defaults, including score-fused IoU cost, inclusive-pixel IoU, and
dropping output boxes that are tiny or wider than 1.6 x their height.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from reidtrack.track.assignment import linear_assignment
from reidtrack.track.boxes import iou_matrix, xyah_to_xyxy, xyxy_to_xyah
from reidtrack.track.kalman import XYAHKalman


class State(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class _Track:
    __slots__ = ("track_id", "box", "score", "mean", "cov", "state", "activated", "frame_id", "start_frame")

    def __init__(self, box: np.ndarray, score: float) -> None:
        self.track_id = 0
        self.box = np.asarray(box, dtype=np.float64)
        self.score = float(score)
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.state = State.NEW
        self.activated = False
        self.frame_id = 0  # last frame with a matched detection
        self.start_frame = 0

    @property
    def xyxy(self) -> np.ndarray:
        return self.box if self.mean is None else xyah_to_xyxy(self.mean[:4])


class ByteTrack:
    def __init__(
        self,
        frame_rate: float = 30,
        track_thresh: float = 0.6,
        match_thresh: float = 0.9,
        track_buffer: int = 30,
        min_box_area: float = 100.0,
        max_aspect: float = 1.6,
        fuse_score: bool = True,
    ) -> None:
        self.track_thresh = track_thresh
        self.new_track_thresh = track_thresh + 0.1
        self.match_thresh = match_thresh
        self.max_time_lost = int(frame_rate / 30 * track_buffer)
        self.min_box_area = min_box_area
        self.max_aspect = max_aspect
        self.fuse_score = fuse_score
        self.kf = XYAHKalman()
        self.tracked: list[_Track] = []
        self.lost: list[_Track] = []
        self.removed_ids: set[int] = set()
        self.frame_id = 0
        self._next_id = 1

    def update(self, xyxy: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.frame_id += 1
        activated, refound, newly_lost, removed = [], [], [], []

        high = scores > self.track_thresh
        low = (scores > 0.1) & (scores < self.track_thresh)
        dets = [_Track(b, s) for b, s in zip(xyxy[high], scores[high])]
        dets_low = [_Track(b, s) for b, s in zip(xyxy[low], scores[low])]
        unconfirmed = [t for t in self.tracked if not t.activated]
        confirmed = [t for t in self.tracked if t.activated]

        # 1. confirmed and lost tracks vs high-score detections
        pool = _join(confirmed, self.lost)
        self._predict(pool)
        cost = self._cost(pool, dets, fuse=self.fuse_score)
        matches, unmatched_tracks, unmatched_dets = linear_assignment(cost, self.match_thresh)
        for t, d in matches:
            self._match(pool[t], dets[d], activated, refound)

        # 2. tracks still active but unmatched vs low-score detections
        remaining = [pool[i] for i in unmatched_tracks if pool[i].state == State.TRACKED]
        matches, unmatched_remaining, _ = linear_assignment(self._cost(remaining, dets_low), 0.5)
        for t, d in matches:
            self._match(remaining[t], dets_low[d], activated, refound)
        for i in unmatched_remaining:
            track = remaining[i]
            if track.state != State.LOST:
                track.state = State.LOST
                newly_lost.append(track)

        # 3. tracks seen only once vs the leftover high-score detections
        dets = [dets[i] for i in unmatched_dets]
        cost = self._cost(unconfirmed, dets, fuse=self.fuse_score)
        matches, unmatched_unconfirmed, unmatched_dets = linear_assignment(cost, 0.7)
        for t, d in matches:
            self._match(unconfirmed[t], dets[d], activated, refound)
        for i in unmatched_unconfirmed:
            unconfirmed[i].state = State.REMOVED
            removed.append(unconfirmed[i])

        # 4. new tracks from confident leftovers
        for i in unmatched_dets:
            det = dets[i]
            if det.score >= self.new_track_thresh:
                self._start(det)
                activated.append(det)

        # 5. forget tracks lost for longer than the buffer
        for track in self.lost:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = State.REMOVED
                removed.append(track)

        self.tracked = [t for t in self.tracked if t.state == State.TRACKED]
        self.tracked = _join(_join(self.tracked, activated), refound)
        self.lost = _subtract(self.lost, self.tracked) + newly_lost
        # The reference drops only tracks removed in earlier frames here, so a track
        # removed this frame can still be re-found on the next one.
        self.lost = [t for t in self.lost if t.track_id not in self.removed_ids]
        self.removed_ids.update(t.track_id for t in removed)
        self.tracked, self.lost = self._remove_duplicates(self.tracked, self.lost)
        return self._output()

    def _predict(self, tracks: list[_Track]) -> None:
        if not tracks:
            return
        mean = np.stack([t.mean for t in tracks])
        cov = np.stack([t.cov for t in tracks])
        not_tracked = np.array([t.state != State.TRACKED for t in tracks])
        mean[not_tracked, 7] = 0  # freeze height velocity while lost
        mean, cov = self.kf.predict(mean, cov)
        for t, m, c in zip(tracks, mean, cov):
            t.mean, t.cov = m, c

    def _cost(self, tracks: list[_Track], dets: list[_Track], fuse: bool = False) -> np.ndarray:
        a = np.array([t.xyxy for t in tracks]).reshape(-1, 4)
        b = np.array([d.xyxy for d in dets]).reshape(-1, 4)
        similarity = iou_matrix(a, b, plus_one=True)
        if fuse and similarity.size:
            similarity = similarity * np.array([d.score for d in dets])[None, :]
        return 1 - similarity

    def _start(self, det: _Track) -> None:
        det.track_id = self._next_id
        self._next_id += 1
        det.mean, det.cov = self.kf.initiate(xyxy_to_xyah(det.box))
        det.state = State.TRACKED
        det.activated = self.frame_id == 1  # otherwise confirmed by a second match
        det.frame_id = det.start_frame = self.frame_id

    def _match(self, track: _Track, det: _Track, activated: list, refound: list) -> None:
        was_tracked = track.state == State.TRACKED
        track.mean, track.cov = self.kf.update(track.mean, track.cov, xyxy_to_xyah(det.box))
        track.state = State.TRACKED
        track.activated = True
        track.frame_id = self.frame_id
        track.score = det.score
        (activated if was_tracked else refound).append(track)

    def _remove_duplicates(self, a: list[_Track], b: list[_Track]) -> tuple[list[_Track], list[_Track]]:
        if not a or not b:
            return a, b
        iou = iou_matrix(np.array([t.xyxy for t in a]), np.array([t.xyxy for t in b]), plus_one=True)
        drop_a, drop_b = set(), set()
        for p, q in zip(*np.nonzero(1 - iou < 0.15)):
            if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
                drop_b.add(q)
            else:
                drop_a.add(p)
        return [t for i, t in enumerate(a) if i not in drop_a], [t for i, t in enumerate(b) if i not in drop_b]

    def _output(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids, boxes, scores = [], [], []
        for t in self.tracked:
            if not t.activated:
                continue
            box = t.xyxy
            w, h = box[2] - box[0], box[3] - box[1]
            if w * h > self.min_box_area and w / h <= self.max_aspect:
                ids.append(t.track_id)
                boxes.append(box)
                scores.append(t.score)
        return (
            np.array(ids, dtype=np.int32),
            np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(scores, dtype=np.float32),
        )


def _join(a: list[_Track], b: list[_Track]) -> list[_Track]:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def _subtract(a: list[_Track], b: list[_Track]) -> list[_Track]:
    drop = {t.track_id for t in b}
    return [t for t in a if t.track_id not in drop]
