"""Retrieval-augmented association tracker.

Per frame:

1. predict every bank entry with its Kalman filter, and move it with the camera
   when camera motion is given;
2. match active entries to confident detections with a fused overlap/appearance
   cost (the rule of BoT-SORT, Aharon et al. 2022);
3. retrieve for the remaining detections: occluded and recently exited entries,
   by overlap with their prediction or by appearance within the motion gate;
   then, in re-entry mode, exited entries by appearance alone. Each detection
   keeps only its ``top_k`` best entries;
4. match low-score detections to still-unmatched active entries by overlap
   (ByteTrack's second pass), and tentative entries to leftovers;
5. write: matched entries always update their motion, but update their
   appearance only for clean sightings (confident, not overlapped);
6. forget: unmatched active entries become occluded or exited; entries are
   removed after their regime's patience, in seconds.

Every stage abstains through a cost limit: a detection with no acceptable entry
starts a new identity, and an entry with no acceptable detection is not seen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from reidtrack.association.features import NAMES, NegativeCalibration, pair_features
from reidtrack.memory.bank import AppearanceMemory, Entry, Regime, exiting, similarity
from reidtrack.track.assignment import linear_assignment
from reidtrack.track.boxes import iou_matrix, xyah_to_xyxy, xyxy_to_xyah
from reidtrack.track.camera import warp_state
from reidtrack.track.kalman import XYAHKalman

CHI2_95_4DOF = 9.4877


@dataclass
class TrackerConfig:
    high_score: float = 0.6
    low_score: float = 0.1
    new_score: float = 0.7
    match_limit: float = 0.8  # fused overlap cost for active entries
    appearance_limit: float = 0.15  # half cosine distance beyond which appearance is ignored
    proximity_limit: float = 1.0  # overlap distance beyond which appearance is ignored; 1 disables
    gate_appearance: bool = True  # allow active matches by appearance anywhere inside the motion gate
    recall_limit: float = 0.2  # half cosine distance for recalling an occluded entry
    recall_veto: float = 0.35  # half cosine distance at which appearance overrules a good overlap
    reentry_limit: float = 0.15  # half cosine distance for recalling an exited entry
    second_limit: float = 0.5
    tentative_limit: float = 0.7
    top_k: int = 10
    accept: float = 0.5  # learned reranker: minimum same-person probability
    hysteresis: float = 0.0  # learned reranker: bonus for continuing last frame's pairing; 0 is off
    patience_occluded: float = 5.0  # seconds
    patience_exited: float = 0.5  # seconds; covers edge jitter and misread exits
    patience_reentry: float = 30.0  # seconds an exited entry is kept in re-entry mode
    reentry: bool = False  # off for MOT17, which gives returning people new ids
    border_margin: float = 0.02
    write_score: float = 0.6
    write_overlap: float = 0.3
    momentum: float = 0.9
    prototypes: int = 4
    novelty: float = 0.1
    fuse_score: bool = True
    min_box_area: float = 100.0
    max_aspect: float = 1.6
    emit_hidden: float = 0.0  # seconds to keep reporting a hidden person's predicted box; 0 is off
    emit_min_hits: int = 5  # only for people seen at least this often


class RetrievalTracker:
    """``reranker`` maps a (N, M, cues) pair tensor to same-person probabilities; when
    given, it replaces the hand-set active and recall stages with one joint assignment
    over all bank entries. ``recorder`` receives each frame's candidate pairs
    (entry ids, detection indices, cues) for building training data."""

    def __init__(
        self,
        width: int,
        height: int,
        frame_rate: float = 30,
        config: TrackerConfig | None = None,
        reranker: Callable[[np.ndarray], np.ndarray] | None = None,
        recorder: Callable[[list[int], np.ndarray, np.ndarray], None] | None = None,
    ) -> None:
        self.width, self.height, self.frame_rate = width, height, frame_rate
        self.cfg = config or TrackerConfig()
        self.reranker = reranker
        self.recorder = recorder
        self.assignments: dict[int, int] = {}  # track id -> detection index, last frame
        self.calibration = NegativeCalibration()  # same-frame pairs: always different people
        self.kf = XYAHKalman()
        self.memory = AppearanceMemory(self.cfg.momentum, self.cfg.prototypes, self.cfg.novelty)
        self.entries: list[Entry] = []
        self.frame = 0
        self._next_id = 1

    @property
    def now(self) -> float:
        return self.frame / self.frame_rate

    def update(
        self,
        xyxy: np.ndarray,
        scores: np.ndarray,
        features: np.ndarray | None = None,
        warp: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One frame of detections. ``warp`` is the camera motion from the previous frame
        (2x3, see ``reidtrack.track.camera``); bank entries are moved with it."""
        if features is None:
            raise ValueError("RetrievalTracker needs appearance features for every detection")
        cfg = self.cfg
        self.frame += 1
        self._predict()
        if warp is not None and self.entries:
            mean, cov = warp_state(np.stack([e.mean for e in self.entries]), np.stack([e.cov for e in self.entries]), warp)
            for e, m, c in zip(self.entries, mean, cov):
                e.mean, e.cov = m, c
                if e.last_box is not None:
                    corners = e.last_box.reshape(2, 2) @ warp[:, :2].T + warp[:, 2]
                    e.last_box = corners.reshape(4)

        high = np.flatnonzero(scores >= cfg.high_score)
        low = np.flatnonzero((scores >= cfg.low_score) & (scores < cfg.high_score))
        crowding = self._crowding(xyxy)
        clean = (scores >= cfg.write_score) & (crowding < cfg.write_overlap)
        feats = np.asarray(features, dtype=np.float32)
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
        self.calibration.observe(feats[high])

        by_regime = {r: [e for e in self.entries if e.regime == r] for r in Regime}
        active, tentative = by_regime[Regime.ACTIVE], by_regime[Regime.TENTATIVE]
        unseen = by_regime[Regime.OCCLUDED] + by_regime[Regime.EXITED]
        matched: dict[int, int] = {}  # id(entry) -> detection index

        pool = active + unseen
        cues = None
        if self.reranker is not None or self.recorder is not None:
            cues = pair_features(pool, xyxy[high], scores[high], feats[high], crowding[high], self.kf, self.now,
                                 self.calibration)
            if self.recorder is not None:
                self.recorder([e.track_id for e in pool], high, cues)
        if self.reranker is not None:
            # one joint assignment over every bank entry, abstaining below ``accept``
            prob = self.reranker(cues).astype(np.float64) if len(pool) and len(high) else np.zeros((len(pool), len(high)))
            if cfg.hysteresis > 0 and prob.size:
                prob += cfg.hysteresis * self._continuing(pool, cues)
            cost = self._top_k(1 - prob) if prob.size else prob
            free = self._match(pool, high, cost, 1 - cfg.accept, matched)
        else:
            cost = self._cost(active, xyxy[high], scores[high], feats[high], cfg.appearance_limit, recall=False)
            free = self._match(active, high, cost, 1.0, matched)
            cost = self._cost(unseen, xyxy[free], scores[free], feats[free], cfg.recall_limit, recall=True)
            free = self._match(unseen, free, cost, 1.0, matched)
            if cfg.reentry:
                exited = [e for e in by_regime[Regime.EXITED] if id(e) not in matched]
                free = self._match(exited, free, self._reentry_cost(exited, feats[free]), 1.0, matched)
        waiting = [e for e in active if id(e) not in matched]
        self._match(waiting, low, 1 - iou_matrix(self._boxes(waiting), xyxy[low]), cfg.second_limit, matched)
        free = self._match(tentative, free, self._overlap_cost(tentative, xyxy[free], scores[free]),
                           cfg.tentative_limit, matched)

        for entry in self.entries:
            j = matched.get(id(entry))
            if j is not None:
                self._write(entry, xyxy[j], scores[j], feats[j], clean[j])
        self.assignments = {e.track_id: matched[id(e)] for e in self.entries if id(e) in matched}
        self._forget(matched)
        shown = set(matched)
        for j in free:
            if scores[j] >= cfg.new_score:
                entry = self._start(xyxy[j], scores[j], feats[j])
                self.assignments[entry.track_id] = int(j)
                if entry.regime == Regime.ACTIVE:  # first frame: reported at once
                    shown.add(id(entry))
        return self._output(shown)

    # -- costs ----------------------------------------------------------------

    def _match(self, entries: list[Entry], dets: np.ndarray, cost: np.ndarray, limit: float, matched: dict) -> np.ndarray:
        """Assign detections ``dets`` (indices) to ``entries``; return the detections left over."""
        if not entries or len(dets) == 0:
            return dets
        pairs, _, unmatched = linear_assignment(cost, limit)
        for r, c in pairs:
            matched[id(entries[r])] = int(dets[c])
        return dets[unmatched]

    def _overlap_cost(self, entries: list[Entry], boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        dist = 1 - iou_matrix(self._boxes(entries), boxes)
        return 1 - (1 - dist) * scores[None, :] if self.cfg.fuse_score and dist.size else dist

    def _cost(
        self, entries: list[Entry], boxes: np.ndarray, scores: np.ndarray, feats: np.ndarray, app_limit: float, recall: bool
    ) -> np.ndarray:
        """Overlap with the prediction or appearance, each scaled by its own limit, so a
        cost of at most 1 is acceptable by at least one cue.

        For active entries appearance counts only near the prediction (``proximity_limit``,
        as in BoT-SORT) or, with ``gate_appearance``, anywhere inside the motion gate.
        For recalls (occluded or exited entries) it counts inside the motion gate, and
        a clear appearance mismatch vetoes the pair: a stranger stepping into the spot
        where a hidden person is predicted must not inherit their identity.
        """
        cfg = self.cfg
        iou_dist = 1 - iou_matrix(self._boxes(entries), boxes)
        overlap = self._overlap_cost(entries, boxes, scores) / cfg.match_limit
        app_dist = (1 - similarity(entries, feats)) / 2
        app = (app_dist / app_limit).astype(np.float64)
        if app.size:
            if recall or cfg.gate_appearance:
                maha = self.kf.gating_distances(np.stack([e.mean for e in entries]), np.stack([e.cov for e in entries]),
                                                xyxy_to_xyah(boxes))
                app[maha > CHI2_95_4DOF] = np.inf
            if not recall and cfg.proximity_limit < 1:
                app[iou_dist > cfg.proximity_limit] = np.inf
        cost = np.minimum(overlap, app)
        if recall:
            cost[app_dist > cfg.recall_veto] = np.inf
            cost = self._top_k(cost)
        return cost

    def _reentry_cost(self, entries: list[Entry], feats: np.ndarray) -> np.ndarray:
        return self._top_k(((1 - similarity(entries, feats)) / 2 / self.cfg.reentry_limit).astype(np.float64))

    def _top_k(self, cost: np.ndarray) -> np.ndarray:
        """Retrieval: each detection keeps only its ``top_k`` cheapest entries."""
        if cost.size and self.cfg.top_k < cost.shape[0]:
            rank = np.argsort(np.argsort(cost, axis=0, kind="stable"), axis=0)
            cost = np.where(rank < self.cfg.top_k, cost, np.inf)
        return cost

    # -- memory ---------------------------------------------------------------

    def _predict(self) -> None:
        if not self.entries:
            return
        mean = np.stack([e.mean for e in self.entries])
        cov = np.stack([e.cov for e in self.entries])
        unseen = np.array([e.regime in (Regime.OCCLUDED, Regime.EXITED) for e in self.entries])
        mean[unseen, 7] = 0  # freeze height velocity while unseen
        mean, cov = self.kf.predict(mean, cov)
        for e, m, c in zip(self.entries, mean, cov):
            e.mean, e.cov = m, c

    @staticmethod
    def _crowding(xyxy: np.ndarray) -> np.ndarray:
        """Each detection's largest overlap with another one: a stand-in for occlusion.
        Only confident sightings below ``write_overlap`` are written to memory."""
        if len(xyxy) < 2:
            return np.zeros(len(xyxy))
        overlap = iou_matrix(xyxy, xyxy)
        np.fill_diagonal(overlap, 0)
        return overlap.max(axis=1)

    def _continuing(self, pool: list[Entry], cues: np.ndarray) -> np.ndarray:
        """1 for each person matched last frame, at the box that best overlaps where they
        were (IoU at least 0.3); 0 elsewhere. A switch then has to win by the bonus."""
        out = np.zeros(cues.shape[:2])
        last = cues[..., NAMES.index("iou_last")]
        best = last.argmax(axis=1)
        just_seen = np.array([self.now - e.last_seen <= 1.5 / self.frame_rate for e in pool])
        rows = np.flatnonzero(just_seen & (last[np.arange(len(pool)), best] >= 0.3))
        out[rows, best[rows]] = 1
        return out

    def _write(self, entry: Entry, box: np.ndarray, score: float, feature: np.ndarray, clean: bool) -> None:
        entry.mean, entry.cov = self.kf.update(entry.mean, entry.cov, xyxy_to_xyah(box))
        if entry.regime in (Regime.OCCLUDED, Regime.EXITED):
            entry.recalls += 1
        entry.regime = Regime.ACTIVE
        entry.last_seen = self.now
        entry.last_box = np.asarray(box, dtype=np.float64)
        entry.score = float(score)
        entry.hits += 1
        if clean:
            self.memory.write(entry, feature)

    def _start(self, box: np.ndarray, score: float, feature: np.ndarray) -> Entry:
        mean, cov = self.kf.initiate(xyxy_to_xyah(box))
        entry = Entry(self._next_id, mean, cov, first_seen=self.now, last_seen=self.now, score=float(score),
                      last_box=np.asarray(box, dtype=np.float64))
        entry.regime = Regime.ACTIVE if self.frame == 1 else Regime.TENTATIVE
        self.memory.write(entry, feature)  # a first view even if crowded: better than none
        self._next_id += 1
        self.entries.append(entry)
        return entry

    def _forget(self, matched: dict) -> None:
        cfg = self.cfg
        kept = []
        for e in self.entries:
            if id(e) not in matched:
                if e.regime == Regime.TENTATIVE:
                    continue  # unconfirmed and unmatched
                if e.regime == Regime.ACTIVE:  # just lost: hidden, or walked out?
                    leaving = exiting(xyah_to_xyxy(e.mean[:4]), e.mean[4:6], self.width, self.height, cfg.border_margin)
                    e.regime = Regime.EXITED if leaving else Regime.OCCLUDED
                if e.regime == Regime.EXITED:
                    patience = cfg.patience_reentry if cfg.reentry else cfg.patience_exited
                else:
                    patience = cfg.patience_occluded
                if self.now - e.last_seen > patience:
                    continue
            kept.append(e)
        self.entries = kept

    def _boxes(self, entries: list[Entry]) -> np.ndarray:
        return np.array([xyah_to_xyxy(e.mean[:4]) for e in entries]).reshape(-1, 4)

    def _output(self, shown: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids, boxes, scores = [], [], []
        for e in self.entries:
            visible = e.regime == Regime.ACTIVE and id(e) in shown
            hidden = (
                e.regime == Regime.OCCLUDED
                and self.cfg.emit_hidden > 0
                and e.hits >= self.cfg.emit_min_hits
                and self.now - e.last_seen <= self.cfg.emit_hidden + 1e-9  # frame / fps rounds
            )
            if not (visible or hidden):
                continue
            box = xyah_to_xyxy(e.mean[:4])
            w, h = box[2] - box[0], box[3] - box[1]
            if w * h > self.cfg.min_box_area and w / h <= self.cfg.max_aspect:
                ids.append(e.track_id)
                boxes.append(box)
                scores.append(e.score)
        return (
            np.array(ids, dtype=np.int32),
            np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(scores, dtype=np.float32),
        )
