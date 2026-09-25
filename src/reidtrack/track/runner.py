"""Run a tracker over every sequence of a split, from precomputed detections."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from reidtrack.data.mot import SeqInfo, Tracks, save_tracks
from reidtrack.data.mot17 import Mot17
from reidtrack.eval.latency import StageTimer
from reidtrack.eval.metrics import split_sequences


class Tracker(Protocol):
    def update(self, xyxy: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Consume one frame of detections; return (track ids, boxes, scores) to report."""
        ...


def run_split(
    make_tracker: Callable[[SeqInfo], Tracker],
    split: str = "val_half",
    detector: str = "FRCNN",
    root: str | Path = "data/mot17",
    timer: StageTimer | None = None,
) -> dict[str, Tracks]:
    """Track every sequence of ``split``; a fresh tracker is made per sequence.

    Only the tracker's ``update`` is timed, under the stage name "track".
    """
    data = Mot17(root)
    results = {}
    for name in split_sequences(split, root):
        seq = data.sequence(name)
        rng = seq.frames(split)
        det = seq.load_det(detector, split)
        bounds = np.searchsorted(det.frame, np.arange(rng.first, rng.last + 2))
        tracker = make_tracker(seq.info)
        frames, ids, boxes, scores = [], [], [], []
        for i, frame in enumerate(range(rng.first, rng.last + 1)):
            lo, hi = bounds[i], bounds[i + 1]
            if timer is None:
                out = tracker.update(det.xyxy[lo:hi].astype(np.float64), det.score[lo:hi])
            else:
                with timer.stage("track"):
                    out = tracker.update(det.xyxy[lo:hi].astype(np.float64), det.score[lo:hi])
                timer.next_frame()
            frames.append(np.full(len(out[0]), frame, dtype=np.int32))
            ids.append(out[0])
            boxes.append(out[1])
            scores.append(out[2])
        results[name] = Tracks(
            frame=np.concatenate(frames).astype(np.int32),
            track_id=np.concatenate(ids).astype(np.int32),
            xyxy=np.concatenate(boxes).astype(np.float32).reshape(-1, 4),
            score=np.concatenate(scores).astype(np.float32),
        )
    return results


def save_results(results: dict[str, Tracks], out_dir: str | Path) -> None:
    for name, tracks in results.items():
        save_tracks(Path(out_dir) / f"{name}.txt", tracks)
