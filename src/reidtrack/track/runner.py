"""Run a tracker over every sequence of a split, from precomputed detections."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from reidtrack.data.mot import SeqInfo, Tracks, save_tracks
from reidtrack.data.mot17 import Mot17
from reidtrack.eval.latency import StageTimer
from reidtrack.eval.metrics import split_sequences


class Tracker(Protocol):
    def update(
        self, xyxy: np.ndarray, scores: np.ndarray, features: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Consume one frame of detections; return (track ids, boxes, scores) to report."""
        ...


def run_split(
    make_tracker: Callable[[SeqInfo], Tracker],
    split: str = "val_half",
    detector: str = "FRCNN",
    root: str | Path = "data/mot17",
    timer: StageTimer | None = None,
    embeddings: str | None = None,
    camera: bool = False,
    stride: int = 1,
    sequences: list[str] | None = None,
) -> dict[str, Tracks]:
    """Track every sequence of ``split``; a fresh tracker is made per sequence.

    ``sequences`` names the videos to track instead, from either subset, over the
    frames of ``split``; "train" covers a whole video, which suits test videos.

    ``stride`` feeds the tracker every n-th frame only, as an edge device that skips
    frames would, and tells it the lower frame rate. Camera motion is not composed across
    skipped frames, so ``camera`` needs ``stride`` 1.

    ``embeddings`` names a feature cache (see ``reidtrack.retrieval.cache``) whose
    rows are passed to the tracker with the detections. ``camera`` passes the cached
    camera motion (see ``reidtrack.track.camera``) as ``warp``. Only the tracker's
    ``update`` is timed, under the stage name "track".
    """
    from reidtrack.retrieval.cache import load_embeddings
    from reidtrack.track.camera import load_warps

    if camera and stride != 1:
        raise ValueError("camera motion is cached per frame; use stride 1 with camera")
    data = Mot17(root)
    results = {}
    for name in sequences or split_sequences(split, root):
        seq = data.sequence(name)
        rng = seq.frames(split)
        det = seq.load_det(detector)
        in_split = (det.frame >= rng.first) & (det.frame <= rng.last)
        features = None
        if embeddings is not None:
            features = load_embeddings(root, embeddings, detector, name, det.frame)[in_split]
        det = det.select(in_split)
        warps = load_warps(root, name) if camera else None
        bounds = np.searchsorted(det.frame, np.arange(rng.first, rng.last + 2))
        info = seq.info if stride == 1 else dataclasses.replace(seq.info, frame_rate=seq.info.frame_rate / stride)
        tracker = make_tracker(info)
        frames, ids, boxes, scores = [], [], [], []
        for frame in range(rng.first, rng.last + 1, stride):
            i = frame - rng.first
            lo, hi = bounds[i], bounds[i + 1]
            args = (det.xyxy[lo:hi].astype(np.float64), det.score[lo:hi], None if features is None else features[lo:hi])
            kwargs = {} if warps is None else {"warp": warps[frame - 1]}
            if timer is None:
                out = tracker.update(*args, **kwargs)
            else:
                with timer.stage("track"):
                    out = tracker.update(*args, **kwargs)
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
