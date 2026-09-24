"""Summary statistics for the prepared MOT17 training sequences.

    python -m reidtrack.data.stats --root data/mot17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from reidtrack.data.mot import GroundTruth
from reidtrack.data.mot17 import DETECTORS, Mot17, Sequence
from reidtrack.report import format_table

HIDDEN = 0.1  # visibility below this counts as fully hidden
LOW_VISIBILITY = 0.25
# Crops below NOISY mostly show whoever is in front of the labelled person;
# crops at or above CLEAN are usable for appearance training.
NOISY = 0.2
CLEAN = 0.3


def occlusion_gaps(frames: np.ndarray, visibility: np.ndarray, threshold: float = HIDDEN) -> list[int]:
    """Lengths, in frames, of hidden stretches after which the person is visible again.

    Takes one track sorted by frame. Stretches before the first visible frame,
    or still open at the end of the track, are not counted.
    """
    gaps = []
    seen = False
    start = None
    for frame, vis in zip(frames.tolist(), visibility.tolist()):
        if vis < threshold:
            if seen and start is None:
                start = frame
        else:
            if start is not None:
                gaps.append(frame - start)
                start = None
            seen = True
    return gaps


def track_gaps(gt: GroundTruth) -> np.ndarray:
    order = np.lexsort((gt.frame, gt.track_id))
    track, frame, vis = gt.track_id[order], gt.frame[order], gt.visibility[order]
    bounds = np.flatnonzero(np.diff(track)) + 1
    gaps = [g for f, v in zip(np.split(frame, bounds), np.split(vis, bounds)) for g in occlusion_gaps(f, v)]
    return np.asarray(gaps, dtype=np.int64)


def truncated(gt: GroundTruth, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = gt.xyxy.T
    return (x1 < 0) | (y1 < 0) | (x2 > width) | (y2 > height)


def sequence_summary(seq: Sequence) -> dict:
    gt = seq.load_gt().targets()
    info = seq.info
    gaps_s = track_gaps(gt) / info.frame_rate
    return {
        "ids": len(np.unique(gt.track_id)),
        "boxes": len(gt),
        "low_vis": int((gt.visibility < LOW_VISIBILITY).sum()),
        "truncated": int(truncated(gt, info.width, info.height).sum()),
        "gaps": len(gaps_s),
        "gaps_1s": int((gaps_s > 1).sum()),
        "gaps_2s": int((gaps_s > 2).sum()),
        "longest_s": float(gaps_s.max()) if len(gaps_s) else 0.0,
    }


def split_summary(seq: Sequence) -> dict:
    train = seq.load_gt("train_half").targets()
    val = seq.load_gt("val_half").targets()
    train_ids = set(train.track_id.tolist())
    val_ids = set(val.track_id.tolist())
    return {
        "train_ids": len(train_ids),
        "val_ids": len(val_ids),
        "shared_ids": len(train_ids & val_ids),
        "train_crops": len(train),
        "clean_crops": int((train.visibility >= CLEAN).sum()),
        "noisy_crops": int((train.visibility < NOISY).sum()),
    }


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "-"


def _sequence_table(sequences: list[Sequence]) -> str:
    rows, total = [], dict.fromkeys(("ids", "boxes", "low_vis", "truncated", "gaps", "gaps_1s", "gaps_2s"), 0)
    frames, longest = 0, 0.0
    for seq in sequences:
        s, info = sequence_summary(seq), seq.info
        rows.append(
            [seq.name, info.frame_rate, info.length, f"{info.width}x{info.height}",
             "moving" if seq.moving_camera else "static", s["ids"], s["boxes"],
             _pct(s["low_vis"], s["boxes"]), _pct(s["truncated"], s["boxes"]),
             s["gaps"], s["gaps_1s"], s["gaps_2s"], f"{s['longest_s']:.1f}"]
        )
        for key in total:
            total[key] += s[key]
        frames += info.length
        longest = max(longest, s["longest_s"])
    footer = [
        ["total", "", frames, "", "", total["ids"], total["boxes"],
         _pct(total["low_vis"], total["boxes"]), _pct(total["truncated"], total["boxes"]),
         total["gaps"], total["gaps_1s"], total["gaps_2s"], f"{longest:.1f}"]
    ]
    return format_table(
        ["sequence", "fps", "frames", "size", "camera", "ids", "boxes",
         f"vis<{LOW_VISIBILITY}", "past border", "gaps", ">1 s", ">2 s", "longest s"],
        rows,
        caption="MOT17 train, pedestrian targets",
        footer=footer,
        note=f"gaps: hidden stretches (vis<{HIDDEN}) that end with the person visible again",
    )


def _split_table(sequences: list[Sequence]) -> str:
    keys = ("train_ids", "val_ids", "shared_ids", "train_crops", "clean_crops", "noisy_crops")
    rows, total = [], dict.fromkeys(keys, 0)

    def cells(name: str, s: dict) -> list[object]:
        return [name, s["train_ids"], s["val_ids"], s["shared_ids"], s["train_crops"],
                f"{s['clean_crops']} ({_pct(s['clean_crops'], s['train_crops'])})",
                f"{s['noisy_crops']} ({_pct(s['noisy_crops'], s['train_crops'])})"]

    for seq in sequences:
        s = split_summary(seq)
        rows.append(cells(seq.name, s))
        for key in keys:
            total[key] += s[key]
    return format_table(
        ["sequence", "train ids", "val ids", "shared", "train crops", f"vis>={CLEAN}", f"vis<{NOISY}"],
        rows,
        caption="MOT17 half split",
        footer=[cells("total", total)],
        note="shared: ids in both halves; crop counts are for the train half",
    )


def _detection_table(sequences: list[Sequence]) -> str:
    rows = []
    for detector in DETECTORS:
        scores = np.concatenate([seq.load_det(detector).score for seq in sequences])
        rows.append([detector, len(scores), f"{scores.min():.2f}", f"{scores.max():.2f}"])
    return format_table(["detector", "boxes", "min score", "max score"], rows, caption="MOT17 train, public detections")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reidtrack.data.stats", description="Statistics for the prepared MOT17 training set."
    )
    parser.add_argument("--root", type=Path, default=Path("data/mot17"), help="prepared dataset root")
    args = parser.parse_args(argv)
    sequences = [s for s in Mot17(args.root).sequences("train") if s.has_gt]
    print(_sequence_table(sequences), _split_table(sequences), _detection_table(sequences), sep="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
