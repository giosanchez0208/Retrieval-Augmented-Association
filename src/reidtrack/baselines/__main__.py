"""Run a baseline tracker on a MOT17 split and score it.

    python -m reidtrack.baselines sort --min-score 0.5
    python -m reidtrack.baselines bytetrack --split val_half
    python -m reidtrack.baselines deepsort --embeddings osnet_x1_0_msmt17

Results go to runs/<name>/<sequence>.txt with a scores.json beside them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reidtrack.baselines import ByteTrack, DeepSort, Sort
from reidtrack.data.mot17 import DETECTORS
from reidtrack.data.splits import SPLITS
from reidtrack.eval.latency import StageTimer
from reidtrack.eval.metrics import Scores, evaluate
from reidtrack.report import format_table
from reidtrack.track.runner import run_split, save_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.baselines", description="Run a baseline tracker.")
    parser.add_argument("--det", choices=DETECTORS, default="FRCNN", help="public detections")
    parser.add_argument("--split", choices=SPLITS, default="val_half")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"), help="prepared dataset root")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="output folder")
    parser.add_argument("--name", help="run name (default: <tracker>_<det>_<split>)")
    trackers = parser.add_subparsers(dest="tracker", required=True)

    sort = trackers.add_parser("sort", help="SORT (IoU + Kalman, no memory)")
    sort.add_argument("--min-score", type=float, default=0.0)
    sort.add_argument("--max-age", type=int, default=1)
    sort.add_argument("--min-hits", type=int, default=3)
    sort.add_argument("--iou", type=float, default=0.3)

    byte = trackers.add_parser("bytetrack", help="ByteTrack (two-stage IoU matching)")
    byte.add_argument("--track-thresh", type=float, default=0.6)
    byte.add_argument("--match-thresh", type=float, default=0.9)
    byte.add_argument("--buffer", type=int, default=30, help="frames a lost track is kept, at 30 fps")
    byte.add_argument("--min-box-area", type=float, default=100.0)

    deep = trackers.add_parser("deepsort", help="DeepSORT (appearance gallery + matching cascade)")
    deep.add_argument("--embeddings", default="osnet_x1_0_msmt17", help="feature cache name")
    deep.add_argument("--min-score", type=float, default=0.3)
    deep.add_argument("--max-cosine", type=float, default=0.2)
    deep.add_argument("--budget", type=int, default=100)
    deep.add_argument("--max-iou-distance", type=float, default=0.7)
    deep.add_argument("--max-age", type=int, default=70)
    deep.add_argument("--n-init", type=int, default=3)

    args = parser.parse_args(argv)
    embeddings = None
    if args.tracker == "sort":
        config = {"min_score": args.min_score, "max_age": args.max_age, "min_hits": args.min_hits,
                  "iou_threshold": args.iou}
        make = lambda info: Sort(frame_rate=info.frame_rate, **config)  # noqa: E731
    elif args.tracker == "bytetrack":
        config = {"track_thresh": args.track_thresh, "match_thresh": args.match_thresh,
                  "track_buffer": args.buffer, "min_box_area": args.min_box_area}
        make = lambda info: ByteTrack(frame_rate=info.frame_rate, **config)  # noqa: E731
    else:
        config = {"min_score": args.min_score, "max_cosine_distance": args.max_cosine, "budget": args.budget,
                  "max_iou_distance": args.max_iou_distance, "max_age": args.max_age, "n_init": args.n_init}
        make = lambda info: DeepSort(frame_rate=info.frame_rate, **config)  # noqa: E731
        embeddings = args.embeddings

    name = args.name or f"{args.tracker}_{args.det}_{args.split}"
    timer = StageTimer(warmup=10)
    results = run_split(make, args.split, args.det, args.root, timer, embeddings)
    ev = evaluate(results, args.split, args.root)

    out = args.runs / name
    save_results(results, out)
    ms = timer.summary()["track"]["mean"]
    (out / "scores.json").write_text(
        json.dumps({"tracker": args.tracker, "detector": args.det, "config": config,
                    "track_ms_per_frame": ms, **ev.to_dict()}, indent=2) + "\n"
    )
    print(
        format_table(
            ["sequence", *Scores.HEADERS],
            [[n, *s.row()] for n, s in ev.sequences.items()],
            caption=f"MOT17 {args.split}, {args.tracker}, {args.det} detections",
            footer=[["combined", *ev.combined.row()]],
            note=f"tracking only: {ms:.2f} ms/frame ({1e3 / ms:.0f} fps); detections precomputed",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
