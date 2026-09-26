"""Run the retrieval-augmented tracker on a MOT17 split and score it.

    python -m reidtrack --embeddings osnet_x1_0_msmt17
    python -m reidtrack --embeddings osnet_x1_0_msmt17 --set patience_occluded=3 --set top_k=5

Any field of ``TrackerConfig`` can be overridden with --set. Results go to
runs/<name>/ with a scores.json beside them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from reidtrack.data.mot17 import ALL_DETECTORS
from reidtrack.data.splits import SPLITS
from reidtrack.eval.latency import StageTimer
from reidtrack.eval.metrics import Scores, evaluate
from reidtrack.report import format_table
from reidtrack.track.runner import run_split, save_results
from reidtrack.tracker import RetrievalTracker, TrackerConfig


def parse_overrides(pairs: list[str]) -> TrackerConfig:
    fields = {f.name: f for f in dataclasses.fields(TrackerConfig)}
    values = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if key not in fields:
            raise SystemExit(f"unknown setting {key!r}; choose from {', '.join(fields)}")
        kind = type(fields[key].default)
        values[key] = raw.lower() in ("1", "true", "yes") if kind is bool else kind(raw)
    return TrackerConfig(**values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack", description="Run the retrieval-augmented tracker.")
    parser.add_argument("--embeddings", default="osnet_x1_0_msmt17", help="feature cache name")
    parser.add_argument("--det", choices=ALL_DETECTORS, default="FRCNN")
    parser.add_argument("--camera", action="store_true", help="compensate camera motion (cached)")
    parser.add_argument("--reranker", type=Path, help="learned reranker checkpoint (default: hand-set costs)")
    parser.add_argument("--interpolate", type=int, default=0, metavar="FRAMES",
                        help="offline: fill gaps up to this many frames inside tracks")
    parser.add_argument("--stride", type=int, default=1,
                        help="track every n-th frame only, as an edge device skipping frames; the skipped "
                             "frames are filled in by interpolation for scoring")
    parser.add_argument("--split", choices=SPLITS, default="val_half")
    parser.add_argument("--sequence", help="track this one whole video instead, test videos included, without scoring")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a TrackerConfig field")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--name", help="run name (default: raa_<embeddings>_<det>_<split>)")
    args = parser.parse_args(argv)

    config = parse_overrides(args.set)
    reranker = None
    if args.reranker:
        from reidtrack.association.reranker import PairwiseReranker

        reranker = PairwiseReranker.load(args.reranker)
    make = lambda info: RetrievalTracker(info.width, info.height, info.frame_rate, config, reranker)  # noqa: E731
    timer = StageTimer(warmup=10)
    if args.sequence:
        results = run_split(make, "train", args.det, args.root, timer, args.embeddings, args.camera, args.stride,
                            sequences=[args.sequence])
        out = args.runs / (args.name or f"raa_{args.sequence}")
        save_results(results, out)
        tracks = results[args.sequence]
        print(f"{args.sequence}: {len(np.unique(tracks.track_id))} IDs over {len(np.unique(tracks.frame))} frames, "
              f"{timer.summary()['track']['mean']:.2f} ms/frame for tracking -> {out}")
        return 0
    results = run_split(make, args.split, args.det, args.root, timer, args.embeddings, args.camera, args.stride)
    if args.stride > 1:
        args.interpolate = max(args.interpolate, args.stride - 1)
    if args.interpolate:
        from reidtrack.track.postprocess import interpolate

        results = {name: interpolate(tracks, args.interpolate) for name, tracks in results.items()}
    ev = evaluate(results, args.split, args.root)

    name = args.name or f"raa_{args.embeddings}_{args.det}_{args.split}"
    out = args.runs / name
    save_results(results, out)
    ms = timer.summary()["track"]["mean"]
    (out / "scores.json").write_text(json.dumps(
        {"tracker": "retrieval-augmented", "embeddings": args.embeddings, "detector": args.det,
         "camera": args.camera, "reranker": str(args.reranker) if args.reranker else None,
         "interpolate": args.interpolate, "stride": args.stride,
         "config": dataclasses.asdict(config), "track_ms_per_frame": ms, **ev.to_dict()}, indent=2) + "\n")
    changed = ", ".join(args.set + (["camera"] if args.camera else []) + (["learned reranker"] if args.reranker else [])
                        + ([f"every {args.stride} frames"] if args.stride > 1 else [])
                        + ([f"interpolate {args.interpolate}"] if args.interpolate else [])) or "defaults"
    print(format_table(
        ["sequence", *Scores.HEADERS],
        [[n, *s.row()] for n, s in ev.sequences.items()],
        caption=f"MOT17 {args.split}, retrieval-augmented tracker ({changed}), {args.embeddings}",
        footer=[["combined", *ev.combined.row()]],
        note=f"tracking only: {ms:.2f} ms/frame; detections and embeddings precomputed",
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
