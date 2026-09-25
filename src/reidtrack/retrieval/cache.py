"""Precompute appearance embeddings for every public detection.

    python -m reidtrack.retrieval.cache --weights data/weights/osnet_x1_0_msmt17.pth

Writes <root>/cache/embeddings/<model>/<detector>/<sequence>.npz with ``frame``
and ``features`` arrays, row-aligned with ``Sequence.load_det(detector)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from reidtrack.data.mot17 import DETECTORS, Mot17
from reidtrack.report import format_table


def cache_path(root: str | Path, model: str, detector: str, sequence: str) -> Path:
    return Path(root) / "cache" / "embeddings" / model / detector / f"{sequence}.npz"


def load_embeddings(root: str | Path, model: str, detector: str, sequence: str, frames: np.ndarray) -> np.ndarray:
    """Cached features for a sequence's detections; ``frames`` must match the cached rows."""
    path = cache_path(root, model, detector, sequence)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python -m reidtrack.retrieval.cache` first")
    data = np.load(path)
    if not np.array_equal(data["frame"], frames):
        raise ValueError(f"{path} does not match the detections of {sequence}; rebuild the cache")
    return data["features"].astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.retrieval.cache", description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--width", default="x1_0")
    parser.add_argument("--model", help="cache name (default: weights file stem)")
    parser.add_argument("--det", choices=DETECTORS, default="FRCNN")
    parser.add_argument("--subset", choices=("train", "test"), default="train")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"), help="prepared dataset root")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from reidtrack.eval.latency import StageTimer
    from reidtrack.retrieval.embedder import Embedder

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    embedder = Embedder(args.weights, args.width, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else None
    model = args.model or args.weights.stem
    rows = []
    for seq in Mot17(args.root).sequences(args.subset):
        det = seq.load_det(args.det)
        bounds = np.searchsorted(det.frame, np.arange(1, seq.info.length + 2))
        features = np.zeros((len(det), embedder.dim), dtype=np.float16)
        timer = StageTimer(warmup=5, sync=sync)
        for f in range(1, seq.info.length + 1):
            lo, hi = bounds[f - 1], bounds[f]
            with timer.stage("read"):
                jpeg = seq.image_path(f).read_bytes()
            with timer.stage("decode"):
                image = embedder.decode(jpeg)
            with timer.stage("embed"):
                features[lo:hi] = embedder(image, det.xyxy[lo:hi])
            timer.next_frame()
        out = cache_path(args.root, model, args.det, seq.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, frame=det.frame, features=features)
        s = timer.summary()
        per_frame = len(det) / seq.info.length
        rows.append([seq.name, len(det), f"{per_frame:.1f}", f"{s['decode']['mean']:.2f}", f"{s['embed']['mean']:.2f}"])
    print(
        format_table(
            ["sequence", "boxes", "boxes/frame", "decode ms", "embed ms"],
            rows,
            caption=f"Embeddings, {model}, {args.det} detections, {device}",
            note=f"written to {cache_path(args.root, model, args.det, '*').parent}",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
