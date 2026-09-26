"""Run a person detector over MOT17 videos and write its boxes as det/<name>.txt.

    python -m reidtrack.detection.detect
    python -m reidtrack.detection.detect --subset test --sequences MOT17-08

The file follows MOT17's detection format, so every tool that takes ``--det`` can read it.
The default weights are RF-DETR small fine-tuned on MOT17's training half, so its boxes
on training-half frames are in-sample.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.detection.detect", description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path,
                        default=Path("data/weights/detectors/rf-detr-small-mot17/checkpoint_best_total.pth"))
    parser.add_argument("--name", default="RFDETR", help="written to det/<name>.txt")
    parser.add_argument("--subset", choices=("train", "test"), default="train")
    parser.add_argument("--sequences", default="", help="only these sequences, comma-separated")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    args = parser.parse_args(argv)

    import torch
    from PIL import Image
    from rfdetr import RFDETRSmall

    from reidtrack.data.mot17 import Mot17

    model = RFDETRSmall(pretrain_weights=str(args.weights), device="cuda" if torch.cuda.is_available() else "cpu")
    chosen = {s for s in args.sequences.split(",") if s}
    person = None
    for seq in Mot17(args.root).sequences(args.subset):
        if chosen and seq.name not in chosen:
            continue
        rows, start = [], time.perf_counter()
        for frame in range(1, seq.info.length + 1):
            d = model.predict(Image.open(seq.image_path(frame)).convert("RGB"), threshold=args.min_score)
            if person is None and len(d):  # a fine-tuned single-class model and the COCO model number people differently
                person = Counter(d.class_id.tolist()).most_common(1)[0][0]
            for (x1, y1, x2, y2), score in zip(d.xyxy[d.class_id == person], d.confidence[d.class_id == person]):
                rows.append((frame, -1, x1 + 1, y1 + 1, x2 - x1, y2 - y1, score))  # MOT boxes are 1-based
        out = seq.root / "det" / f"{args.name}.txt"
        np.savetxt(out, np.array(rows).reshape(-1, 7), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f")
        ms = (time.perf_counter() - start) / seq.info.length * 1e3
        print(f"{seq.name}: {len(rows)} boxes over {seq.info.length} frames, {ms:.1f} ms/frame -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
