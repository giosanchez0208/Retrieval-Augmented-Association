"""MOT17's training half as a COCO-format person dataset, for fine-tuning a detector.

    python -m reidtrack.detection.coco --out data/mot17/coco

Writes ``train``, ``valid`` and ``test`` folders, each with ``_annotations.coco.json``
next to hard links of the frames, so the dataset takes no extra disk space. ``train``
holds the first three quarters of every video's training half; ``valid`` (and ``test``,
a copy of it) holds the last quarter, which decides when training stops. The
validation half is never used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from reidtrack.data.mot import MotClass
from reidtrack.data.mot17 import Mot17
from reidtrack.eval.metrics import split_sequences

PEOPLE = (MotClass.PEDESTRIAN, MotClass.STATIC_PERSON)


def build(root: Path, out: Path, min_visibility: float = 0.1, held_out: float = 0.25) -> dict[str, tuple[int, int]]:
    """Write the dataset; returns images and boxes per folder."""
    data = Mot17(root)
    splits = {"train": ([], []), "valid": ([], [])}
    for name in split_sequences("train_half", root):
        seq = data.sequence(name)
        rng = seq.frames("train_half")
        cut = rng.last - int(held_out * (rng.last - rng.first + 1))
        gt = seq.load_gt("train_half")
        keep = np.isin(gt.cls, PEOPLE) & (gt.visibility >= min_visibility)
        gt = gt.select(keep)
        w, h = seq.info.width, seq.info.height
        for frame in range(rng.first, rng.last + 1):
            split = "train" if frame <= cut else "valid"
            images, boxes = splits[split]
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": f"{name}_{frame:06d}.jpg", "width": w, "height": h,
                           "source": str(seq.image_path(frame))})
            for x1, y1, x2, y2 in np.clip(gt.xyxy[gt.frame == frame], 0, [w, h, w, h]):
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    boxes.append({"id": len(boxes) + 1, "image_id": image_id, "category_id": 1,
                                  "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                  "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0})
    counts = {}
    for split, (images, boxes) in splits.items():
        for folder in (split, "test") if split == "valid" else (split,):
            target = out / folder
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            for image in images:
                os.link(image["source"], target / image["file_name"])
            coco = {"images": [{k: v for k, v in im.items() if k != "source"} for im in images],
                    "annotations": boxes, "categories": [{"id": 1, "name": "person", "supercategory": "none"}]}
            (target / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
            counts[folder] = (len(images), len(boxes))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.detection.coco", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--out", type=Path, default=Path("data/mot17/coco"))
    parser.add_argument("--min-visibility", type=float, default=0.1)
    args = parser.parse_args(argv)
    for folder, (images, boxes) in build(args.root, args.out, args.min_visibility).items():
        print(f"{folder}: {images} images, {boxes} person boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
