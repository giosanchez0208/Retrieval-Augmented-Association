"""Render a sequence with ground truth, detections or tracker output.

    python -m reidtrack.viz MOT17-02 --gt --split val_half
    python -m reidtrack.viz MOT17-02 --results runs/sort/MOT17-02.txt
    python -m reidtrack.viz MOT17-02 --det FRCNN --frame 302 --out fig.png

A single --frame writes a PNG; otherwise an MP4. Ground-truth boxes of people
who are annotated but hidden (visibility < 0.1) are dashed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from reidtrack.data.mot import load_tracks
from reidtrack.data.mot17 import ALL_DETECTORS, Mot17
from reidtrack.data.splits import SPLITS
from reidtrack.viz.render import NEUTRAL, Box, FigureRenderer, id_color

HIDDEN = 0.1


def _overlay(seq, args):
    """Return (frames, per-row boxes, label) for the chosen source."""
    if args.gt:
        gt = seq.load_gt().targets()
        boxes = [Box(tuple(b), id_color(i), str(i), dashed=v < HIDDEN)
                 for b, i, v in zip(gt.xyxy.tolist(), gt.track_id.tolist(), gt.visibility.tolist())]
        return gt.frame, boxes, "ground truth"
    if args.det:
        det = seq.load_det(args.det)
        return det.frame, [Box(tuple(b), NEUTRAL) for b in det.xyxy.tolist()], f"{args.det} detections"
    tracks = load_tracks(args.results)
    boxes = [Box(tuple(b), id_color(i), str(i)) for b, i in zip(tracks.xyxy.tolist(), tracks.track_id.tolist())]
    return tracks.frame, boxes, args.results.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.viz", description="Render a sequence as a figure.")
    parser.add_argument("sequence", help="e.g. MOT17-02")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gt", action="store_true", help="ground-truth pedestrians")
    source.add_argument("--det", choices=ALL_DETECTORS, help="public detections, or RFDETR once written")
    source.add_argument("--results", type=Path, help="tracker output (MOT format)")
    frames = parser.add_mutually_exclusive_group()
    frames.add_argument("--split", choices=SPLITS, help="render the frames of a split")
    frames.add_argument("--frame", type=int, help="render one frame as PNG")
    parser.add_argument("--scale", type=float, default=1.0, help="output scale, e.g. 0.5")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"), help="prepared dataset root")
    parser.add_argument("--out", type=Path, help="output file (default: runs/viz/...)")
    parser.add_argument("--timing", type=Path, help="timing.csv from reidtrack.pipeline: show each frame's stage times")
    args = parser.parse_args(argv)
    timing = {}
    if args.timing:
        with open(args.timing, newline="") as f:
            timing = {int(r["frame"]): r for r in csv.DictReader(f)}

    seq = Mot17(args.root).sequence(args.sequence)
    if args.frame is not None:
        wanted = [args.frame]
    else:
        rng = seq.frames(args.split or "train")
        wanted = list(range(rng.first, rng.last + 1))
    if not all(1 <= f <= seq.info.length for f in wanted):
        parser.error(f"frames must lie in 1..{seq.info.length}")

    row_frames, boxes, label = _overlay(seq, args)
    order = np.argsort(row_frames, kind="stable")
    row_frames = np.asarray(row_frames)[order]
    boxes = [boxes[i] for i in order]

    suffix = ".png" if args.frame is not None else ".mp4"
    tag = label.replace(" ", "-")
    out = args.out or Path("runs/viz") / f"{seq.name}_{tag}{f'_{args.frame:06d}' if args.frame else ''}{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)

    renderer = FigureRenderer(seq.info.width, seq.info.height, args.scale)
    writer = None
    if suffix == ".mp4":
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), seq.info.frame_rate, renderer.figure_size)
        if not writer.isOpened():
            print(f"cannot open video writer for {out}", file=sys.stderr)
            return 1

    for frame in wanted:
        lo, hi = np.searchsorted(row_frames, [frame, frame + 1])
        image = cv2.imread(str(seq.image_path(frame)), cv2.IMREAD_COLOR)
        if image is None:
            print(f"cannot read {seq.image_path(frame)}", file=sys.stderr)
            return 1
        caption = f"{seq.name}  ·  frame {frame}  ·  {(frame - 1) / seq.info.frame_rate:.2f} s"
        note = label
        if frame in timing:
            r = timing[frame]
            note = (f"decode {float(r['decode']):.1f}  ·  detect {float(r['detect']):.1f}  ·  embed {float(r['embed']):.1f}"
                    f"  ·  track {float(r['track']):.1f}  ·  total {float(r['total']):.1f} ms")
        figure = renderer.render(image, boxes[lo:hi], caption, note=note)
        if writer is None:
            cv2.imwrite(str(out), figure)
        else:
            writer.write(figure)
    if writer is not None:
        writer.release()
    print(f"wrote {out} ({len(wanted)} frame{'s' if len(wanted) != 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
