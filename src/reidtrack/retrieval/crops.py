"""Person crops for training and scoring the retriever.

    python -m reidtrack.retrieval.crops --split train_half
    python -m reidtrack.retrieval.crops --split val_half --stride 3

Writes <root>/cache/crops/<split>.npy, (N, 3, H, W) uint8 RGB, and <split>_meta.npz
with one row per crop. Only pedestrian targets at or above ``--min-visibility``
are kept. Crops are cut on the GPU with the same roi_align path used at inference.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reidtrack.data.mot17 import Mot17
from reidtrack.data.splits import SPLITS
from reidtrack.eval.metrics import split_sequences
from reidtrack.report import format_table


@dataclass(frozen=True)
class CropSet:
    """Crops plus per-row metadata. ``images`` is memory-mapped."""

    images: np.ndarray  # (N, 3, H, W) uint8
    sequence: np.ndarray  # (N,) int16 index into ``names``
    track_id: np.ndarray  # (N,) int32, unique within a sequence
    frame: np.ndarray  # (N,) int32
    seconds: np.ndarray  # (N,) float32, (frame - 1) / fps
    visibility: np.ndarray  # (N,) float32
    names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.frame)

    def identities(self) -> np.ndarray:
        """A global id per row: sequence and track id combined."""
        return self.sequence.astype(np.int64) * 100_000 + self.track_id


def crop_paths(root: str | Path, split: str) -> tuple[Path, Path]:
    base = Path(root) / "cache" / "crops"
    return base / f"{split}.npy", base / f"{split}_meta.npz"


def load_crops(root: str | Path, split: str) -> CropSet:
    images_path, meta_path = crop_paths(root, split)
    if not images_path.is_file():
        raise FileNotFoundError(f"{images_path} not found; run `python -m reidtrack.retrieval.crops --split {split}`")
    meta = np.load(meta_path)
    return CropSet(
        images=np.load(images_path, mmap_mode="r"),
        sequence=meta["sequence"],
        track_id=meta["track_id"],
        frame=meta["frame"],
        seconds=meta["seconds"],
        visibility=meta["visibility"],
        names=tuple(meta["names"].tolist()),
    )


def inside_fraction(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Share of each box's area that lies inside the image."""
    x1, y1, x2, y2 = np.asarray(xyxy, dtype=np.float64).T
    inside = (np.clip(x2, 0, width) - np.clip(x1, 0, width)) * (np.clip(y2, 0, height) - np.clip(y1, 0, height))
    return inside / np.maximum((x2 - x1) * (y2 - y1), 1e-9)


def extract(
    root: str | Path,
    split: str,
    min_visibility: float = 0.3,
    min_inside: float = 0.6,
    stride: int = 1,
    size: tuple[int, int] = (256, 128),
) -> list[list[object]]:
    """``min_inside`` drops people mostly outside the frame: MOT17's visibility
    counts occlusion by others but not the image border."""
    import torch
    from torchvision.io import decode_jpeg
    from torchvision.ops import roi_align

    data = Mot17(root)
    names = split_sequences(split, root)
    selections = []
    for s, name in enumerate(names):
        seq = data.sequence(name)
        gt = seq.load_gt(split).targets()
        keep = (
            (gt.visibility >= min_visibility)
            & (inside_fraction(gt.xyxy, seq.info.width, seq.info.height) >= min_inside)
            & ((gt.frame - seq.frames(split).first) % stride == 0)
        )
        selections.append((s, seq, gt.select(keep)))
    total = sum(len(g) for _, _, g in selections)

    images_path, meta_path = crop_paths(root, split)
    images_path.parent.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8, shape=(total, 3, *size))
    columns = {k: [] for k in ("sequence", "track_id", "frame", "seconds", "visibility")}
    rows, offset = [], 0
    for s, seq, gt in selections:
        for frame in np.unique(gt.frame):
            lo, hi = np.searchsorted(gt.frame, [frame, frame + 1])
            jpeg = torch.frombuffer(bytearray(seq.image_path(int(frame)).read_bytes()), dtype=torch.uint8)
            image = decode_jpeg(jpeg, device="cuda").float()[None]
            h, w = image.shape[2:]
            boxes = torch.as_tensor(gt.xyxy[lo:hi], device="cuda")
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            rois = torch.cat([boxes.new_zeros(len(boxes), 1), boxes], dim=1)
            crops = roi_align(image, rois, size, aligned=True).round().clamp(0, 255).to(torch.uint8)
            images[offset : offset + (hi - lo)] = crops.cpu().numpy()
            offset += hi - lo
        columns["sequence"].append(np.full(len(gt), s, dtype=np.int16))
        columns["track_id"].append(gt.track_id)
        columns["frame"].append(gt.frame)
        columns["seconds"].append(((gt.frame - 1) / seq.info.frame_rate).astype(np.float32))
        columns["visibility"].append(gt.visibility)
        rows.append([seq.name, len(gt), len(np.unique(gt.track_id))])
    images.flush()
    np.savez(meta_path, names=np.array(names), **{k: np.concatenate(v) for k, v in columns.items()})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.retrieval.crops", description="Cut person crops.")
    parser.add_argument("--split", choices=SPLITS, default="train_half")
    parser.add_argument("--min-visibility", type=float, default=0.3)
    parser.add_argument("--min-inside", type=float, default=0.6, help="minimum share of the box inside the image")
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"), help="prepared dataset root")
    args = parser.parse_args(argv)
    rows = extract(args.root, args.split, args.min_visibility, args.min_inside, args.stride)
    total = [sum(r[1] for r in rows), sum(r[2] for r in rows)]
    print(format_table(["sequence", "crops", "people"], rows, footer=[["total", *total]],
                       caption=f"Crops, MOT17 {args.split}, visibility >= {args.min_visibility}, "
                               f"inside >= {args.min_inside}, every {args.stride} frame(s)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
