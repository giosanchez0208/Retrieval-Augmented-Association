"""Stress test for appearance models: retrieval on unseen people when the query sighting
is changed one way at a time (lighting, color cast, blocking) and the gallery is not.

    python -m reidtrack.retrieval.probe --models osnet_x0_5_mot17 osnet_x0_5_aug_geometry

The changes are fixed, so every model sees exactly the same altered crops. Blocking uses a
piece of another person's crop, not the noise the blocking augmentation trains on.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from reidtrack.report import format_table


def _scale(x: torch.Tensor, factor) -> torch.Tensor:
    return x * torch.as_tensor(factor, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)


def _contrast(x: torch.Tensor, amount: float) -> torch.Tensor:
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - mean) * amount + mean


def _shadow(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[..., : x.shape[-1] // 2] *= 0.4  # a hard window-frame shadow over one side
    return x


LIGHT: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "dark": lambda x: x * 0.5,
    "overexposed": lambda x: x * 1.6,
    "low contrast": lambda x: _contrast(x, 0.4),
    "warm": lambda x: _scale(x, (1.25, 1.0, 0.8)),
    "cool": lambda x: _scale(x, (0.8, 1.0, 1.25)),
    "shadow": _shadow,
}
BLOCK = {"blocked below": "below", "blocked above": "above", "blocked side": "side", "blocked anywhere": "anywhere"}
PROBES = ("clean", *LIGHT, *BLOCK)


def _region(where: str, row: int, h: int, w: int) -> tuple[slice, slice]:
    """Rows and columns of the crop to cover. Sides alternate by crop; "anywhere" is a
    rectangle of 25 to 40% of the crop at a spot fixed per crop, so every model sees the same."""
    if where == "below":
        return slice(int(0.6 * h), h), slice(0, w)
    if where == "above":
        return slice(0, int(0.4 * h)), slice(0, w)
    if where == "side":
        cols = int(0.35 * w)
        return slice(0, h), (slice(0, cols) if row % 2 == 0 else slice(w - cols, w))
    rng = np.random.default_rng(row)
    area, aspect = rng.uniform(0.25, 0.4) * h * w, rng.uniform(1.0, 3.0)  # taller than wide, like the crops
    eh, ew = min(h, int(np.sqrt(area * aspect))), min(w, int(np.sqrt(area / aspect)))
    top, left = int(rng.integers(0, h - eh + 1)), int(rng.integers(0, w - ew + 1))
    return slice(top, top + eh), slice(left, left + ew)


def still_visible(name: str, visibility: np.ndarray, h: int, w: int) -> np.ndarray:
    """How much of each person still shows after a probe: their annotated visibility times
    the share of the crop left uncovered. Conservative, since it assumes the new block
    never overlaps what was already hidden."""
    if name not in BLOCK:
        return visibility
    where = BLOCK[name]
    uncovered = np.empty(len(visibility))
    for row in range(len(visibility)):
        ys, xs = _region(where, row, h, w)
        uncovered[row] = 1 - (ys.stop - ys.start) * (xs.stop - xs.start) / (h * w)
    return visibility * uncovered


def make_perturb(name: str, images: np.ndarray, device: str) -> Callable[[torch.Tensor, np.ndarray], torch.Tensor] | None:
    """``perturb(crops, rows)`` for ``retrieval_scores``; uint8 in, uint8 out. ``images``
    are all crops of the set, from which blocking takes another person's piece."""
    if name == "clean":
        return None
    if name in LIGHT:
        change = LIGHT[name]
        return lambda crops, rows: (change(crops.float() / 255).clamp(0, 1) * 255).round().to(torch.uint8)
    where = BLOCK[name]
    shift = len(images) // 2  # far away in the set, so almost always someone else

    def block(crops: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        donor = torch.from_numpy(np.array(images[(rows + shift) % len(images)])).to(crops.device)
        out = crops.clone()
        h, w = crops.shape[-2:]
        for k, row in enumerate(rows):
            ys, xs = _region(where, int(row), h, w)
            out[k, :, ys, xs] = donor[k, :, ys, xs]
        return out

    return block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.retrieval.probe", description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", required=True, help="run names under data/weights/retriever")
    parser.add_argument("--weights", type=Path, default=Path("data/weights/retriever"))
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    from reidtrack.retrieval.augment import normalize
    from reidtrack.retrieval.crops import load_crops
    from reidtrack.retrieval.embedder import load_retriever
    from reidtrack.retrieval.scoring import retrieval_scores

    train, val = load_crops(args.root, "train_half"), load_crops(args.root, "val_half")
    seen = set(train.identities().tolist())
    amp = args.device.startswith("cuda")
    h, w = val.images.shape[-2:]
    min_visible = 0.3  # the same rule the crops follow: less than 30% showing is not a fair query
    rows, summary = [], []
    for name in args.models:
        model = load_retriever(args.weights / name / "last.pt").eval().to(args.device)

        def embed(images: torch.Tensor) -> torch.Tensor:
            with torch.autocast(args.device.split(":")[0], dtype=torch.float16, enabled=amp):
                return model(normalize(images))

        clean = retrieval_scores(embed, val, seen, device=args.device)
        changes = []
        for p in PROBES:
            fair = still_visible(p, val.visibility, h, w) >= min_visible
            mask = None if fair.all() else fair
            s = retrieval_scores(embed, val, seen, device=args.device,
                                 perturb=make_perturb(p, val.images, args.device), query_mask=mask)
            base = clean if mask is None else retrieval_scores(embed, val, seen, device=args.device, query_mask=mask)
            changes.append(s["mAP"] - base["mAP"])
            rows.append([name, p, f"{s['mAP']:.1f}", f"{base['mAP']:.1f}", f"{s['mAP'] - base['mAP']:+.1f}",
                         f"{100 * s['queries'] / clean['queries']:.0f}%"])
        summary.append(f"{name}: clean {clean['mAP']:.1f}, " + ", ".join(
            f"{p} {c:+.1f}" for p, c in zip(PROBES[1:], changes[1:])) + f", mean {np.mean(changes[1:]):+.1f}")
        print(summary[-1], file=sys.stderr, flush=True)
        del model
    print(format_table(["model", "probe", "mAP", "clean, same queries", "change", "queries kept"], rows,
                       caption="mAP on unseen validation people, query crop changed one way at a time",
                       note="gallery unchanged; blocking takes a piece of another person's crop and keeps only "
                            "queries still at least 30% visible"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
