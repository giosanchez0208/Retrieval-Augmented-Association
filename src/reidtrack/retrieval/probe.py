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
    rows = []
    for name in args.models:
        model = load_retriever(args.weights / name / "last.pt").eval().to(args.device)

        def embed(images: torch.Tensor) -> torch.Tensor:
            with torch.autocast(args.device.split(":")[0], dtype=torch.float16, enabled=amp):
                return model(normalize(images))

        scores = [retrieval_scores(embed, val, seen, device=args.device,
                                   perturb=make_perturb(p, val.images, args.device))["mAP"] for p in PROBES]
        rows.append([name, *(f"{s:.1f}" for s in scores), f"{np.mean(scores[1:]) - scores[0]:+.1f}"])
        print(f"{name}: " + ", ".join(f"{p} {s:.1f}" for p, s in zip(PROBES, scores)), file=sys.stderr, flush=True)
        del model
    print(format_table(["model", *PROBES, "mean change"], rows,
                       caption="mAP on unseen validation people, query crop changed one way at a time",
                       note="gallery crops unchanged; blocking takes a piece of another person's crop"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
