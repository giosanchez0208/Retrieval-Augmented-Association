"""Verify the MOT17 release and build a deduplicated copy of it.

The release ships every sequence three times, once per public detector, with
the same images and ground truth. This checks that the copies really are
identical, keeps one of each (hard-linked, so no extra disk space), puts the
three detection files side by side, and exports TrackEval ground truth for the
train / train_half / val_half splits.

    python -m reidtrack.data.prepare --raw MOT17 --out data/mot17
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from reidtrack.data.mot import SeqInfo, read_seqinfo
from reidtrack.data.mot17 import DETECTORS, MOVING_CAMERA
from reidtrack.data.splits import SPLITS, frame_range

REFERENCE_DETECTOR = "FRCNN"
_VARIANT_DIR = re.compile(r"^(MOT17-\d{2})-(DPM|FRCNN|SDP)$")


class ReleaseMismatch(ValueError):
    """The per-detector copies of a sequence are not identical."""


@dataclass(frozen=True)
class ReleaseSequence:
    name: str
    subset: str
    info: SeqInfo
    variants: dict[str, Path]  # detector -> release folder

    @property
    def reference(self) -> Path:
        return self.variants[REFERENCE_DETECTOR]

    def images(self, detector: str) -> list[Path]:
        folder = self.variants[detector] / self.info.im_dir
        return sorted(folder.glob(f"*{self.info.im_ext}"))


def discover(raw: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """Map (subset, sequence) to its per-detector release folders."""
    found: dict[tuple[str, str], dict[str, Path]] = {}
    for subset in ("train", "test"):
        subset_dir = raw / subset
        if not subset_dir.is_dir():
            continue
        for entry in sorted(subset_dir.iterdir()):
            match = _VARIANT_DIR.match(entry.name)
            if match and entry.is_dir():
                name, detector = match.groups()
                found.setdefault((subset, name), {})[detector] = entry
    for (subset, name), variants in found.items():
        missing = sorted(set(DETECTORS) - set(variants))
        if missing:
            raise ReleaseMismatch(f"{subset}/{name}: missing detector folders {missing}")
    return found


def verify(subset: str, name: str, variants: dict[str, Path], hash_images: bool = True) -> ReleaseSequence:
    """Check that every detector copy of a sequence holds the same data."""
    info = read_seqinfo(variants[REFERENCE_DETECTOR] / "seqinfo.ini").renamed(name)
    seq = ReleaseSequence(name, subset, info, variants)
    problems = []

    for detector in DETECTORS:
        if read_seqinfo(variants[detector] / "seqinfo.ini").renamed(name) != info:
            problems.append(f"seqinfo.ini of {detector} differs")

    gt_copies = [variants[d] / "gt" / "gt.txt" for d in DETECTORS]
    if any(p.exists() for p in gt_copies):
        if not all(p.exists() for p in gt_copies) or len({p.read_bytes() for p in gt_copies}) != 1:
            problems.append("gt.txt differs between detector copies")

    names = [p.name for p in seq.images(REFERENCE_DETECTOR)]
    if len(names) != info.length:
        problems.append(f"{len(names)} images but seqLength={info.length}")
    for detector in DETECTORS:
        if [p.name for p in seq.images(detector)] != names:
            problems.append(f"image list of {detector} differs")

    if not problems:
        differing = _differing_images(seq, names, hash_images)
        if differing:
            problems.append(f"{len(differing)} images differ between detector copies, e.g. {differing[0]}")

    if problems:
        raise ReleaseMismatch(f"{subset}/{name}: " + "; ".join(problems))
    return seq


def _differing_images(seq: ReleaseSequence, names: list[str], hash_images: bool) -> list[str]:
    key = _digest if hash_images else _size
    folders = [seq.variants[d] / seq.info.im_dir for d in DETECTORS]
    paths = [folder / name for name in names for folder in folders]
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = list(pool.map(key, paths))
    k = len(folders)
    return [name for i, name in enumerate(names) if len(set(keys[i * k : (i + 1) * k])) != 1]


def _digest(path: Path) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def _size(path: Path) -> int:
    return path.stat().st_size


def build(seq: ReleaseSequence, out: Path, link_mode: str) -> Path:
    seq_dir = out / seq.subset / seq.name
    img_dir = seq_dir / seq.info.im_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    for src in seq.images(REFERENCE_DETECTOR):
        _place(src, img_dir / src.name, link_mode)

    _write_text(seq_dir / "seqinfo.ini", seq.info.to_ini())
    gt = seq.reference / "gt" / "gt.txt"
    if gt.exists():
        (seq_dir / "gt").mkdir(exist_ok=True)
        shutil.copyfile(gt, seq_dir / "gt" / "gt.txt")
    (seq_dir / "det").mkdir(exist_ok=True)
    for detector in DETECTORS:
        shutil.copyfile(seq.variants[detector] / "det" / "det.txt", seq_dir / "det" / f"{detector}.txt")
    return seq_dir


def _place(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists():
        if link_mode == "hardlink" and os.path.samefile(src, dst):
            return
        if link_mode == "copy" and _size(dst) == _size(src):
            return
        dst.unlink()
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as err:
            raise OSError(f"cannot hard-link {src} -> {dst} ({err}); retry with --link-mode copy") from err
    else:
        shutil.copy2(src, dst)


def export_trackeval_gt(seq_dir: Path, info: SeqInfo, trackeval_dir: Path, split: str) -> None:
    """Write TrackEval ground truth for one split, renumbering frames from 1."""
    rng = frame_range(split, info.length)
    lines = []
    for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split(",")
        frame = int(float(cols[0]))
        if frame in rng:
            cols[0] = str(frame - rng.offset)
            lines.append(",".join(cols))
    target = trackeval_dir / f"MOT17-{split}" / info.name
    _write_text(target / "gt" / "gt.txt", "".join(f"{line}\n" for line in lines))
    _write_text(target / "seqinfo.ini", replace(info, length=len(rng)).to_ini())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the MOT17 release and build a deduplicated copy.")
    parser.add_argument("--raw", type=Path, default=Path("MOT17"), help="extracted MOT17 release")
    parser.add_argument("--out", type=Path, default=Path("data/mot17"), help="output root")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hard links need the output on the same drive as the release",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="compare image sizes instead of contents (faster, weaker check)",
    )
    args = parser.parse_args(argv)

    try:
        release = discover(args.raw)
        if not release:
            print(f"no MOT17-XX-<detector> folders found under {args.raw}", file=sys.stderr)
            return 1
        sequences = [verify(subset, name, variants, not args.no_hash) for (subset, name), variants in release.items()]
    except ReleaseMismatch as err:
        print(f"release check failed: {err}", file=sys.stderr)
        return 1
    print(f"verified {len(sequences)} sequences: the {', '.join(DETECTORS)} copies are identical")

    duplicate_bytes = 0
    manifest = {}
    trackeval_dir = args.out / "trackeval"
    train_names = []
    for seq in sequences:
        seq_dir = build(seq, args.out, args.link_mode)
        duplicate_bytes += sum(
            _size(p) for d in DETECTORS if d != REFERENCE_DETECTOR for p in seq.images(d)
        )
        entry = {
            "subset": seq.subset,
            "frame_rate": seq.info.frame_rate,
            "length": seq.info.length,
            "width": seq.info.width,
            "height": seq.info.height,
            "moving_camera": seq.name in MOVING_CAMERA,
        }
        if (seq_dir / "gt" / "gt.txt").exists():
            train_names.append(seq.name)
            entry["splits"] = {}
            for split in SPLITS:
                rng = frame_range(split, seq.info.length)
                entry["splits"][split] = [rng.first, rng.last]
                export_trackeval_gt(seq_dir, seq.info, trackeval_dir, split)
        manifest[seq.name] = entry
        print(f"  {seq.subset}/{seq.name}: {seq.info.length} frames")

    for split in SPLITS:
        _write_text(
            trackeval_dir / "seqmaps" / f"MOT17-{split}.txt",
            "name\n" + "".join(f"{name}\n" for name in sorted(train_names)),
        )
    _write_text(
        args.out / "manifest.json",
        json.dumps(
            {
                "dataset": "MOT17",
                "reference_detector": REFERENCE_DETECTOR,
                "detectors": list(DETECTORS),
                "images_compared_by": "size" if args.no_hash else "content",
                "link_mode": args.link_mode,
                "sequences": dict(sorted(manifest.items())),
            },
            indent=2,
        )
        + "\n",
    )

    others = [d for d in DETECTORS if d != REFERENCE_DETECTOR]
    print(f"wrote {args.out}")
    print(
        f"the {' and '.join(others)} image copies in {args.raw} ({duplicate_bytes / 1e9:.1f} GB) are duplicates; "
        f"once you have checked {args.out}, {args.raw} can be deleted"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
