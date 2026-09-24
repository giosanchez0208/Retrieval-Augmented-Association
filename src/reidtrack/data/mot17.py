"""MOT17 metadata and access to the prepared dataset.

``reidtrack.data.prepare`` collapses the release's three per-detector copies of
each sequence into one folder::

    <root>/train/MOT17-02/seqinfo.ini
    <root>/train/MOT17-02/img1/000001.jpg
    <root>/train/MOT17-02/gt/gt.txt
    <root>/train/MOT17-02/det/{DPM,FRCNN,SDP}.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reidtrack.data import mot
from reidtrack.data.splits import FrameRange, frame_range

DETECTORS = ("DPM", "FRCNN", "SDP")

TRAIN_SEQUENCES = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13")
TEST_SEQUENCES = ("MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14")

# From the sequence overview table of the MOT16 paper (Milan et al., 2016).
MOVING_CAMERA = frozenset(
    {"MOT17-05", "MOT17-06", "MOT17-07", "MOT17-10", "MOT17-11", "MOT17-12", "MOT17-13", "MOT17-14"}
)


@dataclass(frozen=True)
class Sequence:
    root: Path
    info: mot.SeqInfo
    subset: str  # "train" or "test"

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def moving_camera(self) -> bool:
        return self.name in MOVING_CAMERA

    @property
    def has_gt(self) -> bool:
        return (self.root / "gt" / "gt.txt").is_file()

    def frames(self, split: str = "train") -> FrameRange:
        return frame_range(split, self.info.length)

    def image_path(self, frame: int) -> Path:
        return self.root / self.info.im_dir / f"{frame:06d}{self.info.im_ext}"

    def load_gt(self, split: str | None = None) -> mot.GroundTruth:
        gt = mot.load_gt(self.root / "gt" / "gt.txt")
        return gt if split is None else gt.in_frames(*self._bounds(split))

    def load_det(self, detector: str = "FRCNN", split: str | None = None) -> mot.Detections:
        if detector not in DETECTORS:
            raise ValueError(f"unknown detector {detector!r}; expected one of {DETECTORS}")
        det = mot.load_det(self.root / "det" / f"{detector}.txt")
        return det if split is None else det.in_frames(*self._bounds(split))

    def _bounds(self, split: str) -> tuple[int, int]:
        rng = self.frames(split)
        return rng.first, rng.last


class Mot17:
    """The prepared MOT17 dataset rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def sequences(self, subset: str = "train") -> list[Sequence]:
        subset_dir = self.root / subset
        if not subset_dir.is_dir():
            raise FileNotFoundError(
                f"{subset_dir} not found; run `python -m reidtrack.data.prepare` first"
            )
        return [
            self._load(subset, seq_dir)
            for seq_dir in sorted(subset_dir.iterdir())
            if (seq_dir / "seqinfo.ini").is_file()
        ]

    def sequence(self, name: str) -> Sequence:
        for subset in ("train", "test"):
            seq_dir = self.root / subset / name
            if (seq_dir / "seqinfo.ini").is_file():
                return self._load(subset, seq_dir)
        raise KeyError(f"{name} not found under {self.root}")

    @staticmethod
    def _load(subset: str, seq_dir: Path) -> Sequence:
        return Sequence(root=seq_dir, info=mot.read_seqinfo(seq_dir / "seqinfo.ini"), subset=subset)
