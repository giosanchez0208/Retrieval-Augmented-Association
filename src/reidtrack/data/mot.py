"""Readers for the MOTChallenge text formats.

MOT files store boxes as ``left, top, width, height`` in 1-based pixel
coordinates. Everything here returns 0-based ``x1, y1, x2, y2`` floats, i.e.
``x1 = left - 1`` and ``x2 = x1 + width``. Writers must add the 1 back, because
the evaluation compares tracker files against the ground truth as written.
"""

from __future__ import annotations

import configparser
import io
from dataclasses import dataclass, fields, replace
from enum import IntEnum
from pathlib import Path

import numpy as np


class MotClass(IntEnum):
    PEDESTRIAN = 1
    PERSON_ON_VEHICLE = 2
    CAR = 3
    BICYCLE = 4
    MOTORBIKE = 5
    NON_MOTORIZED_VEHICLE = 6
    STATIC_PERSON = 7
    DISTRACTOR = 8
    OCCLUDER = 9
    OCCLUDER_ON_GROUND = 10
    OCCLUDER_FULL = 11
    REFLECTION = 12


# Neither rewarded nor penalised by the official evaluation: TrackEval drops
# tracker boxes matched to these before scoring.
DISTRACTOR_CLASSES = frozenset(
    {
        MotClass.PERSON_ON_VEHICLE,
        MotClass.STATIC_PERSON,
        MotClass.DISTRACTOR,
        MotClass.REFLECTION,
    }
)


@dataclass(frozen=True)
class SeqInfo:
    name: str
    frame_rate: int
    length: int
    width: int
    height: int
    im_dir: str = "img1"
    im_ext: str = ".jpg"

    def to_ini(self) -> str:
        return (
            "[Sequence]\n"
            f"name={self.name}\n"
            f"imDir={self.im_dir}\n"
            f"frameRate={self.frame_rate}\n"
            f"seqLength={self.length}\n"
            f"imWidth={self.width}\n"
            f"imHeight={self.height}\n"
            f"imExt={self.im_ext}\n"
        )

    def renamed(self, name: str) -> SeqInfo:
        return replace(self, name=name)


def read_seqinfo(path: str | Path) -> SeqInfo:
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(path)
    seq = parser["Sequence"]
    return SeqInfo(
        name=seq["name"],
        frame_rate=int(seq["frameRate"]),
        length=int(seq["seqLength"]),
        width=int(seq["imWidth"]),
        height=int(seq["imHeight"]),
        im_dir=seq.get("imDir", "img1"),
        im_ext=seq.get("imExt", ".jpg"),
    )


class _Rows:
    """Row selection shared by the column-oriented annotation tables."""

    def __len__(self) -> int:
        return len(self.frame)

    def select(self, mask: np.ndarray):
        return type(self)(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    def in_frames(self, first: int, last: int):
        return self.select((self.frame >= first) & (self.frame <= last))


@dataclass(frozen=True)
class GroundTruth(_Rows):
    """Rows of a ``gt.txt`` file, sorted by frame, then track id."""

    frame: np.ndarray  # (N,) int32, 1-based
    track_id: np.ndarray  # (N,) int32
    xyxy: np.ndarray  # (N, 4) float32, 0-based pixels
    considered: np.ndarray  # (N,) bool; False rows are ignored by the evaluation
    cls: np.ndarray  # (N,) int16, see MotClass
    visibility: np.ndarray  # (N,) float32 in [0, 1]

    def targets(self) -> GroundTruth:
        """The rows the evaluation scores: considered pedestrians."""
        return self.select((self.cls == MotClass.PEDESTRIAN) & self.considered)


@dataclass(frozen=True)
class Detections(_Rows):
    """Rows of a ``det.txt`` file, sorted by frame.

    Score scales are detector-specific: DPM scores are unbounded (roughly -0.5
    to 5 on MOT17), FRCNN and SDP scores lie in [0, 1].
    """

    frame: np.ndarray  # (N,) int32, 1-based
    xyxy: np.ndarray  # (N, 4) float32, 0-based pixels
    score: np.ndarray  # (N,) float32


@dataclass(frozen=True)
class Tracks(_Rows):
    """Tracker output: one box per track and frame, sorted by frame, then track id."""

    frame: np.ndarray  # (N,) int32, 1-based
    track_id: np.ndarray  # (N,) int32
    xyxy: np.ndarray  # (N, 4) float32, 0-based pixels
    score: np.ndarray  # (N,) float32

    @classmethod
    def from_gt(cls, gt: GroundTruth) -> Tracks:
        return cls(gt.frame, gt.track_id, gt.xyxy, np.ones(len(gt), dtype=np.float32))


def load_gt(path: str | Path) -> GroundTruth:
    rows = _read_rows(path, min_cols=9)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    return GroundTruth(
        frame=rows[:, 0].astype(np.int32),
        track_id=rows[:, 1].astype(np.int32),
        xyxy=_tlwh_to_xyxy(rows[:, 2:6]),
        considered=rows[:, 6] != 0,
        cls=rows[:, 7].astype(np.int16),
        visibility=rows[:, 8].astype(np.float32),
    )


def load_det(path: str | Path) -> Detections:
    # DPM files carry three trailing -1 columns; only the first seven matter.
    rows = _read_rows(path, min_cols=7)
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    return Detections(
        frame=rows[:, 0].astype(np.int32),
        xyxy=_tlwh_to_xyxy(rows[:, 2:6]),
        score=rows[:, 6].astype(np.float32),
    )


def load_tracks(path: str | Path) -> Tracks:
    rows = _read_rows(path, min_cols=6)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    score = rows[:, 6] if rows.shape[1] > 6 else np.ones(len(rows))
    return Tracks(
        frame=rows[:, 0].astype(np.int32),
        track_id=rows[:, 1].astype(np.int32),
        xyxy=_tlwh_to_xyxy(rows[:, 2:6]),
        score=score.astype(np.float32),
    )


def save_tracks(path: str | Path, tracks: Tracks) -> None:
    """Write tracks in the MOTChallenge results format (1-based boxes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyxy = tracks.xyxy.astype(np.float64)
    left, top = xyxy[:, 0] + 1, xyxy[:, 1] + 1
    width, height = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    with open(path, "w", newline="\n") as f:
        for row in zip(tracks.frame.tolist(), tracks.track_id.tolist(), left, top, width, height, tracks.score):
            f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.4f},-1,-1,-1\n".format(*row))


def _read_rows(path: str | Path, min_cols: int) -> np.ndarray:
    text = Path(path).read_text().strip()
    if not text:
        return np.zeros((0, min_cols))
    rows = np.loadtxt(io.StringIO(text), delimiter=",", ndmin=2)
    if rows.shape[1] < min_cols:
        raise ValueError(f"{path}: expected at least {min_cols} columns, found {rows.shape[1]}")
    return rows


def _tlwh_to_xyxy(tlwh: np.ndarray) -> np.ndarray:
    x1 = tlwh[:, 0] - 1
    y1 = tlwh[:, 1] - 1
    return np.stack([x1, y1, x1 + tlwh[:, 2], y1 + tlwh[:, 3]], axis=1).astype(np.float32)
