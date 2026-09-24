"""Frame ranges for splitting MOT17 training sequences.

The half split follows CenterTrack's convention, which ByteTrack and most
later MOT17 ablations reuse, so validation numbers stay comparable: with ``n``
frames, frames ``1 .. n//2 + 1`` are for training and ``n//2 + 2 .. n`` for
validation. Splitting by detector folder instead would leak, because the
DPM/FRCNN/SDP folders hold the same video.
"""

from __future__ import annotations

from dataclasses import dataclass

SPLITS = ("train", "train_half", "val_half")


@dataclass(frozen=True)
class FrameRange:
    first: int  # 1-based, inclusive
    last: int  # inclusive

    def __len__(self) -> int:
        return self.last - self.first + 1

    def __contains__(self, frame: int) -> bool:
        return self.first <= frame <= self.last

    @property
    def offset(self) -> int:
        """Subtract from a frame number to renumber the range from 1."""
        return self.first - 1


def frame_range(split: str, length: int) -> FrameRange:
    half = length // 2
    if split == "train":
        return FrameRange(1, length)
    if split == "train_half":
        return FrameRange(1, half + 1)
    if split == "val_half":
        return FrameRange(half + 2, length)
    raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
