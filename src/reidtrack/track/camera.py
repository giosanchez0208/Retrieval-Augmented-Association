"""Camera motion between consecutive frames, as 2x3 similarity transforms.

Estimated like the sparse optical-flow compensation of BoT-SORT (Aharon et al.,
2022): corners on a half-resolution grayscale frame, tracked into the next frame
with pyramidal Lucas-Kanade, then a RANSAC fit of rotation, uniform scale and
translation, so independently moving people are outliers.

    python -m reidtrack.track.camera            # cache transforms for the training sequences

``warps[f]`` maps coordinates in frame f-1 to frame f (``warps[0]`` is the identity).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from reidtrack.data.mot17 import Mot17
from reidtrack.report import format_table

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def estimate(prev: np.ndarray, curr: np.ndarray, downscale: float = 2.0) -> np.ndarray:
    """Transform from ``prev`` to ``curr`` (grayscale, possibly downscaled by ``downscale``)."""
    corners = cv2.goodFeaturesToTrack(prev, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3)
    if corners is None or len(corners) < 4:
        return IDENTITY.copy()
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, corners, None)
    ok = status.ravel() == 1
    if ok.sum() < 4:
        return IDENTITY.copy()
    warp, _ = cv2.estimateAffinePartial2D(corners[ok], moved[ok], method=cv2.RANSAC)
    if warp is None:
        return IDENTITY.copy()
    warp[:, 2] *= downscale
    return warp


def warp_state(mean: np.ndarray, cov: np.ndarray, warp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a similarity transform to Kalman states over (cx, cy, a, h, vcx, vcy, va, vh).

    Centre and velocity are rotated and scaled, height and its velocity scaled, and
    the aspect ratio is unchanged. Works on one state (8,) or a batch (N, 8).
    """
    rot = warp[:, :2]
    scale = float(np.sqrt(abs(np.linalg.det(rot))))
    jac = np.eye(8)
    jac[0:2, 0:2] = rot
    jac[4:6, 4:6] = rot
    jac[3, 3] = jac[7, 7] = scale
    mean = mean @ jac.T
    mean[..., 0:2] += warp[:, 2]
    return mean, jac @ cov @ jac.T


def cache_path(root: str | Path, sequence: str) -> Path:
    return Path(root) / "cache" / "camera" / f"{sequence}.npy"


def load_warps(root: str | Path, sequence: str) -> np.ndarray:
    path = cache_path(root, sequence)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python -m reidtrack.track.camera` first")
    return np.load(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.track.camera", description="Cache camera motion.")
    parser.add_argument("--subset", choices=("train", "test"), default="train")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"), help="prepared dataset root")
    args = parser.parse_args(argv)

    from reidtrack.eval.latency import StageTimer

    rows = []
    for seq in Mot17(args.root).sequences(args.subset):
        timer = StageTimer(warmup=5)
        warps = np.zeros((seq.info.length, 2, 3))
        warps[0] = IDENTITY
        prev = None
        for f in range(1, seq.info.length + 1):
            with timer.stage("decode"):
                gray = cv2.imread(str(seq.image_path(f)), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            if prev is not None:
                with timer.stage("estimate"):
                    warps[f - 1] = estimate(prev, gray)
            prev = gray
            timer.next_frame()
        out = cache_path(args.root, seq.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, warps)
        shift = np.linalg.norm(warps[:, :, 2], axis=1)
        s = timer.summary()
        rows.append([seq.name, "moving" if seq.moving_camera else "static", f"{shift.mean():.2f}", f"{shift.max():.1f}",
                     f"{s['estimate']['mean']:.1f}"])
    print(format_table(["sequence", "camera", "mean shift px", "max shift px", "estimate ms"], rows,
                       caption=f"Camera motion, MOT17 {args.subset}", note="shift: translation between consecutive frames"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
