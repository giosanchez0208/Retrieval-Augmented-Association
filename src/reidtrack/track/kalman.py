"""Constant-velocity Kalman filters for boxes.

``XYAHKalman`` tracks centre, aspect ratio and height, with noise proportional to
the box height: the motion model of DeepSORT and ByteTrack. ``XYSRKalman`` tracks
centre, area and aspect ratio with fixed noise: the motion model of SORT.
"""

from __future__ import annotations

import numpy as np


class XYAHKalman:
    """State ``(cx, cy, a, h, vcx, vcy, va, vh)``; measurement ``(cx, cy, a, h)``."""

    def __init__(self, std_position: float = 1 / 20, std_velocity: float = 1 / 160) -> None:
        self.std_position = std_position
        self.std_velocity = std_velocity
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = xyah[3]
        p, v = 2 * self.std_position * h, 10 * self.std_velocity * h
        std = np.array([p, p, 1e-2, p, v, v, 1e-5, v])
        return np.r_[xyah, np.zeros(4)], np.diag(std**2)

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched prediction: ``mean`` (N, 8), ``cov`` (N, 8, 8)."""
        h = mean[:, 3]
        p, v = self.std_position * h, self.std_velocity * h
        std = np.stack([p, p, np.full_like(h, 1e-2), p, v, v, np.full_like(h, 1e-5), v], axis=1)
        noise = np.zeros_like(cov)
        diag = np.arange(8)
        noise[:, diag, diag] = std**2
        return mean @ self.F.T, self.F @ cov @ self.F.T + noise

    def _innovation_cov(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        p = self.std_position * mean[3]
        return self.H @ cov @ self.H.T + np.diag(np.array([p, p, 1e-1, p]) ** 2)

    def update(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        innovation_cov = self._innovation_cov(mean, cov)
        gain = np.linalg.solve(innovation_cov, self.H @ cov).T
        mean = mean + gain @ (xyah - self.H @ mean)
        cov = cov - gain @ innovation_cov @ gain.T
        return mean, cov

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of measurements (M, 4) from the track's prediction."""
        chol = np.linalg.cholesky(self._innovation_cov(mean, cov))
        z = np.linalg.solve(chol, (np.asarray(xyah).reshape(-1, 4) - self.H @ mean).T)
        return np.sum(z * z, axis=0)

    def gating_distances(self, means: np.ndarray, covs: np.ndarray, xyah: np.ndarray) -> np.ndarray:
        """``gating_distance`` for many tracks at once: means (N, 8), covs (N, 8, 8) -> (N, M)."""
        xyah = np.asarray(xyah).reshape(-1, 4)
        if len(means) == 0 or len(xyah) == 0:
            return np.zeros((len(means), len(xyah)))
        p = self.std_position * means[:, 3]
        noise = np.zeros((len(means), 4, 4))
        noise[:, [0, 1, 3], [0, 1, 3]] = (p**2)[:, None]
        noise[:, 2, 2] = 1e-2
        innovation = self.H @ covs @ self.H.T + noise
        chol = np.linalg.cholesky(innovation)  # (N, 4, 4)
        diff = xyah[None, :, :] - (means @ self.H.T)[:, None, :]  # (N, M, 4)
        z = np.linalg.solve(chol, diff.transpose(0, 2, 1))  # (N, 4, M)
        return np.sum(z * z, axis=1)


class XYSRKalman:
    """State ``(cx, cy, s, r, vcx, vcy, vs)`` with area ``s`` and aspect ratio ``r``;
    measurement ``(cx, cy, s, r)``. The aspect ratio is modelled as constant."""

    def __init__(self) -> None:
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1
        self.H = np.eye(4, 7)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-4])
        self.P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    def initiate(self, xysr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.r_[xysr, np.zeros(3)], self.P0.copy()

    def predict(self, x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = x.copy()
        if x[2] + x[6] <= 0:  # the area would turn negative: stop shrinking
            x[6] = 0
        return self.F @ x, self.F @ P @ self.F.T + self.Q

    def update(self, x: np.ndarray, P: np.ndarray, xysr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S = self.H @ P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ P).T
        x = x + K @ (xysr - self.H @ x)
        I_KH = np.eye(7) - K @ self.H
        return x, I_KH @ P @ I_KH.T + K @ self.R @ K.T  # Joseph form
