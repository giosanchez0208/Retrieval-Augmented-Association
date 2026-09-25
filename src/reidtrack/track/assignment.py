"""One-to-one matching between two sets (e.g. tracks and detections)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost: np.ndarray, limit: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-cost matching in which no matched pair may cost more than ``limit``.

    Returns ``(matches, unmatched_rows, unmatched_cols)``, with ``matches`` of shape (K, 2).

    With a finite ``limit`` the cost matrix is padded with "leave unmatched" slots
    that cost ``limit / 2`` per side, so a pair costing more than ``limit`` loses to
    leaving both unmatched. This is the rule ByteTrack applies through
    ``lap.lapjv(extend_cost=True, cost_limit=limit)``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int), np.arange(n), np.arange(m)
    if np.isfinite(limit):
        padded = np.full((n + m, n + m), limit / 2)
        padded[:n, :m] = cost
        padded[n:, m:] = 0
        rows, cols = linear_sum_assignment(padded)
        keep = (rows < n) & (cols < m)
        rows, cols = rows[keep], cols[keep]
    else:
        rows, cols = linear_sum_assignment(cost)
    matches = np.stack([rows, cols], axis=1).astype(int)
    return matches, np.setdiff1d(np.arange(n), rows), np.setdiff1d(np.arange(m), cols)
