import itertools

import numpy as np
import pytest

from reidtrack.track.assignment import linear_assignment
from reidtrack.track.boxes import iou_matrix, xyah_to_xyxy, xysr_to_xyxy, xyxy_to_xyah, xyxy_to_xysr
from reidtrack.track.kalman import XYAHKalman, XYSRKalman


def test_iou_basic_and_inclusive_pixels():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])

    np.testing.assert_allclose(iou_matrix(a, b), [[1, 50 / 150, 0]])
    # inclusive: 11x11 boxes, overlap 6x11
    np.testing.assert_allclose(iou_matrix(a, b, plus_one=True)[0, 1], 66 / (121 + 121 - 66))
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_box_conversions_round_trip():
    box = np.array([10.0, 20.0, 40.0, 80.0])
    np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box)
    np.testing.assert_allclose(xysr_to_xyxy(xyxy_to_xysr(box)), box)


def _brute_force(cost, limit):
    """Cheapest partial matching, where each unmatched row or column costs limit / 2."""
    n, m = cost.shape
    best = np.inf
    for k in range(min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                total = sum(cost[r, c] for r, c in zip(rows, cols)) + (n + m - 2 * k) * limit / 2
                best = min(best, total)
    return best


@pytest.mark.parametrize("seed", range(20))
def test_capped_assignment_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    cost = rng.uniform(0, 1, size=(rng.integers(1, 4), rng.integers(1, 4)))
    limit = 0.6

    matches, un_rows, un_cols = linear_assignment(cost, limit)

    total = sum(cost[r, c] for r, c in matches) + (len(un_rows) + len(un_cols)) * limit / 2
    assert total == pytest.approx(_brute_force(cost, limit))
    assert all(cost[r, c] <= limit + 1e-12 for r, c in matches)


def test_pairs_above_the_cap_stay_unmatched():
    matches, un_rows, un_cols = linear_assignment(np.array([[0.1, 0.95], [0.95, 0.99]]), 0.9)

    assert matches.tolist() == [[0, 0]]
    assert un_rows.tolist() == [1] and un_cols.tolist() == [1]


def _constant_velocity(kf, to_state, start, step, frames):
    box = np.array(start, dtype=float)
    return [box + i * np.array(step, dtype=float) for i in range(frames)]


def test_xyah_kalman_learns_constant_velocity():
    kf = XYAHKalman()
    boxes = _constant_velocity(kf, xyxy_to_xyah, [100, 100, 140, 200], [5, 0, 5, 0], 20)
    mean, cov = kf.initiate(xyxy_to_xyah(boxes[0]))
    for box in boxes[1:]:
        m, c = kf.predict(mean[None], cov[None])
        mean, cov = kf.update(m[0], c[0], xyxy_to_xyah(box))
    predicted, _ = kf.predict(mean[None], cov[None])

    np.testing.assert_allclose(xyah_to_xyxy(predicted[0, :4]), boxes[-1] + [5, 0, 5, 0], atol=0.5)


def test_xysr_kalman_learns_constant_velocity():
    kf = XYSRKalman()
    boxes = _constant_velocity(kf, xyxy_to_xysr, [100, 100, 140, 200], [5, 0, 5, 0], 20)
    x, P = kf.initiate(xyxy_to_xysr(boxes[0]))
    for box in boxes[1:]:
        x, P = kf.predict(x, P)
        x, P = kf.update(x, P, xyxy_to_xysr(box))
    x, _ = kf.predict(x, P)

    np.testing.assert_allclose(xysr_to_xyxy(x[:4]), boxes[-1] + [5, 0, 5, 0], atol=0.5)
