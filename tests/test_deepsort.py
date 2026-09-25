import numpy as np
import pytest

from reidtrack.baselines import DeepSort
from reidtrack.track.boxes import xyxy_to_xyah
from reidtrack.track.kalman import XYAHKalman


def unit(i, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1
    return v


def step(tracker, boxes, feats, score=0.9):
    boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    feats = np.array(feats, dtype=np.float32).reshape(len(boxes), -1) if len(boxes) else np.zeros((0, 8), np.float32)
    ids, _, _ = tracker.update(boxes, np.full(len(boxes), score), feats)
    return ids.tolist()


def test_gating_distance_is_small_near_the_prediction():
    kf = XYAHKalman()
    mean, cov = kf.initiate(xyxy_to_xyah(np.array([100.0, 100, 150, 250])))
    near, far = kf.gating_distance(mean, cov, np.array([xyxy_to_xyah(np.array([101.0, 100, 151, 250])),
                                                         xyxy_to_xyah(np.array([600.0, 100, 650, 250]))]))
    assert near < 9.4877 < far


def test_identities_survive_a_crossing():
    tracker = DeepSort()
    history = []
    for i in range(20):
        a = [100 + 20 * i, 100, 150 + 20 * i, 250]  # walks right
        b = [500 - 20 * i, 100, 550 - 20 * i, 250]  # walks left, crossing at i = 10
        ids = step(tracker, [a, b], [unit(0), unit(1)])
        history.append(ids)
    assert history[3] == [1, 2]
    assert history[-1] == [1, 2]


def test_a_person_is_recovered_after_twenty_missing_frames():
    tracker = DeepSort()
    box = [100, 100, 150, 250]
    for _ in range(10):
        step(tracker, [box], [unit(0)])
    gap = [step(tracker, [], []) for _ in range(20)]

    assert gap[0] == [1]  # coasts on its prediction for one frame
    assert gap[1:] == [[]] * 19
    assert step(tracker, [box], [unit(0)]) == [1]


def test_a_stranger_in_the_same_place_gets_a_new_id():
    tracker = DeepSort()
    box = [100, 100, 150, 250]
    for _ in range(10):
        step(tracker, [box], [unit(0)])
    for _ in range(20):
        step(tracker, [], [])
    for _ in range(3):
        ids = step(tracker, [box], [unit(1)])

    assert ids == [2]


def test_low_scores_are_ignored_and_features_are_required():
    tracker = DeepSort(min_score=0.5)
    assert step(tracker, [[0, 0, 10, 20]], [unit(0)], score=0.2) == []
    assert tracker.tracks == []
    with pytest.raises(ValueError):
        tracker.update(np.zeros((1, 4)), np.ones(1))
