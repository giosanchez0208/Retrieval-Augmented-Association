import numpy as np

from reidtrack.data.mot import GroundTruth
from reidtrack.data.stats import occlusion_gaps, track_gaps, truncated


def test_only_recovered_hidden_stretches_count():
    frames = np.arange(1, 11)
    # hidden before first seen (ignored), hidden 4-6 then seen at 7 (gap of 3), hidden to the end (ignored)
    visibility = np.array([0.0, 1, 1, 0, 0.05, 0, 1, 0, 0, 0])

    assert occlusion_gaps(frames, visibility) == [3]


def _gt(frame, track_id, visibility, xyxy=None):
    n = len(frame)
    return GroundTruth(
        frame=np.asarray(frame, dtype=np.int32),
        track_id=np.asarray(track_id, dtype=np.int32),
        xyxy=np.asarray(xyxy if xyxy is not None else [[0, 0, 1, 1]] * n, dtype=np.float32),
        considered=np.ones(n, dtype=bool),
        cls=np.ones(n, dtype=np.int16),
        visibility=np.asarray(visibility, dtype=np.float32),
    )


def test_gaps_are_computed_per_track():
    # frame-sorted rows of two interleaved tracks; each has one recovered gap
    gt = _gt(
        frame=[1, 1, 2, 2, 3, 3, 4, 4],
        track_id=[1, 2, 1, 2, 1, 2, 1, 2],
        visibility=[1, 1, 0, 1, 0, 0, 1, 1],
    )

    assert sorted(track_gaps(gt).tolist()) == [1, 2]


def test_truncated_flags_boxes_past_the_border():
    gt = _gt(frame=[1, 1, 1], track_id=[1, 2, 3], visibility=[1, 1, 1],
             xyxy=[[0, 0, 10, 10], [-1, 0, 5, 5], [0, 0, 64, 49]])

    assert truncated(gt, width=64, height=48).tolist() == [False, True, True]
