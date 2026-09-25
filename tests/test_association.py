import numpy as np

from reidtrack.association.features import NAMES, pair_features
from reidtrack.association.reranker import PairwiseReranker
from reidtrack.association.train import ground_truth_ids
from reidtrack.memory import Entry, Regime
from reidtrack.track.boxes import xyxy_to_xyah
from reidtrack.track.kalman import XYAHKalman
from reidtrack.tracker import RetrievalTracker


def make_entry(box, feature, regime=Regime.ACTIVE, last_seen=1.0):
    kf = XYAHKalman()
    mean, cov = kf.initiate(xyxy_to_xyah(np.array(box, dtype=float)))
    e = Entry(1, mean, cov, first_seen=0, last_seen=last_seen, score=0.9, regime=regime)
    e.appearance = feature
    e.prototypes = [feature]
    return e


def test_pair_features_for_a_perfect_match():
    f = np.eye(4, dtype=np.float32)[0]
    e = make_entry([100, 100, 150, 250], f, regime=Regime.OCCLUDED, last_seen=0.5)
    cues = pair_features([e], np.array([[100.0, 100, 150, 250]]), np.array([0.8]), f[None], np.array([0.1]), XYAHKalman(), now=1.5)
    c = dict(zip(NAMES, cues[0, 0]))

    assert cues.shape == (1, 1, len(NAMES))
    assert c["sim_best"] == c["sim_average"] == 1
    assert c["iou"] == 1 and c["dx"] == c["dy"] == c["log_h"] == 0
    assert c["gap_s"] == 0.1  # 1 s of the 10 s scale
    assert c["det_score"] == np.float32(0.8) and c["crowding"] == np.float32(0.1)
    assert (c["active"], c["occluded"], c["exited"]) == (0, 1, 0)


def test_reranker_probabilities_and_round_trip(tmp_path):
    model = PairwiseReranker()
    cues = np.random.default_rng(0).normal(size=(3, 5, len(NAMES))).astype(np.float32)
    p = model(cues)
    path = tmp_path / "r.pt"
    model.save(path)

    assert p.shape == (3, 5) and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(PairwiseReranker.load(path)(cues), p, rtol=1e-6)
    assert model(np.zeros((0, 4, len(NAMES)), np.float32)).shape == (0, 4)


def test_detections_take_the_id_of_the_ground_truth_they_overlap():
    det = np.array([[0, 0, 10, 20], [100, 100, 110, 120], [300, 300, 310, 320]], dtype=float)
    gt = np.array([[101, 100, 111, 120], [0, 0, 10, 21]], dtype=float)

    np.testing.assert_array_equal(ground_truth_ids(det, gt, np.array([7, 3])), [3, 7, -1])


def test_a_tracker_with_a_reranker_follows_a_walking_person():
    def reranker(cues):  # trust appearance and overlap
        return 0.5 * cues[..., 0] + 0.5 * cues[..., 2]

    tracker = RetrievalTracker(1920, 1080, reranker=reranker)
    f = np.eye(8, dtype=np.float32)[:1]
    ids = [tracker.update(np.array([[400.0 + 5 * i, 300, 460 + 5 * i, 460]]), np.array([0.9]), f)[0].tolist()
           for i in range(20)]
    assert all(frame == [1] for frame in ids)


def test_padded_batches_score_like_single_frames():
    import torch

    from reidtrack.association.reranker import AxialReranker, pad_frames

    torch.manual_seed(0)
    model = AxialReranker().eval()
    a = (torch.randn(5, 3, len(NAMES)), torch.zeros(5, 3))
    b = (torch.randn(2, 7, len(NAMES)), torch.zeros(2, 7))
    cues, _, entry_pad, det_pad, valid = pad_frames([a, b])

    with torch.inference_mode():
        batched = model(cues, entry_pad, det_pad)
        torch.testing.assert_close(batched[0, :5, :3], model(a[0]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(batched[1, :2, :7], model(b[0]), atol=1e-5, rtol=1e-5)
    assert int(valid.sum()) == 5 * 3 + 2 * 7
