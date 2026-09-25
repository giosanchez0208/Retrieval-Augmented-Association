import numpy as np
import pytest
import torch

from reidtrack.retrieval.augment import Augment, normalize
from reidtrack.retrieval.crops import CropSet, inside_fraction
from reidtrack.retrieval.embedder import load_retriever
from reidtrack.retrieval.models import ReIDModel, build_backbone
from reidtrack.retrieval.sampling import SameSceneSampler
from reidtrack.retrieval.scoring import average_precision
from reidtrack.retrieval.train import batch_hard_triplet, build_model


def fake_crops(sequences=2, people=20, crops=12):
    rows = [(s, p, f) for s in range(sequences) for p in range(1, people + 1) for f in range(1, crops + 1)]
    seq, tid, frame = (np.array(c) for c in zip(*rows))
    n = len(rows)
    return CropSet(
        images=np.zeros((n, 3, 8, 4), dtype=np.uint8),
        sequence=seq.astype(np.int16), track_id=tid.astype(np.int32), frame=frame.astype(np.int32),
        seconds=(frame / 30).astype(np.float32), visibility=np.ones(n, np.float32), names=("A", "B"),
    )


def test_batches_are_people_by_crops_from_one_scene_and_spread_in_time():
    data = fake_crops()
    batch = SameSceneSampler(data, people=16, crops=4, seed=1).batch()

    assert len(batch) == 64
    assert len(np.unique(data.sequence[batch])) == 1
    ids = data.identities()[batch]
    assert all((ids == i).sum() == 4 for i in np.unique(ids))
    frames = data.frame[batch].reshape(16, 4)
    assert (np.diff(frames, axis=1) > 0).all()  # one crop from each quarter of the track


def test_inside_fraction():
    np.testing.assert_allclose(inside_fraction(np.array([[0, 0, 10, 10], [-10, 0, 10, 10]]), 100, 100), [1, 0.5])


def test_augment_keeps_shape_and_changes_pixels():
    torch.manual_seed(0)
    images = torch.randint(0, 255, (8, 3, 64, 32), dtype=torch.uint8)

    out = Augment()(images)

    assert out.shape == (8, 3, 64, 32) and out.dtype == torch.float32
    assert not torch.allclose(out, normalize(images))


def test_triplet_loss_is_zero_when_people_are_well_separated():
    labels = torch.tensor([0, 0, 1, 1])
    close = torch.tensor([[0.0, 0], [0, 0.1], [10, 10], [10, 10.1]])
    mixed = torch.tensor([[0.0, 0], [10, 10], [0, 0.1], [10, 10.1]])

    assert batch_hard_triplet(close, labels).item() == 0
    assert batch_hard_triplet(mixed, labels).item() > 1


def test_average_precision():
    assert average_precision(np.array([0.9, 0.8, 0.1]), np.array([True, False, True])) == pytest.approx((1 + 2 / 3) / 2)
    assert np.isnan(average_precision(np.array([0.5]), np.array([False])))


def test_checkpoints_round_trip_through_the_embedder_loader(tmp_path):
    model = ReIDModel(build_backbone("osnet_x0_25"), num_classes=7).eval()
    path = tmp_path / "best.pt"
    torch.save({"backbone": "osnet_x0_25", "input_size": [256, 128], "num_classes": 7, "state_dict": model.state_dict()}, path)
    x = torch.randn(2, 3, 256, 128)

    reloaded = load_retriever(path).eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(x), model(x))

    warm = build_model("osnet_x0_25", num_classes=3, init=str(path))
    assert warm.classifier.out_features == 3  # the classifier is re-initialised for the new people


def test_the_sampler_can_be_restricted_to_some_sequences():
    data = fake_crops()
    sampler = SameSceneSampler(data, people=16, crops=4, seed=1, sequences={1})

    for _ in range(5):
        assert set(data.sequence[sampler.batch()].tolist()) == {1}
    assert sampler.batches_per_epoch == (20 * 12) // 64
