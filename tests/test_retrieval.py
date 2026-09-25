import io

import numpy as np
import pytest
import torch
from PIL import Image

from reidtrack.retrieval.cache import cache_path, load_embeddings
from reidtrack.retrieval.embedder import Embedder
from reidtrack.retrieval.osnet import load_osnet, osnet


@pytest.fixture
def weights(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "osnet_x0_25_test.pth"
    torch.save(osnet("x0_25", num_classes=5).state_dict(), path)
    return path


def test_checkpoints_load_strictly_and_embed_in_eval_mode(weights):
    model = load_osnet(weights, "x0_25").eval()
    with torch.inference_mode():
        assert model(torch.randn(2, 3, 256, 128)).shape == (2, 512)


def test_embedder_crops_and_normalises_on_cpu(weights):
    embedder = Embedder(weights, "x0_25", device="cpu")
    jpeg = io.BytesIO()
    Image.new("RGB", (320, 240), (120, 80, 40)).save(jpeg, format="JPEG")
    image = embedder.decode(jpeg.getvalue())

    features = embedder(image, np.array([[10, 20, 60, 140], [-5, 100, 400, 300]], dtype=np.float32))

    assert image.shape == (3, 240, 320)
    assert features.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
    assert embedder(image, np.zeros((0, 4))).shape == (0, 512)


def test_cache_rows_must_match_the_detections(tmp_path):
    path = cache_path(tmp_path, "m", "FRCNN", "MOT17-02")
    path.parent.mkdir(parents=True)
    np.savez(path, frame=np.array([1, 1, 2]), features=np.ones((3, 4), np.float16))

    assert load_embeddings(tmp_path, "m", "FRCNN", "MOT17-02", np.array([1, 1, 2])).shape == (3, 4)
    with pytest.raises(ValueError, match="does not match"):
        load_embeddings(tmp_path, "m", "FRCNN", "MOT17-02", np.array([1, 2, 2]))
    with pytest.raises(FileNotFoundError):
        load_embeddings(tmp_path, "m", "SDP", "MOT17-02", np.array([1]))
