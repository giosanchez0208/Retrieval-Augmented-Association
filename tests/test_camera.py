import cv2
import numpy as np

from reidtrack.track.camera import estimate


def textured(size=(240, 320), seed=0):
    noise = np.random.default_rng(seed).integers(0, 255, size=size, dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 1.5)


def shifted(image, dx, dy):
    return cv2.warpAffine(image, np.array([[1.0, 0, dx], [0, 1, dy]]), image.shape[::-1], borderMode=cv2.BORDER_REFLECT)


def test_estimate_recovers_a_pan_in_full_resolution_pixels():
    a = textured()
    warp = estimate(a, shifted(a, 3, -2), downscale=2)

    np.testing.assert_allclose(warp[:, 2], [6, -4], atol=0.3)
    np.testing.assert_allclose(warp[:, :2], np.eye(2), atol=0.01)

