import numpy as np

from gamma_rppg.signal.image_math import apply_gamma_pixels, make_gamma_table


def test_gamma_one_is_identity():
    image = np.arange(3 * 16 * 16, dtype=np.uint8).reshape(16, 16, 3)
    assert np.array_equal(apply_gamma_pixels(image, make_gamma_table(1.0)), image)


def test_post_roi_is_pixelwise_before_mean():
    pixels = np.array([[20, 40, 60], [200, 220, 240]], dtype=np.uint8)
    table = make_gamma_table(2.0)
    correct = apply_gamma_pixels(pixels, table).mean(axis=0)
    wrong = apply_gamma_pixels(pixels.mean(axis=0).astype(np.uint8), table)
    assert not np.allclose(correct, wrong)
