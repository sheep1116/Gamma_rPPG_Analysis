import numpy as np
import pytest

pytest.importorskip("pyVHR")
pytest.importorskip("mediapipe")
from pyVHR.extraction.sig_extraction_methods import holistic_mean
from pyVHR.extraction.sig_processing import SignalProcessingParams

from gamma_rppg.signal.image_math import make_gamma_table
from gamma_rppg.experiments.post_roi import _post_roi_signal


def test_gamma_one_post_roi_matches_holistic_mean():
    crop = np.zeros((5, 5, 3), dtype=np.uint8)
    crop[1:4, 1:4] = [100, 130, 160]
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    actual, _ = _post_roi_signal(crop, mask, make_gamma_table(1.0))
    expected = holistic_mean(
        crop,
        np.int32(SignalProcessingParams.RGB_LOW_TH),
        np.int32(SignalProcessingParams.RGB_HIGH_TH),
    )
    assert np.allclose(actual, expected)
