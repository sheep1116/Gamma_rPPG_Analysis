import numpy as np

from gamma_rppg.common.statistics import cliffs_delta, holm_adjust


def test_holm_monotonic_and_bounded():
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_cliffs_delta_direction():
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [3, 4, 5]) == -1.0
