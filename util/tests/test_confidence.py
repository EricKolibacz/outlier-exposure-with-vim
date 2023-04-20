"""A test module testing everything related to confidence"""

import numpy as np

from util.confidence import bin_samples, compute_rms_calibration_error


def test_binning_simple():
    x = np.array([0.2, 0.7, 0.3])
    bins = np.array([0.0, 0.5, 1.0])
    expected = [np.array([0.2, 0.3]), np.array([0.7])]
    actual = bin_samples(x, bins)
    for exp, act in zip(expected, actual):
        np.testing.assert_array_equal(exp, act)


def test_binning_empty_bin():
    x = np.array([0.2, 1.2, 0.3])
    bins = np.array([0.0, 0.5, 1.0, 1.5])
    expected = [np.array([0.2, 0.3]), np.array([]), np.array([1.2])]
    actual = bin_samples(x, bins)
    for exp, act in zip(expected, actual):
        np.testing.assert_array_equal(exp, act)

