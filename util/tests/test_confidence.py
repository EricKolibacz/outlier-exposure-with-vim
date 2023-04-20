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


def test_binning_edge_case():
    x = np.array([0.00001, 1, 0.00001])
    bins = np.linspace(0, 1, 4 + 1)

    expected = [np.array([0.00001, 0.00001]), np.array([]), np.array([]), np.array([1])]
    actual = bin_samples(x, bins)
    for exp, act in zip(expected, actual):
        np.testing.assert_array_equal(exp, act)


def test_binning_different_keys_values():
    values = np.array([[0.2, 0.1], [0.7, 0.3], [0.3, -0.1]])
    keys = np.max(values, axis=1)
    bins = np.array([0.0, 0.5, 1.0])
    expected = [np.array([[0.2, 0.1], [0.3, -0.1]]), np.array([[0.7, 0.3]])]
    actual = bin_samples(keys, bins, values=values)
    print(actual)
    for exp, act in zip(expected, actual):
        np.testing.assert_array_equal(exp, act)


def test_rms_calibration_error_perfect_binary():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    expected = 0

    actual = compute_rms_calibration_error(labels, probabilities)

    assert expected == actual


def test_rms_calibration_error_binary_calibrated():
    labels = np.array([0, 0, 0, 1])
    probabilities = np.array(
        [
            [0.75, 0.25],
            [0.75, 0.25],
            [0.75, 0.25],
            [0.75, 0.25],
        ]
    )
    expected = 0

    actual = compute_rms_calibration_error(labels, probabilities)

    assert expected == actual


def test_rms_calibration_error_binary():
    labels = np.array([0])
    probabilities = np.array([[0.75, 0.25]])
    expected = 0.25

    actual = compute_rms_calibration_error(labels, probabilities)

    assert expected == actual


def test_rms_calibration_error_binary_2outputs():
    labels = np.array([0, 1])
    probabilities = np.array([[0.75, 0.25], [0.75, 0.25]])
    expected = 0.25

    actual = compute_rms_calibration_error(labels, probabilities)

    assert expected == actual


def test_rms_calibration_error_binary_2outputs():
    labels = np.array([0, 0, 0])
    probabilities = np.array([[0.9, 0.1], [0.5, 0.499], [0.5, 0.499]])
    expected = np.sqrt(1 / 3 * (0.9 - 1) ** 2 + 1 / 6 * (1 - 2) ** 2)

    actual = compute_rms_calibration_error(labels, probabilities)

    assert expected == actual
