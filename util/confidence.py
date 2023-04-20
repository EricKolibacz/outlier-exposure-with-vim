"""Module containg all relevant functionality for confidence scoring"""

from typing import List

import numpy as np


def bin_samples(keys: np.array, bins: np.array, values: np.array = None) -> list:
    if values is None:
        values = keys

    assert len(keys) == len(values)

    inds = np.digitize(keys, bins, right=True)
    binned_x = [values[np.where(inds == i)] for i in range(1, len(bins))]
    return binned_x


def compute_rms_calibration_error(labels, predictions, bins=10) -> np.ndarray:
    """Computes the root mean square calibration error

    Args:
        labels: label for each sample
        predictions: model predictions on each class for each sample

    Returns:
        np.ndarray: root mean square calibration error
    """

    labels_binned = bin_samples(np.max(predictions, axis=1), np.linspace(0, 1, bins + 1), values=labels)
    preds_binned = bin_samples(np.max(predictions, axis=1), np.linspace(0, 1, bins + 1), values=predictions)

    total_samples = len(labels)

    rms_calibration_error = 0
    for bin_labels, bin_predictions in zip(labels_binned, preds_binned):
        if bin_labels.size == 0:  # empty
            continue

        assert len(bin_predictions[:, 0]) == len(bin_labels), "The predictions and labels don't have the same length"

        bin_size = len(bin_labels)
        predicted_labels = np.argmax(bin_predictions, axis=1)
        correct_classified = bin_labels == predicted_labels
        ratio_correct = np.sum(correct_classified) / bin_size
        mean_confidence = np.mean(np.max(bin_predictions, axis=1))

        rms_calibration_error += bin_size / total_samples * (ratio_correct - mean_confidence) ** 2

    return np.sqrt(rms_calibration_error)
