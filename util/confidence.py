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


