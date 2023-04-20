"""Module containg all relevant functionality for confidence scoring"""

from typing import List

import numpy as np


def bin_samples(x: np.array, bins: np.array) -> list:
    inds = np.digitize(x, bins)
    binned_x = [x[np.where(inds == i)] for i in range(1, len(bins))]
    return binned_x


