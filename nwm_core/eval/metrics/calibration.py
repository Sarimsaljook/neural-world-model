from __future__ import annotations
import numpy as np
from ...common.math import ece

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    return ece(probs, labels, n_bins=n_bins)
