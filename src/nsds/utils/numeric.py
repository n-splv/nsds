from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from numpy.typing import ArrayLike


def gini_inequality_coefficient(x: ArrayLike,
                                w: ArrayLike | None = None) -> np.float64:
    """
    Gini coefficient of inequality, optionally weighted.
    https://stackoverflow.com/a/49571213/17378319
    """

    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def round_half_up(value: float, decimals: int) -> float:
    """
    Avoid 'bankers rounding' problem
    """
    multiplier = 10 ** decimals
    return float(
        Decimal(value * multiplier)
        .quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        / multiplier
    )
