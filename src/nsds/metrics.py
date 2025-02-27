import numpy as np
import numpy.typing as npt
from sklearn.metrics import r2_score


def smape(y_true: npt.NDArray[int | float],
          y_pred: npt.NDArray[int | float],
          *,
          sample_weight: npt.NDArray[int | float] = None) -> np.float64:
    """
    Symmetric Mean Absolute Percentage Error with
    handling of 0 values.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        symmetric_errors = 2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    symmetric_errors = np.nan_to_num(symmetric_errors, nan=0.0)
    return np.average(symmetric_errors, weights=sample_weight)


def r2_adjusted(y_true: npt.NDArray[int | float],
                y_pred: npt.NDArray[int | float],
                n_features: int,
                *,
                sample_weight: npt.NDArray[int | float] = None) -> np.float64:
    n = y_true.shape[0]
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
    return 1 - (1 - r2) * ((n - 1) / (n - n_features - 1))
