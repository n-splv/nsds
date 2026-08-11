import numpy as np
from numpy.typing import ArrayLike


def r2_score(y_true: ArrayLike,
             y_pred: ArrayLike,
             *,
             sample_weight: ArrayLike | None = None) -> np.float64:
    """
    Coefficient of determination, matching sklearn's single-output `r2_score`
    (including its behaviour on a constant `y_true`) without the dependency.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weight = 1.0 if sample_weight is None else np.asarray(sample_weight, dtype=float)

    mean = np.average(y_true, weights=sample_weight)
    residual_sum_of_squares = np.sum(weight * (y_true - y_pred) ** 2)
    total_sum_of_squares = np.sum(weight * (y_true - mean) ** 2)

    if total_sum_of_squares == 0:
        return np.float64(1.0 if residual_sum_of_squares == 0 else 0.0)

    return np.float64(1 - residual_sum_of_squares / total_sum_of_squares)


def r2_adjusted(y_true: ArrayLike,
                y_pred: ArrayLike,
                n_features: int,
                *,
                sample_weight: ArrayLike | None = None) -> np.float64:
    n = np.asarray(y_true).shape[0]
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
    return (
            1 -
            (1 - r2) *
            (n - 1) /
            (n - n_features - 1)
    )


def smape(y_true: ArrayLike,
          y_pred: ArrayLike,
          *,
          sample_weight: ArrayLike | None = None) -> np.float64:
    """
    Symmetric Mean Absolute Percentage Error with
    handling of 0 values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        symmetric_errors = 2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    symmetric_errors = np.nan_to_num(symmetric_errors, nan=0.0)
    return np.average(symmetric_errors, weights=sample_weight)
