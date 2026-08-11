import numpy as np
import pytest

from nsds.metrics import r2_adjusted, r2_score, smape


class TestR2Score:

    @pytest.mark.parametrize(
        ("y_true", "y_pred", "expected"),
        [
            ([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8], 0.98),
            ([1, 2, 3], [1, 2, 3], 1.0),
            ([2, 2, 2], [2, 2, 2], 1.0),
            ([2, 2, 2], [1, 2, 3], 0.0),
            ([1, 2, 3], [3, 2, 1], -3.0),
        ],
        ids=["known", "perfect", "constant-perfect", "constant-imperfect", "inverted"],
    )
    def test_values(self, y_true: list, y_pred: list, expected: float):
        assert r2_score(y_true, y_pred) == pytest.approx(expected)

    def test_sample_weight_is_applied(self):
        result = r2_score([1, 2, 3], [1, 2, 4], sample_weight=[1, 1, 2])
        assert result == pytest.approx(1 - 2 / 2.75)

    def test_accepts_arrays(self):
        assert r2_score(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)


class TestR2Adjusted:

    def test_penalises_features(self):
        assert r2_adjusted([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8], 1) == pytest.approx(0.97)

    def test_equals_r2_without_features(self):
        y_true, y_pred = [1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8]
        assert r2_adjusted(y_true, y_pred, 0) == pytest.approx(r2_score(y_true, y_pred))


class TestSmape:

    @pytest.mark.parametrize(
        ("y_true", "y_pred", "expected"),
        [
            ([1, 2, 3], [1, 2, 3], 0.0),
            ([1], [3], 1.0),
            ([0], [0], 0.0),
            ([0, 2], [0, 2], 0.0),
            ([1, 0], [3, 0], 0.5),
        ],
        ids=["identical", "known", "both-zero", "zero-pair-ignored", "zero-averaged-in"],
    )
    def test_values(self, y_true: list, y_pred: list, expected: float):
        assert smape(y_true, y_pred) == pytest.approx(expected)

    def test_sample_weight_is_applied(self):
        assert smape([1, 1], [3, 1], sample_weight=[3, 1]) == pytest.approx(0.75)
