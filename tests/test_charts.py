import pandas as pd
import pytest

from nsds.charts import Range, calculate_axis_range, dual_y_figure, prediction_scatter_plot


@pytest.fixture
def predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "y_true": [1, 2, 3, 4],
        "y_pred": [1.1, 1.9, 3.2, 3.8],
    })


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 10], Range(-0.5, 10.5)),
        ([5, 5], Range(5.0, 5.0)),
        ([-10, 10], Range(-11.0, 11.0)),
    ],
    ids=["padded", "constant", "negative"],
)
def test_calculate_axis_range(values: list, expected: Range):
    assert calculate_axis_range(values) == pytest.approx(expected)


def test_dual_y_figure_has_a_secondary_axis():
    figure = dual_y_figure()

    assert figure.layout.yaxis2 is not None


class TestPredictionScatterPlot:

    def test_titles_the_figure_with_r2(self, predictions: pd.DataFrame):
        figure = prediction_scatter_plot(predictions, "y_true", "y_pred")

        assert figure.layout.title.text == "r2=0.98"
        assert figure.layout.xaxis.title.text == "y_true"

    def test_adjusted_r2_is_appended(self, predictions: pd.DataFrame):
        figure = prediction_scatter_plot(
            predictions, "y_true", "y_pred", r2_adj_n_features=1
        )

        assert "r2_adj=0.97" in figure.layout.title.text

    def test_equalize_axes_shares_one_range(self, predictions: pd.DataFrame):
        figure = prediction_scatter_plot(
            predictions, "y_true", "y_pred", equalize_axes=True
        )

        assert figure.layout.xaxis.range == figure.layout.yaxis.range
