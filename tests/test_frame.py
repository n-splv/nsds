from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nsds.frame import (
    NDFrameExtensions,
    dt_group,
    extension_names,
    extensions,
    merge_insert_at,
    percentiles,
    read_csv_pyarrow,
    read_csvs,
    set_pandas_options,
)

EXPECTED_EXTENSIONS = {
    "apply_row_wise",
    "explode_all",
    "memory_mb",
    "missing",
    "preview",
    "show",
    "sort",
    "sortd",
    "to_csv_",
    "vc",
}


class TestInstall:

    def test_attaches_every_extension_to_both_types(self):
        assert extension_names() == EXPECTED_EXTENSIONS
        for name in EXPECTED_EXTENSIONS:
            expected = getattr(NDFrameExtensions, name)
            assert getattr(pd.DataFrame, name) is expected
            assert getattr(pd.Series, name) is expected

    @pytest.mark.parametrize("name", ["sort_values", "to_csv", "explode", "value_counts"])
    def test_never_shadows_a_pandas_method(self, name: str):
        assert name not in extension_names()


class TestValueCounts:

    def test_series_report(self, series: pd.Series):
        result = series.vc()

        assert result["count"].tolist() == [2, 1, 1]
        assert result["percentage"].tolist() == [50.0, 25.0, 25.0]

    def test_dataframe_groups_by_all_columns(self):
        df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 2]})

        result = df.vc()

        assert result["count"].tolist() == [2, 1]
        assert isinstance(result.index, pd.MultiIndex)

    def test_cumulative_columns(self, series: pd.Series):
        result = series.vc(show_cumulative=True)

        assert result.columns.tolist() == [
            "count", "count_cumulative", "percentage", "percentage_cumulative",
        ]
        assert result["count_cumulative"].tolist() == [2, 3, 4]
        assert result["percentage_cumulative"].tolist() == [50.0, 75.0, 100.0]

    @pytest.mark.parametrize(
        ("kwargs", "expected_rows"),
        [({}, 3), ({"min_bin_size": 2}, 1), ({"dropna": True}, 2)],
        ids=["all", "min-bin-size", "dropna"],
    )
    def test_row_filtering(self, series: pd.Series, kwargs: dict, expected_rows: int):
        assert len(series.vc(**kwargs)) == expected_rows

    def test_as_index_false_promotes_the_index(self, series: pd.Series):
        assert "group" in series.vc(as_index=False).columns

    def test_rejects_other_types(self):
        with pytest.raises(TypeError):
            NDFrameExtensions.vc(object())


class TestMissing:

    def test_dataframe_report(self, df: pd.DataFrame):
        result = df.missing()

        assert result.columns.tolist() == [
            "isna", "isna_pct", "eq0", "eq0_pct", "empty_str", "empty_str_pct",
        ]
        assert result.loc["group", "isna"] == 1
        assert result.loc["group", "isna_pct"] == 25.0
        assert result.loc["amount", "eq0"] == 1
        assert result.loc["label", "empty_str"] == 1
        assert pd.isna(result.loc["amount", "empty_str"])

    def test_series_drops_inapplicable_columns(self, series: pd.Series):
        result = series.missing()

        assert result.columns.tolist() == ["isna", "isna_pct", "empty_str", "empty_str_pct"]
        assert result.loc["group", "isna"] == 1


class TestSorting:

    def test_sort_delegates_to_sort_values(self, df: pd.DataFrame):
        assert df.sort("amount")["amount"].tolist() == [0, 10, 30, 40]

    def test_sortd_is_descending(self, df: pd.DataFrame):
        assert df.sortd("amount")["amount"].tolist() == [40, 30, 10, 0]

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sortd_rejects_ascending(self, df: pd.DataFrame, ascending: bool):
        with pytest.raises(ValueError, match="always descending"):
            df.sortd("amount", ascending=ascending)


class TestApplyRowWise:

    @staticmethod
    def _total(amount, quantity):
        return amount * quantity

    @staticmethod
    def _scaled(amount, *, factor):
        return amount * factor

    def test_infers_columns_from_the_signature(self, df: pd.DataFrame):
        assert list(df.apply_row_wise(self._total)) == [10, 0, 90, 160]

    def test_keyword_arguments_are_bound_not_looked_up(self, df: pd.DataFrame):
        assert list(df.apply_row_wise(self._scaled, factor=2)) == [20, 0, 60, 80]

    def test_show_progress_wraps_values_in_tqdm(self,
                                                df: pd.DataFrame,
                                                fake_require: Callable):
        tqdm = MagicMock(side_effect=lambda values: values)
        stub = MagicMock(tqdm=tqdm)

        with patch("nsds.frame.extensions.require", fake_require({"tqdm.auto": stub})):
            result = list(df.apply_row_wise(self._total, show_progress=True))

        assert result == [10, 0, 90, 160]
        tqdm.assert_called_once()


class TestDisplayHelpers:

    def test_display_delegates_to_ipython(self,
                                          monkeypatch: pytest.MonkeyPatch,
                                          fake_require: Callable):
        ipython_display = MagicMock()
        monkeypatch.setattr(extensions, "require",
                            fake_require({"IPython.display": ipython_display}))

        extensions._display("payload")

        ipython_display.display.assert_called_once_with("payload")

    def test_preview_uses_a_row_limit(self, df: pd.DataFrame):
        with patch("nsds.frame.extensions._display") as mock_display:
            df.preview(min_rows=2)

        mock_display.assert_called_once_with(df)

    def test_show_truncates_to_nrows(self, df: pd.DataFrame):
        with patch("nsds.frame.extensions._display") as mock_display:
            df.show(nrows=2)

        mock_display.assert_called_once()
        assert len(mock_display.call_args.args[0]) == 2


class TestExplodeAll:

    def test_dataframe_explodes_every_column(self):
        df = pd.DataFrame({"a": [[1, 2]], "b": [[3, 4]]})

        result = df.explode_all()

        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == [3, 4]

    def test_series_explodes_itself(self):
        assert pd.Series([[1, 2]]).explode_all().tolist() == [1, 2]


class TestMemoryMb:

    def test_dataframe_reports_per_column(self, df: pd.DataFrame):
        result = df.memory_mb()

        assert isinstance(result, pd.Series)
        assert (result > 0).all()

    def test_series_reports_a_single_value(self, series: pd.Series):
        assert series.memory_mb() > 0


class TestToCsv:

    def test_defaults_to_excel_friendly_output(self,
                                               df: pd.DataFrame,
                                               tmp_path,
                                               monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)

        df.to_csv_("out.csv")

        content = (tmp_path / "out.csv").read_bytes()
        assert content.startswith(b"\xef\xbb\xbf")
        assert b"group,amount" in content

    def test_datetime_column_supplies_the_suffix(self,
                                                df_dated: pd.DataFrame,
                                                tmp_path,
                                                monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)

        df_dated.to_csv_("out.csv", add_date_to_filename="day")

        assert (tmp_path / "out_20240304_050607.csv").exists()

    def test_true_uses_the_current_time(self,
                                        df_dated: pd.DataFrame,
                                        tmp_path,
                                        monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)

        df_dated.to_csv_("out.csv", add_date_to_filename=True)

        written = list(tmp_path.glob("out_*.csv"))
        assert len(written) == 1

    def test_unknown_column_is_rejected(self,
                                        df_dated: pd.DataFrame,
                                        tmp_path,
                                        monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)

        with pytest.raises(KeyError, match="No datetime column 'missing'"):
            df_dated.to_csv_("out.csv", add_date_to_filename="missing")


class TestReaders:

    def test_read_csvs_concatenates_matches(self,
                                           tmp_path,
                                           monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        pd.DataFrame({"a": [1]}).to_csv("one.csv", index=False)
        pd.DataFrame({"a": [2]}).to_csv("two.csv", index=False)

        result = read_csvs("*.csv", add_filename_column=True)

        assert sorted(result["a"].tolist()) == [1, 2]
        assert set(result["_file"]) == {"one.csv", "two.csv"}

    def test_read_csv_pyarrow_uses_the_arrow_backend(self,
                                                    tmp_path,
                                                    monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        pd.DataFrame({"a": [1]}).to_csv("one.csv", index=False)

        result = read_csv_pyarrow("one.csv")

        assert isinstance(result["a"].dtype, pd.ArrowDtype)


class TestTools:

    def test_dt_group_builds_a_grouper(self):
        grouper = dt_group("day", "ME")

        assert isinstance(grouper, pd.Grouper)
        assert grouper.key == "day"

    @pytest.mark.parametrize(
        ("insert_index", "expected"),
        [
            (1, ["id", "new", "a", "b"]),
            (0, ["new", "id", "a", "b"]),
            (-1, ["id", "a", "b", "new"]),
        ],
        ids=["middle", "start", "negative"],
    )
    def test_merge_insert_at_positions_new_columns(self,
                                                   insert_index: int,
                                                   expected: list[str]):
        df_l = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 4]})
        df_r = pd.DataFrame({"id": [1, 2], "new": [9, 8]})

        result = merge_insert_at(df_l, df_r, insert_index, on="id")

        assert result.columns.tolist() == expected

    def test_percentiles(self):
        assert percentiles.bottom_one[0] == 0.001
        assert percentiles.top_ten[0] == 0.9
        assert len(percentiles.top_one) == 10

    def test_set_pandas_options(self):
        set_pandas_options()

        assert pd.get_option("display.float_format")(1234.5) == "1,234.50"
