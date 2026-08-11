import datetime as dt
from unittest.mock import patch

import pytest

from nsds.utils.dates import datetime_utils
from nsds.utils.introspect import parameter_names
from nsds.utils.mappings import recursively_remove_key
from nsds.utils.numeric import gini_inequality_coefficient, round_half_up
from nsds.utils.system import show_mac_notification

DATETIME = dt.datetime(2024, 1, 2, 3, 4, 5)


class TestDateTimeUtils:

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("data.csv", "data_20240102_030405.csv"),
            ("data", "data_20240102_030405"),
            ("a.b.csv", "a_20240102_030405.b.csv"),
            ("reports/data.csv", "reports/data_20240102_030405.csv"),
        ],
        ids=["extension", "no-extension", "multiple-dots", "with-directory"],
    )
    def test_add_datetime_to_filename(self, filename: str, expected: str):
        assert datetime_utils.add_datetime_to_filename(filename, DATETIME) == expected

    def test_naive_utcnow_has_no_timezone(self):
        assert datetime_utils.naive_utcnow.tzinfo is None

    def test_relative_dates(self):
        today = dt.date.today()
        assert datetime_utils.tomorrow - today == dt.timedelta(days=1)
        assert today - datetime_utils.yesterday == dt.timedelta(days=1)


class TestRoundHalfUp:

    @pytest.mark.parametrize(
        ("value", "decimals", "expected"),
        [
            (2.5, 0, 3.0),
            (1.5, 0, 2.0),
            (0.5, 0, 1.0),
            (-2.5, 0, -3.0),
            (1.25, 1, 1.3),
            (2.4, 0, 2.0),
        ],
        ids=["half-up", "half-up-odd", "half-up-zero", "negative", "decimals", "round-down"],
    )
    def test_values(self, value: float, decimals: int, expected: float):
        assert round_half_up(value, decimals) == expected


class TestGini:

    @pytest.mark.parametrize(
        ("x", "expected"),
        [([1, 1, 1, 1], 0.0), ([0, 0, 0, 1], 0.75)],
        ids=["equality", "inequality"],
    )
    def test_unweighted(self, x: list, expected: float):
        assert gini_inequality_coefficient(x) == pytest.approx(expected)

    def test_equal_weights_match_unweighted(self):
        x = [0, 0, 0, 1]
        weighted = gini_inequality_coefficient(x, [1, 1, 1, 1])
        assert weighted == pytest.approx(gini_inequality_coefficient(x))


class TestParameterNames:

    @staticmethod
    def _function(a, b, *, c):
        return a, b, c

    def test_excludes_keyword_only_by_default(self):
        assert parameter_names(self._function) == ["a", "b"]

    def test_exclude_can_be_overridden(self):
        assert parameter_names(self._function, exclude=()) == ["a", "b", "c"]


class TestRecursivelyRemoveKey:

    def test_removes_every_occurrence(self):
        data = {"keep": 1, "drop": 2, "nested": {"drop": 3, "deeper": {"drop": 4, "keep": 5}}}
        recursively_remove_key(data, "drop")
        assert data == {"keep": 1, "nested": {"deeper": {"keep": 5}}}

    @pytest.mark.parametrize(
        "data",
        [{"keep": 1}, {}, [1, 2], "text", None],
        ids=["absent", "empty", "list", "string", "none"],
    )
    def test_leaves_other_inputs_untouched(self, data: object):
        original = data.copy() if hasattr(data, "copy") else data
        recursively_remove_key(data, "drop")
        assert data == original


class TestShowMacNotification:

    @patch("nsds.utils.system.subprocess.run")
    def test_runs_osascript_on_macos(self, mock_run):
        with patch("nsds.utils.system.sys.platform", "darwin"):
            show_mac_notification("done", title="Job")

        mock_run.assert_called_once()
        command = mock_run.call_args.args[0]
        assert command[:2] == ["osascript", "-e"]
        assert command[2] == 'display notification "done" with title "Job"'

    @patch("nsds.utils.system.subprocess.run")
    def test_escapes_quotes(self, mock_run):
        with patch("nsds.utils.system.sys.platform", "darwin"):
            show_mac_notification('say "hi"')

        assert mock_run.call_args.args[0][2] == (
            'display notification "say \\"hi\\"" with title "Notification"'
        )

    @patch("nsds.utils.system.subprocess.run")
    def test_warns_instead_of_running_elsewhere(self, mock_run):
        with patch("nsds.utils.system.sys.platform", "linux"):
            with pytest.warns(UserWarning, match="only supported on macOS"):
                show_mac_notification("done")

        mock_run.assert_not_called()
