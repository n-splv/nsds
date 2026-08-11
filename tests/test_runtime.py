import pytest

from nsds.runtime import DATABRICKS_ENV_MARKER, detect_runtime


@pytest.mark.parametrize(
    ("marker_value", "expected"),
    [("16.4", "databricks"), ("", "databricks"), (None, "local")],
    ids=["set", "set-empty", "unset"],
)
def test_detect_runtime(monkeypatch: pytest.MonkeyPatch,
                        marker_value: str | None,
                        expected: str):
    if marker_value is None:
        monkeypatch.delenv(DATABRICKS_ENV_MARKER, raising=False)
    else:
        monkeypatch.setenv(DATABRICKS_ENV_MARKER, marker_value)

    assert detect_runtime() == expected
