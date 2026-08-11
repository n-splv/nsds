from unittest.mock import MagicMock

import pandas as pd
import pytest

from nsds import tables


@pytest.fixture
def itables(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    monkeypatch.setattr(tables, "init_notebook_mode", stub.init_notebook_mode)
    monkeypatch.setattr(tables, "itables_show", stub.show)
    monkeypatch.setattr(tables, "_initialized", False)
    return stub


def test_show_loads_the_frontend_once(itables: MagicMock, df: pd.DataFrame):
    tables.show(df)
    tables.show(df)

    itables.init_notebook_mode.assert_called_once_with(all_interactive=False)
    assert itables.show.call_count == 2


def test_show_applies_defaults(itables: MagicMock, df: pd.DataFrame):
    tables.show(df)

    assert itables.show.call_args.kwargs["pageLength"] == 30
    assert itables.show.call_args.args == (df,)


def test_show_defaults_can_be_overridden(itables: MagicMock, df: pd.DataFrame):
    tables.show(df, pageLength=5)

    assert itables.show.call_args.kwargs["pageLength"] == 5
