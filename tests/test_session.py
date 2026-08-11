from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.io
import pytest

from nsds import session as session_module
from nsds.session import setup


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mocks = {}
    for name in ("_load_dotenv", "_setup_pandas", "_setup_plotly"):
        mocks[name] = MagicMock()
        monkeypatch.setattr(session_module, name, mocks[name])
    monkeypatch.setattr(session_module, "IS_DATABRICKS", False)
    return mocks


class TestSetupDispatch:

    def test_defaults_run_every_local_step(self, steps: dict[str, MagicMock]):
        setup()

        for mock in steps.values():
            mock.assert_called_once()

    def test_flags_disable_steps(self, steps: dict[str, MagicMock]):
        setup(pandas=False, plotly=False, dotenv=False)

        for mock in steps.values():
            mock.assert_not_called()

    def test_local_only_steps_are_skipped_on_databricks(self,
                                                        steps: dict[str, MagicMock],
                                                        monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(session_module, "IS_DATABRICKS", True)

        setup()

        steps["_setup_pandas"].assert_called_once()
        steps["_load_dotenv"].assert_not_called()
        steps["_setup_plotly"].assert_not_called()

    def test_itables_is_opt_in(self, steps: dict[str, MagicMock]):
        with patch("nsds.tables.init") as mock_init:
            setup(itables=True)
            setup(itables=False)

        mock_init.assert_called_once()

    def test_logging_is_opt_in(self, steps: dict[str, MagicMock]):
        with patch("nsds.logs.configure_logging") as mock_configure:
            setup(logging=True)
            setup(logging=False)

        mock_configure.assert_called_once()


def test_setup_runs_against_the_real_dependencies():
    setup(dotenv=False, itables=True, logging=True)

    assert hasattr(pd.DataFrame, "vc")
    assert plotly.io.renderers.default == "plotly_mimetype"
    assert pd.get_option("display.float_format")(1234.5) == "1,234.50"


class TestSetupSteps:

    def test_pandas_step(self, monkeypatch: pytest.MonkeyPatch, fake_require: Callable):
        install = MagicMock()
        set_options = MagicMock()
        tqdm = MagicMock()
        monkeypatch.setattr("nsds.frame.install", install)
        monkeypatch.setattr("nsds.frame.set_pandas_options", set_options)
        monkeypatch.setattr(session_module, "require",
                            fake_require({"tqdm.auto": MagicMock(tqdm=tqdm)}))

        session_module._setup_pandas()

        install.assert_called_once()
        set_options.assert_called_once()
        tqdm.pandas.assert_called_once()

    def test_pandas_step_tolerates_missing_tqdm(self,
                                                monkeypatch: pytest.MonkeyPatch,
                                                fake_require: Callable):
        install = MagicMock()
        monkeypatch.setattr("nsds.frame.install", install)
        monkeypatch.setattr("nsds.frame.set_pandas_options", MagicMock())
        monkeypatch.setattr(session_module, "require", fake_require({}))

        session_module._setup_pandas()

        install.assert_called_once()

    def test_plotly_step_selects_the_compact_renderer(self,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     fake_require: Callable):
        plotly_io = MagicMock()
        monkeypatch.setattr(session_module, "require",
                            fake_require({"plotly.io": plotly_io}))

        session_module._setup_plotly()

        assert plotly_io.renderers.default == "plotly_mimetype"

    def test_dotenv_step_ignores_the_working_directory(self,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      fake_require: Callable):
        dotenv = MagicMock()
        dotenv.find_dotenv.return_value = "/somewhere/.env"
        monkeypatch.setattr(session_module, "require", fake_require({"dotenv": dotenv}))

        session_module._load_dotenv()

        dotenv.find_dotenv.assert_called_once_with(usecwd=False)
        dotenv.load_dotenv.assert_called_once_with("/somewhere/.env")
