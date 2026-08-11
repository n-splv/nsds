from contextlib import suppress

from nsds._deps import require
from nsds.runtime import IS_DATABRICKS


def setup(*,
          pandas: bool = True,
          plotly: bool = True,
          dotenv: bool = True,
          itables: bool = False,
          logging: bool = False) -> None:
    """
    Configure a notebook session. This is the only function in `nsds` with side
    effects - importing the package does nothing on its own.

    `dotenv` and the compact `plotly` renderer only make sense locally and are
    skipped on a Databricks cluster. Everything here needs `nsds[notebook]`,
    so turn off what you have not installed.
    """
    if dotenv and not IS_DATABRICKS:
        _load_dotenv()
    if pandas:
        _setup_pandas()
    if plotly and not IS_DATABRICKS:
        _setup_plotly()
    if itables:
        require("itables", "notebook")
        from nsds import tables
        tables.init()
    if logging:
        from nsds.logs import configure_logging
        configure_logging()


def _load_dotenv() -> None:
    dotenv = require("dotenv", "notebook")
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=False), override=True)


def _setup_pandas() -> None:
    from nsds.frame import install, set_pandas_options

    install()
    set_pandas_options()

    # Enables `.progress_apply`, but is not worth failing the whole setup over
    with suppress(ImportError):
        require("tqdm.auto", "notebook").tqdm.pandas()


def _setup_plotly() -> None:
    # Compact JSON via the JupyterLab extension: embedding plotly.js (~4.7MB) into
    # every output bloats the notebook and breaks the collaborative document sync
    require("plotly.io", "notebook").renderers.default = "plotly_mimetype"
