import pandas as pd

from nsds._compat import PANDAS_MAJOR


def set_pandas_options() -> None:
    pd.set_option("display.float_format", lambda x: format(x, ",.2f"))

    # On pandas 3 this is already the default and the option itself is deprecated
    if PANDAS_MAJOR < 3:
        pd.set_option("future.no_silent_downcasting", True)
