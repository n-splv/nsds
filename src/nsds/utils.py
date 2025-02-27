import datetime as dt
from decimal import Decimal, ROUND_HALF_UP
import inspect
from typing import Callable

import numpy as np
from pandas import Timestamp as pd_timestamp


class DateTimeUtils:

    @staticmethod
    def add_datetime_to_filename(filename: str,
                                 datetime: dt.datetime | pd_timestamp) -> str:
        date_string = format(datetime, "_%Y%m%d_%H%M%S")
        dot_index = filename.find(".")
        if dot_index == -1:
            return filename + date_string
        else:
            return filename[:dot_index] + date_string + filename[dot_index:]

    @property
    def naive_utcnow(self) -> dt.datetime:
        """ Since dt.datetime.utcnow() gets deprecated """
        return dt.datetime.now(dt.UTC).replace(tzinfo=None)

    @property
    def tomorrow(self) -> dt.date:
        return dt.date.today() + dt.timedelta(days=1)

    @property
    def yesterday(self) -> dt.date:
        return dt.date.today() - dt.timedelta(days=1)


datetime_utils = DateTimeUtils()


def gini_inequality_coefficient(x, w=None):
    """
    https://stackoverflow.com/a/49571213/17378319
    TODO doc & types
    """

    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def parameter_names(
    f: Callable,
    exclude: tuple[inspect._ParameterKind] = (inspect.Parameter.KEYWORD_ONLY,)
) -> list[str]:
    return [
        name
        for name, param in inspect.signature(f).parameters.items()
        if not param.kind in exclude
    ]


def recursively_remove_key(d, key_to_remove):
    """ https://stackoverflow.com/a/58938747 """
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == key_to_remove:
                del d[key]
            else:
                recursively_remove_key(d[key], key_to_remove)


def __rename_variable(old_name: str, new_name: str):
    """
    Doesn't work in jupyter, probably due to different scope
    """
    if globals().get(new_name):
        raise NameError(f"{new_name} is already used")
    elif new_name in dir(__builtins__):
        raise NameError(f"{new_name} is a builtin")
    globals()[new_name] = globals().pop(old_name)


def round_half_up(value: float, decimals: int) -> float:
    """
    Avoid 'bankers rounding' problem
    """
    multiplier = 10 ** decimals
    return float(
        Decimal(value * multiplier)
        .quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        / multiplier
    )
