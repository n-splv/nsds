from collections.abc import Callable, Iterator
from functools import partial, reduce
from itertools import starmap

import pandas as pd
from pandas.core.generic import NDFrame

from nsds._compat import TEXT_DTYPES
from nsds._deps import require
from nsds.utils.dates import datetime_utils as dtu
from nsds.utils.introspect import parameter_names


def _display(obj: object) -> None:
    require("IPython.display", "notebook").display(obj)


class NDFrameExtensions(NDFrame):
    """
    Extra methods that `install()` attaches to `pd.DataFrame` and `pd.Series`.

    Static analysers cannot see monkey-patched attributes, so editors will not
    autocomplete these on a DataFrame - Jupyter's runtime completion will.
    """

    def apply_row_wise[T](self,
                          func: Callable[..., T],
                          show_progress: bool = False,
                          **kwargs) -> Iterator[T]:
        """
        Infer non-keyword-only arguments from function signature
        to specify dataframe columns
        """
        if any(kwargs):
            func = partial(func, **kwargs)
        values = self[parameter_names(func)].values
        if show_progress:
            values = require("tqdm.auto", "notebook").tqdm(values)
        return starmap(func, values)

    def explode_all(self, *args, **kwargs) -> NDFrame:
        if isinstance(self, pd.DataFrame):
            kwargs |= {"column": self.columns.tolist()}
        return self.explode(*args, **kwargs)

    def memory_mb(self) -> pd.Series | float:
        return self.memory_usage(deep=True) / 1024 ** 2

    def missing(self: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """
        Detailed report on the missing values
        """

        # Series doesn't have .select_dtypes method
        if isinstance(self, pd.Series):
            data = self.to_frame()
        else:
            data = self

        result = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="left"
            ),
            (
                data.isna().sum().rename("isna"),
                data.select_dtypes((int, float)).eq(0).sum().rename("eq0"),
                data.select_dtypes(TEXT_DTYPES).eq("").sum().rename("empty_str"),
            )
        )
        result = result.dropna(axis=1, how="all")
        i = 1
        for col in result.columns:
            result.insert(i, f"{col}_pct", result[col].div(data.shape[0]).mul(100).round(2))
            i += 2
        return result

    def preview(self, min_rows: int = 4):
        context = ("display.min_rows", min_rows, "display.max_rows", min_rows)
        with pd.option_context(*context):
            _display(self)

    def show(self, nrows: int = None, ncols: int = None):
        context = (
            "display.max_colwidth", None,
            "display.max_rows", nrows,
            "display.max_columns", ncols
        )
        with pd.option_context(*context):
            _display(self.iloc[:nrows])

    def sort(self, *args, **kwargs) -> NDFrame:
        return self.sort_values(*args, **kwargs)

    def sortd(self, *args, **kwargs) -> NDFrame:
        if kwargs.get("ascending") is not None:
            raise ValueError(
                "`sortd` is always descending. "
                "If you want to use the keyword argument, use `sort_values`"
            )
        return self.sort_values(*args, ascending=False, **kwargs)

    def to_csv_(self,
                *args,
                add_date_to_filename: bool | str = False,
                **kwargs):
        """
        Saves to csv with an encoding that is more reliable for Excel.

        If `add_date_to_filename` is True, then the current UTC time will
        be added to the filename. This argument can also be a name of a datetime
        column (only for a DataFrame) - in such case, its maximum value will be
        used.
        """

        kwargs.setdefault("index", False)
        kwargs.setdefault("encoding", "utf_8_sig")
        filename = kwargs.pop("path_or_buf", None) or args[0]

        if isinstance(add_date_to_filename, str):
            try:
                self[add_date_to_filename].dt  # noqa check column
                filename = dtu.add_datetime_to_filename(
                    filename,
                    self[add_date_to_filename].max()
                )
            except (KeyError, AttributeError):
                raise KeyError(f"No datetime column '{add_date_to_filename}'") from None

        elif add_date_to_filename:
            filename = dtu.add_datetime_to_filename(filename, dtu.naive_utcnow)

        args = (filename, *args[1:])

        return self.to_csv(*args, **kwargs)

    def vc(self,
           as_index: bool = True,
           dropna: bool = False,
           min_bin_size: int = 1,
           show_cumulative: bool = False) -> pd.DataFrame:
        """
        Advanced version of pandas `value_counts`:
        - Also works on DataFrames;
        - Shows both count and percentage;
        - Can show cumulative values.
        """

        if isinstance(self, pd.Series):
            count = self.value_counts(dropna=dropna)
        elif isinstance(self, pd.DataFrame):
            count = (
                self
                .groupby(self.columns.tolist(), dropna=dropna)
                .size()
                .rename("count")
                .sort_values(ascending=False)
            )
        else:
            raise TypeError

        percentage = (count / sum(count) * 100)
        df = pd.DataFrame({
            "count": count,
            "percentage": percentage,
        }, index=count.index)

        if show_cumulative:
            df.insert(1, "count_cumulative", count.cumsum())
            df.insert(3, "percentage_cumulative", percentage.cumsum())

        if min_bin_size > 1:
            filt = df["count"] >= min_bin_size
            df = df[filt]

        if not as_index:
            df = df.reset_index()

        return df.round(2)


def extension_names() -> set[str]:
    return set(dir(NDFrameExtensions)) - set(dir(NDFrame))


def install() -> None:
    """ No overrides, only new methods """
    for name in extension_names():
        method = getattr(NDFrameExtensions, name)
        setattr(pd.DataFrame, name, method)
        setattr(pd.Series, name, method)
