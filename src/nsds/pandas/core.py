from itertools import chain
from pathlib import Path

from IPython.display import display
import pandas as pd
from pandas.core.generic import NDFrame

from nsds.utils import datetime_utils as dtu


class Percentiles:
    # 0.1%, 0.2% ... 1%
    bottom_one = [i / 1000 for i in range(1, 11)]

    # 1%, 2% ... 10%
    bottom_ten = [i / 100 for i in range(1, 11)]

    # 99.0%, 99.1% ... 99.9%
    top_one = [i / 1000 for i in range(990, 1000)]

    # 90%, 91% ... 99%
    top_ten = [i / 100 for i in range(90, 100)]


percentiles = Percentiles()


def dt_group(key: str, freq: str) -> pd.Grouper:
    return pd.Grouper(key=key, freq=freq)


def merge_insert_at(df_l: pd.DataFrame,
                    df_r: pd.DataFrame,
                    insert_index: int,
                    **kwargs) -> pd.DataFrame:

    # Convert negative index to positive
    if insert_index < 0:
        insert_index = df_l.shape[1] + 1 + insert_index

    # Preserve column order from df_r
    columns_to_add = (col for col in df_r.columns if col not in df_l.columns)

    columns = chain(
        df_l.columns[:insert_index],
        columns_to_add,
        df_l.columns[insert_index:],
    )
    return pd.merge(df_l, df_r, **kwargs)[columns]


def read_csvs(file_mask: str,
              add_filename_column: bool = False,
              **kwargs) -> pd.DataFrame:

    df_generator = (
        (
            pd.read_csv(filepath, **kwargs)
            .assign(**{"_file": filepath.name} if add_filename_column else {})
        )
        for filepath in Path().glob(file_mask)
    )

    return pd.concat(df_generator, ignore_index=True)


class NDFrameExtensions(NDFrame):

    def explode_all(self, *args, **kwargs) -> NDFrame:
        if isinstance(self, pd.DataFrame):
            kwargs |= {"column": self.columns.tolist()}
        return self.explode(*args, **kwargs)

    def preview(self, min_rows: int = 4):
        context = ("display.min_rows", min_rows, "display.max_rows", min_rows)
        with pd.option_context(*context):
            display(self)

    def show(self, nrows: int = None, ncols: int = None):
        context = (
            "display.max_colwidth", None,
            "display.max_rows", nrows,
            "display.max_columns", ncols
        )
        with pd.option_context(*context):
            display(self.iloc[:nrows])

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
                raise KeyError(f"No datetime column '{add_date_to_filename}'")

        elif add_date_to_filename is True:
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


def init_pandas_extensions():
    """ No overrides, only new methods """
    extensions = set(dir(NDFrameExtensions)) - set(dir(NDFrame))
    for ext_name in extensions:
        setattr(pd.DataFrame, ext_name, getattr(NDFrameExtensions, ext_name))
        setattr(pd.Series, ext_name, getattr(NDFrameExtensions, ext_name))
