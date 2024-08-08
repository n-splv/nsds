from IPython.display import display
import pandas as pd
from pandas.core.generic import NDFrame


class DataFrameDisplay:
    """
    TODO
    paginate: bool | int = False
    style_numbers: bool = True
    """

    @staticmethod
    def show(data: pd.DataFrame | pd.Series,
             nrows: int = None,
             ncols: int = None):
        context = (
            "display.max_colwidth", None,
            "display.max_rows", nrows,
            "display.max_columns", ncols
        )
        with pd.option_context(*context):
            display(data)

    @staticmethod
    def preview(df: pd.DataFrame | pd.Series,
                min_rows: int = 4):
        context = ("display.min_rows", min_rows, "display.max_rows", min_rows)
        with pd.option_context(*context):
            display(df)

    @staticmethod
    def style_numbers(df: pd.DataFrame):
        display(df.style.format(thousands=",", precision=2))


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

    columns = (
        df_l.columns[:insert_index].tolist()
        + df_r.columns.difference(df_l.columns).tolist()
        + df_l.columns[insert_index:].tolist()
    )
    return pd.merge(df_l, df_r, **kwargs)[columns]


class PandasExtensions(NDFrame):

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
            display(self)

    def sortd(self, *args, **kwargs):
        return self.sort_values(*args, ascending=False, **kwargs)

    def vc(self,
           as_index: bool = True,
           dropna: bool = True,
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
            )
        else:
            raise TypeError

        percentage = (count / sum(count) * 100)
        df = pd.DataFrame({
            "count": count,
            "percentage": percentage,
        }, index=count.index)

        if show_cumulative:
            df.insert(1, "counts_cumulative", count.cumsum())
            df.insert(3, "percentage_cumulative", percentage.cumsum())

        if min_bin_size > 1:
            filt = df["count"] >= min_bin_size
            df = df[filt]

        if not as_index:
            df = df.reset_index()

        return df.round(2)


def init_pandas_extensions():
    """ No overrides, only new methods """
    extensions = set(dir(PandasExtensions)) - set(dir(NDFrame))
    for ext_name in extensions:
        setattr(pd.DataFrame, ext_name, getattr(PandasExtensions, ext_name))
        setattr(pd.Series, ext_name, getattr(PandasExtensions, ext_name))
