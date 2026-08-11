from itertools import chain

import pandas as pd


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
