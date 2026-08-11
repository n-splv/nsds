from functools import partial, reduce
from pathlib import Path

import pandas as pd

read_csv_pyarrow = partial(pd.read_csv, dtype_backend="pyarrow", engine="pyarrow")


def read_csvs(file_mask: str,
              add_filename_column: bool = False,
              **kwargs) -> pd.DataFrame:

    def _concat(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df1, df2], ignore_index=True)

    df_generator = (
        (
            pd.read_csv(filepath, **kwargs)
            .assign(**{"_file": filepath.name} if add_filename_column else {})
        )
        for filepath in Path().glob(file_mask)
    )

    return reduce(_concat, df_generator)
