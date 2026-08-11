import pandas as pd

PANDAS_MAJOR = int(pd.__version__.split(".")[0])

# pandas 3 moved text columns from `object` to a dedicated `str` dtype
TEXT_DTYPES: list[str] = ["object", "str"] if PANDAS_MAJOR >= 3 else ["object"]
