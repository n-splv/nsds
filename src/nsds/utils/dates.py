from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class DateTimeUtils:

    @staticmethod
    def add_datetime_to_filename(filename: str,
                                 datetime: dt.datetime | pd.Timestamp) -> str:
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
