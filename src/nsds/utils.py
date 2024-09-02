import datetime as dt

from pandas import Timestamp as pd_timestamp


def add_datetime_to_filename(filename: str,
                             datetime: dt.datetime | pd_timestamp) -> str:
    date_string = format(datetime, "_%Y%m%d_%H%M%S")
    dot_index = filename.find(".")
    if dot_index == -1:
        return filename + date_string
    else:
        return filename[:dot_index] + date_string + filename[dot_index:]


def recursively_remove_key(d, key_to_remove):
    """ https://stackoverflow.com/a/58938747 """
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == key_to_remove:
                del d[key]
            else:
                recursively_remove_key(d[key], key_to_remove)


def naive_utcnow() -> dt.datetime:
    """ Since dt.datetime.utcnow() gets deprecated """
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)
