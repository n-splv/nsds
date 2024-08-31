import datetime as dt

from pandas import Timestamp as pd_timestamp


# https://stackoverflow.com/a/58938747
def recursively_remove_key(d, key_to_remove):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == key_to_remove:
                del d[key]
            else:
                recursively_remove_key(d[key], key_to_remove)


def add_datetime_to_filename(filename: str,
                             datetime: dt.datetime | pd_timestamp) -> str:
    date_string = format(datetime, "_%Y%m%d_%H%M%S")
    dot_index = filename.find(".")
    if dot_index == -1:
        return filename + date_string
    else:
        return filename[:dot_index] + date_string + filename[dot_index:]
