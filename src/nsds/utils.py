import datetime as dt


# https://stackoverflow.com/a/58938747
def recursively_remove_key(d, key_to_remove):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == key_to_remove:
                del d[key]
            else:
                recursively_remove_key(d[key], key_to_remove)


def append_datetime_to_filename(filename: str) -> str:
    date_string = format(dt.datetime.utcnow(), "_%Y%m%d_%H%M%S")
    dot_index = filename.find(".")
    if dot_index == -1:
        return filename + date_string
    else:
        return filename[:dot_index] + date_string + filename[dot_index:]
