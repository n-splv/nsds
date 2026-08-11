from collections.abc import Hashable
from typing import Any


def recursively_remove_key(data: Any, key_to_remove: Hashable) -> None:
    """
    Drop every occurrence of a key from nested dicts, in place.
    https://stackoverflow.com/a/58938747
    """
    if isinstance(data, dict):
        for key in list(data.keys()):
            if key == key_to_remove:
                del data[key]
            else:
                recursively_remove_key(data[key], key_to_remove)
