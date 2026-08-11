from typing import Any

import pandas as pd
from itables import init_notebook_mode
from itables import show as itables_show

SHOW_DEFAULTS: dict[str, Any] = {
    "classes": "compact",
    "autoWidth": False,
    "layout": {"top1": "searchBuilder"},
    "stateSave": True,
    "buttons": [
        'columnsToggle',
        {
            'extend': 'colvisGroup',
            'text': 'Hide all',
            'hide': ':visible'
        },
        {
            'extend': 'colvisGroup',
            'text': 'Show all',
            'show': ':hidden'
        }
    ],
    "maxBytes": "1MB",
    "pageLength": 30,
}

_initialized = False


def init(all_interactive: bool = False) -> None:
    """
    Load the itables frontend assets. Called automatically by `show()`.
    """
    global _initialized
    init_notebook_mode(all_interactive=all_interactive)
    _initialized = True


def show(data: pd.DataFrame | pd.Series, **kwargs) -> None:
    if not _initialized:
        init()
    itables_show(data, **(SHOW_DEFAULTS | kwargs))
