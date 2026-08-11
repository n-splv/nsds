import json

import pytest

from nsds._deps import require


def test_returns_the_imported_module():
    assert require("json", "notebook") is json


def test_error_names_the_module_and_the_extra():
    with pytest.raises(ImportError, match=r"'nope_not_here'.*nsds\[gsheets\]"):
        require("nope_not_here", "gsheets")
