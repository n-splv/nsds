from collections.abc import Callable

import pandas as pd
import pytest

from nsds.frame import install


@pytest.fixture(scope="session", autouse=True)
def _install_extensions() -> None:
    install()


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({
        "group": ["a", "a", "b", None],
        "amount": [10, 0, 30, 40],
        "quantity": [1, 2, 3, 4],
        "label": ["x", "", "y", "z"],
    })


@pytest.fixture
def series() -> pd.Series:
    return pd.Series(["a", "a", "b", None], name="group")


@pytest.fixture
def df_dated() -> pd.DataFrame:
    return pd.DataFrame({
        "day": pd.to_datetime(["2024-01-02 03:04:05", "2024-03-04 05:06:07"]),
        "amount": [1, 2],
    })


@pytest.fixture
def fake_require() -> Callable[[dict[str, object]], Callable[[str, str], object]]:
    """
    Builds a replacement for `nsds._deps.require` serving stubs by module name.
    """
    def build(modules: dict[str, object]) -> Callable[[str, str], object]:
        def require(module_name: str, extra: str) -> object:
            if module_name not in modules:
                raise ImportError(f"no stub for '{module_name}' (extra: {extra})")
            return modules[module_name]
        return require
    return build
