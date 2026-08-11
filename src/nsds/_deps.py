import importlib
from types import ModuleType


def require(module_name: str, extra: str) -> ModuleType:
    """
    Import an optional dependency, pointing at the extra that provides it.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"'{module_name}' is needed for this feature but is not installed. "
            f"Install it with: pip install 'nsds[{extra}]'"
        ) from e
