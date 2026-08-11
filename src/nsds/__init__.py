from importlib.metadata import PackageNotFoundError, version

from nsds.session import setup

try:
    __version__ = version("nsds")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__", "setup"]
