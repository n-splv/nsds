import os
from typing import Literal

type RuntimeEnv = Literal["databricks", "local"]

DATABRICKS_ENV_MARKER = "DATABRICKS_RUNTIME_VERSION"


def detect_runtime() -> RuntimeEnv:
    return "databricks" if DATABRICKS_ENV_MARKER in os.environ else "local"


RUNTIME_ENV: RuntimeEnv = detect_runtime()
IS_DATABRICKS: bool = RUNTIME_ENV == "databricks"
