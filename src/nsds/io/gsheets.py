from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from nsds._deps import require

if TYPE_CHECKING:
    import gspread

DEFAULT_KEY_PATH = Path.home() / ".config" / "gspread" / "service_account.json"

ENV_SERVICE_ACCOUNT_KEY = "GOOGLE_SERVICE_ACCOUNT_KEY"
ENV_SECRET_SCOPE = "GSPREAD_SECRET_SCOPE"
ENV_SECRET_KEY = "GSPREAD_SECRET_KEY"


def get_gspread_client(raw: str | dict | None = None,
                       path: Path | str = DEFAULT_KEY_PATH,
                       secret_scope: str | None = None,
                       secret_key: str | None = None) -> gspread.Client:
    """
    Authenticate a gspread client, trying in order:
    1. `raw` service-account JSON, as a string or a dict;
    2. the same JSON in the GOOGLE_SERVICE_ACCOUNT_KEY environment variable;
    3. a key file at `path`;
    4. a Databricks secret, from `secret_scope`/`secret_key` or their
       GSPREAD_SECRET_SCOPE/GSPREAD_SECRET_KEY environment variables.
    """
    gspread_module = require("gspread", "gsheets")

    if raw is not None:
        if isinstance(raw, str):
            raw = json.loads(raw)
        return gspread_module.service_account_from_dict(raw)

    env_raw = os.getenv(ENV_SERVICE_ACCOUNT_KEY)
    if env_raw:
        return gspread_module.service_account_from_dict(json.loads(env_raw))

    path = Path(path)
    if path.exists():
        return gspread_module.service_account(filename=path)

    scope = secret_scope or os.getenv(ENV_SECRET_SCOPE)
    key = secret_key or os.getenv(ENV_SECRET_KEY)
    if not (scope and key):
        raise FileNotFoundError(_missing_credentials_message(path))

    raw = _read_databricks_secret(scope, key)
    return gspread_module.service_account_from_dict(json.loads(raw))


def _missing_credentials_message(path: Path) -> str:
    return (
        "No Google credentials found. Pass the service-account JSON as `raw`, "
        f"save it to {path}, set {ENV_SERVICE_ACCOUNT_KEY}, or - on Databricks - "
        f"point `secret_scope`/`secret_key` (or {ENV_SECRET_SCOPE}/"
        f"{ENV_SECRET_KEY}) at a secret holding it."
    )


def _read_databricks_secret(scope: str, key: str) -> str:
    # `dbutils` is a notebook global - pull it from the IPython namespace
    ipython = require("IPython", "notebook").get_ipython()
    if ipython is None or "dbutils" not in ipython.user_ns:
        raise RuntimeError("`dbutils` is only available inside a Databricks notebook")
    return ipython.user_ns["dbutils"].secrets.get(scope=scope, key=key)
