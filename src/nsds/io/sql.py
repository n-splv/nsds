import os
import warnings
from typing import Any

import pandas as pd

from nsds._deps import require
from nsds.runtime import IS_DATABRICKS

ENV_SERVER_HOSTNAME = "DATABRICKS_SERVER_HOSTNAME"
ENV_HTTP_PATH = "DATABRICKS_HTTP_PATH"
ENV_ACCESS_TOKEN = "DATABRICKS_TOKEN"

_REQUIRED_CONNECT_ARGS = ("server_hostname", "http_path", "access_token")


def read_sql(query: str,
             params: dict[str, Any] | None = None,
             **connect_kwargs) -> pd.DataFrame:
    """
    Run a query against Databricks and return a DataFrame.

    On a cluster the active SparkSession is used. Locally, a connection is
    opened via `databricks-sql-connector`, taking its arguments from
    `connect_kwargs` and falling back to the DATABRICKS_SERVER_HOSTNAME,
    DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN environment variables.
    """
    if IS_DATABRICKS:
        return _read_sql_spark(query, params)
    return _read_sql_connector(query, params, **connect_kwargs)


def _read_sql_spark(query: str, params: dict[str, Any] | None) -> pd.DataFrame:
    from pyspark.sql import SparkSession

    # `spark` is a notebook global - not visible in imported modules
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark.sql(query, params).toPandas()


def _read_sql_connector(query: str,
                        params: dict[str, Any] | None,
                        **connect_kwargs) -> pd.DataFrame:
    sql = require("databricks.sql", "sql")
    connect_kwargs = connection_args_from_env() | connect_kwargs

    missing = [name for name in _REQUIRED_CONNECT_ARGS if not connect_kwargs.get(name)]
    if missing:
        raise ValueError(
            f"Missing connection arguments: {', '.join(missing)}. "
            f"Pass them to `read_sql` or set {ENV_SERVER_HOSTNAME}, "
            f"{ENV_HTTP_PATH} and {ENV_ACCESS_TOKEN}."
        )

    with sql.connect(**connect_kwargs) as connection:
        # pandas warns about any connection that is not SQLAlchemy
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pandas only supports")
            return pd.read_sql(query, params=params, con=connection,
                               dtype_backend="pyarrow")


def connection_args_from_env() -> dict[str, str]:
    args = {
        "server_hostname": os.getenv(ENV_SERVER_HOSTNAME),
        "http_path": os.getenv(ENV_HTTP_PATH),
        "access_token": os.getenv(ENV_ACCESS_TOKEN),
    }
    return {name: value for name, value in args.items() if value is not None}
