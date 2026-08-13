from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, overload

import pandas as pd

from nsds._deps import require
from nsds.runtime import IS_DATABRICKS

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

ENV_SERVER_HOSTNAME = "DATABRICKS_SERVER_HOSTNAME"
ENV_HTTP_PATH = "DATABRICKS_HTTP_PATH"
ENV_ACCESS_TOKEN = "DATABRICKS_TOKEN"

_REQUIRED_CONNECT_ARGS = ("server_hostname", "http_path", "access_token")


@overload
def read_sql(query: str,
             params: dict[str, Any] | None = None,
             *,
             as_spark: Literal[False] = False,
             **connect_kwargs) -> pd.DataFrame: ...
@overload
def read_sql(query: str,
             params: dict[str, Any] | None = None,
             *,
             as_spark: Literal[True],
             **connect_kwargs) -> SparkDataFrame: ...
def read_sql(query: str,
             params: dict[str, Any] | None = None,
             *,
             as_spark: bool = False,
             **connect_kwargs) -> pd.DataFrame | SparkDataFrame:
    """
    Run a query against Databricks and return a DataFrame.

    On a cluster the active SparkSession is used. Locally, a connection is
    opened via `databricks-sql-connector`, taking its arguments from
    `connect_kwargs` and falling back to the DATABRICKS_SERVER_HOSTNAME,
    DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN environment variables.

    Pass `as_spark=True` on a cluster to skip `toPandas()` and get a Spark
    DataFrame. Locally that flag raises.
    """
    if as_spark and not IS_DATABRICKS:
        raise RuntimeError("`as_spark=True` is only available on Databricks")
    if IS_DATABRICKS:
        return _read_sql_spark(query, params, as_spark=as_spark)
    return _read_sql_connector(query, params, **connect_kwargs)


def _read_sql_spark(query: str,
                    params: dict[str, Any] | None,
                    *,
                    as_spark: bool) -> pd.DataFrame | SparkDataFrame:
    from pyspark.sql import SparkSession

    # `spark` is a notebook global - not visible in imported modules
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession")
    sdf = spark.sql(query, params)
    if as_spark:
        return sdf
    _enable_arrow_if_available(spark)
    return _decimals_as_double(sdf).toPandas()


def _decimals_as_double(sdf: SparkDataFrame) -> SparkDataFrame:
    from pyspark.sql import functions as F
    from pyspark.sql.types import DecimalType

    # Warehouse DECIMAL becomes Python Decimal in toPandas, which pandas
    # reports as object — then select_dtypes(include="number") misses cols.
    casts = []
    has_decimal = False
    for field in sdf.schema.fields:
        col = F.col(field.name)
        if isinstance(field.dataType, DecimalType):
            has_decimal = True
            casts.append(col.cast("double").alias(field.name))
        else:
            casts.append(col)
    return sdf.select(*casts) if has_decimal else sdf


def _enable_arrow_if_available(spark: SparkSession):
    from pyspark.errors import PySparkException

    # Speeds up toPandas on older DBR. Serverless / recent runtimes manage Arrow
    # internally and reject this key with CONFIG_NOT_AVAILABLE.
    try:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    except PySparkException:
        pass


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
