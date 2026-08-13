import sys
from collections.abc import Callable
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nsds.io import gsheets as gsheets_module
from nsds.io import sql as sql_module
from nsds.io.gsheets import (
    ENV_SERVICE_ACCOUNT_KEY,
    get_gspread_client,
    overwrite_worksheet,
    overwrite_worksheet_from_spark,
    spark_df_to_rows,
)
from nsds.io.sql import (
    ENV_ACCESS_TOKEN,
    ENV_HTTP_PATH,
    ENV_SERVER_HOSTNAME,
    connection_args_from_env,
    read_sql,
)

CREDENTIALS = {"type": "service_account"}


@pytest.fixture
def env_connection(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    values = {
        ENV_SERVER_HOSTNAME: "host",
        ENV_HTTP_PATH: "/sql/path",
        ENV_ACCESS_TOKEN: "token",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


@pytest.fixture
def no_env_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (ENV_SERVER_HOSTNAME, ENV_HTTP_PATH, ENV_ACCESS_TOKEN):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch, fake_require: Callable) -> MagicMock:
    """
    Stubs `databricks.sql` and forces the local (non-cluster) code path.
    """
    connection = MagicMock()
    stub = MagicMock()
    stub.connect.return_value.__enter__.return_value = connection
    stub.connect.return_value.__exit__.return_value = False
    stub.connection = connection

    monkeypatch.setattr(sql_module, "IS_DATABRICKS", False)
    monkeypatch.setattr(sql_module, "require", fake_require({"databricks.sql": stub}))
    return stub


class PySparkException(Exception):
    pass


@pytest.fixture
def spark(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Stubs `pyspark.sql` and forces the cluster code path.
    """
    session = MagicMock()
    session_class = MagicMock()
    session_class.getActiveSession.return_value = session

    pyspark = ModuleType("pyspark")
    pyspark_sql = ModuleType("pyspark.sql")
    pyspark_errors = ModuleType("pyspark.errors")
    pyspark_errors.PySparkException = PySparkException
    pyspark_sql.SparkSession = session_class
    pyspark.sql = pyspark_sql
    pyspark.errors = pyspark_errors

    monkeypatch.setitem(sys.modules, "pyspark", pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", pyspark_sql)
    monkeypatch.setitem(sys.modules, "pyspark.errors", pyspark_errors)
    monkeypatch.setattr(sql_module, "IS_DATABRICKS", True)

    session.session_class = session_class
    session.PySparkException = PySparkException
    return session


class TestReadSqlLocally:

    def test_connects_from_env_and_reads(self, connector: MagicMock, env_connection: dict):
        expected = pd.DataFrame({"a": [1]})

        with patch.object(pd, "read_sql", return_value=expected) as mock_read:
            result = read_sql("SELECT 1", {"x": 2})

        assert result is expected
        connector.connect.assert_called_once_with(
            server_hostname="host", http_path="/sql/path", access_token="token"
        )
        assert mock_read.call_args.kwargs == {
            "params": {"x": 2},
            "con": connector.connection,
            "dtype_backend": "pyarrow",
        }

    def test_explicit_arguments_win_over_env(self,
                                             connector: MagicMock,
                                             env_connection: dict):
        with patch.object(pd, "read_sql", return_value=pd.DataFrame()):
            read_sql("SELECT 1", http_path="/override")

        assert connector.connect.call_args.kwargs["http_path"] == "/override"

    @pytest.mark.parametrize(
        ("provided", "missing"),
        [
            ({}, ["server_hostname", "http_path", "access_token"]),
            ({"server_hostname": "host"}, ["http_path", "access_token"]),
            ({"server_hostname": "host", "http_path": "/p"}, ["access_token"]),
        ],
        ids=["none", "hostname-only", "token-missing"],
    )
    def test_reports_missing_connection_arguments(self,
                                                  connector: MagicMock,
                                                  no_env_connection: None,
                                                  provided: dict,
                                                  missing: list[str]):
        with pytest.raises(ValueError, match=", ".join(missing)):
            read_sql("SELECT 1", **provided)

        connector.connect.assert_not_called()

    def test_as_spark_is_rejected(self, connector: MagicMock, env_connection: dict):
        with pytest.raises(RuntimeError, match="only available on Databricks"):
            read_sql("SELECT 1", as_spark=True)

        connector.connect.assert_not_called()


class TestReadSqlOnDatabricks:

    def test_uses_the_active_session(self, spark: MagicMock):
        expected = pd.DataFrame({"a": [1]})
        spark.sql.return_value.toPandas.return_value = expected

        result = read_sql("SELECT 1", {"x": 2})

        assert result is expected
        spark.sql.assert_called_once_with("SELECT 1", {"x": 2})
        spark.conf.set.assert_called_once_with(
            "spark.sql.execution.arrow.pyspark.enabled", "true"
        )

    def test_continues_when_arrow_config_is_unavailable(self, spark: MagicMock):
        expected = pd.DataFrame({"a": [1]})
        spark.conf.set.side_effect = spark.PySparkException("CONFIG_NOT_AVAILABLE")
        spark.sql.return_value.toPandas.return_value = expected

        result = read_sql("SELECT 1")

        assert result is expected
        spark.sql.assert_called_once()

    def test_as_spark_skips_to_pandas(self, spark: MagicMock):
        expected = MagicMock(name="spark-df")
        spark.sql.return_value = expected

        result = read_sql("SELECT 1", as_spark=True)

        assert result is expected
        spark.sql.assert_called_once_with("SELECT 1", None)
        expected.toPandas.assert_not_called()
        spark.conf.set.assert_not_called()

    def test_requires_an_active_session(self, spark: MagicMock):
        spark.session_class.getActiveSession.return_value = None

        with pytest.raises(RuntimeError, match="No active SparkSession"):
            read_sql("SELECT 1")


class TestConnectionArgsFromEnv:

    def test_reads_all_three(self, env_connection: dict):
        assert connection_args_from_env() == {
            "server_hostname": "host",
            "http_path": "/sql/path",
            "access_token": "token",
        }

    def test_omits_unset_variables(self,
                                   no_env_connection: None,
                                   monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(ENV_SERVER_HOSTNAME, "host")

        assert connection_args_from_env() == {"server_hostname": "host"}


class TestGetGspreadClient:

    @pytest.fixture
    def gspread(self,
                monkeypatch: pytest.MonkeyPatch,
                fake_require: Callable) -> MagicMock:
        stub = MagicMock()
        monkeypatch.delenv(ENV_SERVICE_ACCOUNT_KEY, raising=False)
        monkeypatch.setattr(
            gsheets_module, "require", fake_require({"gspread": stub})
        )
        return stub

    @pytest.mark.parametrize(
        "raw",
        [CREDENTIALS, '{"type": "service_account"}'],
        ids=["dict", "json-string"],
    )
    def test_raw_credentials(self, gspread: MagicMock, tmp_path, raw: str | dict):
        result = get_gspread_client(raw, path=tmp_path / "absent.json")

        gspread.service_account_from_dict.assert_called_once_with(CREDENTIALS)
        assert result is gspread.service_account_from_dict.return_value

    def test_environment_variable(self,
                                  gspread: MagicMock,
                                  tmp_path,
                                  monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(ENV_SERVICE_ACCOUNT_KEY, '{"type": "service_account"}')

        get_gspread_client(path=tmp_path / "absent.json")

        gspread.service_account_from_dict.assert_called_once_with(CREDENTIALS)

    def test_key_file(self, gspread: MagicMock, tmp_path):
        key_path = tmp_path / "key.json"
        key_path.write_text('{"type": "service_account"}')

        get_gspread_client(path=key_path)

        gspread.service_account.assert_called_once_with(filename=key_path)

    def test_databricks_secret(self,
                               monkeypatch: pytest.MonkeyPatch,
                               fake_require: Callable,
                               tmp_path):
        gspread = MagicMock()
        dbutils = MagicMock()
        dbutils.secrets.get.return_value = '{"type": "service_account"}'
        ipython = MagicMock(user_ns={"dbutils": dbutils})
        monkeypatch.delenv(ENV_SERVICE_ACCOUNT_KEY, raising=False)
        monkeypatch.setattr(gsheets_module, "require", fake_require({
            "gspread": gspread,
            "IPython": MagicMock(get_ipython=MagicMock(return_value=ipython)),
        }))

        get_gspread_client(
            path=tmp_path / "absent.json",
            secret_scope="my-scope",
            secret_key="my-key",
        )

        dbutils.secrets.get.assert_called_once_with(scope="my-scope", key="my-key")
        gspread.service_account_from_dict.assert_called_once_with(CREDENTIALS)

    def test_secret_lookup_needs_a_notebook(self,
                                            monkeypatch: pytest.MonkeyPatch,
                                            fake_require: Callable,
                                            tmp_path):
        monkeypatch.delenv(ENV_SERVICE_ACCOUNT_KEY, raising=False)
        monkeypatch.setattr(gsheets_module, "require", fake_require({
            "gspread": MagicMock(),
            "IPython": MagicMock(get_ipython=MagicMock(return_value=None)),
        }))

        with pytest.raises(RuntimeError, match="Databricks notebook"):
            get_gspread_client(
                path=tmp_path / "absent.json",
                secret_scope="my-scope",
                secret_key="my-key",
            )

    def test_no_credentials_anywhere(self, gspread: MagicMock, tmp_path):
        with pytest.raises(FileNotFoundError, match="No Google credentials found"):
            get_gspread_client(path=tmp_path / "absent.json")


class TestOverwriteWorksheet:

    def test_clears_empty_input(self):
        ws = MagicMock()

        overwrite_worksheet(ws, [])

        ws.clear.assert_called_once_with()
        ws.update.assert_not_called()
        ws.resize.assert_not_called()

    def test_writes_in_chunks_without_shrinking(self):
        ws = MagicMock(row_count=100, col_count=26)
        rows = [["a", "b"], [1, 2], [3, 4], [5, 6]]

        overwrite_worksheet(ws, rows, chunk_rows=2)

        ws.clear.assert_called_once_with()
        ws.resize.assert_not_called()
        assert ws.update.call_count == 2
        first = ws.update.call_args_list[0]
        assert first.args[0] == [["a", "b"], [1, 2]]
        assert first.kwargs == {
            "range_name": "A1",
            "value_input_option": "USER_ENTERED",
        }
        second = ws.update.call_args_list[1]
        assert second.args[0] == [[3, 4], [5, 6]]
        assert second.kwargs["range_name"] == "A3"

    def test_expands_grid_when_data_exceeds_sheet(self):
        ws = MagicMock(row_count=2, col_count=1)
        rows = [["a", "b", "c"], [1, 2, 3], [4, 5, 6]]

        overwrite_worksheet(ws, rows)

        ws.resize.assert_called_once_with(rows=3, cols=3)


class TestSparkDfToRows:

    def test_casts_special_types_and_fills_nulls(self, monkeypatch: pytest.MonkeyPatch):
        from types import SimpleNamespace

        class DateType: ...
        class TimestampType: ...
        class ArrayType: ...
        class MapType: ...
        class StructType: ...
        class DecimalType: ...
        class LongType: ...

        def fake_col(name: str):
            c = MagicMock(name=f"col:{name}")
            c.cast.return_value.alias.side_effect = (
                lambda alias: MagicMock(name=f"{name}.cast.alias({alias})")
            )
            return c

        F = MagicMock()
        F.col.side_effect = fake_col
        F.date_format.return_value.alias.side_effect = (
            lambda alias: MagicMock(name=f"date_format.alias({alias})")
        )
        F.to_json.return_value.alias.side_effect = (
            lambda alias: MagicMock(name=f"to_json.alias({alias})")
        )

        prepared = MagicMock()
        prepared.columns = ["d", "t", "arr", "m", "s", "dec", "n"]
        prepared.toLocalIterator.return_value = [
            ("2024-01-02", "2024-01-02 03:04:05", "[1]", '{"a":1}', '{"x":1}', "1.5", None),
        ]

        sdf = MagicMock()
        sdf.schema.fields = [
            SimpleNamespace(name="d", dataType=DateType()),
            SimpleNamespace(name="t", dataType=TimestampType()),
            SimpleNamespace(name="arr", dataType=ArrayType()),
            SimpleNamespace(name="m", dataType=MapType()),
            SimpleNamespace(name="s", dataType=StructType()),
            SimpleNamespace(name="dec", dataType=DecimalType()),
            SimpleNamespace(name="n", dataType=LongType()),
        ]
        sdf.select.return_value = prepared

        types = ModuleType("pyspark.sql.types")
        types.DateType = DateType
        types.TimestampType = TimestampType
        types.ArrayType = ArrayType
        types.MapType = MapType
        types.StructType = StructType
        types.DecimalType = DecimalType

        functions = ModuleType("pyspark.sql.functions")
        functions.col = F.col
        functions.date_format = F.date_format
        functions.to_json = F.to_json

        pyspark = ModuleType("pyspark")
        pyspark_sql = ModuleType("pyspark.sql")
        pyspark_sql.functions = functions
        pyspark_sql.types = types
        pyspark.sql = pyspark_sql

        monkeypatch.setitem(sys.modules, "pyspark", pyspark)
        monkeypatch.setitem(sys.modules, "pyspark.sql", pyspark_sql)
        monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions)
        monkeypatch.setitem(sys.modules, "pyspark.sql.types", types)

        rows = spark_df_to_rows(sdf)

        assert rows[0] == ["d", "t", "arr", "m", "s", "dec", "n"]
        assert rows[1] == [
            "2024-01-02",
            "2024-01-02 03:04:05",
            "[1]",
            '{"a":1}',
            '{"x":1}',
            "1.5",
            "null",
        ]
        sdf.select.assert_called_once()
        assert len(sdf.select.call_args.args) == 7
        assert F.date_format.called
        assert F.to_json.call_count == 3

    def test_overwrite_from_spark_wires_through(self, monkeypatch: pytest.MonkeyPatch):
        ws = MagicMock(row_count=100, col_count=26)
        sdf = MagicMock()
        monkeypatch.setattr(
            gsheets_module,
            "spark_df_to_rows",
            lambda sdf, fillna="null": [["h"], [1]],
        )

        overwrite_worksheet_from_spark(ws, sdf, chunk_rows=10)

        ws.clear.assert_called_once_with()
        ws.update.assert_called_once()
        assert ws.update.call_args.args[0] == [["h"], [1]]
