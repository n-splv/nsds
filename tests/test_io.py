import sys
from collections.abc import Callable
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nsds.io import gsheets as gsheets_module
from nsds.io import sql as sql_module
from nsds.io.gsheets import ENV_SERVICE_ACCOUNT_KEY, get_gspread_client
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
    pyspark_sql.SparkSession = session_class
    pyspark.sql = pyspark_sql

    monkeypatch.setitem(sys.modules, "pyspark", pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", pyspark_sql)
    monkeypatch.setattr(sql_module, "IS_DATABRICKS", True)

    session.session_class = session_class
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
