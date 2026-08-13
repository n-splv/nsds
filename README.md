# nsds

Personal toolkit for DataScience tasks, runs locally and on Databricks.

## Install

```bash
pip install nsds                # core: pandas helpers, metrics, utils
pip install 'nsds[notebook]'    # plotly, itables, tqdm, IPython, dotenv
pip install 'nsds[sql]'         # databricks-sql-connector
pip install 'nsds[gsheets]'     # gspread
pip install 'nsds[all]'
```

Everything outside the core is imported lazily, so a bare install stays small and
`nsds` never pulls in a `pyspark` that would shadow the one on a cluster.

## Quickstart

```python
import nsds

nsds.setup()
```

`setup()` is the only thing in the package with side effects: importing any module
does nothing on its own. It installs the pandas extensions, sets display options,
selects the compact plotly renderer and loads a `.env`. The last two only apply
locally and are skipped on a Databricks cluster.

```python
nsds.setup(itables=True, logging=True)   # opt in
nsds.setup(plotly=False, dotenv=False)   # opt out
```

## What is in it

| Module | Contents |
| --- | --- |
| `nsds.frame` | `install()`, `read_csvs`, `read_csv_pyarrow`, `merge_insert_at`, `dt_group`, `percentiles` |
| `nsds.charts` | `prediction_scatter_plot`, `dual_y_figure`, `calculate_axis_range`, `Colors` |
| `nsds.tables` | `show()` — itables with sensible defaults |
| `nsds.io.sql` | `read_sql()`, `as_spark=True` |
| `nsds.io.gsheets` | `get_gspread_client()`, `overwrite_worksheet`, `spark_df_to_rows` |
| `nsds.metrics` | `r2_score`, `r2_adjusted`, `smape` |
| `nsds.utils` | `datetime_utils`, `round_half_up`, `gini_inequality_coefficient`, `parameter_names`, `show_mac_notification` |
| `nsds.runtime` | `RUNTIME_ENV`, `IS_DATABRICKS` |

### DataFrame extensions

`nsds.setup()` attaches these to both `pd.DataFrame` and `pd.Series`, without ever
shadowing an existing pandas attribute:

```python
df.vc(show_cumulative=True)        # value_counts with percentages
df.missing()                       # NaN / zero / empty-string report
df.sortd("amount")                 # sort_values, descending
df.preview()                       # display a few rows
df.show(nrows=50)                  # display without truncating columns
df.explode_all()
df.memory_mb()
df.to_csv_("out.csv", add_date_to_filename="day")
df.apply_row_wise(func)            # columns inferred from the signature
```

Static analysers cannot see monkey-patched attributes, so an editor will not
autocomplete these on a DataFrame — Jupyter's runtime completion will. Everything
else in the package is normally typed and ships `py.typed`.

### Reading SQL

Same call in both environments. On a cluster it uses the active `SparkSession`; locally
it opens a `databricks-sql-connector` connection from `DATABRICKS_SERVER_HOSTNAME`,
`DATABRICKS_HTTP_PATH` and `DATABRICKS_TOKEN`, or from arguments you pass directly.

```python
from nsds.io.sql import read_sql

df = read_sql("SELECT * FROM t WHERE day = :day", {"day": "2026-01-01"})
sdf = read_sql("SELECT * FROM t", as_spark=True)   # Databricks only
```

### On Databricks

```python
%pip install 'nsds[gsheets]'
```

`get_gspread_client()` can take its service-account JSON from a Databricks secret:

```python
from nsds.io.gsheets import get_gspread_client

client = get_gspread_client(secret_scope="my-scope", secret_key="gcp-service-account")
```

or from the `GSPREAD_SECRET_SCOPE` and `GSPREAD_SECRET_KEY` environment variables.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests --fix
```

## Releasing

Versions live in `pyproject.toml` and are read back at runtime via package metadata.
Publish is local for now (`UV_PUBLISH_TOKEN` / `UV_PUBLISH_TOKEN_TEST` in a gitignored
`.envrc`). Later: restore a tag-triggered workflow and set `UV_PUBLISH_TOKEN` as a
repo Action secret once 2FA is available on the personal GitHub account.

```bash
uv version --bump patch
git commit -am "Release $(uv version --short)" && git tag "v$(uv version --short)"
git push --follow-tags

rm -rf dist && uv build
uv publish --publish-url https://test.pypi.org/legacy/ --token "$UV_PUBLISH_TOKEN_TEST"  # optional smoke
uv publish --token "$UV_PUBLISH_TOKEN"
```
