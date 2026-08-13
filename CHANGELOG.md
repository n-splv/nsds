# Changelog

## 0.4.1

- Cast Spark `DECIMAL` to `double` in cluster `read_sql()` so spend columns stay numeric in pandas

## 0.4.0

- Add `read_sql(..., as_spark=True)` to return a Spark DataFrame on Databricks
- Ignore `spark.sql.execution.arrow.pyspark.enabled` when the runtime rejects it

## 0.3.0

- Add `overwrite_worksheet`, `spark_df_to_rows`, and `overwrite_worksheet_from_spark`
  for writing Spark DataFrames to Google Sheets
- Expand worksheet grids on write without shrinking existing dimensions (preserves charts)

## 0.2.0

First public release on PyPI.
