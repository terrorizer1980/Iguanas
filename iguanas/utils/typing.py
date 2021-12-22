"""Custom typing objects. Used so packages don't need to be imported."""
from typing import TypeVar

NumpyArrayType = TypeVar('numpy.ndarray')
PandasDataFrameType = TypeVar('pandas.core.frame.DataFrame')
PandasSeriesType = TypeVar('pandas.core.series.Series')
KoalasDataFrameType = TypeVar('databricks.koalas.frame.DataFrame')
KoalasSeriesType = TypeVar('databricks.koalas.series.Series')
PySparkDataFrameType = TypeVar('pyspark.sql.dataframe.DataFrame')
