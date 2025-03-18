from functools import partial
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score

from nsds.metrics import r2_adjusted


dual_y_figure = partial(make_subplots, specs=[[{"secondary_y": True}]])


class Range(NamedTuple):
    start: int | float
    end: int | float


def prediction_scatter_plot(df: pd.DataFrame = None,
                            y_true: str | ArrayLike = None,
                            y_pred: str | ArrayLike = None,
                            *,
                            equalize_axes: bool = False,
                            wh: int = 600,
                            marker_size: int = None,
                            r2_adj_n_features: int = None,
                            **kwargs) -> go.Figure:

    if isinstance(y_true, str):
        y_true = df[y_true]
    if isinstance(y_pred, str):
        y_pred = df[y_pred]
    r2 = r2_score(y_true, y_pred)
    title = f'{r2=:.2f}'

    if r2_adj_n_features:
        r2_adj = r2_adjusted(y_true, y_pred, r2_adj_n_features)
        title += f'\t{r2_adj=:.2f}'

    range_x = calclate_axis_range(y_true)
    range_y = calclate_axis_range(y_pred)
    if equalize_axes:
        range_x = range_y = [
            min(range_x.start, range_y.start),
            max(range_x.end, range_y.end),
        ]

    for (key, value) in (
        ('range_x', range_x),
        ('range_y', range_y),
        ('height', wh),
        ('width', wh),
    ):
        kwargs.setdefault(key, value)

    return (
        px.scatter(
            data_frame=df,
            x=y_true,
            y=y_pred,
            **kwargs
        )
        .add_shape(
            type="line",
            x0=range_x.start, y0=range_y.start,
            x1=range_x.end, y1=range_y.end,
            line=dict(color="red", dash="dash"),
            name="45-degree line"
        )
        .update_layout(
            coloraxis_colorbar_title=None,
            xaxis_title='y_true',
            yaxis_title='y_pred',
            title=title,
            legend_orientation='h',
            legend_yanchor="bottom",
            legend_y=1.02,
            legend_xanchor="right",
            legend_x=1,
        )
        .update_traces(
            marker_line_width=0.5,
            marker_size=marker_size,
        )
    )


def calclate_axis_range(values: ArrayLike) -> Range:
    AXIS_SCALE_PADDING = 0.05
    max_value = np.max(values)
    min_value = np.min(values)
    padding = (max_value - min_value) * AXIS_SCALE_PADDING
    range_start = min_value - padding
    range_end = max_value + padding
    return Range(range_start, range_end)
