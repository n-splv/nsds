from functools import partial

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score

from nsds.metrics import r2_adjusted


dual_y_figure = partial(make_subplots, specs=[[{"secondary_y": True}]])


def prediction_scatter_plot(df: pd.DataFrame = None,
                            y_true: str | pd.Series = None,
                            y_pred: str | pd.Series = None,
                            n_features: int = None,
                            wh: int = 600,
                            marker_size: int = None,
                            **kwargs) -> go.Figure:

    AXIS_SCALE_PADDING = 0.05

    if isinstance(y_true, str):
        y_true = df[y_true]
    if isinstance(y_pred, str):
        y_pred = df[y_pred]
    r2 = r2_score(y_true, y_pred)
    title = f'{r2=:.2f}'

    if n_features:
        r2_adj = r2_adjusted(y_true, y_pred, n_features)
        title += f'\t{r2_adj=:.2f}'

    max_value = np.max(np.maximum(y_true, y_pred))
    min_value = np.min(np.minimum(y_true, y_pred))
    padding = (max_value - min_value) * AXIS_SCALE_PADDING
    range_start = min_value - padding
    range_end = max_value + padding
    range_ = [range_start, range_end]

    for (key, value) in (
            ('range_x', range_),
            ('range_y', range_),
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
            x0=range_start, y0=range_start,
            x1=range_end, y1=range_end,
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
