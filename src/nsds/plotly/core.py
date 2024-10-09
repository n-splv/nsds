from functools import partial
from plotly.subplots import make_subplots

dual_y_figure = partial(make_subplots, specs=[[{"secondary_y": True}]])
