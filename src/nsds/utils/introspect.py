import inspect
from collections.abc import Callable


def parameter_names(
    f: Callable,
    exclude: tuple[inspect._ParameterKind, ...] = (inspect.Parameter.KEYWORD_ONLY,)
) -> list[str]:
    return [
        name
        for name, param in inspect.signature(f).parameters.items()
        if param.kind not in exclude
    ]
