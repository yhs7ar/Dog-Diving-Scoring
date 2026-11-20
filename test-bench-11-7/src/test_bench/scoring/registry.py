"""The scoring-method registry.

One pluggable slot per method key.  ``arc_2d`` registers itself at import time;
``arc_3d`` and ``pc_route`` are reserved (``contract.METHODS``) but intentionally
not implemented yet — adding them is a single ``register`` call with a function
that conforms to :func:`contract.validate`.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from . import contract

#: func(points, plane_doc, config) -> ScoreResult
MethodFunc = Callable[[Sequence[dict], dict | None, dict[str, Any]], dict[str, Any]]

_registry: dict[str, MethodFunc] = {}


def register(method: str, func: MethodFunc) -> None:
    """Register ``func`` under ``method``.  Raises on a bad key or a collision."""
    if method not in contract.METHODS:
        raise ValueError(
            f"method {method!r} is not a reserved slot {contract.METHODS}; "
            "add it to contract.METHODS first")
    if method in _registry:
        raise ValueError(f"method {method!r} is already registered")
    _registry[method] = func


def get(method: str) -> MethodFunc:
    try:
        return _registry[method]
    except KeyError:
        raise KeyError(
            f"method {method!r} is not registered; available: {list(_registry)}") from None


def list_methods() -> list[str]:
    return sorted(_registry)


def run(method: str, points: Sequence[dict], plane_doc: dict | None,
        config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to the registered method and validate its contract on the way out."""
    result = get(method)(list(points), plane_doc, dict(config))
    problems = contract.validate(result)
    if problems:
        raise RuntimeError(f"{method} returned an invalid ScoreResult: {problems}")
    return result
