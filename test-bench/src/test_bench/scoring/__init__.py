"""The scoring package — the standardized, pluggable scoring-method registry.

The bench's one extension point: a scoring method is a single function
conforming to :func:`contract.validate`, registered here.  ``arc_2d`` is the v1
method; ``arc_3d`` and ``pc_route`` are reserved slots for later.
"""
from . import contract, registry
from .arc_2d import arc_2d
from .arc_2d_centerline import arc_2d_centerline

registry.register("arc_2d", arc_2d)
registry.register("arc_2d_centerline", arc_2d_centerline)

__all__ = ["contract", "registry", "arc_2d", "arc_2d_centerline"]
