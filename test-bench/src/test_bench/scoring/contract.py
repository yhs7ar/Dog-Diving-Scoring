"""The standardized scoring contract — one schema every method emits.

Modeled on the parent's ``measure.py`` rule (one output shape whichever route
answered), so the suite runner, the metrics table, and the comparison pane read
exactly one schema and never branch on method internals.

A method is a single function::

    func(points, plane_doc, config) -> ScoreResult (dict)

``points`` are already-assembled accepted detections (``{frame, x, y, kp_conf}``,
see ``dog_diving.trajectory.assemble``).  ``plane_doc`` is the
``dog_diving_pool_calibration/v1`` document.  ``config`` carries the method's
axis knobs (``n_points``, ``selection``, ``frame_size``, …).

Registering a new method (3-D arc, point-cloud) is one ``registry.register``
call; nothing downstream changes.
"""
from __future__ import annotations

from typing import Any, Sequence

SCHEMA = "test_bench.scoring/1"

#: Reserved method keys.  ``arc_2d`` is the v1 implementation; the other two are
#: slots the registry reserves so a later method cannot silently collide.
METHODS = ("arc_2d", "arc_2d_centerline", "arc_3d", "pc_route")

#: Status vocabulary.  Every method returns one of these; the table maps them to
#: a display string and the metrics skip any non-``ok`` row.
STATUS_OK = "ok"
STATUSES = ("ok", "no_calibration", "few_points", "no_fit", "no_crossing", "error")

#: Keys every ScoreResult must carry (measure.py parity aliases included).
REQUIRED_KEYS = (
    "method", "route", "config", "status",
    "entry_px", "entry_frame", "distance_in", "distance_ft",
    "x_px", "y_px", "warnings", "fit",
)


def new_result(method: str, config: dict[str, Any]) -> dict[str, Any]:
    """A blank, schema-valid result for ``method``, ready for a method to fill."""
    return {
        "schema": SCHEMA,
        "method": method,
        "route": method,
        "config": dict(config),
        "status": STATUS_OK,
        "entry_px": None,
        "entry_frame": None,
        "distance_in": None,
        "distance_ft": None,
        "x_px": None,
        "y_px": None,
        "warnings": [],
        "fit": {},
    }


def validate(result: dict[str, Any]) -> list[str]:
    """Schema problems (empty list == valid).  Pure check, no side effects."""
    problems: list[str] = []
    if not isinstance(result, dict):
        return ["result is not a dict"]
    for key in REQUIRED_KEYS:
        if key not in result:
            problems.append(f"missing key {key!r}")
    if result.get("method") not in METHODS:
        problems.append(f"method {result.get('method')!r} not in {METHODS}")
    if result.get("status") not in STATUSES:
        problems.append(f"status {result.get('status')!r} not in {STATUSES}")
    ep = result.get("entry_px")
    if ep is not None:
        ok = (isinstance(ep, (list, tuple)) and len(ep) == 2
              and all(isinstance(v, (int, float)) for v in ep))
        if not ok:
            problems.append(f"entry_px {ep!r} is not a [x, y] pair")
    if not isinstance(result.get("warnings"), Sequence) or isinstance(result.get("warnings"), (str, bytes)):
        problems.append("warnings must be a list")
    if not isinstance(result.get("config"), dict):
        problems.append("config must be a dict")
    return problems
