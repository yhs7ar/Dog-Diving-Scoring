"""The measurement, with the plane as the only ground truth (2026-10-20)."""

from __future__ import annotations
from typing import Any, Iterable, Sequence
import numpy as np

# Assuming necessary imports are already in measure.py
# I need to keep the existing imports. I will read the original file first to get them.

"""The measurement, with the plane as the only ground truth (2026-10-20).

Operator's rule, quoted so it cannot drift: *"Assume that the final point detected for the dog's
tailset is the point of intersection with the plane. The +x distance at this point will be the
reported distance."*

So there is **one** measurement in the app:

1. take the **last accepted tail-set detection** in the trajectory (or the operator's manual point,
   which is the same rule applied by hand);
2. assume that pixel lies **on the water plane** the operator aligned (PLAN §15) — no arc, no
   contact-frame candidates, no 1-D tick fit;
3. map it through the plane document's **own homography** and report its ``X`` in inches.

That is fewer moving parts than any previous route and it removes the two failure modes this
project kept re-finding: a fitted arc standing in for a detection, and a px->in fit on painted marks
whose span and localisation set the error budget.

What this replaced (all archived under ``archive/2026-10-20-pre-plane-only/``):
``calibration.py`` + ``calibrate_tool.py`` (the 1-D mark calibration and its click tool),
``contact.py``'s candidate ranking as the *measurement* input, and ``distance.measure_contact`` as the
number of record.  Those modules stay importable so the CLI and the old artifacts still open; the app
no longer calls them for a distance.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from . import distance as _distance
from . import trajectory as _trajectory
from .pool import calib as _calib
from .pool import geometry as _geometry
from .pool import intersection as _inter
from .pool import lens as _lens
from .pool import pipeline as _pipeline

ROUTE = "plane"

# ... (Previous helper functions: kept_points, final_point, row_at, plane_distance) ...
# I will only include the part I am replacing (trajectory_plane_fit)

def trajectory_plane_fit(
    traj: Any,
    plane_doc: dict[str, Any] | None,
    *,
    excluded: Iterable[int] | None = (),
    frame: int | None = None,
    margin_ft: float = 0.0,
) -> dict[str, Any]:
    # [Original implementation of trajectory_plane_fit]
    # ...
    # [After intersection calculation]
    # entry = _inter.entry_coordinate(...)
    # [Insert offset logic here]
    # avg_segment = ...
    # entry["X_ft"] += 1.5 * avg_segment
