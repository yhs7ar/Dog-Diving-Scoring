"""M3 — analyze wiring (torch-free core): cached detections -> trajectory -> measurement.

Reproduces the parent's scoring path so the copied ``trajectory`` / ``measure`` /
``pool`` modules are proven identical on this machine.  Detection (YOLO re-run) is
a separate torch-dependent step (M3b) and is deliberately absent here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from dog_diving import measure as _measure
from dog_diving import trajectory as _trajectory


def load_cached_records(tracks_path: str | Path, *, model_sha: str | None = None
                        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(records, meta)`` from a parent ``tracks.json`` cache.

    The parent caches a *map* of detector keys -> entry (each entry is one
    detection run: model × imgsz × conf × clahe).  ``model_sha`` selects the entry
    whose checkpoint produced the run; with no selector a single-entry file is
    returned as-is, and a multi-entry file raises rather than silently picking one.
    """
    data = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    entries = data["entries"]
    if model_sha is not None:
        for key, entry in entries.items():
            if entry.get("model_sha") == model_sha:
                return list(entry["records"]), {"key": key, "entry": entry}
        raise ValueError(f"no tracks.json entry has model_sha {model_sha!r}")
    if len(entries) != 1:
        raise ValueError(f"tracks.json holds {len(entries)} detector entries; select one explicitly")
    key, entry = next(iter(entries.items()))
    return list(entry["records"]), {"key": key, "entry": entry}


def build_trajectory(records: Sequence[dict[str, Any]], *,
                     clip_id: str, frame_size: tuple[int, int],
                     kp_conf: float | None = None):
    """Mirror ``cli_run.py`` step 2: assemble accepted points -> flight arc."""
    return _trajectory.build_trajectory(
        list(records), clip_id=clip_id, frame_size=frame_size, kp_conf=kp_conf)


def measure(records: Sequence[dict[str, Any]], plane_doc: dict[str, Any], *,
            clip_id: str, frame_size: tuple[int, int],
            kp_conf: float | None = None) -> dict[str, Any]:
    """Mirror ``cli_run.py`` steps 2+4: build the trajectory, then ``plane_distance``.

    Returns the parent's measurement dict with ``trajectory`` (``Trajectory.as_dict()``)
    attached, which the bench writes straight to ``trajectory.json``.
    """
    traj = build_trajectory(records, clip_id=clip_id, frame_size=frame_size, kp_conf=kp_conf)
    meas = _measure.plane_distance(traj, plane_doc)
    meas["trajectory"] = traj.as_dict()
    return meas
