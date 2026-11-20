"""M3 — analyze wiring (torch-free core): cached detections -> trajectory -> measurement.

Reproduces the parent's scoring path so the copied ``trajectory`` / ``measure`` /
``pool`` modules are proven identical on this machine.  Detection (YOLO re-run) is
a separate torch-dependent step (M3b) and is deliberately absent here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

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


# --- Review deselection (baked into the saved detections) -------------------
# Deselecting a detection means *"this tailset is not the dog"*.  The correction
# is applied to the saved ``detections.json`` itself — the record's ``accepted``
# flag is flipped to False — so that every downstream rebuild of the trajectory
# (``assemble`` -> ``build_trajectory``) drops it with no separate filter step.
# The saved detections are then the only detections scoring ever reads.

def apply_deselection(records: Sequence[dict[str, Any]],
                      excluded_frames: Iterable[int]) -> list[dict[str, Any]]:
    """Mark ``excluded_frames`` as ``accepted=False`` in the record list.

    Re-included frames (previously deselected, now absent from ``excluded_frames``)
    are restored to ``accepted=True``.  A ``deselected`` marker is kept so a
    technician-made exclusion is distinguishable from a detector-rejected one.
    """
    drop = {int(f) for f in (excluded_frames or ())}
    out: list[dict[str, Any]] = []
    for r in records:
        r = dict(r)
        f = int(r.get("frame", -1))
        if f in drop:
            r["accepted"] = False
            r["deselected"] = True
        elif r.pop("deselected", None) is not None:
            r["accepted"] = True
        out.append(r)
    return out


def deselect_detections(clip_dir: str | Path, clip_id: str,
                        excluded_frames: Iterable[int]):
    """Bake deselection into a clip's saved detections and rebuild its trajectory.

    Writes the corrected ``detections.json`` (deselected frames ``accepted=False``),
    rebuilds ``trajectory.json`` from the cleaned records, and updates the
    manifest's ``excluded_frames`` / ``trajectory.n_points``.  Returns the rebuilt
    :class:`dog_diving.trajectory.Trajectory`.
    """
    from . import manifest as _m

    cdir = Path(clip_dir)
    records = _m.read_json(cdir / "detections.json")
    man = _m.read_manifest(cdir)
    w, h = int(man["meta"]["width"]), int(man["meta"]["height"])
    recs = apply_deselection(records, excluded_frames)
    traj = build_trajectory(recs, clip_id=clip_id, frame_size=(w, h))
    _m.write_json(cdir / "detections.json", recs)
    _m.write_json(cdir / "trajectory.json", traj.as_dict())
    man["excluded_frames"] = sorted({int(f) for f in (excluded_frames or ())})
    man["trajectory"]["n_points"] = traj.n_points
    _m.write_manifest(man, cdir)
    return traj
