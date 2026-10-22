"""Report — ``result.json`` + the human summary (M7 deliverable, PLAN.md §5).

PLAN §5 ``report.py``, verbatim:

    - ``result.json``: clip id + hash, model hash, calibration id + residuals, params
      (imgsz/conf/kp_conf/CLAHE), trajectory stats, chosen contact frame + candidate
      source, distance, uncertainty, override flags, timings, per-stage versions.
    - ``report.txt``: one-screen human summary for the judge/reviewer.
    - Optional CSV roll-up across clips for the validation batch (§8).

PLAN §11 M7 then requires the batch artifacts to be **"unchanged by re-runs"** — and
those two clauses cannot both hold literally
------------------------------------------------
A wall-clock ``timings`` block (and a ``created`` stamp) is *by construction* different
on every run, so a ``result.json`` carrying them can never be byte-unchanged by a
re-run.  This module resolves the conflict in the only way that keeps §11 checkable:
``result.json`` is a **pure function of the pipeline's inputs** — no clock, no cache
state, no host, no duration — and every volatile field PLAN §5 lists under
``result.json`` moves to a sibling ``run_meta.json`` (:func:`build_run_meta`), which is
explicitly *not* part of the determinism guarantee.  A test pins the boundary: the same
inputs give identical ``result.json`` bytes, and ``run_meta.json`` is the only artifact
that may differ between two runs.

The same reasoning removes ``from_cache`` from ``result.json``: a cold first run and a
warm second run are the *same measurement*, and putting a cache flag in the result
would make §11's own gate fail for a reason that has nothing to do with the pipeline.

JSON safety
-----------
``json.dumps`` happily emits ``NaN``/``Infinity``, which is **not JSON** and which
breaks any strict reader.  The pipeline produces genuine ``nan`` (an arc RMS over zero
inliers, an unwindowed sigma term), so :func:`jsonable` maps every non-finite float to
``null`` and records the substitution, rather than shipping an unparseable result file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

RESULT_SCHEMA = "dog_diving.result/1"
RESULT_NAME = "result.json"
TEXT_NAME = "report.txt"
META_NAME = "run_meta.json"
ROLLUP_NAME = "batch_rollup.csv"

# PLAN §5's pipeline-stage versions, so a result is reproducible from its own file
STAGE_VERSIONS = {
    "video": "m1",
    "calibration": "m2",
    "detect": "m3",
    "trajectory": "m4",
    "contact": "m5",
    "distance": "m6",
    "annotate": "m7",
    "report": "m7",
}


# --- JSON safety ------------------------------------------------------------
def jsonable(obj: Any) -> Any:
    """Recursively coerce to strict-JSON values (non-finite floats -> ``None``)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def dumps_strict(obj: Any) -> str:
    """Deterministic, strict JSON text (sorted keys, ASCII, trailing newline)."""
    return json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=True,
                      allow_nan=False) + "\n"


def _as_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    fn = getattr(obj, "as_dict", None)
    return fn() if callable(fn) else str(obj)


# --- the result document ----------------------------------------------------
def _ladder_warnings(ladder: dict | None) -> list[str]:
    """The adaptive-threshold ladder's own disclosures (operator decision, 2026-10-12).

    Kept in one place so ``result.json`` and ``report.txt`` cannot disagree about what the ladder
    did.  Absent ladder -> no lines, so a fixed-threshold run's warnings list is unchanged.
    """
    if not ladder:
        return []
    out: list[str] = []
    chosen = ladder.get("chosen") or {}
    n_points = ladder.get("n_points")
    n_low = int(ladder.get("n_low_conf_points") or 0)
    if ladder.get("below_min_points"):
        out.append(
            f"ADAPTIVE THRESHOLD: fewer than {ladder.get('min_points')} accepted points even at the "
            f"floor rung (imgsz={chosen.get('imgsz')} conf={chosen.get('conf')} "
            f"kp_conf={chosen.get('kp_conf')}) — {n_points} point(s) reported and flagged "
            f"below_min_points; the trajectory is not backed by the requested point count"
        )
    elif n_low:
        out.append(
            f"ADAPTIVE THRESHOLD: {n_low} of {n_points} accepted point(s) were admitted only by "
            f"lowering the threshold to conf={chosen.get('conf')} / kp_conf={chosen.get('kp_conf')} "
            f"(base conf/kp_conf {chosen.get('base_conf')}/{chosen.get('base_kp_conf')}); this "
            f"checkpoint's confidence is miscalibrated (measured: lowered rungs admit empty sky and "
            f"people at kp conf >0.95), so those points are marked low_conf in the trajectory and "
            f"the distance may rest on them"
        )
    return out


def build_result(
    *,
    clip_id: str,
    clip_sha: str | None = None,
    clip_name: str | None = None,
    video: Any | None = None,
    detect: Any | None = None,
    traj: Any | None = None,
    contact: Any | None = None,
    measurement: dict | None = None,
    calib: dict | None = None,
    calib_path: str | None = None,
    detect_config: dict | None = None,
    artifacts: dict | None = None,
    still_frame: int | None = None,
    ladder: dict | None = None,
    excluded_frames: Sequence[int] | None = None,
    arc_after_exclusion: dict | None = None,
    entry_frame: int | None = None,
) -> dict[str, Any]:
    """Assemble ``result.json``.  Deterministic: every field is a function of the inputs.

    Deliberately *not* included: timings, ``created``, ``from_cache``, the host name —
    see the module docstring.  ``detect`` is read for its counts and its provenance
    only, so a cache hit and a cache miss yield the same result document.

    ``excluded_frames`` is Review's deselection set (the other-dog case).  It adds a ``review``
    block, and only when it is non-empty — an untouched run keeps exactly its previous key set
    (PLAN §11's byte-identical-across-reruns contract).
    """
    det = _as_dict(detect) or {}
    vid = _as_dict(video) or {}
    trj = _as_dict(traj) or {}
    con = _as_dict(contact) or {}
    m = measurement or {}

    measurement_block = {
        k: m.get(k) for k in (
            "x_px", "y_px", "point_source", "point_origin", "model_point",
            "frame", "contact_source", "arc_is_freefall", "arc_reason",
            "total_inches", "rounded_inches", "feet", "inches", "display",
            "sigma_in", "sigma_complete", "sigma_terms", "sigma_kp_literal_in",
            "sigma_kp_detrended_in", "kp_k", "kp_frames", "in_per_px",
            "waterline_y", "splash_gap_px", "splash_gap_in", "splash_gap_tol_in",
            "calib_setup_id", "calib_fit_type", "calib_residual_rms_in",
            "gates", "flags", "cfg",
        )
    }

    calib_block: dict[str, Any]
    if calib:
        try:
            from . import calibration as _calibration
            gates = {k: {"pass": bool(v.get("pass")), "warn": bool(v.get("warn")),
                         "detail": v.get("detail")}
                     for k, v in _calibration.validate_gates(calib).items()}
        except Exception:  # noqa: BLE001 - a broken calibration must not kill the report
            gates = {}
        calib_block = {
            "present": True,
            "setup_id": calib.get("setup_id"),
            "path": calib_path,
            "fit_type": calib.get("fit", {}).get("type"),
            "residual_rms_in": calib.get("fit", {}).get("residual_rms_in"),
            "residual_max_in": calib.get("fit", {}).get("residual_max_in"),
            "image_size": calib.get("image_size"),
            "n_marks": len(calib.get("marks", []) or []),
            "gates": gates,
            "created": calib.get("created"),
            "notes": calib.get("notes"),
        }
    else:
        calib_block = {
            "present": False, "setup_id": None, "path": calib_path,
            "reason": ("no calibration was supplied: px->in needs PLAN §4.2's marks and "
                       "PLAN §5 #2 needs the waterline row, so the inches fields are null "
                       "and the pixel-domain fields are the real output (data/calib/ is "
                       "empty as of 2026-10-08 — HANDOFF open item 10)"),
        }

    warnings = list(m.get("warnings", []) or [])
    warnings.extend(_ladder_warnings(ladder))

    review_block: dict[str, Any] | None = None
    if excluded_frames or entry_frame is not None:
        exc = sorted({int(f) for f in (excluded_frames or ())})
        kept = max(0, int(trj.get("n_points") or 0) - len(exc))
        arc = dict(arc_after_exclusion or {})
        review_block = {
            "excluded_frames": exc,
            "n_excluded": len(exc),
            "points_kept": kept,
            "entry_frame": None if entry_frame is None else int(entry_frame),
            "entry_is_operator_choice": entry_frame is not None,
            "arc_n": arc.get("n"),
            "arc_rms_px": arc.get("rms_px"),
            "arc_frame_range": arc.get("frame_range"),
            "note": ("operator adjusted the trajectory in Review: deselected detections are out of "
                     "the fitted arc and cannot supply the measured point, and the entry frame (when "
                     "set) is the point measured instead of the final detection"),
        }

    return {
        "schema": RESULT_SCHEMA,
        "stage_versions": dict(STAGE_VERSIONS),
        "clip": {
            "id": clip_id,
            "name": clip_name,
            "sha256": clip_sha,
            "width": vid.get("width"),
            "height": vid.get("height"),
            "fps": vid.get("fps"),
            "codec": vid.get("codec"),
            "n_frames": vid.get("n_frames"),
            "vfr": vid.get("vfr"),
            "declared_vs_decoded": [vid.get("n_frames_declared"), vid.get("n_frames_decoded")],
        },
        "model": {"path": det.get("config", {}).get("model_path") if isinstance(det.get("config"), dict) else None,
                  "sha256": det.get("model_sha"),
                  # which keypoint mapping produced the tailset, and whether it was a labelled
                  # keypoint or derived from another joint (a COCO person-pose run derives it from
                  # the hips — see dog_diving/pose_layout.py)
                  "layout": det.get("model_layout"),
                  "tailset_derived": det.get("tailset_derived")},
        "config": detect_config,
        "calibration": calib_block,
        "detection": {
            "n_frames_clip": det.get("n_frames_clip"),
            "n_frames_processed": det.get("n_frames_processed"),
            "n_detections": det.get("n_detections"),
            "n_accepted": det.get("n_accepted"),
            "dominant_track": det.get("dominant_track"),
            "tracking_available": det.get("tracking_available"),
            "track_coverage": det.get("track_coverage"),
            "cache_key_hash": det.get("cache_key_hash"),
        },
        "trajectory": {
            "n_points": trj.get("n_points"),
            "runs": trj.get("runs"),
            "flight_run": trj.get("flight_run"),
            "airborne_frame_range": trj.get("airborne_frame_range"),
            "n_before_leave": trj.get("n_before_leave"),
            "n_before_entry": trj.get("n_before_entry"),
            "n_inliers": len(trj.get("inliers", []) or []),
            "n_outliers": trj.get("n_outliers"),
            "arc_rms_px": trj.get("arc_rms_px"),
            "arc_rms_all_px": trj.get("arc_rms_all_px"),
            "max_inlier_res_px": trj.get("max_inlier_res_px"),
            "last_arc_frame": trj.get("last_arc_frame"),
            "coeffs": trj.get("coeffs"),
            "inliers": trj.get("inliers"),
            "flight_points": trj.get("flight_points"),
            "last_contact_candidate": trj.get("last_contact_candidate"),
        },
        "contact": {
            "chosen_frame": con.get("chosen_frame"),
            "chosen_source": con.get("chosen_source"),
            "window": con.get("window"),
            "n_frames": con.get("n_frames"),
            "waterline_y": con.get("waterline_y"),
            "arc_is_freefall": con.get("arc_is_freefall"),
            "arc_reason": con.get("arc_reason"),
            "candidates": con.get("candidates"),
        },
        # Present only for the plane route (PLAN §16): a 1-D run's result.json keeps exactly its
        # previous key set (PLAN §11's byte-identical-across-reruns contract).
        **({"measurement_route": m["route"]} if m.get("route") else {}),
        "measurement": measurement_block,
        "still_frame": still_frame,
        "overrides": {
            "point_source": m.get("point_source"),
            "contact_source": m.get("contact_source") or con.get("chosen_source"),
            "manual_point": bool(m.get("point_source") == "manual"),
        },
        "warnings": warnings,
        "artifacts": artifacts or {},
        # Present only when the adaptive ladder ran, so a fixed-threshold run's result.json keeps
        # exactly its previous key set (PLAN §11's byte-identical-across-reruns contract).
        **({"detection_ladder": ladder} if ladder is not None else {}),
        # Present only when the operator deselected detections in Review (the other-dog case), for
        # the same key-set reason as the ladder block above.
        **({"review": review_block} if review_block is not None else {}),
    }


def write_result_json(path: str | Path, result: dict) -> Path:
    """Write ``result.json`` deterministically (atomic replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps_strict(result), encoding="utf-8")
    tmp.replace(path)
    return path


# --- the human summary ------------------------------------------------------
def _gates_line(calib_block: dict | None) -> str:
    cb = calib_block or {}
    if not cb.get("present"):
        return "calibration  NONE — data/calib/ is empty (PLAN §12 item 4): the pixel-domain\n" \
               "             fields below are real, the inches are not computable"
    gates = cb.get("gates", {}) or {}
    if not gates:
        return f"calibration  setup_id={cb.get('setup_id')} (gates not evaluated)"
    txt = " ".join(f"{k}={'ok' if v.get('pass') else 'FAIL'}"
                   f"{'*' if v.get('warn') else ''}" for k, v in sorted(gates.items()))
    return (f"calibration  setup_id={cb.get('setup_id')} "
            f"fit={cb.get('fit_type')} "
            f"RMS={cb.get('residual_rms_in')} in  [{txt}]")


def build_report_text(result: dict) -> str:
    """PLAN §5's ``report.txt``: one screen, for the judge/reviewer.  Deterministic."""
    c = result.get("clip", {})
    cal = result.get("calibration", {}) or {}
    det = result.get("detection", {}) or {}
    trj = result.get("trajectory", {}) or {}
    con = result.get("contact", {}) or {}
    m = result.get("measurement", {}) or {}

    sha = (c.get("sha256") or "")[:16]
    L: list[str] = []
    L.append("Dog Diving jump-distance result")
    L.append("=" * 30)
    L.append(f"clip         {c.get('id')}  (sha256 {sha}...)  {c.get('name')}")
    L.append(f"video        {c.get('width')}x{c.get('height')} @ {c.get('fps')} fps, "
             f"{c.get('codec')}, {c.get('n_frames')} frames"
             + ("  [VFR]" if c.get("vfr") else ""))
    L.append(f"model        sha256 {str(result.get('model', {}).get('sha256') or '')[:16]}...")
    cfg = result.get("config") or {}
    L.append(f"config       imgsz={cfg.get('imgsz')} conf={cfg.get('conf')} "
             f"kp_conf={cfg.get('kp_conf')} clahe={cfg.get('clahe')}")
    L.append("")
    L.append(_gates_line(result.get("calibration")))
    L.append("")
    L.append(f"detection    {det.get('n_frames_processed')} frames processed, "
             f"{det.get('n_detections')} detections, {det.get('n_accepted')} accepted")
    lad = result.get("detection_ladder") or None
    if lad:
        if lad.get("disabled"):
            L.append("ladder       off: fixed thresholds (min_points=0)")
        else:
            ch = lad.get("chosen") or {}
            L.append(
                f"ladder       min {lad.get('min_points')} pt -> imgsz={ch.get('imgsz')} "
                f"conf={ch.get('conf')} kp_conf={ch.get('kp_conf')}: "
                f"{lad.get('n_points')} pt, {lad.get('n_low_conf_points')} low-conf, "
                f"{len(lad.get('rungs') or [])} rung(s) tried"
                + (f", imgsz escalated {lad.get('imgsz_tried')}" if lad.get("imgsz_escalated") else "")
                + ("  [BELOW MINIMUM]" if lad.get("below_min_points") else ""))
    L.append(f"trajectory   {trj.get('n_points')} accepted points, "
             f"{len(trj.get('runs') or [])} run(s), flight run {trj.get('flight_run')} "
             f"{trj.get('airborne_frame_range')}, "
             f"{trj.get('n_inliers')} inliers, arc RMS {trj.get('arc_rms_px')} px")
    L.append(f"contact      chosen frame {con.get('chosen_frame')} "
             f"from '{con.get('chosen_source')}'; arc_is_freefall={con.get('arc_is_freefall')}")
    for cand in (con.get("candidates") or []):
        frame = cand.get("frame")
        L.append(f"             #{cand.get('rank')} {cand.get('source'):17s} "
                 f"{('f' + str(frame)) if frame is not None else 'n/a':>7s}  "
                 f"avail={cand.get('available')} conf={cand.get('confident')}")
    L.append("")
    L.append(f"measurement  x={m.get('x_px')} y={m.get('y_px')} "
             f"point_source={m.get('point_source')} ({m.get('point_origin')})")
    if m.get("display"):
        bound = "" if m.get("sigma_complete") else "  [LOWER BOUND: a budget term is missing]"
        L.append(f"distance     {m.get('display')}  +/- {m.get('sigma_in')} in{bound}")
        L.append(f"             total_inches={m.get('total_inches')} "
                 f"sigma_complete={m.get('sigma_complete')} terms={m.get('sigma_terms')}")
    else:
        L.append(f"distance     n/a — {cal.get('reason') or 'no calibration'}")
    rev = result.get("review") or {}
    if rev:
        exc = rev.get("excluded_frames") or []
        shown = ", ".join(str(int(f)) for f in exc[:24]) + ("…" if len(exc) > 24 else "")
        L.append(f"review       {rev.get('n_excluded')} detection(s) deselected by the operator "
                 f"[{shown}]; {rev.get('points_kept')} kept, "
                 f"arc over {rev.get('arc_n')} point(s), RMS "
                 f"{'n/a' if rev.get('arc_rms_px') is None else round(float(rev['arc_rms_px']), 2)} px")
        if rev.get("entry_frame") is not None:
            L.append(f"             entry frame {rev['entry_frame']} chosen by the operator — the "
                     f"measured point is that frame's detection, not the final one")
        else:
            L.append("             the measured point is the highest-frame KEPT detection")
    warns = result.get("warnings") or []
    L.append("")
    L.append(f"warnings     {len(warns)}")
    for w in warns:
        L.append(f"  - {w}")
    arts = result.get("artifacts") or {}
    if arts:
        L.append("")
        L.append("artifacts")
        for k in sorted(arts):
            L.append(f"  {k:12s} {arts[k]}")
    return "\n".join(L) + "\n"


def write_report_txt(path: str | Path, result: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(build_report_text(result), encoding="utf-8")
    tmp.replace(path)
    return path


# --- volatile run metadata (explicitly NOT part of the §11 guarantee) -------
def build_run_meta(
    *,
    clip_id: str,
    timings: dict | None = None,
    from_cache: bool | None = None,
    created: str | None = None,
    host: str | None = None,
    argv: Sequence[str] | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    """``run_meta.json``: wall-clock, cache state, host.  May differ between runs."""
    import datetime as _dt
    import platform
    import socket

    return jsonable({
        "clip_id": clip_id,
        "created": created or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": host if host is not None else socket.gethostname(),
        "platform": platform.platform(),
        "from_cache": from_cache,
        "timings": timings or {},
        "argv": list(argv) if argv is not None else None,
        "extra": extra or {},
    })


def write_run_meta(path: str | Path, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps_strict(meta), encoding="utf-8")
    tmp.replace(path)
    return path


# --- CSV roll-up (PLAN §5's optional batch artefact, PLAN §8 V4) ------------
ROLLUP_COLUMNS = (
    "clip_id", "n_frames", "n_detections", "n_accepted", "flight_run",
    "airborne_frame_range", "arc_rms_px", "arc_is_freefall", "chosen_frame",
    "chosen_source", "x_px", "y_px", "point_source", "total_inches", "display",
    "sigma_in", "sigma_complete", "calib_setup_id", "n_warnings", "annotated",
)


def rollup_row(result: dict) -> dict[str, Any]:
    """One clip's deterministic summary row (the V4 batch table's columns)."""
    c = result.get("clip", {}) or {}
    det = result.get("detection", {}) or {}
    trj = result.get("trajectory", {}) or {}
    con = result.get("contact", {}) or {}
    m = result.get("measurement", {}) or {}
    arts = result.get("artifacts", {}) or {}
    return {
        "clip_id": c.get("id"),
        "n_frames": c.get("n_frames"),
        "n_detections": det.get("n_detections"),
        "n_accepted": det.get("n_accepted"),
        "flight_run": trj.get("flight_run"),
        "airborne_frame_range": trj.get("airborne_frame_range"),
        "arc_rms_px": trj.get("arc_rms_px"),
        "arc_is_freefall": con.get("arc_is_freefall"),
        "chosen_frame": con.get("chosen_frame"),
        "chosen_source": con.get("chosen_source"),
        "x_px": m.get("x_px"),
        "y_px": m.get("y_px"),
        "point_source": m.get("point_source"),
        "total_inches": m.get("total_inches"),
        "display": m.get("display"),
        "sigma_in": m.get("sigma_in"),
        "sigma_complete": m.get("sigma_complete"),
        "calib_setup_id": m.get("calib_setup_id"),
        "n_warnings": len(result.get("warnings") or []),
        "annotated": bool(arts.get("annotated")),
    }


def _csv_quote(text: str) -> str:
    """RFC 4180 quoting.  Without it a list cell like ``[146,159]`` shifts every later column.

    Measured on the real batch roll-up (2026-10-08): ``airborne_frame_range`` is written as
    ``[146,159]``, whose comma was read as a field separator, so ``chosen_frame`` onward were
    off by one in any spreadsheet or ``csv`` reader.  The fix is quoting, not a different
    list separator: a quoted field round-trips through :mod:`csv` unchanged.
    """
    if any(ch in text for ch in ',"\n\r'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return _csv_quote(f"{v:.6g}")
    if isinstance(v, (list, tuple)):
        return _csv_quote("[" + ",".join(_csv_cell(x) for x in v) + "]")
    if isinstance(v, dict):
        return _csv_quote("{" + ",".join(f"{k}:{_csv_cell(x)}" for k, x in sorted(v.items())) + "}")
    return _csv_quote(str(v))


def write_rollup_csv(path: str | Path, results: Iterable[dict]) -> Path:
    """PLAN §8 V4's batch roll-up.  Row order is the caller's (clip order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(ROLLUP_COLUMNS)]
    for r in results:
        row = rollup_row(r)
        lines.append(",".join(_csv_cell(row.get(c)) for c in ROLLUP_COLUMNS))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
