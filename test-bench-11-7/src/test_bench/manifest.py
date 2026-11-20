"""The analyzed-clips record schema (``test_bench.manifest/1``) and its read/write.

One clip = one directory under ``analyzed-clips/``::

    <clip_id>/
      # (no clip copy — ``source.original_path`` maps to the file in place)
      manifest.json            this record — the record of record
      meta.json                the camera-metadata block
      plane.json               plane geometry  (written at M3; absent until then)
      detections.json          frame-by-frame records (M3)
      trajectory.json          built trajectory   (M3)
      ground_truth.json        GT point + judge  (M4)
      results/<run_id>.json    suite runs       (M6)

The manifest is the only file other stages are allowed to *derive* from; the
artifact files are caches that a re-do overwrites.  Its ``schema`` tag is the
one thing that must never drift (a bench that reads an old manifest under a new
schema silently compares the wrong numbers).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from dog_diving.config import PLANES_DIR  # test-bench/planes/ — the saved-planes folder

#: Bump only with a migration; readers refuse an unknown major.
SCHEMA = "test_bench.manifest/1"

#: The artifact files a manifest may reference (relative to the clip directory).
ARTIFACTS: tuple[str, ...] = (
    "manifest.json",
    "meta.json",
    "plane.json",
    "detections.json",
    "trajectory.json",
    "ground_truth.json",
)


def sanitize_clip_id(stem: str) -> str:
    """A filesystem-safe clip id from a clip stem (keeps alnum, dot, dash, underscore)."""
    keep = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(stem))
    keep = keep.strip("._-")
    if not keep:
        raise ValueError(f"clip stem {stem!r} produced an empty id")
    return keep


def clip_dir(analyzed_root: str | Path, clip_id: str) -> Path:
    """The directory for ``clip_id`` under ``analyzed_root`` (not created)."""
    return Path(analyzed_root) / clip_id


def clip_video_path(manifest: dict[str, Any],
                    clip_dir: str | Path | None = None) -> Path | None:
    """Resolve a clip's video file — the *mapped original*, never a repo copy.

    New ingests store only ``source.original_path`` (the clip stays where the
    operator put it; nothing is copied into ``analyzed-clips/``).  Pre-migration
    manifests carry a ``source.clip_file`` (the old in-repo copy); resolve
    ``original_path`` first and fall back to that copy only when the original has
    since moved/deleted and the copy is all that remains.  ``None`` = unreachable.
    """
    source = manifest.get("source") or {}
    orig = source.get("original_path")
    if orig:
        p = Path(orig)
        if p.is_file():
            return p
    clip_file = source.get("clip_file")
    if clip_dir is not None and clip_file:
        c = Path(clip_dir) / clip_file
        if c.is_file():
            return c
    return None


def new_manifest(clip_id: str, *, source: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """A fresh manifest in the *pre-analysis* state (no plane/detections/GT yet)."""
    return {
        "schema": SCHEMA,
        "clip_id": clip_id,
        "source": dict(source),          # {original_path, sha256, kind}
        "meta": dict(meta),              # the probe_mp4 / probe_clip block
        "plane": {"setup_id": None, "present": False},
        "detections": {},
        "trajectory": {},
        "ground_truth": {"path": "ground_truth.json", "judge_present": False},
        "results": [],
    }


# --- saved-planes folder (test-bench/planes/) --------------------------------
# A clip references its plane by ``plane.setup_id``; the plane document itself
# lives in ``PLANES_DIR/<setup_id>.pool.json`` — never next to the clip file.

def plane_setup_path(setup_id: str) -> Path:
    # A setup_id may already carry the ".pool" suffix (it is the plane document's
    # stem).  Normalise so we never write "….pool.pool.json".
    sid = str(setup_id)
    if sid.endswith(".pool"):
        sid = sid[:-len(".pool")]
    return Path(PLANES_DIR) / f"{sid}.pool.json"



def list_planes() -> list[str]:
    d = Path(PLANES_DIR)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.pool.json"))


def load_plane(setup_id: str | None) -> dict[str, Any] | None:
    if not setup_id:
        return None
    p = plane_setup_path(setup_id)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def clip_plane(manifest: dict[str, Any], *, allow_camera_fallback: bool = True
               ) -> dict[str, Any] | None:
    """Resolve a clip's plane document (see :func:`plane_resolution`).

    Order: the manifest's ``plane.setup_id``, then — when that id names no saved
    plane — the newest saved plane whose id starts with the clip's camera tag
    (``camera0_right`` -> ``camera0_right_2026-11-20_01-44-19``).  The fallback
    exists because a re-saved calibration gets a timestamped id while every
    manifest still carries the older one; without it the same clips silently lose
    their plane on every machine.  Pass ``allow_camera_fallback=False`` to insist
    on the exact id.

    Raises ``ValueError`` when the named plane exists but its ``image_size``
    disagrees with the clip (see :func:`plane_size_mismatch`); returns ``None``
    when no plane resolves at all.
    """
    info = plane_resolution(manifest, allow_camera_fallback=allow_camera_fallback)
    if info.get("error_kind") == "size_mismatch":
        raise ValueError(f"{info['error']} (set DD_ALLOW_PLANE_SIZE_MISMATCH=1 to override)")
    return info["doc"]


_CAMERA_TAG = re.compile(r"(camera\d+[A-Za-z_]*)")


def camera_tag(manifest: dict[str, Any]) -> str | None:
    """The camera a clip belongs to (``camera0_right``), from labels or clip id."""
    cam = (manifest.get("labels") or {}).get("camera")
    for candidate in (cam, manifest.get("clip_id")):
        if not candidate:
            continue
        m = _CAMERA_TAG.search(str(candidate))
        if m:
            return m.group(1)
    return None


def plane_documents() -> list[tuple[str, dict[str, Any]]]:
    """Every saved plane as ``(stem, document)``, skipping unreadable files."""
    out: list[tuple[str, dict[str, Any]]] = []
    d = Path(PLANES_DIR)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.pool.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            out.append((p.stem, doc))
    return out


def plane_resolution(manifest: dict[str, Any], *, allow_camera_fallback: bool = True
                     ) -> dict[str, Any]:
    """Resolve a clip's plane, reporting *which* plane was used and why.

    Returns ``{doc, setup_id, requested_id, source, error, error_kind, note}``:
    ``source`` is ``"manifest"`` (the clip's own id resolved), ``"camera-auto"``
    (fallback by camera tag), or ``None``.  Callers that display the plane use
    this rather than :func:`clip_plane` so an auto-matched plane is never silent.
    """
    requested = (manifest.get("plane") or {}).get("setup_id")
    info: dict[str, Any] = {"doc": None, "setup_id": requested, "requested_id": requested,
                            "source": None, "error": None, "error_kind": None, "note": None}

    doc = load_plane(requested)
    if doc is not None:
        info["setup_id"] = doc.get("setup_id") or requested
        mismatch = plane_size_mismatch(manifest, doc)
        if mismatch:
            info["error"] = mismatch.replace(" (set DD_ALLOW_PLANE_SIZE_MISMATCH=1 to override)", "")
            info["error_kind"] = "size_mismatch"
            return info
        info["doc"] = doc
        info["source"] = "manifest"
        return info

    if not allow_camera_fallback:
        info["error"] = f"plane {requested!r} is not saved under {PLANES_DIR}"
        info["error_kind"] = "not_found"
        return info

    tag = camera_tag(manifest)
    if not tag:
        info["error"] = (f"plane {requested!r} is not saved under {PLANES_DIR} and the clip "
                         f"names no camera tag to fall back on")
        info["error_kind"] = "not_found"
        return info

    candidates = [(stem, d) for stem, d in plane_documents()
                  if str(d.get("setup_id") or stem).startswith(tag)
                  and not plane_size_mismatch(manifest, d)]
    if not candidates:
        info["error"] = (f"plane {requested!r} is not saved and no saved plane matches camera "
                         f"tag {tag!r}")
        info["error_kind"] = "not_found"
        return info

    candidates.sort(key=lambda item: (str(item[1].get("created") or ""), item[0]))
    stem, doc = candidates[-1]
    info.update(doc=doc, setup_id=doc.get("setup_id") or stem, source="camera-auto",
                note=(f"manifest plane {requested!r} is not saved; using {info.get('setup_id')!r} "
                      f"matched by camera tag {tag!r}"))
    info["setup_id"] = doc.get("setup_id") or stem
    return info


def plane_size_mismatch(manifest: dict[str, Any], plane_doc: dict[str, Any] | None) -> str | None:
    """A description of a frame-size mismatch between a clip and its plane, or ``None``.

    The plane document records the pixel space its homography was fitted in
    (``image_size`` = ``[w, h]``).  A clip of a different size maps to a different
    pixel space, so the homography cannot be reused.  ``None`` when the plane has no
    ``image_size``, the manifest has no ``meta``, or the sizes agree.  Honours
    ``DD_ALLOW_PLANE_SIZE_MISMATCH=1`` as the only override.
    """
    if not plane_doc:
        return None
    if os.environ.get("DD_ALLOW_PLANE_SIZE_MISMATCH", "").strip() == "1":
        return None
    psize = plane_doc.get("image_size")
    if not isinstance(psize, (list, tuple)) or len(psize) < 2:
        return None
    meta = manifest.get("meta") or {}
    w, h = meta.get("width"), meta.get("height")
    if w is None or h is None:
        return None
    if (int(psize[0]), int(psize[1])) != (int(w), int(h)):
        return (f"plane {plane_doc.get('setup_id')!r} is {int(psize[0])}x{int(psize[1])} "
                f"but clip {manifest.get('clip_id')!r} is {int(w)}x{int(h)}")
    return None


def write_json(path: str | Path, obj: Any) -> Path:
    """Write JSON atomically (same-dir temp + replace) so a crash never halves a manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, Any], analyzed_dir: str | Path) -> Path:
    """Validate the schema tag, then write ``<analyzed_dir>/manifest.json``."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"refusing to write schema {manifest.get('schema')!r} "
                         f"(this build writes {SCHEMA!r})")
    return write_json(Path(analyzed_dir) / "manifest.json", manifest)


def read_manifest(analyzed_dir_or_path: str | Path) -> dict[str, Any]:
    """Read a manifest from a clip directory or a ``manifest.json`` path."""
    p = Path(analyzed_dir_or_path)
    if p.is_dir():
        p = p / "manifest.json"
    man = read_json(p)
    if man.get("schema") != SCHEMA:
        raise ValueError(f"unknown manifest schema {man.get('schema')!r} (expected {SCHEMA!r}) at {p}")
    return man


def migrate_drop_copied_clips(analyzed_root: str | Path) -> list[dict[str, Any]]:
    """One-time migration: stop keeping the in-repo ``clip.<ext>`` copy.

    Every manifest already records ``source.original_path``; this deletes the
    copied clip file and drops the now-stale ``source.clip_file`` key, so the clip
    is resolved from its original location.  A copy is **kept** when its original
    no longer resolves (a moved/deleted original must not orphan the only clip).
    Idempotent — a re-run finds nothing to do.
    """
    root = Path(analyzed_root)
    report: list[dict[str, Any]] = []
    if not root.is_dir():
        return report
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (cdir / "manifest.json").is_file():
            continue
        man = read_manifest(cdir)
        source = man.get("source") or {}
        orig = source.get("original_path")
        actions: list[str] = []
        kept_copy = False
        for f in sorted(cdir.glob("clip.*")):
            if orig and Path(orig).is_file():
                f.unlink()
                actions.append(f"removed {f.name}")
            else:
                kept_copy = True
                actions.append(f"kept {f.name} (original missing: {orig!r})")
        # Drop the stale clip_file only when no copy remains to fall back to.
        if not kept_copy and source.pop("clip_file", None) is not None:
            write_manifest(man, cdir)
        if actions:
            report.append({"clip_id": man.get("clip_id"),
                           "original_path": orig, "actions": actions})
    return report


# --- Review deselections (the technician's excluded detections) -------------
# A clip's Review state: which detection frames the technician marked faulty.
# Stored as a bare frame-int list in ``<clip>/excluded_frames.json`` so it is a
# first-class input to scoring (the suite drops those frames from the point set).

# --- Derived clip state (artifacts are the source of truth) -----------------
# A manifest's ``detections`` / ``trajectory`` / ``ground_truth`` blocks are
# *caches*, written only by the stage that last ran.  These 116 clips were found
# carrying ``detections: {}`` while ``detections.json`` held real accepted
# points, which made the GUI hide every processed clip.  Anything that decides
# "is this clip ready?" therefore reads the artifact files; the cached block is
# consulted only when the artifact is missing.

#: Fewest accepted points a clip needs before the suite can fit a trajectory.
MIN_SUITE_POINTS = 4


def accepted_frames(records: Any, excluded: Iterable[int] = ()) -> list[int]:
    """Frames of the accepted, non-deselected detections (minus ``excluded``)."""
    skip = {int(f) for f in excluded or ()}
    frames: set[int] = set()
    for r in records if isinstance(records, (list, tuple)) else ():
        if not isinstance(r, dict) or r.get("frame") is None:
            continue
        if not r.get("accepted", True) or r.get("deselected"):
            continue
        frame = int(r["frame"])
        if frame not in skip:
            frames.add(frame)
    return sorted(frames)


_STATE_CACHE: dict[tuple[str, tuple], dict[str, Any]] = {}


def _state_fingerprint(cdir: Path) -> tuple:
    """(mtime_ns, size) of every artifact a clip_state depends on."""
    marks = []
    for name in ("manifest.json", "detections.json", "trajectory.json", "ground_truth.json"):
        try:
            st = (cdir / name).stat()
            marks.append((st.st_mtime_ns, st.st_size))
        except OSError:
            marks.append(None)
    return tuple(marks)


def clear_state_cache() -> None:
    """Drop every memoised clip_state (tests, or after an out-of-band file edit)."""
    _STATE_CACHE.clear()


def clip_state(clip_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the panes need about one clip, read from its artifact files.

    ``{clip_id, has_detections_file, n_detections, n_accepted, accepted_frames,
    has_trajectory, n_trajectory_points, has_gt_file, gt, gt_placed, gt_in,
    judge_in, plane_doc, plane_id, plane_source, plane_error, plane_note,
    excluded_frames, ready, blockers}``.  ``ready``/``blockers`` come from
    :func:`suite_blockers`, so the Analyze pane and the Run-suite list agree by
    construction instead of each re-deriving the rule.
    """
    cdir = Path(clip_dir)
    # Memoised on each artifact's mtime+size: a 1200-record detections.json costs
    # ~15 ms, and the Run-suite list walks every clip on each refresh.
    use_cache = manifest is None
    key = None
    if use_cache:
        key = (str(cdir), _state_fingerprint(cdir))
        hit = _STATE_CACHE.get(key)
        if hit is not None:
            return dict(hit)
    man = manifest if manifest is not None else read_manifest(cdir)
    excluded = sorted({int(f) for f in (man.get("excluded_frames") or [])})

    det_path = cdir / "detections.json"
    has_det = det_path.is_file()
    records: Any = []
    if has_det:
        try:
            records = read_json(det_path)
        except (OSError, ValueError):
            records = []
    frames = accepted_frames(records, excluded)
    n_det = len(records) if isinstance(records, list) else 0
    n_acc = len(frames)
    if not has_det:  # legacy clip: fall back to the manifest's cached block
        cache = man.get("detections") or {}
        n_det = int(cache.get("n_detections") or 0)
        n_acc = int(cache.get("n_accepted") or 0)

    traj_path = cdir / "trajectory.json"
    traj_points: list[Any] = []
    if traj_path.is_file():
        try:
            traj_points = read_json(traj_path).get("points") or []
        except (OSError, ValueError, AttributeError):
            traj_points = []

    gt_path = cdir / "ground_truth.json"
    gt: dict[str, Any] | None = None
    if gt_path.is_file():
        try:
            gt = read_json(gt_path)
        except (OSError, ValueError):
            gt = None
    gt = gt if isinstance(gt, dict) else None
    x, y = (gt or {}).get("x_px"), (gt or {}).get("y_px")
    gt_placed = isinstance(x, (int, float)) and isinstance(y, (int, float))
    gt_in = (gt or {}).get("gt_in")
    if gt_in is None and gt is None:  # nothing on disk — trust the cached number
        gt_in = (man.get("ground_truth") or {}).get("gt_in")

    plane = plane_resolution(man)

    state: dict[str, Any] = {
        "clip_id": man.get("clip_id") or cdir.name,
        "has_detections_file": has_det,
        "n_detections": n_det,
        "n_accepted": n_acc,
        "accepted_frames": frames,
        "has_trajectory": bool(traj_points),
        "n_trajectory_points": len(traj_points),
        "has_gt_file": gt_path.is_file(),
        "gt": gt,
        "gt_placed": gt_placed,
        "gt_in": gt_in,
        "judge_in": (gt or {}).get("judge_in"),
        "plane_doc": plane["doc"],
        "plane_id": plane["setup_id"],
        "plane_source": plane["source"],
        "plane_error": plane["error"],
        "plane_note": plane["note"],
        "excluded_frames": excluded,
    }
    state["blockers"] = suite_blockers(state)
    state["ready"] = not state["blockers"]
    if key is not None:
        _STATE_CACHE[key] = dict(state)
    return state


def suite_blockers(state: dict[str, Any]) -> list[str]:
    """Why this clip cannot enter the suite yet (``[]`` == ready).

    The single source of truth for suite readiness: the Run-suite list, the
    Analyze readout and the tests all call it, so a stale manifest cache can no
    longer hide a prepared clip (or admit an unprepared one).
    """
    out: list[str] = []
    n_acc = int(state.get("n_accepted") or 0)
    if not state.get("has_detections_file"):
        out.append("no detections.json (run Analyze)")
    elif n_acc < MIN_SUITE_POINTS:
        out.append(f"{n_acc} accepted point(s), need {MIN_SUITE_POINTS}")
    if state.get("plane_doc") is None:
        out.append(state.get("plane_error") or "no plane assigned")
    if not state.get("has_gt_file"):
        out.append("no ground_truth.json")
    elif not state.get("gt_placed"):
        out.append("ground-truth point not placed")
    return out


def read_excluded_frames(clip_dir: str | Path) -> list[int]:
    """The technician's deselected detection frames, or ``[]`` (none yet).

    Stored in the manifest (``excluded_frames``), the bench's record of record, so
    scoring provenance reads it directly rather than from a side file.
    """
    man = read_manifest(clip_dir)
    return sorted({int(f) for f in (man.get("excluded_frames") or [])})
