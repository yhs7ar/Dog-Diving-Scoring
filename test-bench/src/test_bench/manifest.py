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
from pathlib import Path
from typing import Any

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


def clip_plane(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a clip's plane document from its ``plane.setup_id`` (or None)."""
    return load_plane((manifest.get("plane") or {}).get("setup_id"))


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

def read_excluded_frames(clip_dir: str | Path) -> list[int]:
    """The technician's deselected detection frames, or ``[]`` (none yet).

    Stored in the manifest (``excluded_frames``), the bench's record of record, so
    scoring provenance reads it directly rather than from a side file.
    """
    man = read_manifest(clip_dir)
    return sorted({int(f) for f in (man.get("excluded_frames") or [])})
