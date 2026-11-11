"""The analyzed-clips record schema (``test_bench.manifest/1``) and its read/write.

One clip = one directory under ``analyzed-clips/``::

    <clip_id>/
      clip.mp4 | clip.svo2     the copied clip (self-contained)
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


def new_manifest(clip_id: str, *, source: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """A fresh manifest in the *pre-analysis* state (no plane/detections/GT yet)."""
    return {
        "schema": SCHEMA,
        "clip_id": clip_id,
        "source": dict(source),          # {original_path, sha256, kind}
        "meta": dict(meta),              # the probe_mp4 / probe_clip block
        "plane": {"path": "plane.json", "setup_id": None, "present": False},
        "detections": {},
        "trajectory": {},
        "ground_truth": {"path": "ground_truth.json", "judge_present": False},
        "results": [],
    }


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
