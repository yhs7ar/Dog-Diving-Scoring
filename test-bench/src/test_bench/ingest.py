"""Ingest: copy a clip into ``analyzed-clips/<clip_id>/`` and write its record.

Per the locked decision, the clip file is **copied** (self-contained); the bench
never stores only a path.  Ingest is idempotent: re-ingesting the same source
into the same id overwrites ``clip.*`` + ``manifest.json`` but preserves any
``plane.json`` / ``ground_truth.json`` / ``results/`` already present (so a
re-ingest cannot silently drop placed ground truth).

MP4/MOV is the M2 path.  SVO2 raises a clean "needs the ZED X SDK" notice from
:mod:`test_bench.meta` when the SDK is absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from dog_diving import video as _video

from . import manifest as _manifest
from . import meta as _meta


def ingest_clip(
    source: str | Path,
    analyzed_root: str | Path,
    *,
    clip_id: str | None = None,
    preserve_existing: bool = True,
) -> dict[str, Any]:
    """Copy ``source`` into ``analyzed-clips/<clip_id>/`` and return its manifest.

    ``clip_id`` defaults to the sanitized source stem.  ``preserve_existing`` keeps
    any already-present ``plane.json``/``ground_truth.json``/``results/`` and
    ``manifest.json`` state (plane/detections/GT/results sections) — a re-ingest
    refreshes the clip + camera metadata without discarding analysis.
    """
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"source clip not found: {src}")

    kind = _meta.clip_kind(src)
    cid = _manifest.sanitize_clip_id(clip_id or src.stem)
    dest = _manifest.clip_dir(analyzed_root, cid)
    dest.mkdir(parents=True, exist_ok=True)

    # Probe BEFORE copying: metadata is of the source; sha256 is the clip identity
    # the parent's result.json also records for this file.
    mblock = _meta.probe_clip(src)
    sha = _video.sha256(src)

    dest_clip = dest / f"clip{src.suffix.lower()}"
    shutil.copy2(src, dest_clip)

    _manifest.write_json(dest / "meta.json", mblock)

    prior: dict[str, Any] | None = None
    man_path = dest / "manifest.json"
    if preserve_existing and man_path.is_file():
        prior = _manifest.read_manifest(dest)

    man = _manifest.new_manifest(
        cid,
        source={"original_path": str(src), "sha256": sha, "kind": kind,
                "filename": src.name, "clip_file": dest_clip.name},
        meta=mblock,
    )
    if prior is not None:
        # Keep analysis state; refresh only source+meta (the clip may differ).
        for key in ("plane", "detections", "trajectory", "ground_truth", "results"):
            if key in prior:
                man[key] = prior[key]

    _manifest.write_manifest(man, dest)
    return man
