"""Ingest: register a clip under ``analyzed-clips/<clip_id>/`` *by reference*.

Per the locked decision, the clip file is **not copied** into the bench; the
manifest maps ``source.original_path`` to where the operator left it.  Every other
artifact (``manifest.json``, ``meta.json``, ``plane.json``, ``detections.json``,
``trajectory.json``, ``ground_truth.json``, ``results/``) still lives in
``analyzed-clips/<clip_id>/``.  Ingest is idempotent: re-ingesting the same source
into the same id refreshes ``manifest.json`` (source + camera metadata) but
preserves any placed analysis (``plane``/``ground_truth``/``results``).

MP4/MOV is the M2 path.  SVO2 raises a clean "needs the ZED X SDK" notice from
:mod:`test_bench.meta` when the SDK is absent.
"""

from __future__ import annotations

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
    """Register ``source`` under ``analyzed-clips/<clip_id>/`` (no copy) and return its manifest.

    ``clip_id`` defaults to the sanitized source stem.  The clip is mapped by
    ``source.original_path`` (nothing is copied).  ``preserve_existing`` keeps any
    already-present ``plane.json``/``ground_truth.json``/``results/`` and
    ``manifest.json`` state (plane/detections/GT/results sections) — a re-ingest
    refreshes the source + camera metadata without discarding analysis.
    """
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"source clip not found: {src}")

    kind = _meta.clip_kind(src)
    cid = _manifest.sanitize_clip_id(clip_id or src.stem)
    dest = _manifest.clip_dir(analyzed_root, cid)
    dest.mkdir(parents=True, exist_ok=True)

    # Probe BEFORE writing: metadata is of the source; sha256 is the clip identity
    # the parent's result.json also records for this file.
    mblock = _meta.probe_clip(src)
    sha = _video.sha256(src)

    _manifest.write_json(dest / "meta.json", mblock)

    prior: dict[str, Any] | None = None
    man_path = dest / "manifest.json"
    if preserve_existing and man_path.is_file():
        prior = _manifest.read_manifest(dest)

    man = _manifest.new_manifest(
        cid,
        source={"original_path": str(src), "sha256": sha, "kind": kind,
                "filename": src.name},
        meta=mblock,
    )
    if prior is not None:
        # Keep analysis state; refresh only source+meta (the clip may differ).
        for key in ("plane", "detections", "trajectory", "ground_truth", "results"):
            if key in prior:
                man[key] = prior[key]

    _manifest.write_manifest(man, dest)
    return man
