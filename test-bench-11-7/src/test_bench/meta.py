"""Camera-metadata probe for the test bench.

Thin wrapper over the parent's ``dog_diving.video`` (copied verbatim into
``test-bench/src/dog_diving/video.py``) so the bench reports *exactly* the
metadata the parent's ``result.json`` records — width/height/fps/codec, the
declared-vs-decoded frame counts, and VFR flag.

Why wrap instead of re-probe: ``video.probe`` already cross-checks ffprobe
against OpenCV, counts the decoded frames (VFR containers over-declare), and
warns on disagreement.  A second probe implementation is a second set of
numbers, which is how ``width`` stops matching ``result.json``.
"""

from __future__ import annotations

from pathlib import Path

from dog_diving import video as _video

#: Extensions the bench treats as OpenCV-readable video.
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv"})
#: Extensions that need the ZED X SDK to read.
SVO2_EXTS: frozenset[str] = frozenset({".svo2"})


def clip_kind(path: str | Path) -> str:
    """``"mp4"`` or ``"svo2"`` for ``path``; raise on anything the bench can't ingest."""
    ext = Path(path).suffix.lower()
    if ext in SVO2_EXTS:
        return "svo2"
    if ext in VIDEO_EXTS:
        return "mp4"
    raise ValueError(f"unsupported clip extension {ext!r}: use one of "
                     f"{sorted(VIDEO_EXTS)} or {sorted(SVO2_EXTS)}")


def probe_mp4(path: str | Path) -> dict:
    """The ``meta`` block the manifest stores for an MP4/MOV clip.

    Keys mirror the parent ``result.json``'s ``clip`` block (minus ``path`` and
    ``name``, which the bench keeps in the manifest's ``source``)::

        {width, height, fps, codec, n_frames, vfr,
         declared_vs_decoded: [declared, decoded], duration_s, backend, probe_warnings}

    ``n_frames`` is the *decoded* count when available (``video.VideoInfo.n_frames``),
    because a VFR container's declared count leaves trailing holes.
    """
    info = _video.probe(path)
    return {
        "width": int(info.width),
        "height": int(info.height),
        "fps": float(info.fps),
        "codec": info.codec,
        "n_frames": int(info.n_frames),
        "vfr": bool(info.vfr),
        "declared_vs_decoded": [int(info.n_frames_declared),
                                None if info.n_frames_decoded is None else int(info.n_frames_decoded)],
        "duration_s": float(info.duration_s),
        "backend": info.backend,
        "probe_warnings": list(info.warnings),
    }


def _probe_svo2(path: str | Path) -> dict:  # pragma: no cover - needs the SDK
    """SVO2 metadata — only when the ZED X SDK (``pyzed.sl``) is importable.

    Not exercised on an SDK-less desktop (this machine).  The ingest path treats
    the raised error as a clean "SVO2 unavailable" notice rather than a crash.
    """
    try:
        import pyzed.sl as sl  # type: ignore
    except Exception as exc:  # ImportError, OSError, ...
        raise RuntimeError(
            "SVO2 input needs the ZED X SDK (pyzed.sl); it is not installed on this "
            "machine. Use an MP4/MOV clip here, or run ingest on a ZED-SDK host."
        ) from exc
    raise NotImplementedError(
        "SVO2 metadata extraction is wired in M2.5 (needs a ZED-SDK host to verify); "
        "the MP4 path is M2."
    )


def probe_clip(path: str | Path) -> dict:
    """Dispatch on extension: MP4/MOV → :func:`probe_mp4`, SVO2 → SDK-guarded probe."""
    kind = clip_kind(path)
    return probe_mp4(path) if kind == "mp4" else _probe_svo2(path)
