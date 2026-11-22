"""Video I/O — metadata probe, frame iteration, clip hashing.

M1 deliverable (PLAN.md §5).

Design notes that matter downstream
-----------------------------------
``ffprobe`` is the *authoritative* metadata source when it is on PATH; cv2 is
used to prove the file actually decodes.  The two disagree in ways that matter,
measured on this corpus (2026-10-07) by ``oracle_m1.py``:

* the 8 h264 sample clips agree completely — 1920x1080 @ 30.000, declared frame
  count == decoded frame count.
* the phone MOVs (HEVC) are variable-frame-rate and their metadata *lies*:
  ``IMG_6070.MOV`` declares 68 frames but decodes 60; ``IMG_6072.MOV`` declares
  172 and decodes 171.  ``r_frame_rate`` (60) != ``avg_frame_rate`` (57.38).

Consequently "the frame count" is ambiguous and ``duration = n_frames / fps`` is
wrong on VFR input.  ``probe`` reports the declared *and* the decoded count and
raises a warning when they disagree, instead of quietly picking one.

``iter_frames`` decodes through cv2 by default, with an ffmpeg rawvideo pipe as
an alternative backend.  cv2 handles every file in this corpus, so the ffmpeg
path would otherwise be untested dead code — the M1 gate forces it and compares
the two backends frame-for-frame.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# --- tool discovery ---------------------------------------------------------
# ffmpeg/ffprobe may be on PATH (MSYS) or only as a native .exe; resolve both.
_FFPROBE_NAMES = ("ffprobe", "ffprobe.exe")
_FFMPEG_NAMES = ("ffmpeg", "ffmpeg.exe")


def _find_tool(names: tuple[str, ...]) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    for base in (
        Path.home() / "AppData/Local/Microsoft/WinGet/Links",
        Path("C:/Program Files/ffmpeg/bin"),
    ):
        for n in names:
            p = base / n
            if p.is_file():
                return str(p)
    return None


FFPROBE = _find_tool(_FFPROBE_NAMES)
FFMPEG = _find_tool(_FFMPEG_NAMES)


# --- hashing ----------------------------------------------------------------
def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of a media file — clip identity for cache keys + provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# --- metadata ---------------------------------------------------------------
@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float                      # average fps (ffprobe avg_frame_rate when available)
    n_frames_declared: int          # what the container claims
    n_frames_decoded: int | None    # what actually comes out, when counted
    codec: str
    duration_s: float
    vfr: bool
    backend: str                    # where the metadata came from: 'ffprobe' | 'cv2'
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_frames(self) -> int:
        """Frame count to size arrays with: the decoded count when known.

        Deliberately prefers the measured count — the container over-declares on
        VFR input, and sizing a trajectory array by the declared count leaves
        trailing holes (or, worse, silently shifts every frame index).
        """
        return self.n_frames_decoded if self.n_frames_decoded is not None else self.n_frames_declared

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["path"] = str(self.path)
        d["warnings"] = list(self.warnings)
        d["n_frames"] = self.n_frames
        return d


def _ratio(s: str | None) -> float | None:
    if not s or s in ("0/0", "N/A"):
        return None
    if "/" in s:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else None
    return float(s)


def _ffprobe_raw(path: Path) -> dict | None:
    if not FFPROBE:
        return None
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
        d = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    st = (d.get("streams") or [None])[0]
    if not st:
        return None
    return {
        "codec": st.get("codec_name"),
        "width": st.get("width"),
        "height": st.get("height"),
        "avg_fps": _ratio(st.get("avg_frame_rate")),
        "r_fps": _ratio(st.get("r_frame_rate")),
        "nb_frames": int(st["nb_frames"]) if st.get("nb_frames", "").isdigit() else None,
        "duration": float(st["duration"]) if st.get("duration") else None,
        "format_duration": float(d["format"]["duration"]) if (d.get("format") or {}).get("duration") else None,
    }


def _cv2_open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    return cap


def _count_cv2_frames(cap: cv2.VideoCapture) -> int:
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    return n


def probe(path: str | Path, *, count_frames: bool = True) -> VideoInfo:
    """Metadata for ``path``, cross-checked between ffprobe and cv2.

    ``count_frames`` decodes the whole file to establish the *real* frame count.
    Keep it on for short clips (this corpus is 7-15 s); turn it off for very long
    input, where the declared count is the only cheap answer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    ff = _ffprobe_raw(path)
    warnings: list[str] = []

    cap = _cv2_open(path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cv_codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    cv_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cv_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv_decoded = _count_cv2_frames(cap) if count_frames else None
    cap.release()

    if ff is None:
        warnings.append("ffprobe unavailable — metadata from cv2 only (no VFR detection)")
        decoded = cv_decoded
        fps = cv_fps
        return VideoInfo(
            path=path,
            width=cv_w, height=cv_h, fps=fps,
            n_frames_declared=cv_declared, n_frames_decoded=decoded,
            codec=cv_codec, duration_s=(decoded / fps if (decoded and fps) else 0.0),
            vfr=False, backend="cv2", warnings=tuple(warnings),
        )

    width, height = ff["width"] or cv_w, ff["height"] or cv_h
    fps = ff["avg_fps"] or cv_fps
    declared = ff["nb_frames"] if ff["nb_frames"] is not None else cv_declared
    decoded = cv_decoded if cv_decoded is not None else declared
    duration = ff["format_duration"] or ff["duration"] or (decoded / fps if fps else 0.0)

    # --- cross-checks: every one of these fired on real files in this corpus ---
    if ff["r_fps"] and ff["avg_fps"] and abs(ff["r_fps"] - ff["avg_fps"]) > 0.5:
        warnings.append(f"VFR: r_frame_rate={ff['r_fps']:.3f} != avg_frame_rate={ff['avg_fps']:.3f}")
    if abs(fps - cv_fps) > 0.05:
        warnings.append(f"fps disagreement: ffprobe={fps:.3f} cv2={cv_fps:.3f}")
    if (width, height) != (cv_w, cv_h):
        warnings.append(f"size disagreement: ffprobe={width}x{height} cv2={cv_w}x{cv_h}")
    if count_frames and decoded != declared:
        warnings.append(
            f"declared frame count != decoded: declared={declared} decoded={decoded} "
            "(container metadata is unreliable; use VideoInfo.n_frames)"
        )

    return VideoInfo(
        path=path, width=width, height=height, fps=fps,
        n_frames_declared=declared, n_frames_decoded=decoded,
        codec=ff["codec"] or cv_codec, duration_s=duration,
        vfr=any(w.startswith("VFR") for w in warnings),
        backend="ffprobe", warnings=tuple(warnings),
    )


# --- frame iteration --------------------------------------------------------
def _iter_cv2(path: Path, start: int, stop: int | None) -> Iterator[tuple[int, np.ndarray]]:
    cap = _cv2_open(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while stop is None or idx < stop:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def _iter_ffmpeg(
    path: Path, start: int, stop: int | None, width: int, height: int, fps: float
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode through an ffmpeg rawvideo pipe.

    ``-fps_mode passthrough`` keeps 1:1 frame correspondence with the source; the
    default would resample VFR input to CFR and invent or drop frames.

    ``start`` is a frame index but ``-ss`` is a *timestamp*, so it is converted
    via ``fps``; the seek is keyframe-accurate, not frame-exact.  stderr goes to
    DEVNULL rather than a pipe — an undrained pipe can deadlock ffmpeg.
    """
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found; cannot use the ffmpeg backend")
    cmd = [FFMPEG, "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start / fps:.6f}" if fps else str(start)]
    cmd += ["-i", str(path), "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    nbytes = width * height * 3
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=nbytes * 4
    )
    idx = start
    try:
        stdout = proc.stdout
        assert stdout is not None
        while stop is None or idx < stop:
            buf = stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield idx, np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()
            idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def iter_frames(
    path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
    backend: str = "auto",
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` from ``path``.

    ``backend``: ``"cv2"``, ``"ffmpeg"``, or ``"auto"`` (cv2, falling back to
    ffmpeg when cv2 cannot open or cannot produce the first frame).
    ``start``/``stop`` are half-open ``[start, stop)``.  ``start`` on the ffmpeg
    backend is a *timestamp* seek, i.e. keyframe-accurate, not frame-exact.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if backend not in ("auto", "cv2", "ffmpeg"):
        raise ValueError(f"unknown backend: {backend!r}")

    if backend == "cv2":
        yield from _iter_cv2(path, start, stop)
        return

    if backend == "ffmpeg":
        info = probe(path, count_frames=False)
        yield from _iter_ffmpeg(path, start, stop, info.width, info.height, info.fps)
        return

    # auto: prefer cv2, but pull one frame to prove it actually decodes before
    # committing — a generator only fails when iterated, so opening is not enough.
    gen = _iter_cv2(path, start, stop)
    try:
        first = next(gen)
    except (StopIteration, Exception):  # noqa: B014 - explicit: any cv2 failure falls back
        gen = None
        first = None
    if gen is not None:
        yield first
        yield from gen
        return
    info = probe(path, count_frames=False)
    yield from _iter_ffmpeg(path, start, stop, info.width, info.height, info.fps)
