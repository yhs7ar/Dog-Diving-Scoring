"""
zed_recorder.py — Dual ZED X Rolling SVO2 & Clip Buffer (rig-proven config)

Capture standard (see scripts/dual_recording/WORKAROUNDS_AND_LIMITATIONS.md):

  * Resolution: **SVGA (1200x720)** @ **30 FPS** — the only dual-camera config proven
    on the Jetson Orin Nano (HD1080@30, VGA, and 15 FPS all FAIL on the rig).
  * Compression: **LOSSLESS** only. Orin Nano has NO NVENC, so H264/H265 crash the
    daemon / corrupt SVO2.
  * Depth: captured **NONE** and computed off-line from the recorded SVO2.

The rolling buffer is a **true ring** (a ``collections.deque`` capped at
``chunk_seconds * fps`` frames).  A save therefore always emits the **last N
seconds ending at the request**, independent of where the request lands relative
to an SVO2 slot boundary — the pre-ring design returned "the chunk I happen to be
in", which shrank to a near-empty window when the trigger landed just after a
rollover.  The duration is parameterised (default **10 s**) and live-adjustable via
``set_capture_seconds``.

The module also ships two testability aids so the recorder state machine is
verifiable offline (no pyzed, no hardware):

  * ``ZEDRecorder`` — the abstract capture seam the app depends on.
  * ``FakeZEDRecorder`` — a synthetic-camera stand-in that mirrors the hardware
    recorder's public API (rolling ring, save-on-request, MP4 export).
  * ``ComputeGovernor`` — gates capture while a heavy stage (detect/measure) runs,
    plus an optional psutil-based CPU load watch.
"""

from __future__ import annotations

import collections
import json
import shutil
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

# Safe ZED SDK import
try:
    import pyzed.sl as sl
    HAS_ZED_SDK = True
except ImportError:
    sl = None
    HAS_ZED_SDK = False


# ---------------------------------------------------------------------------
# Rig-proven capture standard (single source of truth for the recorder).
# ---------------------------------------------------------------------------
CAPTURE_RESOLUTION = "SVGA"       # 1200x720 — the only dual-cam config proven on Orin Nano
CAPTURE_FPS = 30                  # 30 FPS; the ZED X sensor natively streams 30/60 only
CAPTURE_COMPRESSION = "LOSSLESS"  # no NVENC on Orin Nano -> CPU PNG/RAW only
CAPTURE_DEPTH_MODE = "NONE"       # depth is computed off-line from the SVO2

# Frame dimensions of the rig-proven resolution (used by the fake camera seam).
CAPTURE_WIDTH = 1200
CAPTURE_HEIGHT = 720

# ---------------------------------------------------------------------------
# Pipeline-clip naming convention (step 2c): one clip PER camera.
#   cam1's LEFT view -> "{prefix}_right.mp4"  (right == cam1, matching the test-data
#                                              video004_right / video004_left convention)
#   cam2's LEFT view -> "{prefix}_left.mp4"
# SVO2 filenames stay "{prefix}_cam1.svo2" / "{prefix}_cam2.svo2" (serial-keyed, not view-named).
# ---------------------------------------------------------------------------
CLIP_CAM1_SUFFIX = "right"   # cam1's LEFT view
CLIP_CAM2_SUFFIX = "left"    # cam2's LEFT view


def clip_name(prefix: str, camera: str) -> str:
    """Pipeline-clip filename for one camera: ``{prefix}_{camera}.mp4``."""
    return f"{prefix}_{camera}.mp4"


def _write_mp4(path: Path, frames: list, fps: int) -> Path:
    """Write ``frames`` (BGR numpy arrays) to ``path`` as an mp4v-encoded MP4."""
    if frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
    return path


def _sl_resolution():
    """Return the sl.RESOLUTION enum for ``CAPTURE_RESOLUTION``, or None offline."""
    if not HAS_ZED_SDK:
        return None
    return getattr(sl.RESOLUTION, CAPTURE_RESOLUTION, sl.RESOLUTION.SVGA)


class ZEDRecorder(ABC):
    """The capture seam the app depends on.

    Concrete backends: ``DualZEDXLiveBuffer`` (hardware) and ``FakeZEDRecorder``
    (offline). The informal contract also includes the ``running`` attribute and
    the ``last_saved_mp4`` attribute (set by ``trigger_capture``; the cam1/right clip),
    plus ``last_saved_clips`` — a ``{"right": cam1, "left": cam2}`` path dict (step 2c).
    """

    @abstractmethod
    def start(self, cam1_sn: int | None = None, cam2_sn: int | None = None) -> bool:
        """Start live capture. True on success, False when offline/failed."""

    @abstractmethod
    def trigger_capture(self, prefix: str) -> Path | None:
        """Freeze the last-N rolling window to disk; return the MP4 path or None."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capture and release camera handles."""


class ComputeGovernor:
    """Gates capture to protect the Jetson Orin Nano from overload.

    Two independent mechanisms, both consulted by ``allow_capture()``:

    * **explicit suspension** — a heavy stage (detect/measure) holds the governor
      via ``suspend()``/``resume()`` while it runs; ref-counted and re-entrant so
      nested stages cannot release it early.
    * **load watch** — when ``psutil`` is importable, ``overloaded()`` reflects live
      system CPU load against ``max_cpu_percent``. Without psutil it is always
      False (the suspension path still works).
    """

    def __init__(self, max_cpu_percent: float = 90.0, poll_interval: float = 0.5):
        self.max_cpu_percent = float(max_cpu_percent)
        self.poll_interval = float(poll_interval)
        self._suspend_count = 0
        self._lock = threading.Lock()
        self._last_poll = 0.0
        self._last_cpu = -1.0
        try:
            import psutil  # type: ignore
            self._psutil = psutil
        except ImportError:
            self._psutil = None

    # -- explicit suspension (primary mechanism) ----------------------------
    def suspend(self, stage: str = "") -> None:
        with self._lock:
            self._suspend_count += 1

    def resume(self, stage: str = "") -> None:
        with self._lock:
            self._suspend_count = max(0, self._suspend_count - 1)

    @property
    def suspended(self) -> bool:
        with self._lock:
            return self._suspend_count > 0

    # -- load watch (secondary; degrades gracefully without psutil) ---------
    def cpu_percent(self) -> float:
        """Live system-wide CPU load (0-100), or -1.0 when unmeasurable."""
        if self._psutil is None:
            return -1.0
        now = time.monotonic()
        if now - self._last_poll >= self.poll_interval:
            self._last_cpu = self._psutil.cpu_percent(interval=None)
            self._last_poll = now
        return self._last_cpu

    def overloaded(self) -> bool:
        cpu = self.cpu_percent()
        return cpu >= 0.0 and cpu > self.max_cpu_percent

    # -- combined gate ------------------------------------------------------
    def allow_capture(self) -> bool:
        """True when capture may proceed (not suspended and not overloaded)."""
        return (not self.suspended) and (not self.overloaded())


class DualZEDXLiveBuffer(ZEDRecorder):
    """Dual ZED X rolling SVO2 buffer + clip exporter (hardware backend).

    Records two lossless SVO2 streams (one per camera) into alternating temp
    slots; the completed slot is committed to ``output_dir`` only when a jump
    save is requested.  Yields to scoring compute via an optional ``governor``.

    The pipeline clips (MP4) are built from **ring buffers** of the last
    ``chunk_seconds`` of cam1 and cam2 frames (one per camera, step 2c),
    snapshotted at trigger time — so each saved clip is the last N seconds
    *ending at the request*, not "the chunk the request happened to land in".
    """

    DEFAULT_FPS = CAPTURE_FPS
    DEFAULT_CHUNK_SECONDS = 10   # plan default: save the last 10 s (was 5)
    DEFAULT_RESOLUTION = CAPTURE_RESOLUTION
    DEFAULT_COMPRESSION = CAPTURE_COMPRESSION

    def __init__(
        self,
        output_dir: str | Path = "data/svo_recordings",
        clips_dir: str | Path = "data/clips",
        chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
        fps: int = DEFAULT_FPS,
        governor: ComputeGovernor | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.clips_dir = Path(clips_dir)
        self.temp_dir = self.output_dir / "temp_rolling_buffer"

        self.chunk_seconds = chunk_seconds
        self.fps = fps
        self.chunk_frames = int(chunk_seconds * fps)
        self.governor = governor

        self.running = False
        self.lock = threading.Lock()
        self.last_saved_mp4: Path | None = None
        #: The saved clip pair, keyed by camera role: {"right": cam1 clip, "left": cam2 clip}.
        self.last_saved_clips: dict[str, Path] = {}
        #: Ring buffer of the most recent ``chunk_frames`` cam1 frames (BGR).  Persists
        #: across SVO2 slot boundaries so a save always holds the last N seconds, never
        #: just "the chunk I happen to be in".
        self._ring: collections.deque = collections.deque(maxlen=self.chunk_frames)
        #: Ring buffer of the most recent ``chunk_frames`` cam2 frames (BGR), kept in lock
        #: step with ``_ring`` so both cameras' clips span the same last-N window (step 2c).
        self._ring2: collections.deque = collections.deque(maxlen=self.chunk_frames)
        #: The trigger-time snapshot of the ring, taken under lock so the clip is the
        #: last N seconds *ending at the request*, not the next completed chunk.
        self._pending_frames: list = []
        self._pending_frames2: list = []

        if not HAS_ZED_SDK:
            print("[ZED] pyzed not installed. Operating in OFFLINE / STUB mode.")
            self.zed1 = None
            self.zed2 = None
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.zed1 = sl.Camera()
        self.zed2 = sl.Camera()
        self.active_slot = 0
        self.save_requested = False
        self.save_prefix = ""

    def _init_cam(self, zed: sl.Camera, serial: int | None = None) -> None:
        init_p = sl.InitParameters()
        init_p.camera_resolution = _sl_resolution()  # SVGA 1200x720 (rig-proven)
        init_p.camera_fps = self.fps
        init_p.depth_mode = sl.DEPTH_MODE.NONE
        init_p.coordinate_units = sl.UNIT.METER

        if serial:
            init_p.set_from_serial_number(serial)

        status = zed.open(init_p)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED X ({serial}): {status}")

    def _enable_lossless_svo(self, zed: sl.Camera, path: Path) -> None:
        # LOSSLESS only — Orin Nano has no NVENC; H264/H265 are forbidden.
        rec_p = sl.RecordingParameters(
            video_filename=str(path),
            compression_mode=sl.SVO_COMPRESSION_MODE.LOSSLESS,
        )
        zed.enable_recording(rec_p)

    def _governor_allows(self) -> bool:
        return self.governor is None or self.governor.allow_capture()

    def set_capture_seconds(self, seconds: int) -> None:
        """Live-adjust the rolling buffer depth (the last-N window), in seconds."""
        seconds = int(seconds)
        if seconds <= 0:
            raise ValueError(f"capture seconds must be positive, got {seconds}")
        self.chunk_seconds = seconds
        self.chunk_frames = int(seconds * self.fps)
        with self.lock:
            self._ring = collections.deque(self._ring, maxlen=self.chunk_frames)
            self._ring2 = collections.deque(self._ring2, maxlen=self.chunk_frames)

    def start(self, cam1_sn: int | None = None, cam2_sn: int | None = None) -> bool:
        """Starts live capture. Returns True if successful, False if offline/failed."""
        if not HAS_ZED_SDK:
            print("[ZED] Cannot start live stream: pyzed library is not available.")
            return False

        try:
            print(f"[ZED] Initializing Dual ZED X cameras at SVGA 1200x720 @ {self.fps} FPS...")
            devs = sl.Camera.get_device_list()
            sn1 = cam1_sn or (devs[0].serial_number if len(devs) > 0 else None)
            sn2 = cam2_sn or (devs[1].serial_number if len(devs) > 1 else None)

            self._init_cam(self.zed1, sn1)
            time.sleep(2.0)
            self._init_cam(self.zed2, sn2)
            print(f"[ZED] Camera 1 (SN: {sn1}) & Camera 2 (SN: {sn2}) online.")

            self.running = True
            self.thread = threading.Thread(target=self._record_worker, daemon=True)
            self.thread.start()
            print(f"[ZED] Rolling {self.chunk_seconds}-second LOSSLESS SVO2 cache active.")
            return True
        except Exception as e:
            print(f"[ZED] Failed to initialize physical cameras (Offline mode): {e}")
            self.running = False
            return False

    def _record_worker(self) -> None:
        runtime_p = sl.RuntimeParameters()
        img1 = sl.Mat()
        img2 = sl.Mat()

        while self.running:
            # Backpressure: yield to scoring compute while suspended/overloaded.
            while self.running and not self._governor_allows():
                time.sleep(0.05)
            if not self.running:
                break

            slot = self.active_slot
            c1_svo = self.temp_dir / f"chunk_{slot}_cam1.svo2"
            c2_svo = self.temp_dir / f"chunk_{slot}_cam2.svo2"

            self._enable_lossless_svo(self.zed1, c1_svo)
            self._enable_lossless_svo(self.zed2, c2_svo)

            frame_count = 0
            while frame_count < self.chunk_frames and self.running:
                if not self._governor_allows():
                    time.sleep(0.05)
                    continue
                if (
                    self.zed1.grab(runtime_p) == sl.ERROR_CODE.SUCCESS
                    and self.zed2.grab(runtime_p) == sl.ERROR_CODE.SUCCESS
                ):
                    frame_count += 1
                    self.zed1.retrieve_image(img1, sl.VIEW.LEFT)
                    self.zed2.retrieve_image(img2, sl.VIEW.LEFT)
                    with self.lock:
                        self._ring.append(cv2.cvtColor(img1.get_data(), cv2.COLOR_BGRA2BGR))
                        self._ring2.append(cv2.cvtColor(img2.get_data(), cv2.COLOR_BGRA2BGR))

            self.zed1.disable_recording()
            self.zed2.disable_recording()

            with self.lock:
                if self.save_requested:
                    self._commit_jump_data(slot, self.save_prefix)
                    self.save_requested = False

            self.active_slot = 1 if self.active_slot == 0 else 0

    def _commit_jump_data(self, slot: int, prefix: str) -> None:
        c1_src = self.temp_dir / f"chunk_{slot}_cam1.svo2"
        c2_src = self.temp_dir / f"chunk_{slot}_cam2.svo2"
        dst_c1 = self.output_dir / f"{prefix}_cam1.svo2"
        dst_c2 = self.output_dir / f"{prefix}_cam2.svo2"

        if c1_src.exists():
            shutil.copy2(c1_src, dst_c1)
        if c2_src.exists():
            shutil.copy2(c2_src, dst_c2)

        # The pipeline clips come from the trigger-time ring snapshots — the last N
        # seconds ending at the request, independent of which SVO2 chunk is current.
        # One clip per camera (step 2c): cam1 -> "{prefix}_right.mp4", cam2 -> "{prefix}_left.mp4".
        frames1 = self._pending_frames if self._pending_frames else list(self._ring)
        frames2 = self._pending_frames2 if self._pending_frames2 else list(self._ring2)

        right_path = self.clips_dir / clip_name(prefix, CLIP_CAM1_SUFFIX)
        left_path = self.clips_dir / clip_name(prefix, CLIP_CAM2_SUFFIX)
        _write_mp4(right_path, frames1, self.fps)
        _write_mp4(left_path, frames2, self.fps)

        self.last_saved_mp4 = right_path
        self.last_saved_clips = {
            CLIP_CAM1_SUFFIX: right_path,
            CLIP_CAM2_SUFFIX: left_path,
        }
        print(
            f"[ZED] Saved SVO2 pair & pipeline clips: "
            f"{right_path.name} (cam1) + {left_path.name} (cam2)"
        )

    def trigger_capture(self, prefix: str) -> Path | None:
        if not self.running:
            print("[ZED] Capture unavailable: camera buffer is not running.")
            return None

        with self.lock:
            # Snapshot both rings AT the trigger so the clips are the last N seconds ending
            # now — not whatever the next completed chunk happens to contain.
            self._pending_frames = list(self._ring)
            self._pending_frames2 = list(self._ring2)
            self.save_prefix = prefix
            self.save_requested = True

        for _ in range(60):
            time.sleep(0.1)
            with self.lock:
                if not self.save_requested and self.last_saved_mp4:
                    return self.last_saved_mp4
        return None

    def stop(self) -> None:
        self.running = False
        if hasattr(self, "thread") and self.thread:
            self.thread.join(timeout=2.0)
        if self.zed1:
            self.zed1.close()
        if self.zed2:
            self.zed2.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


class FakeZEDRecorder(ZEDRecorder):
    """Offline stand-in for ``DualZEDXLiveBuffer``.

    Simulates a dual-camera rolling SVO2 buffer with synthetic frames so the
    recorder state machine — ring wrap, save-on-request, governor backpressure,
    and MP4 export — is testable without pyzed or hardware.

    The ``.svo2`` outputs are JSON metadata stubs (not real SVO2); they carry the
    synthetic capture metadata (fps / resolution / compression / frame count) so
    tests can assert the recording parameters end-to-end.
    """

    DEFAULT_FPS = CAPTURE_FPS
    DEFAULT_CHUNK_SECONDS = 10
    DEFAULT_RESOLUTION = CAPTURE_RESOLUTION
    DEFAULT_COMPRESSION = CAPTURE_COMPRESSION

    def __init__(
        self,
        output_dir: str | Path = "data/svo_recordings",
        clips_dir: str | Path = "data/clips",
        chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
        fps: int = CAPTURE_FPS,
        width: int = CAPTURE_WIDTH,
        height: int = CAPTURE_HEIGHT,
        cam_serials: tuple = (42250580, 40847387),
        governor: ComputeGovernor | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.clips_dir = Path(clips_dir)
        self.temp_dir = self.output_dir / "temp_rolling_buffer"
        self.chunk_seconds = chunk_seconds
        self.fps = fps
        self.width = width
        self.height = height
        self.cam_serials = cam_serials
        self.chunk_frames = int(chunk_seconds * fps)
        self.governor = governor

        self.running = False
        self.lock = threading.Lock()
        self.last_saved_mp4: Path | None = None
        #: The saved clip pair, keyed by camera role: {"right": cam1 clip, "left": cam2 clip}.
        self.last_saved_clips: dict[str, Path] = {}
        self.frames_written = 0
        self._frame_seq = 0
        #: Ring buffer of the most recent ``chunk_frames`` synthetic frames — the true
        #: "save the last N seconds" window that survives slot (wrap) boundaries.
        self._ring: collections.deque = collections.deque(maxlen=self.chunk_frames)
        #: Second ring for cam2 (step 2c) — filled in lock step with ``_ring``.
        self._ring2: collections.deque = collections.deque(maxlen=self.chunk_frames)
        self._started = threading.Event()

    @staticmethod
    def _synthetic_frame(i: int, width: int, height: int, cam: int = 1) -> np.ndarray:
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = (i * 37) % max(1, width - 40)
        y = (i * 23) % max(1, height - 40)
        if cam == 1:
            frame[y : y + 40, x : x + 40] = (0, 200, 255)  # BGR bright yellow marker
        else:
            frame[y : y + 40, x : x + 40] = (255, 120, 0)  # BGR blue marker — distinct from cam1
        return frame

    def _governor_allows(self) -> bool:
        return self.governor is None or self.governor.allow_capture()

    @property
    def ring_frames(self) -> int:
        """Number of frames currently in the ring (0..chunk_frames)."""
        with self.lock:
            return len(self._ring)

    @property
    def ring2_frames(self) -> int:
        """Number of cam2 frames currently in the second ring (0..chunk_frames)."""
        with self.lock:
            return len(self._ring2)

    @property
    def ring_full(self) -> bool:
        """True once the ring holds a full last-N window."""
        return self.ring_frames >= self.chunk_frames

    def set_capture_seconds(self, seconds: int) -> None:
        """Live-adjust the rolling buffer depth (the last-N window), in seconds."""
        seconds = int(seconds)
        if seconds <= 0:
            raise ValueError(f"capture seconds must be positive, got {seconds}")
        self.chunk_seconds = seconds
        self.chunk_frames = int(seconds * self.fps)
        with self.lock:
            self._ring = collections.deque(self._ring, maxlen=self.chunk_frames)
            self._ring2 = collections.deque(self._ring2, maxlen=self.chunk_frames)

    def start(self, cam1_sn: int | None = None, cam2_sn: int | None = None) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.running = True
        with self.lock:
            self._ring.clear()
            self.frames_written = 0
            self._frame_seq = 0
        self.thread = threading.Thread(target=self._record_worker, daemon=True)
        self.thread.start()
        return True

    def _record_worker(self) -> None:
        self._started.set()
        while self.running:
            while self.running and not self._governor_allows():
                time.sleep(0.02)
            if not self.running:
                break

            frame1 = self._synthetic_frame(self._frame_seq, self.width, self.height, cam=1)
            frame2 = self._synthetic_frame(self._frame_seq, self.width, self.height, cam=2)
            self._frame_seq += 1
            with self.lock:
                self._ring.append(frame1)
                self._ring2.append(frame2)
                self.frames_written += 1

    def _commit_jump_data(self, prefix: str, frames1: list, frames2: list) -> Path:
        dst_c1 = self.output_dir / f"{prefix}_cam1.svo2"
        dst_c2 = self.output_dir / f"{prefix}_cam2.svo2"

        for path, serial, n in zip((dst_c1, dst_c2), self.cam_serials, (len(frames1), len(frames2))):
            meta = {
                "format": "svo2",
                "synthetic": True,
                "serial": serial,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "compression": CAPTURE_COMPRESSION,
                "frames": n,
            }
            path.write_text(json.dumps(meta))

        # One pipeline clip per camera (step 2c): cam1 -> "{prefix}_right.mp4",
        # cam2 -> "{prefix}_left.mp4".  No bare "{prefix}.mp4" is written anymore.
        right_path = self.clips_dir / clip_name(prefix, CLIP_CAM1_SUFFIX)
        left_path = self.clips_dir / clip_name(prefix, CLIP_CAM2_SUFFIX)
        _write_mp4(right_path, frames1, self.fps)
        _write_mp4(left_path, frames2, self.fps)

        self.last_saved_mp4 = right_path
        self.last_saved_clips = {
            CLIP_CAM1_SUFFIX: right_path,
            CLIP_CAM2_SUFFIX: left_path,
        }
        return right_path

    def trigger_capture(self, prefix: str) -> Path | None:
        if not self.running:
            return None

        # Wait (bounded) for the ring to fill so the save is a full last-N window rather
        # than whatever partial buffer happens to exist at the request.
        deadline = time.monotonic() + max(2.0, self.chunk_seconds + 1.0)
        while not self.ring_full and time.monotonic() < deadline:
            time.sleep(0.01)

        with self.lock:
            frames1 = list(self._ring)
            frames2 = list(self._ring2)
        return self._commit_jump_data(prefix, frames1, frames2)

    def stop(self) -> None:
        self.running = False
        if hasattr(self, "thread") and self.thread:
            self.thread.join(timeout=2.0)
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
