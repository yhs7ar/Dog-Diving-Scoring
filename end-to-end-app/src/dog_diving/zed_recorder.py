"""
zed_recorder.py — Dual ZED X Rolling SVO2 & Clip Buffer (rig-proven config)

Capture standard (see scripts/dual_recording/WORKAROUNDS_AND_LIMITATIONS.md):

  * Resolution: **SVGA (1200x720)** @ **30 FPS** — the only dual-camera config proven
    on the Jetson Orin Nano (HD1080@30, VGA, and 15 FPS all FAIL on the rig).
  * Compression: **LOSSLESS** only. Orin Nano has NO NVENC, so H264/H265 crash the
    daemon / corrupt SVO2.
  * Depth: captured **NONE** and computed off-line from the recorded SVO2.

The module also ships two testability aids so the recorder state machine is
verifiable offline (no pyzed, no hardware):

  * ``ZEDRecorder`` — the abstract capture seam the app depends on.
  * ``FakeZEDRecorder`` — a synthetic-camera stand-in that mirrors the hardware
    recorder's public API (rolling slot buffer, save-on-request, MP4 export).
  * ``ComputeGovernor`` — gates capture while a heavy stage (detect/measure) runs,
    plus an optional psutil-based CPU load watch.
"""

from __future__ import annotations

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


def _sl_resolution():
    """Return the sl.RESOLUTION enum for ``CAPTURE_RESOLUTION``, or None offline."""
    if not HAS_ZED_SDK:
        return None
    return getattr(sl.RESOLUTION, CAPTURE_RESOLUTION, sl.RESOLUTION.SVGA)


class ZEDRecorder(ABC):
    """The capture seam the app depends on.

    Concrete backends: ``DualZEDXLiveBuffer`` (hardware) and ``FakeZEDRecorder``
    (offline). The informal contract also includes the ``running`` attribute and
    the ``last_saved_mp4`` attribute (set by ``trigger_capture``).
    """

    @abstractmethod
    def start(self, cam1_sn: int | None = None, cam2_sn: int | None = None) -> bool:
        """Start live capture. True on success, False when offline/failed."""

    @abstractmethod
    def trigger_capture(self, prefix: str) -> Path | None:
        """Freeze the current rolling chunk to disk; return the MP4 path or None."""

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
    save is requested. Yields to scoring compute via an optional ``governor``.
    """

    DEFAULT_FPS = CAPTURE_FPS
    DEFAULT_CHUNK_SECONDS = 5
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
        self.chunk_frames = chunk_seconds * fps
        self.governor = governor

        self.running = False
        self.lock = threading.Lock()
        self.last_saved_mp4: Path | None = None

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

            recent_frames_cam1 = []

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
                    recent_frames_cam1.append(cv2.cvtColor(img1.get_data(), cv2.COLOR_BGRA2BGR))

            self.zed1.disable_recording()
            self.zed2.disable_recording()

            with self.lock:
                if self.save_requested:
                    self._commit_jump_data(slot, self.save_prefix, recent_frames_cam1)
                    self.save_requested = False

            self.active_slot = 1 if self.active_slot == 0 else 0

    def _commit_jump_data(self, slot: int, prefix: str, frames: list) -> None:
        c1_src = self.temp_dir / f"chunk_{slot}_cam1.svo2"
        c2_src = self.temp_dir / f"chunk_{slot}_cam2.svo2"
        dst_c1 = self.output_dir / f"{prefix}_cam1.svo2"
        dst_c2 = self.output_dir / f"{prefix}_cam2.svo2"

        if c1_src.exists():
            shutil.copy2(c1_src, dst_c1)
        if c2_src.exists():
            shutil.copy2(c2_src, dst_c2)

        mp4_path = self.clips_dir / f"{prefix}.mp4"
        if frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(mp4_path), fourcc, self.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

        self.last_saved_mp4 = mp4_path
        print(f"[ZED] Saved SVO2 pair & pipeline clip: {mp4_path.name}")

    def trigger_capture(self, prefix: str) -> Path | None:
        if not self.running:
            print("[ZED] Capture unavailable: camera buffer is not running.")
            return None

        with self.lock:
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
    recorder state machine — slot rotation, save-on-request, governor
    backpressure, and MP4 export — is testable without pyzed or hardware.

    The ``.svo2`` outputs are JSON metadata stubs (not real SVO2); they carry the
    synthetic capture metadata (fps / resolution / compression / frame count) so
    tests can assert the recording parameters end-to-end.
    """

    DEFAULT_FPS = CAPTURE_FPS
    DEFAULT_RESOLUTION = CAPTURE_RESOLUTION
    DEFAULT_COMPRESSION = CAPTURE_COMPRESSION

    def __init__(
        self,
        output_dir: str | Path = "data/svo_recordings",
        clips_dir: str | Path = "data/clips",
        chunk_seconds: int = 5,
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
        self.chunk_frames = chunk_seconds * fps
        self.governor = governor

        self.running = False
        self.lock = threading.Lock()
        self.last_saved_mp4: Path | None = None
        self.frames_written = 0
        self.slots_completed = 0
        self._started = threading.Event()

    @staticmethod
    def _synthetic_frame(i: int, width: int, height: int) -> np.ndarray:
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = (i * 37) % max(1, width - 40)
        y = (i * 23) % max(1, height - 40)
        frame[y : y + 40, x : x + 40] = (0, 200, 255)  # BGR bright marker
        return frame

    def _governor_allows(self) -> bool:
        return self.governor is None or self.governor.allow_capture()

    def start(self, cam1_sn: int | None = None, cam2_sn: int | None = None) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.active_slot = 0
        self.save_requested = False
        self.save_prefix = ""
        self.running = True
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

            slot = self.active_slot
            frames = []
            frame_count = 0
            while frame_count < self.chunk_frames and self.running:
                if not self._governor_allows():
                    time.sleep(0.02)
                    continue
                frames.append(self._synthetic_frame(frame_count, self.width, self.height))
                frame_count += 1
                with self.lock:
                    self.frames_written += 1

            with self.lock:
                self.slots_completed += 1
                if self.save_requested:
                    self._commit_jump_data(slot, self.save_prefix, frames)
                    self.save_requested = False

            self.active_slot = 1 if self.active_slot == 0 else 0

    def _commit_jump_data(self, slot: int, prefix: str, frames: list) -> None:
        dst_c1 = self.output_dir / f"{prefix}_cam1.svo2"
        dst_c2 = self.output_dir / f"{prefix}_cam2.svo2"

        for path, serial in zip((dst_c1, dst_c2), self.cam_serials):
            meta = {
                "format": "svo2",
                "synthetic": True,
                "serial": serial,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "compression": CAPTURE_COMPRESSION,
                "frames": len(frames),
            }
            path.write_text(json.dumps(meta))

        mp4_path = self.clips_dir / f"{prefix}.mp4"
        if frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(mp4_path), fourcc, self.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

        self.last_saved_mp4 = mp4_path

    def trigger_capture(self, prefix: str) -> Path | None:
        if not self.running:
            return None

        with self.lock:
            self.save_prefix = prefix
            self.save_requested = True

        for _ in range(200):
            time.sleep(0.01)
            with self.lock:
                if not self.save_requested and self.last_saved_mp4:
                    return self.last_saved_mp4
        return None

    def stop(self) -> None:
        self.running = False
        if hasattr(self, "thread") and self.thread:
            self.thread.join(timeout=2.0)
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
