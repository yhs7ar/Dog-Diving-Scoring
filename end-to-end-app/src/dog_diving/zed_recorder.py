"""
zed_recorder.py — Dual ZED X 5-Second Rolling SVO2 & Clip Buffer
Includes graceful fallback for Windows/offline development without pyzed.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
import cv2

# Safe ZED SDK import
try:
    import pyzed.sl as sl
    HAS_ZED_SDK = True
except ImportError:
    sl = None
    HAS_ZED_SDK = False


class DualZEDXLiveBuffer:
    def __init__(
        self,
        output_dir: str | Path = "data/svo_recordings",
        clips_dir: str | Path = "data/clips",
        chunk_seconds: int = 5,
        fps: int = 60,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.clips_dir = Path(clips_dir)
        self.temp_dir = self.output_dir / "temp_rolling_buffer"
        
        self.chunk_seconds = chunk_seconds
        self.fps = fps
        self.chunk_frames = chunk_seconds * fps

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
        init_p.camera_resolution = sl.RESOLUTION.HD720
        init_p.camera_fps = self.fps
        init_p.depth_mode = sl.DEPTH_MODE.NONE
        init_p.coordinate_units = sl.UNIT.METER

        if serial:
            init_p.set_from_serial_number(serial)

        status = zed.open(init_p)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED X ({serial}): {status}")

    def _enable_lossless_svo(self, zed: sl.Camera, path: Path) -> None:
        rec_p = sl.RecordingParameters(
            video_filename=str(path),
            compression_mode=sl.SVO_COMPRESSION_MODE.LOSSLESS
        )
        zed.enable_recording(rec_p)

    def start(self, cam1_sn: int | None = None, cam2_sn: int | None = None) -> bool:
        """Starts live capture. Returns True if successful, False if offline/failed."""
        if not HAS_ZED_SDK:
            print("[ZED] Cannot start live stream: pyzed library is not available.")
            return False

        try:
            print("[ZED] Initializing Dual ZED X cameras at 720p @ 60 FPS...")
            devs = sl.Camera.get_device_list()
            sn1 = cam1_sn or (devs[0].serial_number if len(devs) > 0 else None)
            sn2 = cam2_sn or (devs[1].serial_number if len(devs) > 1 else None)

            self._init_cam(self.zed1, sn1)
            self._init_cam(self.zed2, sn2)
            print(f"[ZED] Camera 1 (SN: {sn1}) & Camera 2 (SN: {sn2}) online.")

            self.running = True
            self.thread = threading.Thread(target=self._record_worker, daemon=True)
            self.thread.start()
            print("[ZED] Rolling 5-second SVO2 cache active.")
            return True
        except Exception as e:
            print(f"[ZED] Failed to initialize physical cameras (Offline mode): {e}")
            self.running = False
            return False

    def _record_worker(self) -> None:
        runtime_p = sl.RuntimeParameters()
        img1 = sl.Mat()

        while self.running:
            slot = self.active_slot
            c1_svo = self.temp_dir / f"chunk_{slot}_cam1.svo2"
            c2_svo = self.temp_dir / f"chunk_{slot}_cam2.svo2"

            self._enable_lossless_svo(self.zed1, c1_svo)
            self._enable_lossless_svo(self.zed2, c2_svo)

            recent_frames_cam1 = []

            frame_count = 0
            while frame_count < self.chunk_frames and self.running:
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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=2.0)
        if self.zed1:
            self.zed1.close()
        if self.zed2:
            self.zed2.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)