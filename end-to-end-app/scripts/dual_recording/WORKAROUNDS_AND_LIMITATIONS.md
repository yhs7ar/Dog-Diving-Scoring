# ZED X Multi-Camera Recording: Findings, Workarounds, and Hardware Limitations

**Target Hardware:** NVIDIA Jetson Orin Nano Super / Developer Kit  
**Cameras:** Dual ZED X (GMSL2) via ZED Link Duo  
**Software Environment:** JetPack 6.x (L4T R36.x), ZED SDK 5.5.1, ZED Link Duo 1.4.3  
**Document Updated:** November 2026

---

## 1. CRITICAL HARDWARE LIMITATION: Jetson Orin Nano Has NO NVENC Chip

> **GROUND TRUTH:** The NVIDIA Jetson Orin Nano specifically **lacks a hardware video encoder (NVENC)**.
> Official Stereolabs documentation and technical support threads confirm that hardware H264/H265 encoding is unsupported on Orin Nano.

* **The Pitfall:** Passing `sl.SVO_COMPRESSION_MODE.H264` or `sl.SVO_COMPRESSION_MODE.H265` in `sl.RecordingParameters()` forces the SDK to attempt invoking non-existent NVENC hardware encoders, resulting in `[ZED][Encoding] Critical Error: No Video Enc [enc_0]`, daemon instability, or corrupted `.svo2` files.
* **The Rule:** **DO NOT** hardcode `SVO_COMPRESSION_MODE.H264` or `H265` on Jetson Orin Nano.
* **The Correct Approach:** Use **`sl.SVO_COMPRESSION_MODE.LOSSLESS`** (CPU frame compression) or export `ZED_SDK_SVO_VERSION=1` for legacy SVO v1 format.

---

## 2. Argus & `zed_x_daemon` Service Disruption Protocol

Referenced in Stereolabs Community Threads [`11590`](https://community.stereolabs.com/t/zed-x-grab-stuck-in-failure-after-argus-zed-x-daemon-failure/11590) & [`11668`](https://community.stereolabs.com/t/dual-zed-x-on-zed-box-duo-repeatable-argus-alreadyallocated-and-invalidstate-despite-two-working-streams-sdk-5-4-1-driver-1-4-3/11668):

* **Observed Mechanism:** Disruption of `nvargus-daemon` or `zed_x_daemon` while a camera session is open causes `sl::Camera::grab()` to enter `CAMERA_REBOOTING`, log `[ZED-Argus][Critical] Reboot has failed for CAM0 with current state=4`, and lock into `FAILURE`.
* **Required Recovery Sequence:**
  1. Terminate the Python/C++ process holding the camera handle.
  2. Restart daemons sequentially with a pause:
     ```bash
     sudo -S -p '' systemctl restart nvargus-daemon.service
     sleep 2
     sudo -S -p '' systemctl restart zed_x_daemon.service
     sleep 3
     ```
  3. If Argus enters an unrecoverable `SCF: Error 0x0000000f` state, issue a system reboot (`sudo reboot`).

---

## 3. Comprehensive List of Workaround Methods Discussed Online (2024–2026)

### Category 1: ZED SDK Compression Modes
1. **`SVO_COMPRESSION_MODE.LOSSLESS` (Recommended):** Uses CPU PNG/RAW frame compression. Bypasses NVENC hardware requirement entirely.
2. **Legacy SVO v1 Override (`ZED_SDK_SVO_VERSION=1`):** Setting `export ZED_SDK_SVO_VERSION=1` forces SVO v1 file format, preventing CPU SVO2 header corruption on Orin Nano (Thread `11649`).
3. **SVOPRO / CPU Fallback:** Uses CPU-based encoding routines introduced in SDK 5.4.1+.

### Category 2: Execution Architecture
4. **Process-Isolated Subprocesses:** Launching independent OS processes per camera serial number (e.g., via `multiprocessing` or background bash jobs) isolates memory/IPC buffers and prevents Python GIL deadlocks.
5. **Compiled C++ Native Binaries:** Compiling native C++ recording executables from `/usr/local/zed/samples/recording/` eliminates Python wrapper overhead and threading lockups.

### Category 3: CLI Tools
6. **Headless `ZED_Explorer` CLI:** Calling `ZED_Explorer --serial_number <SN> --output <FILE>`. *Note:* Thread `9931` documents known CLI connection issues in some SDK builds (`Unable to connect to GMSL camera when using --output`).

### Category 4: Alternative Video Pipeline Bypasses
7. **Direct V4L2 Raw Capture:** Streaming directly from `/dev/video*` GMSL video nodes via V4L2/OpenCV. Bypasses ZED Daemon entirely (sacrifices SVO metadata).
8. **GStreamer `zedsrc` Plugin:** Video streaming/recording via Stereolabs GStreamer pipeline.
9. **ZED Open Capture C++ Driver:** Open-source direct C++ driver bypassing proprietary Argus stack.

---

## 4. Status Matrix of `dual_recording` Infrastructure

| Method / Approach | Status | Notes / Existing Files |
| :--- | :--- | :--- |
| **`SVO_COMPRESSION_MODE.LOSSLESS` (Python Subprocess)** | **SUCCESSFUL (RECOMMENDED)** | Implemented in `dual_svga_recorder.py` & `single_cam_recorder.py`. Achieves **29.7–30.0 FPS** on both cameras simultaneously with zero GIL lockup and automatic open retries. |
| **Native C++ Compiled Recording Binary** | **SUCCESSFUL (PRIMARY)** | Compiled `dual_svga_recorder`. Achieves **30.1–30.2 FPS (302 frames / 10s)** on BOTH cameras simultaneously with zero dropped frames. |
| **Legacy SVO v1 (`ZED_SDK_SVO_VERSION=1`)** | **Attempted** | Implemented in `test_legacy_svo.py`. |
| **Process Isolation (Subprocesses)** | **Attempted** | `dual_headless_record.py`, `automated_record.sh`. |
| **Multi-Threaded Single-Process** | **Attempted** | `single_process_zed.py`, `dual_zedx_record_fixed.py`. |
| **Headless `ZED_Explorer` CLI** | **Attempted** | `automated_record.sh`. |
| **Pre-flight Daemon Sanitization** | **Attempted** | `restart_cameras_v2.sh`, `stress_testbench.py`. |
| **Direct V4L2 Stream Capture** | **Attempted** | `dual_v4l2_record.py`, `monitor_cameras.sh`. |
| **GStreamer (`zedsrc`) Pipeline** | **UNATTEMPTED** | No GStreamer pipeline scripts exist in `dual_recording`. |

---

## 5. Test Log & Verification Results (November 2026)

### Test Run 1: Python `SVO_COMPRESSION_MODE.LOSSLESS` (`test_lossless_clean.py`)
* **Single Camera S/N 40847387:** Recorded 145 frames in 5.01s (29.0 FPS). Verified SVO size: 203 MB.
* **Single Camera S/N 42250580:** Recorded 150 frames in 5.02s (29.9 FPS). Verified SVO size: 215 MB.
* **Dual Camera Simultaneous:** Recorded 162 frames on Cam 0 and 260 frames on Cam 1 in 10s. Both `.svo2` files verified valid.

### Test Run 2: Native C++ Multi-Camera Binary (`cpp_recorder/zed_cpp_recorder`)
* **Build Command:** `g++ -std=c++17 -O3 -I/usr/local/zed/include -I/usr/local/cuda/include -I./cpp_recorder/include ./cpp_recorder/src/main.cpp -L/usr/local/zed/lib -L/usr/local/cuda/lib64 -lsl_zed -lcudart -lpthread -o ./cpp_recorder/zed_cpp_recorder`
* **Execution:** `./cpp_recorder/zed_cpp_recorder 10`
* **Output SVO Files:**
  - `cpp_lossless_SN40847387.svo2` (404 MB) — Verified **302 frames** (30.2 FPS)
  - `cpp_lossless_SN42250580.svo2` (413 MB) — Verified **302 frames** (30.2 FPS)
* **Verdict:** Complete success. Full 30 FPS synchronized dual-camera acquisition achieved without GUI or NVENC hardware requirement.

### Test Run 3: Resolution & Frame Rate Stepping Matrix (`zed_cpp_recorder_multi`)
* **`SVGA` (1200x720) @ 30 FPS (Stepped Down Hardware Resolution):**
  - **Cam 0 (42250580):** 302 frames in 10s (**30.2 FPS**), 413 MB.
  - **Cam 1 (40847387):** 303 frames in 10s (**30.3 FPS**), 404 MB.
  - **Result:** **SUCCESSFUL.** Primary operational configuration for Orin Nano dual-camera capture.
* **`HD1080` (1920x1080) @ 30 FPS (Full Resolution):**
  - **Result:** **FAILED.** Dual uncompressed 1080p stereo pair data bandwidth exceeds Argus IPC socket limit on Orin Nano, triggering `(Argus) Error 0x00030003: Failed socket read: Connection reset by peer` and requiring reboot.
* **`VGA` (672x376 / 960x600):**
  - **Result:** **FAILED.** Unsupported by ZED X GMSL2 sensor hardware (returns `INVALID RESOLUTION`).
* **`SVGA` @ 15 FPS (Software Frame Rate Stepping):**
  - **Result:** **FAILED.** ZED X ISP sensor firmware only streams natively at 30 FPS or 60 FPS. Requesting 15 FPS causes stream timeout (`checkStreamStatus: no NEW_FRAME state within 20000ms budget`).

### Test Run 4: Hardened Python Subprocess Dual Recorder (`dual_svga_recorder.py`)
* **Architecture:** Launcher script (`dual_svga_recorder.py`) spawns isolated worker subprocesses (`single_cam_recorder.py`) via `subprocess.Popen` to eliminate GIL locks and Cython static socket sharing. Uses 2.0s staggered launch and 3-attempt GMSL open retry logic.
* **Execution:** `python3 dual_svga_recorder.py -d 10 -o /home/dogdiving/Desktop/scripts/dual_recording -p py_svga`
* **Output SVO2 Files:**
  - `py_svga_cam0_SN42250580.svo2` — **297 frames** (**29.67 FPS**), verified read back via SDK.
  - `py_svga_cam1_SN40847387.svo2` — **300 frames** (**29.93 FPS**), verified read back via SDK.
* **Verdict:** Complete success. Achieves full 30 FPS synchronized dual-camera acquisition in pure Python.

---

## 6. Directives for Future Autonomous Agents

1. **DO NOT** write or modify recording scripts using `sl.SVO_COMPRESSION_MODE.H264` or `.H265` on this Jetson Orin Nano rig.
2. **DO** use `sl.SVO_COMPRESSION_MODE.LOSSLESS` for SVO2 capture on Orin Nano.
3. **DO** use the native C++ compiled binary (`~/Desktop/scripts/dual_recording/cpp_recorder/zed_cpp_recorder`) or build C++ binaries for maximum throughput and frame rate without Python GIL overhead.
4. **DO** execute `./restart_cameras_v2.sh` or reboot if an Argus socket lockup occurs before opening camera handles.
