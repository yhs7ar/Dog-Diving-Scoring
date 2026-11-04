#!/usr/bin/env python3
"""
======================================================================================
SINGLE ZED X CAMERA SVO2 WORKER (PYTHON PROCESS WORKER)
======================================================================================
Target Platform: NVIDIA Jetson Orin Nano / Orin Developer Kit
Camera: Single Stereolabs ZED X (GMSL2 interface)
SDK Version: Stereolabs pyzed SDK 5.x / L4T R36.x (JetPack 6.x)

HARDWARE CONSTRAINTS & DESIGN RATIONALE:
1. NO NVENC ENCODER: Hardcoded sl.SVO_COMPRESSION_MODE.LOSSLESS (CPU PNG/RAW frame compression).
2. HARDWARE RESOLUTION: SVGA (1200x720) @ 30 FPS native sensor rate.
3. CLEAN SIGNAL HANDLING: Traps SIGINT/SIGTERM to guarantee zed.disable_recording() and
   zed.close() are called before exit, protecting SVO2 file headers from corruption.
======================================================================================
"""

import argparse
import json
import signal
import sys
import time

# Deferred import flag
import pyzed.sl as sl

stop_recording = False

def signal_handler(signum, frame):
    global stop_recording
    print(f"[WORKER {sys.argv}] Signal {signum} caught. Stopping recording...", file=sys.stderr)
    stop_recording = True

def main():
    parser = argparse.ArgumentParser(description="Single ZED X Camera SVO2 Worker")
    parser.add_argument("--serial", type=int, required=True, help="Camera serial number")
    parser.add_argument("--output", type=str, required=True, help="Output SVO2 file path")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds (0 = continuous)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[CAM {args.serial}] Initializing camera handle...")
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.SVGA # 1200x720 native SVGA
    init_params.camera_fps = 30                        # 30 FPS native sensor rate
    init_params.depth_mode = sl.DEPTH_MODE.NONE       # Disable depth for maximum throughput
    init_params.input.set_from_serial_number(args.serial)

    # 3-attempt open retry loop for transient GMSL setup states
    opened = False
    max_retries = 3
    for attempt in range(max_retries):
        err = zed.open(init_params)
        if err == sl.ERROR_CODE.SUCCESS:
            opened = True
            break
        print(f"[WARNING] S/N {args.serial} open attempt {attempt+1}/{max_retries} failed: {err}. Retrying in 2.0s...", file=sys.stderr)
        time.sleep(2.0)

    if not opened:
        print(f"[ERROR] Could not open Camera S/N {args.serial} after {max_retries} attempts.", file=sys.stderr)
        result = {"serial": args.serial, "frames": 0, "fps": 0.0, "success": False, "error": "Open failed"}
        print(json.dumps(result))
        sys.exit(1)

    recording_params = sl.RecordingParameters()
    recording_params.video_filename = args.output
    recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.LOSSLESS # CPU PNG/RAW mode for Orin Nano

    err = zed.enable_recording(recording_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to enable LOSSLESS recording on S/N {args.serial}: {err}", file=sys.stderr)
        zed.close()
        result = {"serial": args.serial, "frames": 0, "fps": 0.0, "success": False, "error": str(err)}
        print(json.dumps(result))
        sys.exit(1)

    print(f"[SUCCESS] Camera S/N {args.serial} recording active -> {args.output}")

    frames_captured = 0
    start_time = time.time()
    runtime_params = sl.RuntimeParameters()

    while not stop_recording:
        elapsed = time.time() - start_time
        if args.duration > 0 and elapsed >= args.duration:
            break

        grab_err = zed.grab(runtime_params)
        if grab_err == sl.ERROR_CODE.SUCCESS:
            frames_captured += 1
        else:
            time.sleep(0.001)

    total_time = time.time() - start_time
    print(f"[SHUTDOWN] Closing S/N {args.serial} (Captured {frames_captured} frames in {total_time:.2f}s)...")
    
    zed.disable_recording()
    zed.close()
    
    fps = frames_captured / total_time if total_time > 0 else 0.0
    result = {"serial": args.serial, "frames": frames_captured, "fps": fps, "success": True, "output": args.output}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
