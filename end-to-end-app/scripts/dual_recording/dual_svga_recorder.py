#!/usr/bin/env python3
"""
======================================================================================
HARDENED ZED X DUAL SVGA SVO2 RECORDER (PYTHON SUBPROCESS LAUNCHER)
======================================================================================
Target Platform: NVIDIA Jetson Orin Nano / Orin Developer Kit
Cameras: Dual Stereolabs ZED X (GMSL2 interface via ZED Link Duo)
SDK Version: Stereolabs pyzed SDK 5.x / L4T R36.x (JetPack 6.x)

CORE ARCHITECTURE:
1. SUBPROCESS ISOLATION: Launches independent single_cam_recorder.py Python processes
   via subprocess.Popen. This guarantees distinct OS memory spaces, completely isolated
   C++ SDK static singletons, and zero GIL contention during frame acquisition.

2. STAGGERED INITIALIZATION: Process 0 launches immediately; Process 1 launches after
   a 2.0s delay to prevent IPC lock contention on /tmp/.zed_gmsl_list_lock.

3. HARDWARE CAPABILITIES: Forces SVGA (1200x720) @ 30 FPS using LOSSLESS CPU compression
   (NVENC hardware video encoder is NOT available on Jetson Orin Nano).

4. AUTOMATED INTEGRITY VERIFICATION: Runs SVO file verification on all output .svo2 files
   upon completion.
======================================================================================
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

active_processes = []

def signal_handler(signum, frame):
    """Forward signals (SIGINT/SIGTERM) to child worker processes."""
    print(f"\n[SIGNAL] Received signal ({signum}). Terminating child processes...")
    for p in active_processes:
        if p.poll() is None:
            p.send_signal(signum)

def check_svo_integrity(file_path):
    """Verify SVO2 file opening and frame count using pyzed SDK."""
    try:
        import pyzed.sl as sl
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(file_path)
        cam = sl.Camera()
        err = cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            return False, 0
        frames = cam.get_svo_number_of_frames()
        cam.close()
        return True, frames
    except Exception as e:
        return False, 0

def main():
    parser = argparse.ArgumentParser(description="Hardened Python Dual ZED X SVGA SVO2 Recorder")
    parser.add_argument("-d", "--duration", type=int, default=10, help="Recording duration in seconds (0 = continuous)")
    parser.add_argument("-o", "--output-dir", type=str, default="/home/dogdiving/Desktop/scripts/dual_recording", help="Output directory")
    parser.add_argument("-p", "--prefix", type=str, default="py_svga", help="Filename prefix")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=======================================================")
    print("     HARDENED PYTHON DUAL ZED X SVGA SVO2 RECORDER     ")
    print("=======================================================")
    print(f"Platform: NVIDIA Jetson Orin Nano (Python Subprocess Isolation)")
    print(f"Resolution: SVGA (1200x720) @ 30 FPS")
    print(f"Compression: LOSSLESS (CPU PNG/RAW)")
    print(f"Duration: {args.duration if args.duration > 0 else 'Continuous (Ctrl+C to stop)'} seconds")
    print(f"Output Directory: {args.output_dir}")
    print(f"Prefix: {args.prefix}")
    print("-------------------------------------------------------")

    serials = [42250580, 40847387]
    print(f"[INFO] Target dual camera serial numbers: {serials}")

    os.makedirs(args.output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    worker_script = os.path.join(script_dir, "single_cam_recorder.py")

    procs_info = []

    # Launch subprocess per camera handle with 2.0s stagger delay
    for idx, sn in enumerate(serials):
        out_file = os.path.join(args.output_dir, f"{args.prefix}_cam{idx}_SN{sn}.svo2")
        if idx > 0:
            print(f"[INIT] Staggering Camera [{idx}] launch (2.0s pause for IPC socket safety)...")
            time.sleep(2.0)

        cmd = [
            sys.executable, worker_script,
            "--serial", str(sn),
            "--output", out_file,
            "--duration", str(args.duration)
        ]

        print(f"[LAUNCH] Starting worker process for S/N {sn} -> {out_file}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        active_processes.append(p)
        procs_info.append({"proc": p, "serial": sn, "output": out_file})

    print("\n[RECORDING ACTIVE] Subprocesses actively capturing frames across all cameras...")

    results = []
    # Monitor child processes and collect JSON output
    for info in procs_info:
        p = info["proc"]
        stdout_data, _ = p.communicate()
        
        # Parse output for JSON result line
        result_json = None
        for line in stdout_data.splitlines():
            line_str = line.strip()
            if line_str.startswith("{") and line_str.endswith("}"):
                try:
                    result_json = json.loads(line_str)
                except Exception:
                    pass
            else:
                print(f"  [{info['serial']}] {line_str}")

        if result_json:
            results.append(result_json)
        else:
            results.append({"serial": info["serial"], "frames": 0, "fps": 0.0, "success": False, "output": info["output"]})

    print("\n=======================================================")
    print("             PYTHON RECORDING SUMMARY                  ")
    print("=======================================================")
    
    all_success = True
    for res in results:
        sn = res["serial"]
        frames = res["frames"]
        fps = res["fps"]
        success = res["success"]
        out_path = res.get("output", "")

        status_str = "SUCCESS" if success else "FAILED"
        print(f"  * Camera S/N {sn}: {frames} frames ({fps:.2f} FPS) -> [{status_str}]")
        
        # Integrity check verified output file
        if success and os.path.exists(out_path):
            valid, verified_frames = check_svo_integrity(out_path)
            if valid:
                print(f"    [VERIFY] SVO2 File Validated: {os.path.basename(out_path)} ({verified_frames} frames read back)")
            else:
                print(f"    [VERIFY] SVO2 File Validation Failed for {out_path}")
                all_success = False
        else:
            all_success = False

    if all_success and len(results) >= 2:
        print("\n[COMPLETE] All Python SVO2 dual-camera recordings verified valid.")
        sys.exit(0)
    else:
        print("\n[ERROR] One or more camera recordings failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
