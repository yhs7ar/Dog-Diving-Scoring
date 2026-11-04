"""calibrate_capture.py — headless calibration-capture entry point (step 1a).

A **distinct** CLI from the scorer and from the live-capture path: it produces the
two-camera Fusion JSON the runtime fusion reads, by (1) capturing two synchronized
SVO2 with the proven SVGA recorder and (2) running extrinsics on that pair, then
(3) verifying the Fusion JSON loads back through ``fusion.FusionConfig.from_file``.

Run on the rig (needs pyzed + the ZED X pair):

    python src/calibrate_capture.py --duration 10 --extrinsics icp

Run offline (synthetic fixture, for testing the wiring):

    python src/calibrate_capture.py --fake

This is **not** "Score Live Jump": no rolling buffer, no pipeline clip — it is the
once-per-rig-setup operator action that populates ``config.FUSION_JSON_PATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dog_diving.calibration_capture import (  # noqa: E402
    METHODS,
    FakeCalibrationCaptureBackend,
    RigCalibrationCaptureBackend,
    default_prefix,
    run_calibration_capture,
)
from dog_diving.config import FUSION_JSON_PATH, SVO_RECORDINGS_DIR  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Calibration capture: two SVO2 -> Fusion JSON.")
    ap.add_argument("--prefix", default=None,
                    help="capture filename prefix (default: timestamped 'calib_YYYYMMDD_HHMMSS')")
    ap.add_argument("--duration", type=int, default=10,
                    help="recording duration in seconds (default 10)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help=f"SVO2 output directory (default {SVO_RECORDINGS_DIR})")
    ap.add_argument("--fusion-path", type=Path, default=None,
                    help=f"Fusion JSON output path (default {FUSION_JSON_PATH})")
    ap.add_argument("--extrinsics", choices=METHODS, default="icp",
                    help="extrinsics method: 'icp' (3x3 matrix) or 'sensor_placer' (Euler degrees)")
    ap.add_argument("--fake", action="store_true",
                    help="use the offline synthetic backend (no cameras, no pyzed)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = FakeCalibrationCaptureBackend() if args.fake else RigCalibrationCaptureBackend()
    prefix = args.prefix or default_prefix()

    try:
        result = run_calibration_capture(
            backend,
            prefix=prefix,
            duration=args.duration,
            output_dir=args.output_dir,
            fusion_path=args.fusion_path,
            method=args.extrinsics,
        )
    except Exception as exc:  # noqa: BLE001 — a clean refusal, not a traceback
        print(f"calibration capture FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = {k: v for k, v in result.items() if k != "config"}
    print(json.dumps(summary, indent=2))
    print(f"\nFusion JSON: {result['fusion_path']}")
    print(f"  reference  : SN {result['reference_serial']}")
    print(f"  method     : {result['method']} (rotation convention: {result['rotation_convention']})")
    for cam in result["cameras"]:
        print(f"  camera     : SN {cam['serial']}  T={cam['translation']}")
    print("\ncalibration capture complete — fusion runtime will read this Fusion JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
