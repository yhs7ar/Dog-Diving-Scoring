#!/usr/bin/env python3
"""Generate a point cloud from a SINGLE SVO2 (offline NEURAL depth) — step 12 (PC-GEN).

Opens **one** SVO2 file, computes depth off-line (``DEPTH_MODE.NEURAL``), and saves, per
``{stem}`` of the SVO2 path:

  * ``{stem}.npy`` (default) or ``{stem}.xyz`` — the merged XYZ point cloud (metres,
    ``RIGHT_HANDED_Y_UP``, invalid/NaN points dropped), one ``(N, 3)`` float64 array.
  * ``{stem}_intrinsics.json`` — ``fx/fy/cx/cy`` + ``width/height`` + ``serial_number``.
  * ``{stem}_timestamps.csv`` — ``frame_index,timestamp_ms`` for every *saved* frame.
  * ``{stem}_frames/frame_000000.npy`` — **only with ``--save-per-frame``**: the image-shaped
    ``(H, W, 3)`` float32 cloud per frame (invalid depth = NaN), named by the same
    ``frame_index`` as the timestamps CSV, so row *i* of the CSV ↔ ``frame_{i:06d}.npy``.
    This is the layout ``get_3d_points_at_pixels`` (sensor_placer_auto.py) already indexes by
    ``[row, col]`` — i.e. exactly what the trajectory lookup in steps 13–14 needs.

Single-SVO2 rule: this opens ONE file. Never feed it two SVO2 files as a stereo pair — the
raw files were not cut with this pipeline and are not synchronized across cameras.

Proven SVO2-open pattern (mirrors ``sensor_placer_auto.py::open_svo`` and
``fusion.py::SVO2PointCloudSource._cloud_from_svo``): ``set_from_svo_file``,
``svo_real_time_mode=False``, ``coordinate_units=UNIT.METER``,
``coordinate_system=RIGHT_HANDED_Y_UP``, ``retrieve_measure(..., MEASURE.XYZRGBA, MEM.CPU)``.

**Runs on the Jetson only** (NEURAL depth needs its GPU). On a CPU host ``HAS_ZED_SDK`` is
False and :func:`generate` raises a clear ``RuntimeError`` rather than inventing a cloud.

Usage (on the rig, ``src/recording/``):

    python3 generate_point_cloud.py --svo /path/to/SVGA_SN40847387_19-01-49.svo2

Recommended SVO2 (smallest SVGA = the working capture config, per
``WORKAROUNDS_AND_LIMITATIONS.md``): **``SVGA_SN40847387_19-01-49.svo2`` (~93 MB)**.
The smallest SVGA files are ~93–183 MB; HD1080/VGA/15 FPS files are failed configs and
must not be re-driven, though any single SVGA file is fine.  Copy the file to the Jetson's
local disk before running (a cloud-synced read hangs on download).

Exit codes: 0 = success; 1 = pyzed missing / argument error; 2 = SVO2 open or grab failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Safe ZED SDK import (mirrors fusion.py / zed_recorder.py).  Depth generation (NEURAL
# mode) requires the Jetson's GPU; on a CPU host this stays False and the hardware path
# refuses to run.
try:
    import pyzed.sl as sl  # type: ignore

    HAS_ZED_SDK = True
except ImportError:  # pragma: no cover - exercised on CPU hosts only
    sl = None  # type: ignore
    HAS_ZED_SDK = False


DEPTH_MODES = {
    "NEURAL": sl.DEPTH_MODE.NEURAL if HAS_ZED_SDK else "NEURAL",
    "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS if HAS_ZED_SDK else "NEURAL_PLUS",
    "ULTRA": sl.DEPTH_MODE.ULTRA if HAS_ZED_SDK else "ULTRA",
    "QUALITY": sl.DEPTH_MODE.QUALITY if HAS_ZED_SDK else "QUALITY",
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE if HAS_ZED_SDK else "PERFORMANCE",
}

VALID_FORMATS = ("npy", "xyz", "both")


# ---------------------------------------------------------------------------
# Pure, offline-testable helpers (no pyzed).
# ---------------------------------------------------------------------------

def output_paths(svo_path: str | Path, out_dir: str | Path, fmt: str) -> dict[str, list[Path] | Path]:
    """Derive every output path for one SVO2. ``cloud`` is a list (1 or 2 entries)."""
    if fmt not in VALID_FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {VALID_FORMATS}")
    stem = Path(svo_path).stem
    out = Path(out_dir)
    if fmt == "npy":
        clouds = [out / f"{stem}.npy"]
    elif fmt == "xyz":
        clouds = [out / f"{stem}.xyz"]
    else:  # both
        clouds = [out / f"{stem}.npy", out / f"{stem}.xyz"]
    return {
        "cloud": clouds,
        "intrinsics": out / f"{stem}_intrinsics.json",
        "timestamps": out / f"{stem}_timestamps.csv",
        "frames_dir": out / f"{stem}_frames",
    }


def write_intrinsics(doc: dict, path: str | Path) -> Path:
    """Write the intrinsics document as pretty JSON; returns the path written."""
    p = Path(path)
    p.write_text(json.dumps(doc, indent=2))
    return p


def write_timestamps_csv(rows: list[tuple[int, int]], path: str | Path) -> Path:
    """Write ``frame_index,timestamp_ms`` rows; returns the path written."""
    p = Path(path)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_ms"])
        for frame_index, ts in rows:
            w.writerow([frame_index, ts])
    return p


def save_cloud(xyz: np.ndarray, path: str | Path) -> Path:
    """Save an ``(N, 3)`` float64 cloud as ``.npy`` or ASCII ``.xyz`` (x y z per line)."""
    p = Path(path)
    arr = np.asarray(xyz, dtype=np.float64)
    if p.suffix == ".npy":
        np.save(p, arr)
    elif p.suffix == ".xyz":
        np.savetxt(p, arr, fmt="%.6f")
    else:
        raise ValueError(f"unsupported cloud suffix {p.suffix!r}; expected .npy or .xyz")
    return p


def load_cloud(path: str | Path) -> np.ndarray:
    """Read back a cloud saved by :func:`save_cloud` (used by tests)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".xyz":
        return np.loadtxt(p)
    raise ValueError(f"unsupported cloud suffix {p.suffix!r}")


def save_frame_cloud(xyz: np.ndarray, path: str | Path) -> Path:
    """Save one image-shaped ``(H, W, 3)`` cloud as float32 ``.npy`` (invalid -> NaN)."""
    p = Path(path)
    arr = np.asarray(xyz, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.nan
    np.save(p, arr)
    return p


def load_frame_cloud(path: str | Path) -> np.ndarray:
    """Read back a frame cloud saved by :func:`save_frame_cloud` (used by tests)."""
    return np.load(Path(path))


def merge_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Flatten + concatenate per-frame ``(H, W, 3)`` clouds, dropping non-finite points."""
    blocks = []
    for f in frames:
        flat = np.asarray(f, dtype=np.float64).reshape(-1, 3)
        blocks.append(flat[np.isfinite(flat).all(axis=1)])
    if not blocks:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# Hardware path (Jetson only).
# ---------------------------------------------------------------------------

def open_svo(svo_path: str | Path, depth_mode: str = "NEURAL"):
    """Open one SVO2 with the proven rig settings; returns ``(zed, runtime_params)``."""
    if not HAS_ZED_SDK:
        raise RuntimeError(
            "pyzed is not installed — cannot read an SVO2 or generate depth on this host"
        )
    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_mode = DEPTH_MODES.get(depth_mode.upper(), sl.DEPTH_MODE.NEURAL)

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"failed to open SVO2 {svo_path}: {err!r}")
    return zed, sl.RuntimeParameters()


def intrinsics_from_zed(zed) -> dict:
    """Read fx/fy/cx/cy + resolution + serial from an open camera."""
    info = zed.get_camera_information()
    cam = info.camera_configuration.calibration_parameters.left_cam
    res = info.camera_configuration.resolution
    return {
        "serial_number": int(info.serial_number),
        "fx": float(cam.fx),
        "fy": float(cam.fy),
        "cx": float(cam.cx),
        "cy": float(cam.cy),
        "width": int(res.width),
        "height": int(res.height),
    }


def generate(
    svo_path: str | Path,
    out_dir: str | Path | None = None,
    depth_mode: str = "NEURAL",
    fmt: str = "npy",
    stride: int = 1,
    max_frames: int = 0,
    save_per_frame: bool = False,
    skip_frames: int = 0,
) -> dict:
    """Run the full single-SVO2 → depth → cloud generation.

    Returns a summary dict: ``{"serial_number", "intrinsics", "frames_saved",
    "points", "paths"}``.  Raises ``RuntimeError`` on a CPU host (no pyzed).
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if max_frames < 0:
        raise ValueError(f"max_frames must be >= 0, got {max_frames}")
    if skip_frames < 0:
        raise ValueError(f"skip_frames must be >= 0, got {skip_frames}")

    svo = Path(svo_path)
    out = Path(out_dir) if out_dir is not None else svo.parent
    paths = output_paths(svo, out, fmt)

    zed, runtime = open_svo(svo, depth_mode)
    try:
        intrinsics = intrinsics_from_zed(zed)

        # Settle auto-exposure / white-balance before measuring.
        for _ in range(skip_frames):
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                break

        frames_dir = Path(paths["frames_dir"])
        if save_per_frame:
            frames_dir.mkdir(parents=True, exist_ok=True)

        blocks: list[np.ndarray] = []
        rows: list[tuple[int, int]] = []
        frame_index = 0

        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                break  # end of SVO2

            pc = sl.Mat()
            zed.retrieve_measure(pc, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            arr = pc.get_data()  # (H, W, 4) float32
            xyz = arr[..., :3].astype(np.float64)

            ts = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds())
            rows.append((frame_index, ts))

            if save_per_frame:
                save_frame_cloud(xyz, frames_dir / f"frame_{frame_index:06d}.npy")
            blocks.append(xyz)

            frame_index += 1
            if max_frames and frame_index >= max_frames:
                break

            # Skip (stride - 1) frames so we keep every stride-th one.
            for _ in range(stride - 1):
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    break

        if frame_index == 0:
            raise RuntimeError(f"no frames grabbed from SVO2 {svo}")

        merged = merge_frames(blocks)

        doc = dict(intrinsics)
        doc.update(
            {
                "depth_mode": depth_mode.upper(),
                "coordinate_system": "RIGHT_HANDED_Y_UP",
                "unit": "meter",
                "frames_saved": frame_index,
                "stride": stride,
            }
        )
        write_intrinsics(doc, paths["intrinsics"])
        write_timestamps_csv(rows, paths["timestamps"])
        for cloud_path in paths["cloud"]:
            save_cloud(merged, cloud_path)

        return {
            "serial_number": intrinsics["serial_number"],
            "intrinsics": doc,
            "frames_saved": frame_index,
            "points": int(merged.shape[0]),
            "paths": paths,
        }
    finally:
        zed.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a point cloud from a single SVO2 (offline NEURAL depth)."
    )
    parser.add_argument("--svo", required=True, help="path to the single SVO2 file")
    parser.add_argument("--out-dir", default=None, help="output dir (default: SVO2's parent)")
    parser.add_argument("--depth-mode", default="NEURAL", choices=sorted(DEPTH_MODES),
                        help="ZED depth mode (default NEURAL)")
    parser.add_argument("--format", default="npy", choices=VALID_FORMATS,
                        help="merged-cloud format: npy, xyz, or both")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame (>=1)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N saved frames (0 = whole SVO2)")
    parser.add_argument("--save-per-frame", action="store_true",
                        help="also write image-shaped (H,W,3) clouds to {stem}_frames/")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="frames to skip at the start (exposure settling)")
    args = parser.parse_args(argv)

    try:
        summary = generate(
            args.svo,
            out_dir=args.out_dir,
            depth_mode=args.depth_mode,
            fmt=args.format,
            stride=args.stride,
            max_frames=args.max_frames,
            save_per_frame=args.save_per_frame,
            skip_frames=args.skip_frames,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"serial_number : {summary['serial_number']}")
    print(f"frames saved  : {summary['frames_saved']}")
    print(f"points        : {summary['points']:,}")
    for key in ("intrinsics", "timestamps"):
        print(f"{key:13}: {summary['paths'][key]}")
    for cloud_path in summary["paths"]["cloud"]:
        print(f"cloud         : {cloud_path}")
    if args.save_per_frame:
        print(f"frames_dir    : {summary['paths']['frames_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
