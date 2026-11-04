"""calibration_capture.py — the calibration-capture mode (step 1a).

A **distinct** entry point a human runs once on the rig to produce the two-camera
Fusion JSON that the runtime fusion (``fusion.py``, step 2d) consumes:

  1. capture two synchronized SVO2 — ``scripts/dual_recording/dual_svga_recorder.py``,
     the proven **SVGA 1200×720 @ 30 FPS LOSSLESS** config (subprocess-isolated);
  2. compute extrinsics from that pair — ``run_icp_calibration.py`` (a 3×3 rotation
     matrix) or ``sensor_placer_auto.py --format fusion`` (Euler degrees);
  3. save the Fusion JSON to a **known path** and verify it loads back through
     ``fusion.FusionConfig.from_file`` — the exact loader ``fuse_point_clouds`` uses.

This is deliberately *not* the live-capture path (``zed_recorder.py`` +
"Score Live Jump"): calibration is a once-per-rig-setup operator action, so it
shells out to the proven subprocess-isolated recorder instead of touching the
in-process rolling buffer.

Hardware sits behind a seam (:class:`CalibrationCaptureBackend`) so the whole flow is
testable offline: :class:`RigCalibrationCaptureBackend` drives the proven scripts and
refuses cleanly on a CPU host (no pyzed); :class:`FakeCalibrationCaptureBackend`
writes synthetic SVO2 stubs + a synthetic Fusion JSON with a known ``(R, T)``.

Rotation convention — the recorded ambiguity, resolved explicitly, never guessed:

  * ``run_icp_calibration.py`` writes ``position.rotation`` as a **3×3 matrix**
    (9 values) — ``FusionConfig`` reads that by arity, convention not consulted.
  * ``sensor_placer_auto.py --format fusion`` writes **Euler degrees** (3 values) —
    ``FusionConfig.from_file(..., rotation_convention="euler_deg")`` is required.

:func:`fusion_convention` maps a method name to the loader convention it needs, and
:func:`run_calibration_capture` records which method ran so the convention it hands
the loader is the one the file actually encodes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from dog_diving.config import FUSION_JSON_PATH, PROJECT_ROOT, SVO_RECORDINGS_DIR
from dog_diving.fusion import FusionConfig, HAS_ZED_SDK

#: Where the proven rig scripts live (in-repo, read-only reference).
DUAL_RECORDING_DIR = PROJECT_ROOT / "src" / "recording"
DUAL_SVGA_RECORDER = DUAL_RECORDING_DIR / "dual_svga_recorder.py"
ICP_CALIBRATION = DUAL_RECORDING_DIR / "run_icp_calibration.py"
SENSOR_PLACER = DUAL_RECORDING_DIR / "sensor_placer_auto.py"

#: The recorder hardcodes these serials (``dual_svga_recorder.py``).  The app
#: does not invent a second source of truth for which cameras exist.
RIG_SERIALS = (42250580, 40847387)

#: Extrinsics methods this module can drive, keyed by CLI name.
METHODS = ("icp", "sensor_placer")


def fusion_convention(method: str) -> str:
    """The ``rotation_convention`` the loader needs for a given extrinsics method.

    ``"icp"`` produces a 3×3 matrix (arity 9) — ``FusionConfig`` reads that
    directly, so the convention value is the loader default.  ``"sensor_placer"``
    produces Euler *degrees* (arity 3) and must be told so.  Anything else is a
    configuration error, raised, not guessed.
    """
    key = str(method).strip().lower().replace("-", "_")
    if key in ("icp", "kabsch", "run_icp_calibration"):
        return "rodrigues"  # 3x3 matrix output; arity-9, convention not consulted
    if key in ("sensor_placer", "sensorplacer"):
        return "euler_deg"
    raise ValueError(f"unknown extrinsics method {method!r} — known: {', '.join(METHODS)}")


class CalibrationCaptureBackend(ABC):
    """The seam between the calibration-capture orchestration and the hardware.

    Concrete backends: :class:`RigCalibrationCaptureBackend` (shells out to the
    proven scripts) and :class:`FakeCalibrationCaptureBackend` (offline fixture).
    """

    @abstractmethod
    def capture(self, prefix: str, duration: int, output_dir: str | Path) -> list[Path]:
        """Capture two synchronized SVO2; return ``[cam0, cam1]`` (cam0 = reference)."""

    @abstractmethod
    def compute_extrinsics(
        self, svo0: str | Path, svo1: str | Path, output_path: str | Path, method: str
    ) -> Path:
        """Compute extrinsics from the pair and write a Fusion JSON to ``output_path``."""


class RigCalibrationCaptureBackend(CalibrationCaptureBackend):
    """Hardware backend: drive the proven subprocess-isolated recorder + extrinsics.

    **hardware-verify-pending.**  Every method refuses cleanly (``RuntimeError``)
    on a host without pyzed — the scripts themselves need the ZED SDK, and a CPU
    host must never produce a fake calibration.
    """

    def _require_rig(self) -> None:
        if not HAS_ZED_SDK:
            raise RuntimeError(
                "pyzed is not installed — the calibration-capture mode needs the ZED "
                "SDK and the rig's cameras; nothing was captured or computed"
            )

    def _capture_argv(self, prefix: str, duration: int, output_dir: str | Path) -> list[str]:
        return [
            sys.executable,
            str(DUAL_SVGA_RECORDER),
            "-d", str(int(duration)),
            "-o", str(output_dir),
            "-p", prefix,
        ]

    def _extrinsics_argv(
        self, svo0: str | Path, svo1: str | Path, output_path: str | Path, method: str
    ) -> list[str]:
        key = str(method).strip().lower().replace("-", "_")
        if key in ("icp", "kabsch", "run_icp_calibration"):
            return [
                sys.executable, str(ICP_CALIBRATION),
                "--svo0", str(svo0), "--svo1", str(svo1), "--output", str(output_path),
            ]
        if key in ("sensor_placer", "sensorplacer"):
            return [
                sys.executable, str(SENSOR_PLACER),
                "--svo0", str(svo0), "--svo1", str(svo1),
                "--output", str(output_path), "--format", "fusion",
            ]
        raise ValueError(f"unknown extrinsics method {method!r} — known: {', '.join(METHODS)}")

    def capture(self, prefix: str, duration: int, output_dir: str | Path) -> list[Path]:
        self._require_rig()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = self._capture_argv(prefix, duration, out_dir)
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"dual SVGA recorder failed (exit {proc.returncode}): "
                f"{proc.stdout.strip() or proc.stderr.strip()}"
            )
        svos = _discover_svo_pair(out_dir, prefix)
        if len(svos) != 2:
            raise RuntimeError(
                f"recorder produced {len(svos)} SVO2 for prefix {prefix!r}; expected exactly 2"
            )
        return svos

    def compute_extrinsics(
        self, svo0: str | Path, svo1: str | Path, output_path: str | Path, method: str
    ) -> Path:
        self._require_rig()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        argv = self._extrinsics_argv(svo0, svo1, out, method)
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"extrinsics ({method}) failed (exit {proc.returncode}): "
                f"{proc.stdout.strip() or proc.stderr.strip()}"
            )
        if not out.is_file():
            raise RuntimeError(f"extrinsics ({method}) reported success but wrote no {out}")
        return out


class FakeCalibrationCaptureBackend(CalibrationCaptureBackend):
    """Offline stand-in that writes synthetic SVO2 stubs + a known Fusion JSON.

    The SVO2 outputs are JSON metadata stubs (not real SVO2) carrying the capture
    metadata, mirroring ``FakeZEDRecorder``.  The Fusion JSON is the 3×3-matrix
    schema ``run_icp_calibration.py`` emits, with a known ``(R, T)`` so the
    orchestration's read-back assertion is a real check, not a smoke test.
    """

    #: The known transform the fake writes (cam1 -> cam0 frame): Z+90° rotation.
    R = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    T = np.array([1.0, 0.5, 0.0], dtype=np.float64)  # metres
    #: ``R`` in Euler degrees (XYZ, ``rotation_from_json``'s ``euler_deg`` order) — the
    #: ``sensor_placer_auto.py --format fusion`` representation of the same rotation.
    EULER_DEG = [0.0, 0.0, 90.0]

    def __init__(self, cam_serials: tuple = RIG_SERIALS) -> None:
        self.cam_serials = tuple(int(s) for s in cam_serials)
        self.last_capture: dict | None = None
        self.last_extrinsics: dict | None = None

    def capture(self, prefix: str, duration: int, output_dir: str | Path) -> list[Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, serial in enumerate(self.cam_serials):
            p = out_dir / f"{prefix}_cam{i}_SN{serial}.svo2"
            p.write_text(json.dumps({
                "format": "svo2", "synthetic": True, "serial": serial,
                "cam_index": i, "duration": int(duration),
                "resolution": "SVGA", "fps": 30, "compression": "LOSSLESS",
            }))
            paths.append(p)
        self.last_capture = {"prefix": prefix, "duration": int(duration),
                             "output_dir": str(out_dir), "svos": [str(p) for p in paths]}
        return paths

    def compute_extrinsics(
        self, svo0: str | Path, svo1: str | Path, output_path: str | Path, method: str
    ) -> Path:
        convention = fusion_convention(method)  # validate the method name up front
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Write the rotation in the representation the real method produces, so the
        # orchestration's read-back exercises the right convention, not a matrix in
        # every case.
        if convention == "euler_deg":
            rotation = list(self.EULER_DEG)
        else:
            rotation = self.R.tolist()
        doc = {
            "world": {"override_gravity": True, "rotation": [0.0, 0.0, 0.0],
                      "translation": [0.0, 0.0, 0.0]},
            "cameras": [
                {"serial_number": self.cam_serials[0],
                 "position": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}},
                {"serial_number": self.cam_serials[1],
                 "position": {"translation": self.T.tolist(), "rotation": rotation}},
            ],
        }
        out.write_text(json.dumps(doc, indent=2))
        self.last_extrinsics = {"svo0": str(svo0), "svo1": str(svo1),
                                "method": method, "output": str(out)}
        return out


# ---------------------------------------------------------------------------
# The orchestration (pure over the backend)
# ---------------------------------------------------------------------------

def _discover_svo_pair(output_dir: str | Path, prefix: str) -> list[Path]:
    """Find ``{prefix}_cam0…`` / ``{prefix}_cam1…`` SVO2 files, sorted cam0 first."""

    def cam_index(p: Path) -> int:
        name = p.name
        marker = f"{prefix}_cam"
        if marker not in name:
            return 99
        tail = name.split(marker, 1)[1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else 99

    found = sorted(Path(output_dir).glob(f"{prefix}_cam*.svo2"), key=cam_index)
    return found[:2]


def run_calibration_capture(
    backend: CalibrationCaptureBackend,
    *,
    prefix: str = "calib_capture",
    duration: int = 10,
    output_dir: str | Path | None = None,
    fusion_path: str | Path | None = None,
    method: str = "icp",
) -> dict:
    """Capture two SVO2, compute extrinsics, and verify the Fusion JSON loads.

    The Fusion JSON is written to ``fusion_path`` (default: the known
    ``config.FUSION_JSON_PATH`` the fusion runtime reads) and is immediately
    re-loaded through ``fusion.FusionConfig.from_file`` with the convention the
    ``method`` requires — so a sensor-placer (Euler-degree) file can never be
    silently misread as Rodrigues.

    Returns a provenance dict carrying the SVO2 paths, the Fusion JSON path, the
    method, the convention actually used, and the parsed :class:`FusionConfig`
    (plus a serializable camera summary for logging).  Never returns a fake
    calibration: a missing SVO2 pair or a load failure raises.
    """
    method = str(method).strip().lower().replace("-", "_")
    convention = fusion_convention(method)
    out_dir = Path(output_dir) if output_dir is not None else SVO_RECORDINGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fusion_out = Path(fusion_path) if fusion_path is not None else FUSION_JSON_PATH
    fusion_out.parent.mkdir(parents=True, exist_ok=True)

    svos = backend.capture(prefix, int(duration), out_dir)
    if len(svos) < 2:
        raise RuntimeError(
            f"calibration capture needs two SVO2 (got {len(svos)}); "
            "the two cameras were not both recorded"
        )

    backend.compute_extrinsics(svos[0], svos[1], fusion_out, method)
    if not fusion_out.is_file():
        raise RuntimeError(f"extrinsics did not produce {fusion_out}")

    # Verify, don't claim: read it back with the convention the method produced.
    config = FusionConfig.from_file(fusion_out, rotation_convention=convention)
    cameras = []
    for cam in config.cameras:
        cameras.append({
            "serial": cam.serial_number,
            "translation": [round(float(v), 6) for v in cam.transform.translation],
            "rotation_matrix": [
                [round(float(v), 6) for v in row] for row in cam.transform.rotation
            ],
        })
    return {
        "svos": [str(p) for p in svos],
        "fusion_path": str(fusion_out),
        "method": method,
        "rotation_convention": convention,
        "reference_serial": config.reference_serial,
        "cameras": cameras,
        "config": config,
    }


def load_fusion_config(
    path: str | Path | None = None, rotation_convention: str = "rodrigues"
) -> FusionConfig:
    """Load the saved Fusion JSON (the known path by default) for the fusion runtime.

    Thin wrapper over :class:`FusionConfig.from_file` so ``fuse_point_clouds`` has a
    single known path to read.  ``rotation_convention`` must match how the file was
    written (``"icp"`` → default; ``"sensor_placer"`` → ``"euler_deg"``) — see
    :func:`fusion_convention`.
    """
    p = Path(path) if path is not None else FUSION_JSON_PATH
    return FusionConfig.from_file(p, rotation_convention=rotation_convention)


def default_prefix() -> str:
    """A timestamped capture prefix, so repeated calibrations never collide."""
    return f"calib_{time.strftime('%Y%m%d_%H%M%S')}"
