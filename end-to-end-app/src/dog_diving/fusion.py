"""fusion.py — runtime fusion of two ZED X point clouds (step 2d).

``scripts/dual_recording/run_icp_calibration.py`` and ``sensor_placer_auto.py``
compute a two-camera rigid transform (rotation + translation) from a synchronized
SVO2 pair and save it as a **Fusion JSON**.  What was missing is the *runtime*
side: take the two SVO2 files the recorder saves (``{prefix}_cam1.svo2`` /
``{prefix}_cam2.svo2``), build a point cloud per camera from offline NEURAL depth,
and merge them through that transform into one fused cloud in the reference
camera's frame.

The module splits cleanly so the geometry is verifiable without any hardware:

  * :func:`load_fusion_config` / :class:`FusionConfig` — parse the Fusion JSON and
    expose one :class:`FusionTransform` per camera (pure ``numpy``).
  * :func:`fuse_point_clouds` — apply each camera's transform and concatenate
    (pure ``numpy``, unit-tested against a synthetic fixture).
  * :class:`PointCloudSource` — the seam behind which the *real* SVO2→depth→cloud
    path hides.  :class:`SVO2PointCloudSource` is the hardware backend
    (**hardware-verify-pending** — needs pyzed + a Jetson GPU); the offline
    :class:`SyntheticPointCloudSource` returns pre-loaded clouds so the fusion
    core is testable end-to-end.

Fusion JSON schema (pinned from the two calibration scripts — do not guess):

    {
      "world":    {"override_gravity": true, "rotation": [...], "translation": [...]},
      "cameras": [
        {"serial_number": <int>,            # cameras[0] is the REFERENCE frame
         "position": {"translation": [x,y,z], "rotation": ...}},   # metres
        ...
      ]
    }

Transform semantics (both scripts): ``p_cam0 = R @ p_cam1 + T`` — i.e. the
transform maps **camera 1 into camera 0's frame**, so cameras[0] is the reference
and every other camera carries a non-identity ``position``.

**Rotation representation — a real ambiguity the loader resolves explicitly.**
The two scripts do not agree on what ``position.rotation`` means:

  * ``run_icp_calibration.py`` writes ``R_opt.tolist()`` → a **3×3 rotation matrix**
    (9 values, nested).
  * ``sensor_placer_auto.py`` (``--format fusion``) writes **Euler degrees** (3 values);
    its ``sensor_placer`` format writes **Rodrigues** (3 values).
  * The canonical ZED Fusion API uses a **Rodrigues rotation vector in radians**
    (3 values).

The loader therefore keys on *arity*: 9 values → rotation matrix; 3 values →
Rodrigues vector (radians, the canonical convention).  ``sensor_placer_auto.py``'s
``fusion`` output (Euler **degrees**) is the one case this does **not** match — a
caller with such a file must pass ``rotation_convention="euler_deg"``.  That
discrepancy is a defect in the calibration script to reconcile on the rig, not
something the fusion silently papers over.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# Safe ZED SDK import (mirrors zed_recorder.py).  Depth generation (NEURAL mode)
# requires the Jetson's GPU; on a CPU host ``HAS_ZED_SDK`` is False and the
# hardware source refuses to run rather than invent a cloud.
try:
    import pyzed.sl as sl  # type: ignore
    HAS_ZED_SDK = True
except ImportError:
    sl = None  # type: ignore
    HAS_ZED_SDK = False


# ---------------------------------------------------------------------------
# Rotation parsing (arity-keyed; see module docstring for the ambiguity).
# ---------------------------------------------------------------------------

def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation vector (radians) → 3×3 rotation matrix (numpy-only)."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _is_rotation_matrix(a: np.ndarray) -> bool:
    return a.ndim == 2 and a.shape == (3, 3)


def rotation_from_json(value, convention: str = "rodrigues") -> np.ndarray:
    """Coerce a Fusion-JSON ``position.rotation`` value to a 3×3 matrix.

    ``value`` may be:

      * a 3×3 nested list / array (9 values) → used as a rotation matrix verbatim;
      * a flat list of 9 values → reshaped row-major to 3×3;
      * a 3-element list → ``convention`` decides:
        ``"rodrigues"`` (default, radians, canonical ZED Fusion) or
        ``"euler_deg"`` (``sensor_placer_auto.py``'s ``fusion`` output).

    Returns an orthonormal 3×3 ``float64`` matrix.
    """
    a = np.asarray(value, dtype=np.float64)
    if a.size == 9:
        a = a.reshape(3, 3)
        if not _is_rotation_matrix(a):
            raise ValueError(f"rotation has 9 values but is not 3x3: shape {a.shape}")
        return a
    if a.size == 3:
        a = a.reshape(3)
        if convention == "euler_deg":
            rx, ry, rz = np.deg2rad(a)
            Rx = _rodrigues_to_matrix(np.array([rx, 0.0, 0.0]))
            Ry = _rodrigues_to_matrix(np.array([0.0, ry, 0.0]))
            Rz = _rodrigues_to_matrix(np.array([0.0, 0.0, rz]))
            return Rz @ Ry @ Rx
        if convention == "rodrigues":
            return _rodrigues_to_matrix(a)
        raise ValueError(f"unknown rotation convention {convention!r}")
    raise ValueError(
        f"rotation must have 3 or 9 values, got {a.size} (shape {a.shape})"
    )


# ---------------------------------------------------------------------------
# Parsed Fusion config
# ---------------------------------------------------------------------------

@dataclass
class FusionTransform:
    """One camera's rigid pose relative to the reference frame."""

    rotation: np.ndarray  # 3×3
    translation: np.ndarray  # (3,) metres

    @classmethod
    def identity(cls) -> "FusionTransform":
        return cls(np.eye(3), np.zeros(3))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map ``points`` (N×3) from this camera's frame into the reference frame."""
        pts = np.asarray(points, dtype=np.float64)
        return (self.rotation @ pts.T).T + self.translation


@dataclass
class CameraPose:
    serial_number: int
    transform: FusionTransform


@dataclass
class FusionConfig:
    """A parsed Fusion JSON: one :class:`CameraPose` per camera + the reference.

    ``cameras[0]`` is the reference frame (identity pose); every other camera's
    ``transform`` maps *its* points into that reference frame.
    """

    cameras: list[CameraPose]
    reference_serial: int
    rotation_convention: str = "rodrigues"

    @property
    def reference_transform(self) -> FusionTransform:
        return FusionTransform.identity()

    def transform_for(self, serial_number: int) -> FusionTransform:
        for cam in self.cameras:
            if cam.serial_number == int(serial_number):
                return cam.transform
        raise KeyError(
            f"camera serial {serial_number} is not in the Fusion config "
            f"(declared: {[c.serial_number for c in self.cameras]})"
        )

    @classmethod
    def from_dict(cls, doc: Mapping, rotation_convention: str = "rodrigues") -> "FusionConfig":
        cameras_raw = doc.get("cameras")
        if not isinstance(cameras_raw, list) or len(cameras_raw) < 2:
            raise ValueError(
                "Fusion config must declare at least two cameras (a `cameras` list)"
            )
        cameras: list[CameraPose] = []
        for entry in cameras_raw:
            serial = entry.get("serial_number")
            if serial is None:
                raise ValueError(f"camera entry missing `serial_number`: {entry!r}")
            position = entry.get("position", {}) or {}
            rotation = position.get("rotation", [0.0, 0.0, 0.0])
            translation = np.asarray(
                position.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
            ).reshape(3)
            if rotation is None:
                rotation = [0.0, 0.0, 0.0]
            R = rotation_from_json(rotation, convention=rotation_convention)
            cameras.append(CameraPose(int(serial), FusionTransform(R, translation)))
        return cls(cameras=cameras, reference_serial=cameras[0].serial_number,
                   rotation_convention=rotation_convention)

    @classmethod
    def from_file(cls, path: str | Path, rotation_convention: str = "rodrigues") -> "FusionConfig":
        doc = json.loads(Path(path).read_text())
        return cls.from_dict(doc, rotation_convention=rotation_convention)


# ---------------------------------------------------------------------------
# The fusion core (pure numpy)
# ---------------------------------------------------------------------------

@dataclass
class FusedCloud:
    """The merged point cloud + per-camera provenance."""

    points: np.ndarray  # (N, 3) metres, in the reference camera's frame
    counts: dict[int, int]  # serial -> number of points contributed
    reference_serial: int

    def __len__(self) -> int:
        return int(self.points.shape[0])


def fuse_point_clouds(
    clouds: Mapping[int, np.ndarray],
    config: FusionConfig,
    *,
    skip_missing: bool = False,
) -> FusedCloud:
    """Merge per-camera point clouds into the reference frame.

    ``clouds`` maps serial number → N×3 point cloud (metres, in that camera's own
    frame).  Each cloud is transformed by its camera's pose (identity for the
    reference) and concatenated.  A serial not declared in ``config`` raises
    ``KeyError`` unless ``skip_missing`` is set, in which case it is dropped —
    matching the "refuse, don't guess" rule: a cloud whose transform you do not
    know must not be silently treated as already-in-reference.
    """
    blocks: list[np.ndarray] = []
    counts: dict[int, int] = {}
    for serial, cloud in clouds.items():
        serial = int(serial)
        try:
            transform = config.transform_for(serial)
        except KeyError:
            if skip_missing:
                continue
            raise
        pts = np.asarray(cloud, dtype=np.float64)
        if pts.size == 0:
            pts = np.empty((0, 3), dtype=np.float64)
        elif pts.ndim == 1:
            pts = pts.reshape(1, 3)
        elif pts.ndim == 2 and pts.shape[1] == 3:
            pass
        else:
            raise ValueError(
                f"cloud for serial {serial} has shape {pts.shape}; expected (N, 3)"
            )
        blocks.append(transform.apply(pts))
        counts[serial] = int(pts.shape[0])

    if not blocks:
        points = np.empty((0, 3), dtype=np.float64)
    else:
        points = np.vstack(blocks)
    return FusedCloud(points=points, counts=counts, reference_serial=config.reference_serial)


# ---------------------------------------------------------------------------
# The source seam (SVO2 -> depth -> cloud), hardware behind an ABC.
# ---------------------------------------------------------------------------

class PointCloudSource(ABC):
    """The seam between the fusion core and wherever per-camera clouds come from.

    Concrete backends: :class:`SVO2PointCloudSource` (hardware, needs pyzed + GPU)
    and :class:`SyntheticPointCloudSource` (offline fixture).  Both return
    ``{serial_number: N×3 cloud (metres)}`` so the caller can hand the result
    straight to :func:`fuse_point_clouds`.
    """

    @abstractmethod
    def load(self) -> dict[int, np.ndarray]:
        """Return per-camera point clouds keyed by serial number."""


class SyntheticPointCloudSource(PointCloudSource):
    """Offline fixture: returns pre-loaded clouds keyed by serial."""

    def __init__(self, clouds: Mapping[int, np.ndarray]) -> None:
        self._clouds = {int(k): np.asarray(v, dtype=np.float64) for k, v in clouds.items()}

    def load(self) -> dict[int, np.ndarray]:
        return dict(self._clouds)


class SVO2PointCloudSource(PointCloudSource):
    """Real path: SVO2 → offline NEURAL depth → per-camera point cloud.

    **hardware-verify-pending.**  Depth generation (``DEPTH_MODE.NEURAL``) needs
    the Jetson's GPU; on a CPU host ``HAS_ZED_SDK`` is False and ``load()`` raises
    a clear ``RuntimeError`` instead of returning a fake cloud.

    The serial is read from each SVO2's own camera information, so the returned
    keys always match what the Fusion JSON declares — the caller does not map
    filenames to serials by hand.
    """

    def __init__(
        self,
        svo_paths: Sequence[str | Path],
        depth_mode: str = "NEURAL",
        skip_frames: int = 15,
    ) -> None:
        self.svo_paths = [Path(p) for p in svo_paths]
        self.depth_mode = depth_mode
        self.skip_frames = int(skip_frames)

    def _cloud_from_svo(self, path: Path) -> tuple[int, np.ndarray]:
        """Open one SVO2, retrieve one stabilized frame's XYZ cloud, return (serial, cloud)."""
        if not HAS_ZED_SDK:
            raise RuntimeError(
                "pyzed is not installed — cannot read an SVO2 or generate depth on this host"
            )
        init = sl.InitParameters()
        init.set_from_svo_file(str(path))
        init.svo_real_time_mode = False
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init.depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode.upper(), sl.DEPTH_MODE.NEURAL)

        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            zed.close()
            raise RuntimeError(f"failed to open SVO2 {path}: {status!r}")

        runtime = sl.RuntimeParameters()
        try:
            serial = int(zed.get_camera_information().serial_number)
            # Skip frames so auto-exposure / white-balance settle before measuring.
            for _ in range(self.skip_frames):
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    break
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"could not grab a frame from SVO2 {path}")

            pc = sl.Mat()
            zed.retrieve_measure(pc, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            arr = pc.get_data()
            if arr.ndim == 3:
                arr = arr.reshape(-1, 4)
            xyz = arr[:, :3].astype(np.float64)
            # Drop invalid / non-finite measurements (ZED encodes invalid as NaN/Inf).
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            if xyz.shape[0] == 0:
                raise RuntimeError(f"no valid depth points in SVO2 {path}")
            return serial, xyz
        finally:
            zed.close()

    def load(self) -> dict[int, np.ndarray]:
        clouds: dict[int, np.ndarray] = {}
        for path in self.svo_paths:
            serial, cloud = self._cloud_from_svo(path)
            clouds[serial] = cloud
        return clouds


def fuse_from_source(
    source: PointCloudSource,
    config: FusionConfig,
    *,
    skip_missing: bool = False,
) -> FusedCloud:
    """Convenience: load clouds from a :class:`PointCloudSource` and fuse them."""
    return fuse_point_clouds(source.load(), config, skip_missing=skip_missing)
