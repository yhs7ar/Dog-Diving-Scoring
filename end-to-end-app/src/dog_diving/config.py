from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = PROJECT_ROOT / "data" / "calib"
CLIP_DIR = PROJECT_ROOT / "data" / "clips"
OUT_DIR = PROJECT_ROOT / "out"

#: The trajectory's pose checkpoint, resolved **inside** ``models/``.
#:
#: Declared in :data:`MODEL_REGISTRY` below.  The model of record is ``yolo26n-pose_dogs_v5.pt``:
#: 46/46 detected, tailset median **8.18 px**, 89.1 % within 20 % of the box diagonal on the 46-frame
#: labelled still set (``data/testing/test-clip-video004_right_pose.zip``, imgsz 1280 / conf 0.12 /
#: kp_conf 0.30; ``models/yolo26n-pose_dogs_v5.PROVENANCE.md``, sha256 ``e6e0005e…f0ba``).  The
#: vendored upstream ``best.pt`` (45/46, 10.10 px, 68.9 %) is kept for provenance and for
#: ``DD_MODEL=best.pt`` reproductions.  A second keypoint family runs through ``pose_layout``: the
#: stock COCO pose model (``models/yolo26s-pose.pt``, 17 keypoints, class ``person``) with nose =
#: row 0 and the tailset **derived** as the hips midpoint with ``min(hip confidence)`` — no dog
#: class, so on Dog Diving footage it measures the handler, and ``DetectResult.tailset_derived`` says so.
#:
#: Selectors: ``DD_MODEL`` (an absolute path or a file name inside ``models/`` — e.g.
#: ``DD_MODEL=best.pt`` reproduces the pre-2026-10-19 behaviour) and ``DD_MODEL_SCALE`` (``n`` /
#: ``s``, resolved through the registry).  The GUI sets the same two things: a **backbone** selector
#: beside a pose-model combobox.  A chosen file that is absent still falls back so the tree stays
#: runnable, and an unusable ``DD_MODEL_SCALE`` is recorded in ``MODEL_SELECTION_NOTE`` rather than
#: silently obeyed.  A ``kpt_shape`` with no layout is refused at load (``dog_diving/pose_layout.py``).
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# The model registry — the one place a pose checkpoint is declared
# ---------------------------------------------------------------------------
# Why a registry (2026-10-22)
# ---------------------------
# Two backbones are in use — **yolo26n** (the model of record) and **yolo26s** (a fine-tune on the
# same 2-keypoint dog labels) — and they differ in *scale*, not in keypoint semantics: both carry
# ``kpt_shape [2, 3]`` / ``names {0: 'dog'}``, so ``pose_layout`` maps both to ``dog-2kpt``.  What
# used to vary from place to place was the *list of checkpoints*: this module had one
# ``MODEL_PATH``, ``measure_pose_models.py`` had its own hardcoded list, and the GUI globbed
# ``models/*.pt``.  Three declarations of one fact is how "switch the model" turns into three
# different runs.
#
# So every checkpoint the project knows about is declared **once**, here, with what it is, which
# keypoint layout it must resolve to, and how it actually measured on the 46-frame labelled still
# set.  ``MODEL_PATH``, the GUI's backbone selector, ``eval_frames --scale`` and
# ``measure_pose_models.py`` all read this registry; there is no second list.
#
# Adding a checkpoint = one ``ModelSpec`` here (plus a ``PoseLayout`` in ``pose_layout.py`` if it is
# a new keypoint family).  Renaming a file without updating its spec makes the declaration lie, so
# ``test_m28_model_registry.py`` pins the file names and the registry's resolution order.
SCALES: tuple[str, ...] = ("n", "s")

#: Checkpoint families.  ``dog`` = the 2-keypoint fine-tunes this pipeline measures with;
#: ``person`` = a stock COCO-pose model kept for A/B, which has no dog class and therefore cannot
#: be selected by scale (see :func:`default_for_scale`).
FAMILY_DOG = "dog"
FAMILY_PERSON = "person"


@dataclass(frozen=True)
class ModelSpec:
    """One selectable pose checkpoint: what it is, what its keypoints mean, how it measured.

    ``scale`` is the backbone (``n`` or ``s``) — the axis an operator selects on.  ``layout`` is
    the *expected* ``pose_layout.PoseLayout.key``; the authoritative mapping is still resolved from
    the loaded checkpoint's ``kpt_shape``, so a mis-declared layout is caught at load rather than
    trusted here.
    """

    key: str
    filename: str
    scale: str
    label: str
    layout: str
    family: str = FAMILY_DOG
    role: str = "alternative"
    #: Measured on ``data/testing/test-clip-video004_right_pose.zip`` (46 labelled stills, imgsz 1280,
    #: conf 0.12, kp_conf 0.30) — numbers, not adjectives.  Empty when never measured.
    measured: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename

    def is_file(self) -> bool:
        return self.path.is_file()

    def measured_line(self) -> str:
        """One-line accuracy summary, or ``""`` when this checkpoint was never measured."""
        m = dict(self.measured or {})
        if not m:
            return ""
        bits = []
        if "detected" in m:
            bits.append(f"detected {m['detected']}/46")
        if "tailset_median_px" in m:
            bits.append(f"tailset median {m['tailset_median_px']} px")
        if "within_20pct" in m:
            bits.append(f"{m['within_20pct']} % within 20 % of box diagonal")
        if "note" in m:
            bits.append(str(m["note"]))
        return ", ".join(bits)

    def describe(self) -> str:
        return (f"{self.key}: {self.filename} scale={self.scale} family={self.family} "
                f"layout={self.layout} role={self.role}")


#: Order is meaningful: the first ``dog`` spec of a scale is that scale's default.
MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="dogs-n-v5",
        filename="yolo26n-pose_dogs_v5.pt",
        scale="n",
        label="yolo26n · dogs v5",
        layout="dog-2kpt",
        role="model of record",
        measured={"detected": 46, "tailset_median_px": 8.18, "within_20pct": 89.1},
        note="local real-background rotation model; default for the trajectory",
    ),
    ModelSpec(
        key="dogs-s-clahe",
        filename="yolo26s-pose_dogs_clahe.pt",
        scale="s",
        label="yolo26s · dogs (CLAHE fine-tune)",
        layout="dog-2kpt",
        role="alternative",
        measured={"detected": 1, "tailset_median_px": 196.37, "within_20pct": 0.0,
                  "note": "at conf 0.05: 34/46, median 253 px, 0 % within 20 %"},
        note="same 2-kpt dog labels on a yolo26s backbone; selectable, NOT a replacement for v5",
    ),
    ModelSpec(
        key="coco-person-s",
        filename="yolo26s-pose.pt",
        scale="s",
        label="yolo26s · stock COCO person (17 kpt)",
        layout="coco-person-17",
        family=FAMILY_PERSON,
        role="a/b only",
        note="no dog class; on Dog Diving footage it tracks the handler, and the tailset is derived "
             "from the hips",
    ),
)


def dog_specs() -> tuple[ModelSpec, ...]:
    """Every checkpoint that measures a dog tailset (the ones a scale choice can land on)."""
    return tuple(s for s in MODEL_REGISTRY if s.family == FAMILY_DOG)


def specs_for_scale(scale: str) -> tuple[ModelSpec, ...]:
    """Declared dog-family checkpoints for one backbone scale, registry order."""
    return tuple(s for s in dog_specs() if s.scale == str(scale))


def default_for_scale(scale: str) -> ModelSpec:
    """The checkpoint a backbone-scale choice selects.

    Only the dog family is selectable: the stock COCO pose model has no tailset, so "use the s
    weights" can never mean "use the 17-keypoint person model by accident".
    """
    key = str(scale).strip().lower()
    if key not in SCALES:
        raise ValueError(f"unknown model scale {scale!r} — known scales are {', '.join(SCALES)}")
    specs = specs_for_scale(key)
    if not specs:
        raise ValueError(f"no dog-pose checkpoint declared for scale {key!r} "
                         f"(models/ or MODEL_REGISTRY is missing it)")
    return specs[0]


def spec_for_path(path: str | Path) -> ModelSpec | None:
    """The registry entry a checkpoint path belongs to, or ``None`` for an undeclared file."""
    try:
        name = Path(str(path)).name
    except (TypeError, ValueError):
        return None
    for s in MODEL_REGISTRY:
        if s.filename == name:
            return s
    return None


def resolve_model_path(name_or_path: str | Path) -> Path:
    """Absolute path, or a file name resolved inside ``models/`` (``DD_MODEL``'s contract)."""
    p = Path(str(name_or_path)).expanduser()
    return p if p.is_absolute() else (MODELS_DIR / p)


#: The trajectory's pose checkpoint, resolved **inside** ``models/``.
#:
#: Selection order — explicit beats implicit, and every step lands on a declared checkpoint:
#:
#: 1. ``DD_MODEL`` — an absolute path or a file name inside ``models/`` (``DD_MODEL=best.pt``
#:    reproduces the pre-2026-10-19 behaviour).
#: 2. ``DD_MODEL_SCALE`` — ``n`` or ``s``; resolves through :func:`default_for_scale` to that
#:    backbone's dog-pose checkpoint (the GUI's backbone selector writes the same choice).
#: 3. the registry's ``n``-scale model of record (``yolo26n-pose_dogs_v5.pt``), falling back to the
#:    vendored ``best.pt`` when it is absent so the tree stays runnable.
#:
#: **Which checkpoint actually ran is recorded, not assumed:** it is part of ``detect.DetectConfig``
#: and therefore of the cache key, and the result carries ``DetectResult.model_sha`` (also written to
#: ``tracks.json`` and ``run_meta.json``) — a change here invalidates every warm ``tracks.json`` and
#: forces a re-run, which is the intended behaviour.  ``MODEL_SPEC`` / ``MODEL_SCALE`` say which
#: registry entry the resolved path is, so a run can state its backbone rather than infer it from a
#: file name.
MODEL_PATH = MODELS_DIR / "best.pt"                     # the vendored upstream checkpoint
MODEL_SELECTION_NOTE: str | None = None                 # set when an env override was ignored
_env_model = os.environ.get("DD_MODEL", "").strip()
_env_scale = os.environ.get("DD_MODEL_SCALE", "").strip().lower()
if _env_model:
    MODEL_PATH = resolve_model_path(_env_model)
elif _env_scale:
    try:
        MODEL_PATH = default_for_scale(_env_scale).path
    except ValueError as exc:
        MODEL_SELECTION_NOTE = f"DD_MODEL_SCALE ignored: {exc}"
        _default = default_for_scale("n")
        MODEL_PATH = _default.path if _default.is_file() else MODELS_DIR / "best.pt"
else:
    _default = default_for_scale("n")
    MODEL_PATH = _default.path if _default.is_file() else MODELS_DIR / "best.pt"

#: Which registry entry ``MODEL_PATH`` is (``None`` for a checkpoint declared nowhere).
MODEL_SPEC: ModelSpec | None = spec_for_path(MODEL_PATH)
#: The backbone scale the resolved checkpoint runs at — ``"n"``, ``"s"``, or ``"custom"``.
MODEL_SCALE: str = MODEL_SPEC.scale if MODEL_SPEC is not None else "custom"

# reference-recipe defaults (PLAN §6 / draw_tail_trajectory.py)
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.12
DEFAULT_KP_CONF = 0.30
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_DEVICE = "cpu"
DEFAULT_CLAHE_CLIP = 2.5
DEFAULT_CLAHE_GRID = (8, 8)
TAILSET_INDEX = 1
NOSE_INDEX = 0

# trajectory.py (PLAN §6 lists arc_outlier_px; the other two are the M4 knobs)
ARC_OUTLIER_PX = 20          # drop arc points with residual > this (PLAN §6)
TRAJ_MAX_FRAME_GAP = 10      # split accepted points into runs at a larger frame gap
TRAJ_SEED = 0                # fixed RANSAC seed -> reproducible arc fits

# distance.py (PLAN §6 lists waterline_tol_in; the rest are M6's knobs)
WATERLINE_TOL_IN = 6             # PLAN §6 / §5: splash-centre gap warning threshold, inches
DISTANCE_KP_WINDOW = 5           # K of PLAN §4.4's sigma_kp: "the last K accepted tailset frames"
DISTANCE_KP_MODE = "literal"     # "literal" = PLAN §4.4 verbatim; "detrended" = vs the fitted arc
DISTANCE_ROUNDING = "nearest"    # feet+inch display; "floor"/"plus" also available (see distance.py)

# contact.py (PLAN §6 lists none of these; M5's knobs)
CONTACT_KP_CONF = DEFAULT_KP_CONF            # PLAN §5 #3's "tailset conf" threshold (0.30)
CONTACT_COLLAPSE_RUN = 2                     # PLAN §5 #3: "for >= 2 consecutive frames"
CONTACT_VEL_TOL_PX = 20.0                    # PLAN §5 #4 dy/dt departure tolerance (cf. ARC_OUTLIER_PX)
CONTACT_FREEFALL_CURVATURE_MIN = 0.0         # image-y curvature a free-fall arc must exceed
CONTACT_MIN_DESCENT_PX_PER_FRAME = 1.0       # what counts as "the arc was still descending"
CONTACT_MAX_EXTRAP_FRAMES = 90               # bound on candidate #2's extrapolation past the data
CONTACT_BOX_CONF_FLOOR = None         # PLAN §5 #3's "box conf" trigger; measured unusable (<0.25)

# ---------------------------------------------------------------------------
# pool-translation-plan.txt (the 3-D route; docs/POOL_TRANSLATION_IMPACT.md)
# ---------------------------------------------------------------------------
# Phase 1 — a focal prior when nothing better exists.  ``source="assumed"`` is recorded in the
# document, so a distance can never be quoted without saying the intrinsics were not measured.
POOL_FOCAL_RATIO = 1.0                # f / max(image_width, image_height)
# Phase 3 — the plan's "standard dock diving pool dimensions (e.g. 14 to 20 feet wide)".  A PRIOR,
# and route B's only transverse anchor; ``homography.width_sensitivity`` prices it in inches per
# foot of width so the cost is reported rather than assumed.
POOL_WIDTH_FT = 16.0
POOL_WIDTH_PRIOR_BAND_FT = (14.0, 20.0)
# 2026-10-20 operator order — the metric axis is FIXED: +X always runs from the dock lip to 35 ft.
# ``PlaneSession.length_ft`` is no longer the measured pool length; it is only the anchor saying
# where the operator's quad far edge sits, so the grid can be lined up against detail visible in
# the video.  Everything that reports a distance, draws a grid or clips to a pool extent uses this
# constant, never the anchor.
POOL_LENGTH_FT = 35.0
POOL_MARKS_ON_EDGE = "near"           # which long edge the painted marks sit on
# Phase 5 — used only when the operator supplies no rim pair and no contact: Dog Diving' standard
# dock height above the water.  Recorded as a specification, never as a measurement.
POOL_FREEBOARD_IN = 19.5
POOL_WATER_METHOD = "rim_pair"        # rim_pair | wall_contact | known_freeboard
POOL_WALL_AXIS = "y"                  # the vertical plane the water contact lies on
POOL_WALL_VALUE = 0.0                 # its world offset in feet
POOL_EXTENT_MARGIN_FT = 1.0           # slack on the (X, Y)-inside-the-pool sanity gate
# ---------------------------------------------------------------------------
# Adaptive detection threshold (operator instruction, 2026-10-12)
# ---------------------------------------------------------------------------
# "Dynamically change the threshold so that a minimum of 8 points are identified, with the
# threshold being lowered if 8 points can't be found with the current confidence level."
#
# Rungs are tried in order of conservatism — `conf` first, then `kp_conf` — and the FIRST rung
# whose accepted-keypoint-row count reaches MIN_ACCEPTED_POINTS wins.  The search starts at the
# base config's own thresholds and can only lower them, never raise them, so a clip that already
# has >= MIN_ACCEPTED_POINTS accepted rows is untouched and costs exactly what it costs today.
#
# MEASURED CAVEAT — slow-mo clip `Dog Dock Diving Slow Mo 2-...-seg07` (149 frames), 2026-10-12.
# This checkpoint's confidence does not separate its right answers from its wrong ones, so a
# lowered rung buys points of unknown provenance: conf 0.12 -> 3 accepted, conf 0.06 -> 78
# accepted, and the added points include EMPTY SKY (frame 50, kp conf 0.960) and a PERSON at the
# pool edge (frame 130, kp conf 0.995).  Raising imgsz 1280 -> 1920 did NOT help there (3 accepted
# either way, different frames).  Every point admitted by a lowered rung is therefore marked
# `low_conf` in the records and counted in result.json / report.txt, so a distance can never be
# quoted without knowing which points the ladder bought.
MIN_ACCEPTED_POINTS = 8              # 0 disables the ladder and restores the fixed-threshold behaviour
CONF_LADDER = (0.12, 0.06, 0.03, 0.015, 0.01)
KP_CONF_LADDER = (0.30, 0.20, 0.10, 0.05)
IMGSZ_LADDER = (1280, 1920)          # the final rung, tried only when every conf/kp rung is short

# Phase 6 — the plank length reported in the banner when a pool calibration is present.
POOL_GRID_UPTO_FT = 35.0              # the metric axis runs to 35 ft (see POOL_LENGTH_FT)
