"""app_gui — Tkinter wiring for PLAN §5's A→F flow (M8 deliverable).

PLAN §5, ``app_gui.py`` — Tkinter flow (D5)::

    1. Load      — file picker; show frame 0, resolution/fps/frame count; run the
                   stability check; block with a clear message if `unusable`.
    2. Calibrate — New Plane: one button that (re)makes the quad, runs the waterline detector on
                   the clip's first frames, snaps the quad onto the detected lines (far edge short,
                   near edge long) and arms the Snap to Waterline toggle; the pool's two short ends
                   are detected as well as its two long waterlines, and X = 0 lands on whichever end
                   the orientation control selects.  Plus two anchors (near dock / opposite dock),
                   Re-detect Waterline, Flip Plane (left-right, which changes the NUMBER scale only)
                   and Complete; Save New Plane sits with the saved-plane picker and keys the
                   document to the clip's sha256 instead of a typed setup id.  A clip that has a
                   saved plane gets it automatically on load.
    3. Process   — `Run` in a worker thread; progress bar with frame count and ETA
                   (measured throughput, not guessed); `Cancel`; "using cached
                   tracks" indicator; a quick `Preview on this frame` button.
    4. Review    — frame slider + prev/next/+/-5/+/-10; frame canvas with the trajectory, the arc,
                   the lost points and the tailset X drawn as zoom-scaled canvas items (1 px dashed at
                   fit, 2-3 px closed when zoomed in) under an *element opacity* slider that starts at
                   10 %; the plan view where the candidate table used to be; a point list plus a
                   drag-a-box deselect mode; Correct Entry, which arms a drag mode that moves the
                   tailset X (a small dot under the pointer) and cuts the trajectory where it is
                   dropped; Set entry frame / entry = slider frame for the frame-based override;
                   export buttons.
    5. Results   — distance in ft/in with +/-, calibration residuals, stability
                   verdict, uncertainty breakdown, export annotated MP4 / still PNG
                   / JSON / TXT; session log pane.

    Threading rule: inference and rendering never run on the Tk main thread; the GUI
    stays responsive and cancellable.

Where this file closes a gap PLAN §5 does not have
---------------------------------------------------
PLAN §5 asks for per-frame progress **with ETA** and a **Cancel** button, but
``cli_run.run_clip`` — the Run-button call — exposes neither: its ``progress`` is
``progress(stage, {})`` with no frame counts, and it has **no cancel parameter at
all**.  The long stage by far is detection, and ``detect.detect_clip`` *does* take
``progress_cb(done, total, eta_s)`` and a ``cancel_event``.  So the worker drives
detection directly and then calls ``run_clip`` with a warm cache to emit PLAN §7's
artifact set.  Established by ``out/_m8_gate/gui_contract_probe.py`` (Q1/Q2/Q5).

The honest consequences, surfaced in the UI rather than hidden:

* **Cancel reaches inference only.**  It cannot interrupt trajectory / contact /
  annotate / encode, because ``run_clip`` offers no hook.  The Process pane says so.
* **A warm cache never consults ``cancel_event``** — ``detect_clip`` returns from the
  cache immediately, which is also why Run is fast on a second pass.  A cancelled run
  never writes the cache, so it can never be mistaken for a complete one.
* **M8's PLAN §11 gate is "full manual run end-to-end on one clip without touching the
  CLI"** — that click-through IS the gate and it needs a person.  This file makes the
  flow possible, and its logic (frame selection, live readout, calibration wiring,
  cancellable pipeline, artifact export) is exercised headlessly by
  ``src/tests/test_m8_gui.py``.  It does not and cannot self-certify the milestone;
  see HANDOFF.

Deliberately NOT implemented here: the "current team / previous team / accept score"
session model from ``GUI Brainstorm - Dog Diving.pdf``.  HANDOFF open item 16 records
it as a **team** decision ("do not invent it").  This file implements PLAN §5's flow.

Run::

    PYTHONPATH=src <venv-python> src/app_gui.py [--clip data/clips/18-07.mp4]
"""

from __future__ import annotations

import argparse
import math
import queue
import shutil
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import cli_run  # noqa: E402
from dog_diving.pool import pipeline as pipeline_mod  # noqa: E402
from dog_diving.pool import plane as plane_mod  # noqa: E402
from dog_diving import annotate as annotate_mod  # noqa: E402
from dog_diving import calibration as calibration_mod  # noqa: E402
from dog_diving import detect as detect_mod  # noqa: E402
from dog_diving import distance as distance_mod  # noqa: E402
from dog_diving import measure as measure_mod  # noqa: E402
from dog_diving import report as report_mod  # noqa: E402
from dog_diving import stability as stability_mod  # noqa: E402
from dog_diving import waterline as waterline_mod  # noqa: E402
from dog_diving.pool import seed as seed_mod  # noqa: E402
from dog_diving import trajectory as trajectory_mod  # noqa: E402
from dog_diving import video as video_mod  # noqa: E402
from dog_diving.pool import calib as pool_calib_mod  # noqa: E402
from dog_diving.config import (  # noqa: E402
    CALIB_DIR,
    CLIP_DIR,
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_KP_CONF,
    MIN_ACCEPTED_POINTS,
    MODELS_DIR,
    MODEL_SELECTION_NOTE,
    OUT_DIR,
    POOL_WIDTH_FT,
    dog_specs,
    spec_for_path,
)

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
CALIB_FRAME_DEFAULT = 0
MAX_ZOOM = 16.0          # 1 image px becomes 16 canvas px; above this the frame is unreadable
ZOOM_STEPS = (1.0, 2.0, 4.0, 8.0, 16.0)
#: Shown in the backbone combobox when the selected checkpoint is not declared in
#: ``config.MODEL_REGISTRY`` (a browsed file, or the vendored ``best.pt``) — the file choice then
#: decides, and the scale box says so rather than guessing a backbone from the file name.
SCALE_CUSTOM_LABEL = "custom (file choice)"

#: How much of a clip the waterline detector looks at (operator order 2026-10-21: *"use the first 90
#: frames"*, after trying 10).  The old path searched for a 60-150 frame calm window guided by the
#: pose detector, which needed a finished run; the opening frames are the ones the operator is looking
#: at and nothing has swum into the water yet.  Measured on video004 seg01 (92 frames, 960x600):
#: 10 frames finds only the near waterline in 0.4 s, 90 frames finds far+near in 2.1 s — the temporal
#: noise map needs the longer window to separate the far wall from the sky.
SEED_FRAME_COUNT = 90

#: The two +x anchors, in feet from the dock: the quad's dock-side edge and its far edge.  The
#: defaults are the whole 35 ft pool, which is what a new plane starts as.
SEED_ANCHOR_NEAR_FT = 0.0
SEED_ANCHOR_FAR_FT = float(plane_mod.METRIC_LENGTH_FT)

#: The snap toggle's four boxes, in the order they appear: plane side -> label.  ``back``/``front``
#: are the two long waterlines (far/near), ``left``/``right`` the pool's two short ends (X = 0/X = L).
SNAP_SIDE_LABELS = (("back", "Top"), ("front", "Bottom"), ("left", "Left"), ("right", "Right"))

#: Which box is checked when the pane is built: the far waterline only.  Operator order 2026-10-21 —
#: *"start with Snap on only for the top waterline"* — a fresh quad is placed against the far line
#: first, and the other three edges stay free for the operator to drag.
SNAP_DEFAULT_SIDE = "back"

#: The blue primary-action colours.  ``tk.Button`` is used for these: the Windows ``vista`` ttk theme
#: paints ttk buttons itself and ignores ``background``, so a ttk "blue" button is a grey one.
PRIMARY_BG = "#1f6feb"
PRIMARY_BG_ACTIVE = "#388bfd"

#: Review's element opacity at startup, in percent (operator order 2026-10-21, revised: *"set opacity
#: at 90% default"*).  Tk cannot alpha-composite canvas items over a photo, so :meth:`App._blend`
#: applies the same arithmetic an alpha blend would — colour = a*ink + (1-a)*the frame pixel under it.
OPACITY_DEFAULT_PCT = 90.0

#: Grab radius, in SCREEN px, for taking hold of the entry marker.  Screen px so it behaves the same
#: at every zoom instead of demanding a pixel-perfect hit on a 3 px mark.
ENTRY_GRAB_PX = 14.0

#: The entry/tailset marker's red — the same red ``annotate.C_POINT`` puts in the exported still, so
#: what Review shows and what the still shows are one style rather than two.
ENTRY_INK = "#ff4040"

#: The Review lattice's colours — the same values ``plane.draw_grid_on`` bakes into an exported
#: still, so the canvas lattice and the still's lattice are one lattice rather than two greens.
GRID_MAJOR = "#78d278"
GRID_MINOR = "#557355"
GRID_LABEL = "#96f096"
GRID_DATUM = "#ff783c"
#: How much of the element opacity the lattice keeps ABOVE the landing row (that function's own
#: ``fade_alpha``), so the dog on its final frame is not covered by the grid behind it.
GRID_FADE_ALPHA = 0.35


# ===========================================================================
# Non-Tk core — everything below this divider is importable and testable
# without a display.
# ===========================================================================
@dataclass
class Progress:
    """One progress report from the worker to the UI."""

    stage: str
    done: int = 0
    total: int = 0
    eta_s: float | None = None
    from_cache: bool | None = None
    message: str = ""

    @property
    def fraction(self) -> float:
        return (self.done / self.total) if self.total else 0.0

    def label(self) -> str:
        if self.stage == "detect" and self.total:
            eta = f"  ETA {self.eta_s:.0f}s" if self.eta_s is not None else ""
            return f"detect {self.done}/{self.total} frames{eta}"
        return self.stage


@dataclass
class PipelineOutcome:
    detect: Any | None = None
    run: dict[str, Any] | None = None
    cancelled: bool = False
    error: str | None = None
    tb: str | None = None


def run_pipeline(
    clip: str | Path,
    *,
    calib: dict | None = None,
    calib_path: str | None = None,
    pool_calib: dict | None = None,
    pool_calib_path: str | None = None,
    pool_height_ft: float = 0.0,
    cfg: detect_mod.DetectConfig | None = None,
    out_dir: str | Path | None = None,
    contact: str | int |None = "auto",
    manual_point: tuple[float, float] | None = None,
    subclip: tuple[int, int] | None = None,
    use_cache: bool = True,
    model: Any | None = None,
    draw: bool = True,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    min_points: int | None = None,
) -> PipelineOutcome:
    """Detect (cancellable, per-frame) then finish through ``run_clip`` (warm cache).

    ``min_points`` is the operator's adaptive-threshold minimum (2026-10-12): the detection stage
    here is ``detect.detect_clip_adaptive``, and ``run_clip`` is called with the same value so the
    artifacts describe the same detection this pane displayed.  Both calls resolve to the same cache
    entries, so the second one is free.  ``min_points=0`` restores the fixed thresholds.

    Never raises: a failure comes back as ``PipelineOutcome.error`` so the worker
    thread can always report something to the UI.
    """
    clip = Path(clip)
    cfg = cfg or detect_mod.DetectConfig(
        imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF, kp_conf=DEFAULT_KP_CONF
    )

    def emit(**kw: Any) -> None:
        if on_progress is not None:
            on_progress(Progress(**kw))

    def frame_cb(done: int, total: int, eta_s: float) -> None:
        emit(stage="detect", done=int(done), total=int(total), eta_s=eta_s)

    def stage_cb(stage: str, _payload: dict) -> None:
        emit(stage=stage, message=stage)

    try:
        det = detect_mod.detect_clip_adaptive(
            clip, cfg, progress_cb=frame_cb, cancel_event=cancel_event,
            frame_range=subclip, out_dir=out_dir, use_cache=use_cache, model=model,
            min_points=min_points,
        )
        cancelled = bool((getattr(det, "timings", None) or {}).get("cancelled"))
        emit(
            stage="detect",
            done=int(det.n_frames_processed),
            total=int(det.n_frames_processed),
            from_cache=bool(det.from_cache),
            message="cancelled" if cancelled else ("cached" if det.from_cache else "detected"),
        )
        if cancelled:
            return PipelineOutcome(detect=det, cancelled=True)

        run = cli_run.run_clip(
            clip, calib=calib, calib_path=calib_path, detect_cfg=cfg, out_dir=out_dir,
            contact=contact, manual_point=manual_point, subclip=subclip,
            use_cache=use_cache, model=model, draw=draw, progress=stage_cb,
            pool_calib=pool_calib, pool_calib_path=pool_calib_path,
            pool_height_ft=float(pool_height_ft), min_points=min_points,
        )
        emit(stage="done", message="complete")
        return PipelineOutcome(detect=det, run=run)
    except Exception as exc:  # noqa: BLE001 - the UI must always get a report
        emit(stage="error", message=f"{type(exc).__name__}: {exc}")
        return PipelineOutcome(error=f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())


def read_frame(clip: str | Path, index: int):
    """One frame by index, via ``video.iter_frames`` (never CAP_PROP_POS_FRAMES seeking —
    that is unreliable on the VFR MOVs, M1)."""
    idx = max(0, int(index))
    for _i, fr in video_mod.iter_frames(clip, start=idx, stop=idx + 1):
        return fr
    return None


def annotated_frame(
    clip: str | Path,
    index: int,
    *,
    traj: Any | None = None,
    contact: Any | None = None,
    calib: dict | None = None,
    measurement: dict | None = None,
    show_grid: bool = True,
    pool_doc: dict | None = None,
    pool_result: dict | None = None,
    entry_mark: bool = True,
    fade_above_y: float | None = None,
    arc: dict | None = None,
    excluded: set[int] | tuple[int, ...] = (),
):
    """The overlay the Review canvas shows for an arbitrary frame."""
    fr = read_frame(clip, index)
    if fr is None:
        return None
    return annotate_mod.draw_frame(
        fr, frame_idx=int(index), traj=traj, contact=contact, calib=calib,
        measurement=measurement, show_grid=show_grid, show_candidates=False,
        pool_doc=pool_doc, pool_result=pool_result, entry_mark=entry_mark,
        fade_above_y=fade_above_y, arc=arc, excluded=excluded,
    )


def measurement_rows(meas: dict | None) -> list[tuple[str, str]]:
    """Results-pane rows.  Inches stay ``None`` without a calibration — never invented."""
    if not meas:
        return []
    def num(key: str, unit: str = "", prec: int = 2) -> str:
        v = meas.get(key)
        return "—" if v is None else f"{v:.{prec}f}{unit}"
    total = meas.get("total_inches")
    route = str(meas.get("route") or "")
    if total is not None:
        disp = meas.get("display")
    elif route == measure_mod.ROUTE:
        disp = "n/a (no final detection)"
    else:
        disp = "n/a (no plane calibration)"
    rows = [
        ("Distance", str(disp or "—")),
        ("Total inches", "—" if total is None else f"{total:.2f}"),
        ("Route", route or "—"),
        ("Frame (final detection)", "—" if meas.get("frame") is None else str(meas.get("frame"))),
        ("Transverse", "—" if meas.get("transverse_ft") is None
            else f"{meas['transverse_ft']:.2f} ft"),
    ]
    if route == measure_mod.ROUTE:
        extra = [
            ("Point source", str(meas.get("point_source") or "—")),
            ("Measured at (px)", "—" if meas.get("x_px") is None
                else f"({meas['x_px']:.1f}, {meas['y_px']:.1f})"),
            ("Calibration", "plane (PLAN §15)"),
        ]
        if meas.get("excluded_frames"):
            extra.append(("Deselected in Review",
                          f"{len(meas['excluded_frames'])} point(s) out of the trajectory"))
        if meas.get("entry_override"):
            extra.append(("Entry frame (operator)",
                          f"{meas.get('frame')} — this point, not the final detection"))
        return rows + extra
    return rows + [
        ("Uncertainty", "—" if meas.get("sigma_in") is None else f"± {meas['sigma_in']:.2f} in"),
        ("Uncertainty complete", str(meas.get("sigma_complete"))),
        ("Point source", str(meas.get("point_source") or "—")),
        ("Measured at (px)", "—" if meas.get("x_px") is None
            else f"({meas['x_px']:.1f}, {meas['y_px']:.1f})"),
        ("Calibration", str(meas.get("calib_setup_id") or "none")),
        ("Waterline gap", num("waterline_gap_in", " in")),
    ]


def residual_rows(calib: dict | None) -> list[tuple[str, str, str]]:
    """PLAN §5 step 2's residual table: mark, x_px, per-mark error."""
    if not calib:
        return []
    marks = calib.get("marks") or []
    errs = (calib.get("fit") or {}).get("per_mark_error_in") or []
    rows = []
    for i, m in enumerate(marks):
        err = errs[i] if i < len(errs) else None
        rows.append((
            str(m.get("label", f"#{i}")),
            f"{float(m['x_px']):.2f}",
            "—" if err is None else f"{err:+.2f}",
        ))
    return rows


def ladder_rows(ladder: dict | None) -> list[tuple[str, str]]:
    """The adaptive-threshold ladder's rows for the Results pane (operator feature, 2026-10-12).

    Shown rather than hidden because a lowered threshold changes *which* points the distance rests
    on: on the slow-mo corpus clip 75 of the 78 accepted points were bought by lowering `conf` to
    0.06, and the measured eye-check on that clip found the added points on empty sky and on a
    person at the pool edge.
    """
    if not ladder:
        return []
    if ladder.get("disabled"):
        return [("threshold ladder", "off — fixed thresholds (min_points=0)")]
    ch = ladder.get("chosen") or {}
    rows = [
        ("threshold ladder",
         f"min {ladder.get('min_points')} pt -> imgsz={ch.get('imgsz')} conf={ch.get('conf')} "
         f"kp_conf={ch.get('kp_conf')}"),
        ("ladder points",
         f"{ladder.get('n_points')} accepted, {ladder.get('n_low_conf_points')} low-conf (admitted "
         f"only below conf={ch.get('base_conf')} / kp_conf={ch.get('base_kp_conf')})"),
        ("ladder rungs",
         f"{len(ladder.get('rungs') or [])} rung(s) tried"
         + (f"; imgsz escalated {ladder.get('imgsz_tried')}"
            if ladder.get("imgsz_escalated") else "")),
    ]
    if ladder.get("below_min_points"):
        rows.append(("ladder WARNING", "BELOW MINIMUM even at the floor rung"))
    return rows


def pool_rows(pool_result: dict | None, pool_doc: dict | None) -> list[tuple[str, str]]:
    """Results-pane rows for the 3-D route.  Empty when the route did not run.

    Kept separate from :func:`measurement_rows` so the 1-D measurement's rows stay exactly as they
    were — the 3-D entry is an *addition* to the pane, not a replacement, because the cross-check
    between the two routes is the evidence that either is right.
    """
    rows: list[tuple[str, str]] = []
    if pool_doc is not None:
        water = pool_doc.get("water") or {}
        rows.append(("pool setup", f"{pool_doc.get('setup_id')} ({pool_doc.get('route')})"))
        if water.get("z_ft") is not None:
            rows.append(("water plane", f"z = {float(water['z_ft']):.3f} ft "
                                        f"(method {water.get('method')})"))
    if pool_result is None:
        return rows
    entry = pool_result.get("entry") or {}
    if entry.get("X_ft") is not None and entry.get("Y_ft") is not None:
        rows.append(("3-D entry (X, Y)", f"({float(entry['X_ft']):.2f}, {float(entry['Y_ft']):.2f}) ft"))
        rows.append(("3-D plank length", f"{float(entry['distance_ft']):.2f} ft "
                                         f"= {float(entry['distance_in']):.2f} in"))
        rows.append(("3-D transverse", f"{float(entry['transverse_ft']):.2f} ft"))
        rows.append(("3-D entry source", f"{entry.get('point_source')} @ {entry.get('frame')}"))
        if entry.get("height_assumed_ft") is not None:
            rows.append(("height assumed", f"{float(entry['height_assumed_ft']):.2f} ft"))
        pb = entry.get("parallax") or {}
        if pb.get("ok"):
            rows.append(("parallax if height real",
                         f"{abs(pb['dX_in']):.1f} in "
                         f"({'short' if pb['short'] else 'long'})"))
    else:
        rows.append(("3-D entry", "none (see notes)"))
    cc = pool_result.get("cross_check") or {}
    if cc.get("compared"):
        rows.append(("route cross-check",
                     f"3-D {cc['d3d_in']:.2f} in vs 1-D {cc['d1d_in']:.2f} in "
                     f"({cc['diff_in']:+.2f} in, {'AGREE' if cc['agree'] else 'DISAGREE'})"))
    flags = [k for k, v in (pool_result.get("flags") or {}).items() if v]
    if flags:
        rows.append(("3-D flags", ", ".join(sorted(flags))))
    return rows


def gate_rows(calib: dict | None) -> list[tuple[str, str, bool]]:
    """PLAN §4.3's G1–G5 as (gate, detail, passed)."""
    if not calib:
        return []
    out = []
    for name, g in calibration_mod.validate_gates(calib).items():
        label = f"{name}: {'PASS' if g['pass'] else 'FAIL'}"
        if g.get("warn"):
            label += " (warn)"
        out.append((label, str(g.get("detail", "")), bool(g["pass"])))
    return out


def _finite2(p) -> bool:
    """True when a projected point is a real, finite 2-D pixel."""
    try:
        return bool(math.isfinite(float(p[0])) and math.isfinite(float(p[1])))
    except (TypeError, ValueError, IndexError):
        return False


def guard_frame(canvas_w: int, canvas_h: int, img_w: int, img_h: int) -> float:
    """Scale factor that fits an image inside a canvas without upscaling."""
    if img_w <= 0 or img_h <= 0:
        return 1.0
    return min(1.0, canvas_w / img_w, canvas_h / img_h)


def _hex_rgb(colour: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> ``(r, g, b)``.  Tk colour names fall back to mid-grey rather than raising."""
    try:
        s = str(colour).lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return (128, 128, 128)


@dataclass
class ReviewState:
    """PLAN §5 step 4's frame picker + live readout, with no Tk in it."""

    clip: Path
    traj: Any
    contact: Any
    calib: dict | None
    image_size: tuple[int, int]
    n_frames: int = 0
    manual_point: tuple[float, float] | None = None
    #: The pool document the run used.  When it is a PLANE document, Review measures from ITS
    #: homography — the same map the calibration claims — and the 1-D number becomes a comparison.
    pool_doc: dict | None = None
    #: Frames the operator clicked OUT of the fitted arc.  The *measurement* is untouched (PLAN §16.2
    #: takes the final detection, not the arc) — this only changes the arc that is drawn and reported.
    excluded: set[int] = field(default_factory=set)
    #: The frame whose detection is taken as the water-entry point (operator order 2026-10-21).  None
    #: means the PLAN §16.2 rule: the final accepted detection.  This one DOES change the number —
    #: it is the scoring override, and it is recorded as one.
    entry_frame: int | None = None
    #: When True the entry follows the frame slider, so stepping frames walks the 2D->3D translation
    #: live instead of needing a click per candidate frame.
    entry_live: bool = False

    def cutoff(self) -> int | None:
        """The frame the trajectory is drawn to and measured at, or None for the whole flight."""
        if self.entry_live and self.frame is not None:
            return self._snap(int(self.frame))
        if self.entry_frame is not None:
            return int(self.entry_frame)
        return None

    def _snap(self, frame: int) -> int:
        """The nearest KEPT detection's frame — an entry frame is never a frame with no point."""
        pts = measure_mod.kept_points(self.traj, self.excluded)
        if not pts:
            return int(frame)
        return int(min(pts, key=lambda p: abs(int(p["frame"]) - int(frame)))["frame"])

    def set_entry_frame(self, frame: int | None) -> int | None:
        """Pin (or clear) the entry frame, snapped to a detection that exists and is kept."""
        self.entry_frame = None if frame is None else self._snap(int(frame))
        return self.entry_frame

    def draw_points(self, *, kept_only: bool = True) -> list[dict]:
        """The points the canvas draws: kept, and no later than the entry frame."""
        pts = (measure_mod.kept_points(self.traj, self.excluded) if kept_only
               else list(getattr(self.traj, "points", None) or []))
        cut = self.cutoff()
        if cut is not None:
            pts = [p for p in pts if int(p["frame"]) <= int(cut)]
        return pts

    def arc(self) -> dict:
        """The fitted arc over the points the operator kept, ending at the entry frame.

        With nothing excluded and no entry frame this is the trajectory's own arc, unchanged;
        otherwise it is a refit of what is left, so what Review *draws* and what it *reports* cannot
        disagree.
        """
        pts = self.draw_points()
        if not self.excluded and self.cutoff() is None:
            return {"coeffs": getattr(self.traj, "coeffs", None),
                    "inliers": getattr(self.traj, "inliers", []) or [],
                    "frame_range": getattr(self.traj, "airborne_frame_range", None),
                    "n": len(pts), "rms_px": getattr(self.traj, "arc_rms_px", None)}
        return trajectory_mod.refit(pts)

    def point_rows(self) -> list[dict]:
        """Every accepted detection, frame order, with whether it is currently included.

        Backs Review's point list: the click on a marker and the row in the table are two views of
        one set (``excluded``), so keeping them in step needs no second source of truth.
        """
        rows = []
        for p in (getattr(self.traj, "points", None) or []):
            f = int(p["frame"])
            rows.append({"frame": f, "x": float(p["x"]), "y": float(p["y"]),
                         "kp_conf": p.get("kp_conf"), "included": f not in self.excluded,
                         "low_conf": bool(p.get("low_conf"))})
        return rows


    def include_all(self) -> None:
        self.excluded.clear()

    def toggle_point(self, frame: int) -> bool:
        """Include/exclude one frame's detection from the arc.  Returns True if it is now excluded."""
        frame = int(frame)
        if frame in self.excluded:
            self.excluded.discard(frame)
            return False
        self.excluded.add(frame)
        return True

    def nearest_point(self, x: float, y: float) -> tuple[float, dict] | None:
        """The accepted detection closest to an image-pixel click, with its distance."""
        best: tuple[float, dict] | None = None
        for p in (getattr(self.traj, "points", None) or []):
            d = float(np.hypot(float(p["x"]) - float(x), float(p["y"]) - float(y)))
            if best is None or d < best[0]:
                best = (d, p)
        return best

    def choose(self, frame: int, source: str = "manual"):
        self.contact = self.contact.choose(int(frame), source)
        return self.contact

    def nudge(self, delta: int):
        self.contact = self.contact.nudge(int(delta))
        return self.contact

    @property
    def frame(self) -> int | None:
        return self.contact.chosen_frame

    def measurement(self) -> dict:
        """The distance: the entry point, assumed on the plane; its +X.

        No contact-frame candidates and no 1-D fit — the operator's rule (PLAN §16) is that the last
        accepted detection *is* the intersection with the plane, so the number is
        ``measure.plane_distance``.  A manual click overrides the point under the same assumption.

        ``excluded`` (Review's deselections) is passed through, so a confident detection of a second
        dog cannot be the point that scores: the distance falls back to the highest-frame KEPT point.
        ``cutoff()`` (the operator's entry frame) pins the sample to that frame instead — the
        operator's rule for a mistimed last detection.
        """
        cut = self.cutoff()
        return measure_mod.plane_distance(
            self.traj, self.pool_doc, manual_point=self.manual_point, excluded=self.excluded,
            frame=cut, frame_is_operator_choice=cut is not None)

    def readout(self) -> str:
        m = self.measurement()
        frame = self.contact.chosen_frame
        tail = (f"   [{len(self.excluded)} point(s) deselected]") if self.excluded else ""
        if self.entry_frame is not None or (self.entry_live and self.frame is not None):
            tail += f"   [entry f{self.cutoff()} by operator]"
        if m.get("total_inches") is None:
            why = (m.get("warnings") or ["no measurement"])[0]
            return f"frame {frame}: distance n/a — {why}{tail}"
        at = m.get("frame")
        where = "final detection" if m.get("point_source") == "model" else str(m.get("point_source"))
        return (f"{m.get('display')}   [plane·{where}"
                + (f" f{at}" if at is not None else "") + "]" + tail)


# ===========================================================================
# Tk layer
# ===========================================================================
class FrameView:
    """A Canvas showing one BGR frame with **zoom + pan**, mapping clicks back to image px.

    Zoom is not a nicety here.  A 1920-px frame shown at fit-to-window on a 900-px canvas
    is drawn at ~0.4x, so **one canvas pixel spans 2.5 image pixels** — the operator
    cannot express even 1 px of placement precision, let alone the ~0.2 px the M2
    measurement says a foot-mark calibration needs.  (Measured, M2 tool section: mark
    localisation error converts to distance at ~5.1 in per px.)

    Rendering crops to the visible region *before* scaling.  Scaling the whole frame at
    16x would be a 530-megapixel image, so only the viewport is ever resized.

    Click mapping goes through ``Canvas.canvasx/canvasy``, which fold in the scroll
    position, so zoom, pan and scrollbars all stay consistent with where the operator
    actually pressed.
    """

    def __init__(self, canvas, max_w: int = 900, max_h: int = 520) -> None:
        self.canvas = canvas
        # ``max_w``/``max_h`` are the *minimum* pane size (used before the widget is mapped); the
        # live viewport drives the fit once Tk reports a size, so the frame scales with the window.
        self.max_w, self.max_h = max_w, max_h
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.on_change: Callable[["FrameView"], None] | None = None
        #: Optional vector overlay, re-drawn AFTER the photo on every render.  Drawing the plane as
        #: canvas items instead of into the image keeps its lines 1-2 *screen* px at any zoom and
        #: means a nudge does not have to re-resize the frame.
        self.overlay: Callable[["FrameView"], None] | None = None
        self.cursor_var = None            # optional tk.StringVar for an image-px readout
        self._src = None                  # RGB numpy array of the current frame
        self._photo = None                # keep a reference or Tk garbage-collects it
        self._shown = False               # False until the first frame is displayed
        self._at_fit = True               # True while the operator has not zoomed away from fit
        self._last_size: tuple[int, int] | None = None
        #: Optional (x0, y0, x1, y1) in image px of the region worth looking at.  The Calibrate pane
        #: shows a frame padded by 35 % so corners can sit outside it, and fitting *that* wastes
        #: three quarters of the pane on black margin — so fit and the zoom buttons are anchored on
        #: this box instead, and the padding is reached by panning.
        self.content_box: tuple[float, float, float, float] | None = None
        self._forced_viewport: tuple[int, int] | None = None   # for tests

    # -- geometry -----------------------------------------------------------
    def _viewport(self) -> tuple[int, int]:
        if self._forced_viewport is not None:
            return self._forced_viewport
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _fit_size(self) -> tuple[int, int]:
        """The size the fit scale is computed against: the live canvas, else the constructor size.

        An unmapped Tk widget reports 1x1, so using it would make the first frame invisible;
        falling back to the constructor size keeps the pre-map render sane and ``on_resize`` fixes
        it the moment Tk maps the canvas.
        """
        if self._forced_viewport is not None:
            return self._forced_viewport
        cw, ch = int(self.canvas.winfo_width()), int(self.canvas.winfo_height())
        if cw <= 1 or ch <= 1:
            return self.max_w, self.max_h
        return cw, ch

    @property
    def image_size(self) -> tuple[int, int] | None:
        return None if self._src is None else (self._src.shape[1], self._src.shape[0])

    def _clamp_zoom(self, z: float) -> float:
        return max(self.fit_zoom, min(MAX_ZOOM, float(z)))

    def _fit_source_size(self) -> tuple[int, int] | None:
        """The (w, h) the fit scale is computed over: the content box if set, else the whole image."""
        if self._src is None:
            return None
        h, w = self._src.shape[:2]
        box = self.content_box
        if box is None:
            return w, h
        bw = max(1.0, float(box[2]) - float(box[0]))
        bh = max(1.0, float(box[3]) - float(box[1]))
        return int(round(bw)), int(round(bh))

    def _centre_on_content(self) -> None:
        """Scroll so the content box sits in the middle of the pane."""
        box, size = self.content_box, self.image_size
        if box is None or size is None:
            return
        W, H = size
        z = self.zoom or 1.0
        sw, sh = max(1.0, W * z), max(1.0, H * z)
        cx = 0.5 * (float(box[0]) + float(box[2])) * z
        cy = 0.5 * (float(box[1]) + float(box[3])) * z
        cw, ch = self._viewport()
        self.canvas.xview_moveto(min(1.0, max(0.0, (cx - cw / 2.0) / sw)))
        self.canvas.yview_moveto(min(1.0, max(0.0, (cy - ch / 2.0) / sh)))

    def _update_scrollregion(self) -> None:
        size = self.image_size
        if size is None:
            return
        w, h = size
        self.canvas.configure(scrollregion=(0, 0, w * self.zoom, h * self.zoom))

    # -- display ------------------------------------------------------------
    def show(self, bgr, *, reset_view: bool = False) -> None:
        """Display ``bgr``.  Zoom/pan are kept unless ``reset_view`` (used on load)."""
        from PIL import Image  # local: keeps the module importable without Pillow

        if bgr is None:
            self._src = None
            self.canvas.delete("all")
            self.canvas.create_text(self.max_w // 2, self.max_h // 2, text="no frame", fill="#888")
            self._notify()
            return
        self._src = bgr[:, :, ::-1]                     # BGR -> RGB
        h, w = self._src.shape[:2]
        cw, ch = self._fit_size()
        src_w, src_h = self._fit_source_size() or (w, h)
        self.fit_zoom = guard_frame(cw, ch, src_w, src_h)
        first = reset_view or not self._shown
        if first:
            # first frame of a pane: fit it.  Later redraws keep the operator's zoom.
            self.zoom = self.fit_zoom
            self._shown = True
            self._at_fit = True
        else:
            self.zoom = self._clamp_zoom(self.zoom)     # a new clip can change the fit scale
            self._at_fit = abs(self.zoom - self.fit_zoom) < 1e-6
        self._update_scrollregion()
        if first:
            # scroll BEFORE rendering: _render crops to what the scrollbars currently show
            self._centre_on_content()
        self._render()

    def on_resize(self, event=None) -> None:
        """Re-fit to the window.  A pane left at Fit follows the window; a zoomed pane keeps its
        zoom (clamped) so resizing does not throw away the operator's detail work."""
        if self._src is None:
            return
        if event is not None and getattr(event, "widget", None) is not self.canvas:
            return
        size = (int(getattr(event, "width", 0) or self._fit_size()[0]),
                int(getattr(event, "height", 0) or self._fit_size()[1]))
        if size[0] <= 1 or size[1] <= 1 or size == self._last_size:
            return
        self._last_size = size
        h, w = self._src.shape[:2]
        src_w, src_h = self._fit_source_size() or (w, h)
        self.fit_zoom = guard_frame(size[0], size[1], src_w, src_h)
        was_fit = self._at_fit
        self.zoom = self.fit_zoom if was_fit else self._clamp_zoom(self.zoom)
        self._update_scrollregion()
        if was_fit:
            self._centre_on_content()
        self._render()

    def refresh(self) -> None:
        """Re-draw the current frame at the current zoom (after a zoom/pan change)."""
        self._render()

    def redraw_overlay(self) -> None:
        """Re-draw the vector overlay without touching the photo.

        This is the path a corner nudge takes: the frame image is unchanged, so re-resizing it
        (which is what a full ``show()`` does) would be pure cost and would drop the zoom.
        """
        self.canvas.delete("overlay")
        if self.overlay is not None and self._src is not None:
            self.overlay(self)

    def _render(self) -> None:
        from PIL import Image, ImageTk

        if self._src is None:
            return
        H, W = self._src.shape[:2]
        z = self.zoom
        cw, ch = self._viewport()
        self.canvas.delete("all")

        # visible region in image coordinates, plus a small pad so panning is smooth
        x0, y0 = self.canvas.canvasx(0) / z, self.canvas.canvasy(0) / z
        pad = 4
        cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        cx1, cy1 = min(W, int(x0 + cw / z) + pad + 1), min(H, int(y0 + ch / z) + pad + 1)
        if cx1 <= cx0 or cy1 <= cy0:
            cx0, cy0, cx1, cy1 = 0, 0, min(W, max(1, cx1)), min(H, max(1, cy1))

        crop = self._src[cy0:cy1, cx0:cx1]
        resample = Image.NEAREST if z > 1.0 else Image.BILINEAR
        # NEAREST above 1:1 so the operator sees actual pixels, not a smoothed guess
        img = Image.fromarray(crop).resize(
            (max(1, int(round(crop.shape[1] * z))), max(1, int(round(crop.shape[0] * z)))),
            resample,
        )
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(cx0 * z, cy0 * z, anchor="nw", image=self._photo)
        self.redraw_overlay()
        self._notify()

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change(self)

    # -- zoom / pan ---------------------------------------------------------
    def set_zoom(self, z: float, anchor: tuple[float, float] | None = None) -> float:
        """Zoom, keeping the image point under ``anchor`` (canvas px) fixed."""
        if self._src is None:
            return self.zoom
        if anchor is None and self.content_box is not None:
            # The zoom buttons carry no cursor position, so anchor on the PHOTO rather than on the
            # padded canvas: otherwise 2x/4x zoom into the black margin the padding creates.
            self.zoom = self._clamp_zoom(z)
            self._at_fit = abs(self.zoom - self.fit_zoom) < 1e-6
            self._update_scrollregion()
            self._centre_on_content()
            self._render()
            return self.zoom
        if anchor is None:
            cw, ch = self._viewport()
            anchor = (cw / 2.0, ch / 2.0)
        ix = self.canvas.canvasx(anchor[0]) / self.zoom
        iy = self.canvas.canvasy(anchor[1]) / self.zoom
        self.zoom = self._clamp_zoom(z)
        self._at_fit = abs(self.zoom - self.fit_zoom) < 1e-6
        self._update_scrollregion()
        H, W = self._src.shape[:2]
        self.canvas.xview_moveto(min(1.0, max(0.0, (ix * self.zoom - anchor[0]) / max(1.0, W * self.zoom))))
        self.canvas.yview_moveto(min(1.0, max(0.0, (iy * self.zoom - anchor[1]) / max(1.0, H * self.zoom))))
        self._render()
        return self.zoom

    def zoom_by(self, factor: float, anchor: tuple[float, float] | None = None) -> float:
        return self.set_zoom(self.zoom * factor, anchor=anchor)

    def fit(self) -> float:
        return self.set_zoom(self.fit_zoom)

    def on_wheel(self, event) -> None:
        delta = getattr(event, "delta", 0) or 0
        if not delta:
            return
        self.zoom_by(1.25 ** (delta / 120.0), anchor=(event.x, event.y))

    def bind_zoom(self) -> None:
        """Mouse-wheel zoom, middle-drag pan, window-resize re-fit, and the image-px readout."""
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Button-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: (self.canvas.scan_dragto(e.x, e.y, gain=1),
                                                   self._render()))
        self.canvas.bind("<Motion>", self.on_motion)

    def on_motion(self, event) -> None:
        if self.cursor_var is None:
            return
        if self._src is None:
            self.cursor_var.set("")
            return
        ix, iy = self.to_image(event.x, event.y)
        h, w = self._src.shape[:2]
        inside = 0 <= ix < w and 0 <= iy < h
        self.cursor_var.set(f"x={ix:7.2f}  y={iy:7.2f} px" + ("" if inside else "  (outside)"))

    # -- mapping ------------------------------------------------------------
    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        """Canvas pixel -> original image pixel, honouring zoom *and* scroll offset."""
        z = self.zoom or 1.0
        return self.canvas.canvasx(cx) / z, self.canvas.canvasy(cy) / z


class App:
    """The M8 window: PLAN §5's A→F flow, one notebook tab per step."""

    def __init__(self, root, clip: Path | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title("Dog Diving jump-distance scorer — prototype (PLAN §5)")
        root.geometry("1180x820")

        # --- state ---------------------------------------------------------
        self.clip: Path | None = None
        self.info = None
        self.stability = None
        self.stability_record: dict | None = None
        self.stability_overridden = False
        self.calib: dict | None = None
        self.calib_path: Path | None = None
        # The 3-D route's document and its per-clip result.  Kept apart from ``calib`` because they
        # are different schemas answering different questions, and a mix-up would produce a plausible
        # number instead of an error.
        self.pool_calib: dict | None = None
        self.pool_calib_path: Path | None = None
        self.pool_result: dict | None = None
        self.ladder: dict | None = None
        # The single aligned plane driving the 3-D route (2026-10-19, replacing the four bars), on
        # the same zoomable canvas the 1-D marks use.  One sequence at a time.
        self.plane_session: plane_mod.PlaneSession | None = None
        #: Last waterline detection, kept so its three fitted lines can be drawn over the photo even
        #: when the seed was refused (that is the case where seeing them matters most).
        self.seed_edges: waterline_mod.WaterlineEdges | None = None
        #: The reset prompt for *New Plane*.  ``None`` means "ask with a real dialog"; a test (or an
        #: unattended run) may set a callable returning True/False instead of blocking on a modal.
        self.confirm_reset: Callable[[], bool] | None = None
        #: Waterline detections cached per clip sha256 — "detect once per clip", so New Plane on a
        #: clip already looked at places its quad instantly instead of re-running the detector.
        self._seed_cache: dict[str, tuple[Any, Any]] = {}
        #: The clip's sha256, computed on load and reused as the key that finds this clip's plane.
        self.clip_sha: str = ""
        #: The saved plane's identity: its setup id and the clips it has been accepted for.
        self._plane_setup_id: str | None = None
        self._plane_clips: list[dict] = []
        #: Where saved planes live.  A real run uses ``CALIB_DIR``; tests point it at a tmp dir so a
        #: save cannot leak into the operator's calibrations.
        self.plane_dir = Path(CALIB_DIR)
        #: Group-deselect band state on the Review canvas (armed by the Deselect points button).
        self.marquee_mode = False
        self._band0: tuple[float, float] | None = None
        self._band1: tuple[float, float] | None = None
        # An optional MEASURED lens (a checkerboard still).  None means the pane supplies the
        # focal prior, and the document records ``lens.source == "assumed"`` — the largest single
        # accuracy lever left on this route, so it has a hook rather than being invisible.
        self.pool_lens = None
        #: Loaded pose checkpoints, keyed by path — the trajectory fork's pattern: loading a .pt
        #: costs a second or two and picking a different one must not re-load it every run.
        self._models: dict[str, Any] = {}
        self.pool_frame = None
        #: What the Calibrate pane's photo currently shows — (id(plane), padded shape).  The photo
        #: is only re-rendered when this changes; plane edits re-draw the vector overlay instead.
        self._plane_shown_key: tuple[int, tuple[int, ...]] | None = None
        self.outcome: PipelineOutcome | None = None
        self.review: ReviewState | None = None
        self.show_grid = True
        #: *Correct Entry* mode: while it is armed the entry X can be hovered and dragged, and the
        #: trajectory is cut where it is dropped.  (Named ``correct_point_mode`` until 2026-10-21;
        #: the pane's button is now "Correct Entry" and ``correct_point_mode`` reads the same flag.)
        self.correct_entry_mode = False
        #: Cursor position in image px while the mode is armed — the dot preview follows it.
        self.entry_cursor: tuple[float, float] | None = None
        #: The drag in progress, as an image-px point, or None.
        self._entry_drag: tuple[float, float] | None = None
        #: Cached Review lattice (``plane.grid_spec``) keyed by (document, length, width).
        self._grid_key: tuple | None = None
        self._grid_cache: dict | None = None
        self._syncing = False       # re-entrancy guard for tk.Scale.set()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.q: queue.Queue = queue.Queue()

        self._build_ui()
        self.root.after(80, self._drain)
        if clip is not None:
            self.load_clip(clip)

    # -- ui construction ----------------------------------------------------
    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk

        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="Open clip…", command=self.on_open).pack(side="left")
        self.clip_label = ttk.Label(bar, text="no clip loaded")
        self.clip_label.pack(side="left", padx=12)

        self.status = ttk.Label(self.root, text="ready", anchor="w", relief="sunken")
        self.status.pack(side="bottom", fill="x")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.tab_load = ttk.Frame(self.nb, padding=8)
        self.tab_calib = ttk.Frame(self.nb, padding=8)
        self.tab_process = ttk.Frame(self.nb, padding=8)
        self.tab_review = ttk.Frame(self.nb, padding=8)
        self.tab_results = ttk.Frame(self.nb, padding=8)
        for tab, name in ((self.tab_load, "1 Load"), (self.tab_calib, "2 Calibrate"),
                          (self.tab_process, "3 Process"), (self.tab_review, "4 Review"),
                          (self.tab_results, "5 Results")):
            self.nb.add(tab, text=name)

        self._build_load()
        self._build_calibrate()
        self._build_process()
        self._build_review()
        self._build_results()

        log_frame = ttk.LabelFrame(self.root, text="session log", padding=4)
        log_frame.pack(side="bottom", fill="x", padx=8, pady=(0, 4))
        self.log_text = tk.Text(log_frame, height=7, wrap="none")
        self.log_text.pack(fill="x")
        if MODEL_SELECTION_NOTE:
            self.log(MODEL_SELECTION_NOTE)       # e.g. a DD_MODEL_SCALE the registry could not honour

    def _zoomable(self, parent, w: int, h: int):
        """Canvas + scrollbars + zoom toolbar + image-px cursor readout.

        Shared by the Calibrate and Review panes.  The caller packs the returned frame.
        The readout states the achieved precision ("1 px = N image px") because that is
        the number that matters: at fit-to-window it is ~2.5, and the M2 measurement says
        foot-mark placement needs ~0.2.
        """
        tk, ttk = self.tk, self.ttk
        holder = ttk.Frame(parent)
        # The canvas is the pane, not a fixed 900-px picture: it grows with the window and the view
        # re-fits on <Configure>, so a wider window means more frame (and more usable precision).
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        canvas = tk.Canvas(holder, width=w, height=h, bg="#202020",
                           highlightthickness=1, highlightbackground="#444")
        hbar = ttk.Scrollbar(holder, orient="horizontal", command=canvas.xview)
        vbar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        view = FrameView(canvas, w, h)
        view.bind_zoom()

        bar = ttk.Frame(holder)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Button(bar, text="Fit", width=4, command=view.fit).pack(side="left", padx=1)
        for z in ZOOM_STEPS:
            ttk.Button(bar, text=f"{z:g}\u00d7", width=4,
                       command=lambda zz=z: view.set_zoom(zz)).pack(side="left", padx=1)
        zoom_var = tk.StringVar(value="zoom \u2014")
        ttk.Label(bar, textvariable=zoom_var).pack(side="left", padx=8)
        cursor = tk.StringVar(value="")
        ttk.Label(bar, textvariable=cursor, width=26).pack(side="left", padx=8)
        view.cursor_var = cursor

        def changed(v: FrameView) -> None:
            zz = v.zoom or 1.0
            zoom_var.set(f"zoom {zz * 100:.0f}%   1 px = {1.0 / zz:.2f} image px")

        view.on_change = changed
        ttk.Label(bar, text="wheel = zoom at cursor \u00b7 middle-drag = pan").pack(
            side="left", padx=8)
        return view, holder

    def _build_load(self) -> None:
        tk, ttk = self.tk, self.ttk
        f = self.tab_load
        self.load_info = ttk.Label(f, text="Open a clip to begin.", justify="left")
        self.load_info.pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(anchor="w", pady=6)
        ttk.Button(row, text="Run stability check", command=self.on_stability).pack(side="left")
        self.override_btn = ttk.Button(row, text="Override 'unusable'",
                                       command=self.on_override_stability, state="disabled")
        self.override_btn.pack(side="left", padx=6)
        self.stability_label = ttk.Label(f, text="", justify="left", wraplength=1000)
        self.stability_label.pack(anchor="w", pady=4)
        self.load_canvas = tk.Canvas(f, width=900, height=420, bg="#202020",
                                     highlightthickness=1, highlightbackground="#444")
        self.load_canvas.pack(anchor="w", pady=4)
        self.load_view = FrameView(self.load_canvas, 900, 420)
        self.load_canvas.bind("<Configure>", self.load_view.on_resize)   # re-fits with the window

    def _build_calibrate(self) -> None:
        tk, ttk = self.tk, self.ttk
        f = self.tab_calib
        # The calibration frame and the 1-ft grid are assumed, not asked (operator order 2026-10-21:
        # *"hide calibration frame index and show 1-ft grid, these should be assumed"*).  The two
        # variables stay — the review/annotate/report paths read them — but the widgets that made the
        # operator think about them are gone.
        self.calib_frame_idx = tk.IntVar(value=CALIB_FRAME_DEFAULT)
        self.grid_var = tk.BooleanVar(value=True)

        # Which saved plane this clip uses.  "auto" is the normal state and the default: the clip's
        # own plane is adopted when it is opened (Load pool doc, 2026-10-21).  The list exists for the
        # one case auto cannot cover — pointing a clip at a plane that was saved for OTHER clips, so
        # that saving adds this clip to that plane's accepted list instead of making a copy.
        pick = ttk.Frame(f)
        pick.pack(anchor="w")
        ttk.Label(pick, text="saved plane:").pack(side="left")
        self.plane_pick_var = tk.StringVar(value="auto")
        self.plane_pick = ttk.Combobox(pick, textvariable=self.plane_pick_var, width=34,
                                       state="readonly")
        self.plane_pick.pack(side="left", padx=4)
        self.plane_pick.bind("<<ComboboxSelected>>", self.on_pick_plane)
        # Save New Plane belongs beside the picker (operator order): the picker decides which plane is
        # loaded, so the button that commits the current quad goes with it rather than in the button
        # row, where it was read as part of the alignment workflow.
        self.save_plane_btn = ttk.Button(pick, text="Save New Plane", command=self.on_save_pool)
        self.save_plane_btn.pack(side="left", padx=(8, 0))
        # Whether this clip has a saved plane at all — the document is found by CLIP, not named by
        # hand (operator order 2026-10-21).
        self.pool_status = ttk.Label(pick, text="no saved plane for this clip")
        self.pool_status.pack(side="left", padx=(12, 0))

        # --- the 3-D route: one plane on the water surface (2026-10-19) ------
        # The operator aligns a quad whose corners are four points of a metric rectangle on the pool
        # — X = 0 along the dock lip, +X away down the pool — and the homography that quad defines IS
        # the distance mapping.  The canvas extends beyond the frame so corners can sit outside it.
        # No heights are inferred: see ``dog_diving/pool/plane.py`` and PLAN §15.
        plane_row = ttk.Frame(f)
        plane_row.pack(anchor="w", pady=(8, 0))
        # One entry point (operator order 2026-10-21: *"'New Plane', 'Make parallelogram' and 'Fit to
        # frame' are redundant"*).  New Plane now does the whole job: it makes the quad, runs the
        # detector on the clip's first frames, snaps the quad onto the detected waterlines and turns
        # the snap toggle on.  ``on_make_parallelogram`` / ``on_plane_fit`` remain as module-level
        # tools — the seed and the flip use the same primitives — but they are no longer buttons.
        self.new_plane_btn = self._blue_button(plane_row, "New Plane", self.on_new_plane)
        self.new_plane_btn.pack(side="left")
        # The waterline detector runs on demand and its result is cached per clip, so New Plane on
        # the same clip is instant afterwards; this button forces a fresh look.
        self.detect_btn = ttk.Button(plane_row, text="Re-detect Waterline",
                                     command=self.on_detect_waterline)
        self.detect_btn.pack(side="left", padx=(8, 0))
        # Snapping is its own button (operator order) because it is a MODE: on, every edit that moves
        # an edge is followed by sliding that edge back onto the detected waterline, which is what the
        # operator wants while placing and exactly what they do not want while hand-correcting.
        self.snap_on_var = tk.BooleanVar(value=False)
        self.snap_btn = ttk.Button(plane_row, text="TOGGLE: Snap to Waterline (OFF)",
                                   command=self.on_toggle_snap)
        self.snap_btn.pack(side="left", padx=(8, 0))
        # One box per plane edge: Top = the far waterline, Bottom = the near one, Left/Right = the
        # pool's two short ends (X = 0 and X = L).  Unchecking one leaves that edge free.  Only TOP is
        # checked to begin with (operator order 2026-10-21): the far waterline is the one the seed's
        # far edge is built from, so holding it is a real constraint, while holding all four would
        # pin corners the operator still wants to drag.
        self.snap_side_vars: dict[str, Any] = {}
        for side, label in SNAP_SIDE_LABELS:
            var = tk.BooleanVar(value=(side == SNAP_DEFAULT_SIDE))
            self.snap_side_vars[side] = var
            ttk.Checkbutton(plane_row, text=label, variable=var).pack(side="left", padx=(4, 0))
        # Kept for the tests and for the log line that named it: the far edge is "Top".
        self.snap_far_var = self.snap_side_vars["back"]
        # Changes the NUMBER scale only (operator order 2026-10-21): the quad keeps every corner where
        # the operator put it and only which end means X=0 changes.
        ttk.Button(plane_row, text="Flip Plane (left-right)",
                   command=self.on_plane_flip_x).pack(side="left", padx=(10, 0))
        self.complete_btn = self._blue_button(plane_row, "Complete", self.on_build_plane)
        self.complete_btn.pack(side="left", padx=(10, 0))
        self.plane_label = ttk.Label(f, text="no plane yet")
        self.plane_label.pack(anchor="w", pady=(4, 0))

        dim_row = ttk.Frame(f)
        dim_row.pack(anchor="w", pady=(6, 0))
        # The two anchors, named for what the operator sees rather than for the maths (operator order
        # 2026-10-21): the near anchor is the X of the quad's DOCK-SIDE edge, the opposite one is the
        # X of its far edge.  Defaults are the whole pool: 0 ft at the dock, 35 ft at the far wall.
        ttk.Label(dim_row, text="• Anchor (near dock):").pack(side="left")
        self.plane_near_var = tk.StringVar(value=f"{SEED_ANCHOR_NEAR_FT:g}")
        ttk.Entry(dim_row, textvariable=self.plane_near_var, width=6).pack(side="left", padx=2)
        ttk.Label(dim_row, text="• Anchor (opposite dock):").pack(side="left", padx=(8, 2))
        self.plane_len_var = tk.StringVar(value=f"{SEED_ANCHOR_FAR_FT:g}")
        ttk.Entry(dim_row, textvariable=self.plane_len_var, width=6).pack(side="left", padx=2)
        ttk.Label(dim_row, text="width (ft):").pack(side="left", padx=(8, 2))
        self.plane_wid_var = tk.StringVar(value=f"{plane_mod.DEFAULT_WIDTH_FT:g}")
        ttk.Entry(dim_row, textvariable=self.plane_wid_var, width=6).pack(side="left")
        ttk.Button(dim_row, text="Apply size", command=self.on_plane_dims).pack(side="left", padx=4)
        # Which end of the pool the dock is on.  "auto" uses the clip's first accepted detections —
        # only available after a run — so the control exists to make the seed usable in the normal
        # order (calibrate, *then* process).  It also says which SHORT-END waterline is X = 0.
        ttk.Label(dim_row, text="dock side:").pack(side="left", padx=(10, 2))
        self.dock_side_var = tk.StringVar(value="auto")
        ttk.Combobox(dim_row, textvariable=self.dock_side_var,
                     values=("auto", "left", "right"), width=6,
                     state="readonly").pack(side="left")

        ttk.Label(f, text=(f"grid always on: 1 ft minor, 5 ft major · the +x axis always runs 0 → "
                           f"{plane_mod.METRIC_LENGTH_FT:g} ft from the dock edge; the anchors only "
                           "place this quad's two edges against visible pool marks")).pack(
            anchor="w", pady=(2, 0))

        self.calib_prompt = ttk.Label(f, text="not calibrating", font=("", 11, "bold"))
        self.calib_prompt.pack(anchor="w", pady=4)

        self.calib_view, _cal_holder = self._zoomable(f, 900, 430)
        _cal_holder.pack(fill="both", expand=True)
        self.calib_view.overlay = self._draw_plane_overlay
        self.calib_canvas = self.calib_view.canvas
        self.calib_canvas.bind("<Button-1>", self.on_calib_click)
        # Press grabs the nearest corner (corner before side midpoint); motion moves it; release
        # drops it.  The selection survives the release, because the arrow pad below moves it one
        # image px at a time.
        self.calib_canvas.bind("<B1-Motion>", self.on_calib_drag)
        self.calib_canvas.bind("<ButtonRelease-1>", self.on_calib_release)
        for key, dx, dy in (("<Left>", -1, 0), ("<Right>", 1, 0), ("<Up>", 0, -1), ("<Down>", 0, 1)):
            self.calib_canvas.bind(key, lambda e, a=dx, b=dy: self.on_arrow_key(e, a, b))
        self._build_nudge_pad(f)


    # -- fine placement: the arrow pad ---------------------------------------
    def _blue_button(self, parent, text: str, command):
        """A primary action: blue with white text (operator order — New Plane and Complete).

        ``tk.Button``, not ``ttk.Button``: the Windows ``vista`` theme paints ttk buttons itself and
        ignores ``background``, so a themed "blue" button comes out grey.
        """
        return self.tk.Button(parent, text=text, command=command, bg=PRIMARY_BG, fg="white",
                              activebackground=PRIMARY_BG_ACTIVE, activeforeground="white",
                              relief="flat", bd=0, padx=12, pady=3, cursor="hand2",
                              highlightthickness=0)

    def _build_nudge_pad(self, parent) -> None:
        """A fixed arrow pad under the canvas: one click = one IMAGE px on the selected handle.

        Shift-click = 10 px (coarse), Ctrl/Alt-click = 0.1 px (sub-pixel, which is the precision the
        placement budget needs).  It is a fixed toolbar rather than a floating control because the
        pointer is busy dragging a corner while the operator wants to nudge it.
        """
        ttk = self.ttk
        pad = ttk.Frame(parent)
        pad.pack(anchor="w", pady=(4, 0))
        ttk.Label(pad, text="fine placement:", font=("", 9, "bold")).pack(side="left", padx=(0, 6))
        cross = ttk.Frame(pad)
        cross.pack(side="left")
        self._nudge_button(cross, "\u25b2", 0, -1).grid(row=0, column=1, padx=1)
        self._nudge_button(cross, "\u25c0", -1, 0).grid(row=1, column=0, padx=1)
        self._nudge_button(cross, "\u25bc", 0, 1).grid(row=1, column=1, padx=1)
        self._nudge_button(cross, "\u25b6", 1, 0).grid(row=1, column=2, padx=1)
        self.nudge_var = self.tk.StringVar(value="no corner selected — click one on the canvas")
        ttk.Label(pad, textvariable=self.nudge_var).pack(side="left", padx=10)
        ttk.Label(pad, text="1 image px per click · Shift 10 · Ctrl 0.1 · arrow keys work too",
                  foreground="#777").pack(side="left", padx=6)

    def _nudge_button(self, parent, text: str, dx: int, dy: int):
        b = self.ttk.Button(parent, text=text, width=2)
        # A ``command=`` fires with no modifier state; binding the press keeps Shift/Ctrl/Alt.
        b.bind("<Button-1>", lambda e, a=dx, c=dy: self.on_nudge_handle(a, c, e))
        return b

    @staticmethod
    def _nudge_step(state: int) -> float:
        """Shift = coarse (10 px), Ctrl/Alt = sub-pixel (0.1 px), plain = 1 image px."""
        mods = int(state or 0)
        if mods & 0x1:                       # Shift
            return 10.0
        if mods & 0x4 or mods & 0x20000:     # Control, or Alt on Windows
            return 0.1
        return 1.0

    def on_nudge_handle(self, dx: float, dy: float, event=None) -> None:
        """Move the selected handle by one step.  Image px, not screen px — the unit the doc uses.

        Named ``on_nudge_handle``, not ``on_nudge``: ``App.on_nudge(delta)`` already steps the
        Review slider by whole frames, and a name collision silently replaced it.
        """
        s = self.plane_session
        if s is None:
            self.log("start the plane first")
            return
        if s.selected is None:
            placed = [t for t in plane_mod.CORNER_TAGS if s.corners.get(t) is not None]
            if not placed:
                self.log("no corner to move — click a corner on the canvas first")
                return
            s.select(placed[0])
        step = self._nudge_step(getattr(event, "state", 0) if event is not None else 0)
        if not s.nudge(dx * step, dy * step):
            self.log(f"nothing movable at {s.selected!r}")
            return
        self._maybe_snap_edges()
        self._redraw_plane()
        self._sync_plane_widgets()

    def on_arrow_key(self, event, dx: int, dy: int):
        """Arrow keys nudge on the Calibrate tab only, so Review keeps its own key bindings."""
        try:
            if int(self.nb.index(self.nb.select())) != 1:
                return None
        except Exception:                                            # noqa: BLE001
            return None
        self.on_nudge_handle(dx, dy, event)
        return "break"

    def _sync_nudge_label(self) -> None:
        var = getattr(self, "nudge_var", None)
        if var is None:
            return
        s = self.plane_session
        if s is None or s.selected is None:
            var.set("no corner selected — click one on the canvas")
            return
        p = (s.corners.get(s.selected) if s.selected in plane_mod.CORNER_TAGS
             else s.side_midpoint(s.selected))
        where = "" if p is None else f"  ({p[0]:.2f}, {p[1]:.2f}) px"
        var.set(f"moving {s.selected}{where}")

    def _build_process(self) -> None:
        ttk = self.ttk
        f = self.tab_process
        # --- which .pt pose model the trajectory uses (2026-10-19) --------------
        # Taken from ``dog-diving-trajectory``'s app: a combobox over ``models/*.pt`` plus a browse
        # button and a per-path model cache.  The checkpoint is part of DetectConfig and therefore
        # of the detection cache key, so switching it invalidates warm ``tracks.json`` — which is
        # the correct behaviour, and the log says so rather than silently mixing checkpoints.
        mrow = ttk.Frame(f)
        mrow.pack(anchor="w", pady=(6, 0))
        ttk.Label(mrow, text="pose model:").pack(side="left")
        self.model_var = self.tk.StringVar(value=str(detect_mod.MODEL_PATH))
        self.cmb_model = ttk.Combobox(mrow, textvariable=self.model_var, width=56,
                                      values=self._model_choices())
        self.cmb_model.pack(side="left", padx=4)
        self.cmb_model.bind("<<ComboboxSelected>>", self.on_model_pick)
        ttk.Button(mrow, text="…", width=3, command=self.on_model_browse).pack(side="left")
        self.model_status = ttk.Label(mrow, text=Path(str(detect_mod.MODEL_PATH)).name)
        self.model_status.pack(side="left", padx=6)

        # --- backbone scale: yolo26n or yolo26s (2026-10-22, operator request) -----
        # "The user can select whether detection is using YOLO26s weights or YOLO26n weights
        # alongside the existing selection of the model weights."  The scale resolves through
        # ``config.MODEL_REGISTRY`` — n -> yolo26n-pose_dogs_v5.pt (model of record), s ->
        # yolo26s-pose_dogs_clahe.pt — and moves the file selection above onto that scale's declared
        # checkpoint.  Picking a file moves this box, so the two controls can never disagree about
        # what will be loaded.  Only dog-family checkpoints are selectable here: the stock COCO pose
        # model has no tailset, so "the s weights" can never quietly mean the 17-keypoint person
        # model (that one stays reachable through the file combobox, marked a/b in the registry).
        srow = ttk.Frame(f)
        srow.pack(anchor="w", pady=(2, 0))
        ttk.Label(srow, text="backbone:").pack(side="left")
        self.scale_var = self.tk.StringVar(value=self._scale_choice_for(detect_mod.MODEL_PATH))
        self.cmb_scale = ttk.Combobox(srow, textvariable=self.scale_var, width=40,
                                      state="readonly", values=self._scale_choices())
        self.cmb_scale.pack(side="left", padx=4)
        self.cmb_scale.bind("<<ComboboxSelected>>", self.on_scale_pick)
        self.scale_status = ttk.Label(srow, text=self._scale_status_text(detect_mod.MODEL_PATH))
        self.scale_status.pack(side="left", padx=6)

        row = ttk.Frame(f)
        row.pack(anchor="w", pady=6)
        self.run_btn = ttk.Button(row, text="Run", command=self.on_run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self.on_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        ttk.Button(row, text="Preview on this frame",
                   command=self.on_preview).pack(side="left", padx=6)
        self.contact_mode = ttk.Label(f, text="contact frame: auto (PLAN §5 candidate #1)")
        self.contact_mode.pack(anchor="w")
        row2 = ttk.Frame(f)
        row2.pack(anchor="w", pady=(6, 0))
        ttk.Label(row2, text="minimum accepted points:").pack(side="left")
        # 0 = fixed thresholds, lifted from ``dog-diving-trajectory``: its app never lowers conf, and
        # the ladder is what produced "a cursor that moves up and diagonal with no attempt to look
        # for a dog" — it buys points of unknown provenance (measured: empty sky at kp 0.96, a
        # bystander at 0.995).  The ladder stays available as an explicit opt-in.
        self.min_points_var = self.tk.IntVar(value=0)
        ttk.Spinbox(row2, from_=0, to=500, width=5,
                    textvariable=self.min_points_var).pack(side="left", padx=4)
        ttk.Label(row2, justify="left", wraplength=620, text=(
            "0 = fixed thresholds. Below this many accepted points the detection confidence "
            "threshold is lowered until the minimum is reached (imgsz last), and every point that "
            "buys is marked low_conf in the artifacts.")).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(f, length=880, maximum=1.0)
        self.progress.pack(anchor="w", pady=8)
        self.progress_label = ttk.Label(f, text="idle")
        self.progress_label.pack(anchor="w")
        self.cache_label = ttk.Label(f, text="")
        self.cache_label.pack(anchor="w", pady=4)
        ttk.Label(f, justify="left", wraplength=900, text=(
            "Note (honest limitation): Cancel reaches the inference loop, because that is "
            "the only stage exposing a cancel hook. It cannot interrupt the trajectory / "
            "contact / annotate / encode stages, and a warm cache never consults it — the "
            "run returns immediately instead."
        )).pack(anchor="w", pady=(12, 0))
        self.process_canvas = self.tk.Canvas(f, width=900, height=360, bg="#202020",
                                             highlightthickness=1, highlightbackground="#444")
        self.process_canvas.pack(anchor="w")
        self.process_view = FrameView(self.process_canvas, 900, 360)

    def on_jump_last_detection(self) -> None:
        """Jump to the last accepted detection — the fork's ``jump_last``, ported."""
        if self.review is None:
            return
        pts = list(getattr(self.review.traj, "points", []) or [])
        if not pts:
            self.log("no accepted detections in this clip's trajectory")
            return
        frame = int(pts[-1]["frame"])
        self.review.choose(frame, "jump-last")
        self.log(f"jumped to the last accepted detection: frame {frame}")
        self._after_review_change()

    def on_jump_apex(self) -> None:
        """Jump to the fitted arc's apex — the fork's ``jump_apex``, ported."""
        if self.review is None:
            return
        traj = self.review.traj
        coeffs = getattr(traj, "coeffs", None)
        flight = list(getattr(traj, "flight_points", []) or [])
        if not coeffs or not flight:
            self.log("no fitted arc to find an apex on")
            return
        fs = trajectory_mod.solve_frame_for_y(coeffs, max(p["y"] for p in flight))
        if fs is None:
            self.log("the arc has no apex in the fitted range")
            return
        frame = int(round(fs))
        self.review.choose(frame, "jump-apex")
        self.log(f"jumped to the arc apex: frame {frame}")
        self._after_review_change()

    def _build_review(self) -> None:
        tk, ttk = self.tk, self.ttk
        f = self.tab_review
        row = ttk.Frame(f)
        row.pack(anchor="w")
        # the trajectory fork's two jumps: land on a frame that HAS a detection, or on the apex
        ttk.Button(row, text="last detection",
                   command=self.on_jump_last_detection).pack(side="left", padx=(10, 2))
        ttk.Button(row, text="apex",
                   command=self.on_jump_apex).pack(side="left", padx=(0, 8))
        for txt, delta in (("<<", -10), ("<", -5), ("prev", -1), ("next", +1),
                           ("+5", +5), (">>", +10)):
            ttk.Button(row, text=txt, width=5,
                       command=lambda d=delta: self.on_nudge(d)).pack(side="left", padx=1)
        ttk.Button(row, text="Set entry frame", command=self.on_set_entry).pack(side="left", padx=6)
        # The entry point is a 2D->3D translation, so the operator wants to see the number move as
        # they step frames: this ties the measured point to the slider until they pin it.
        self.entry_live_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="entry = slider frame", variable=self.entry_live_var,
                        command=self.on_entry_live).pack(side="left")
        ttk.Button(row, text="Auto entry", command=self.on_clear_entry).pack(side="left", padx=4)
        # Group deselect: a mode, because a plain click already toggles one point and middle-drag pans.
        self.marquee_btn = ttk.Button(row, text="Deselect points", command=self.on_toggle_marquee)
        self.marquee_btn.pack(side="left", padx=(10, 0))
        self.correct_btn = ttk.Button(row, text="Correct Entry",
                                      command=self.on_toggle_correct)
        self.correct_btn.pack(side="left", padx=(4, 0))

        self.slider = tk.Scale(f, from_=0, to=1, orient="horizontal", length=880,
                               label="frame", command=self.on_slider)
        self.slider.pack(anchor="w", pady=4)
        self.readout = ttk.Label(f, text="no run yet", font=("", 12, "bold"))
        self.readout.pack(anchor="w")

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True)
        self.review_view, _rev_holder = self._zoomable(body, 880, 420)
        _rev_holder.pack(side="left", fill="both", expand=True)
        self.review_view.overlay = self._draw_review_overlay
        self.review_canvas = self.review_view.canvas
        self.review_canvas.bind("<Button-1>", self.on_review_click)
        self.review_canvas.bind("<B1-Motion>", self.on_review_drag)
        self.review_canvas.bind("<ButtonRelease-1>", self.on_review_release)
        # Hover matters only in *Correct Entry* mode: the X turns into a small dot under the pointer
        # so the operator can see they have hold of it before they press (operator order 2026-10-21).
        self.review_canvas.bind("<Motion>", self.on_review_motion)
        self.review_canvas.bind("<Leave>", self.on_review_leave)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        # The plan view lives here now (2026-10-21): it replaces the water-entry candidate table,
        # which the final-point rule had already made redundant.  Where the dog landed belongs next
        # to the frame it landed on.
        plan = ttk.LabelFrame(right, text="plan view — landing on the pool", padding=4)
        plan.pack(fill="both", expand=True)
        self.plan_canvas = tk.Canvas(plan, width=360, height=250, background="#101418",
                                     highlightthickness=1, highlightbackground="#2a3038")
        self.plan_canvas.pack(fill="both", expand=True)
        self.plan_canvas.bind("<Configure>", lambda _e: self._redraw_plan())
        self.plan_note = ttk.Label(plan, text="", font=("", 8))
        self.plan_note.pack(anchor="w")

        # The other-dog fix (2026-10-21): every accepted detection is listed and can be taken out of
        # the trajectory.  The canvas click and this list write the SAME set, so they cannot disagree.
        pts = ttk.LabelFrame(right, text="trajectory points — deselect what is not your dog", padding=4)
        pts.pack(fill="both", expand=True, pady=(6, 0))
        bar = ttk.Frame(pts)
        bar.pack(fill="x")
        self.pts_note = ttk.Label(bar, text="all detections included")
        self.pts_note.pack(side="left")
        ttk.Button(bar, text="Include all", command=self.on_include_all).pack(side="right", padx=(4, 0))
        ttk.Button(bar, text="Exclude selected",
                   command=self.on_exclude_selected).pack(side="right")
        holder = ttk.Frame(pts)
        holder.pack(fill="both", expand=True)
        self.pts_tree = ttk.Treeview(holder, columns=("keep", "frame", "x", "y", "conf"),
                                     show="headings", height=9, selectmode="extended")
        for col, txt, w in (("keep", "kept", 58), ("frame", "frame", 56), ("x", "x px", 62),
                            ("y", "y px", 62), ("conf", "kp conf", 62)):
            self.pts_tree.heading(col, text=txt)
            self.pts_tree.column(col, width=w, anchor="w")
        self.pts_tree.tag_configure("out", foreground="#8a8f98")
        pts_scroll = ttk.Scrollbar(holder, orient="vertical", command=self.pts_tree.yview)
        self.pts_tree.configure(yscrollcommand=pts_scroll.set)
        self.pts_tree.pack(side="left", fill="both", expand=True)
        pts_scroll.pack(side="right", fill="y")
        # click the "kept" cell to toggle; Space toggles the selection; double-click jumps the frame
        self.pts_tree.bind("<Button-1>", self.on_point_click)
        self.pts_tree.bind("<space>", self.on_point_space)
        self.pts_tree.bind("Double-1", self.on_point_goto)

        # Element opacity, under the table (operator order 2026-10-21).  Review's trajectory, arc,
        # dots and the entry mark are drawn ON TOP of the frame, and at full strength they cover the
        # dog they are describing; the slider is the operator's own control over how much of the
        # picture they are willing to lose.  10 % to start: the elements are findable, the frame is
        # readable.
        op = ttk.Frame(right)
        op.pack(fill="x", pady=(6, 0))
        ttk.Label(op, text="element opacity:").pack(side="left")
        self.entry_opacity = tk.DoubleVar(value=OPACITY_DEFAULT_PCT)
        self.opacity_scale = tk.Scale(op, from_=0, to=100, orient="horizontal",
                                      variable=self.entry_opacity, showvalue=True,
                                      length=200, width=10,
                                      command=self.on_opacity)
        self.opacity_scale.pack(side="left", padx=(6, 0))
        ttk.Label(op, text="%").pack(side="left", padx=(2, 0))

    def _build_results(self) -> None:
        tk, ttk = self.tk, self.ttk
        f = self.tab_results
        self.res_tree = ttk.Treeview(f, columns=("k", "v"), show="headings", height=9)
        self.res_tree.heading("k", text="field")
        self.res_tree.heading("v", text="value")
        self.res_tree.column("k", width=200, anchor="w")
        self.res_tree.column("v", width=560, anchor="w")
        self.res_tree.pack(anchor="w")

        self.res_warn = tk.Text(f, height=6, wrap="word")
        self.res_warn.pack(fill="x", pady=6)

        exp = ttk.LabelFrame(f, text="export", padding=4)
        exp.pack(fill="x", pady=6)
        for label, key, ext in (("annotated MP4…", "annotated", ".mp4"),
                                ("final frame PNG…", "still", ".png"),
                                ("result JSON…", "result", ".json"),
                                ("report TXT…", "report", ".txt"),
                                ("run meta JSON…", "meta", ".json")):
            ttk.Button(exp, text=label, width=20,
                       command=lambda k=key, e=ext: self.on_export(k, e)).pack(side="left", padx=3)
        ttk.Button(exp, text="Open output folder",
                   command=self.on_open_outdir).pack(side="left", padx=8)

    # -- logging / status ---------------------------------------------------
    def log(self, msg: str) -> None:
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.status.configure(text=msg)

    # -- Load ---------------------------------------------------------------
    def on_open(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Open a dock-diving clip",
            initialdir=str(CLIP_DIR if Path(CLIP_DIR).is_dir() else Path.cwd()),
            filetypes=[("video", " ".join(f"*{s}" for s in VIDEO_SUFFIXES)), ("all", "*.*")],
        )
        if path:
            self.load_clip(Path(path))

    def load_clip(self, clip: Path) -> None:
        try:
            self.info = video_mod.probe(clip, count_frames=False)
            self.clip = clip
        except Exception as exc:  # noqa: BLE001
            self.log(f"could not open {clip}: {type(exc).__name__}: {exc}")
            return
        self.clip_label.configure(text=str(clip))
        self.load_info.configure(text=(
            f"{self.info.width}x{self.info.height} @ {self.info.fps:.3f} fps   "
            f"{self.info.n_frames} frames   codec={self.info.codec}   "
            f"duration={self.info.duration_s:.2f}s"
            + ("   " + " | ".join(self.info.warnings) if self.info.warnings else "")
        ))
        for w in self.info.warnings:
            self.log(f"clip warning: {w}")
        self.load_view.show(read_frame(clip, 0), reset_view=True)
        self.slider.configure(to=max(1, int(self.info.n_frames) - 1))
        self.stability = None
        self.stability_overridden = False
        self.override_btn.configure(state="disabled")
        self.stability_label.configure(text="")
        # A different clip is a different plane until proven otherwise: drop the previous one, then
        # let the clip itself find its saved plane (Load pool doc, now automatic).
        self.pool_calib = None
        self.pool_calib_path = None
        self.pool_result = None
        self.plane_session = None
        self.seed_edges = None
        self._plane_setup_id = None
        self._plane_clips = []
        self.clip_sha = ""
        self._clip_sha()
        if hasattr(self, "plane_pick_var"):
            self.plane_pick_var.set("auto")
        self._refresh_plane_choices()
        self._sync_plane_widgets()
        self.log(f"loaded {clip.name} ({self.info.n_frames} frames)")
        self.root.after(1, self.auto_load_plane_for_clip)

    def on_stability(self) -> None:
        if not self.clip:
            return
        self.log("stability check…")

        def work() -> None:
            try:
                rep = stability_mod.analyze(self.clip)
                self.q.put(("stability", rep))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("log", f"stability check failed: {type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _show_stability(self, rep) -> None:
        self.stability = rep
        self.stability_label.configure(text="stability: " + rep.summary())
        if rep.verdict == "unusable":
            self.override_btn.configure(state="normal")
            self.log("STABILITY: unusable — measurement is blocked. Override explicitly to continue.")
        else:
            self.override_btn.configure(state="disabled")
            self.log(f"stability: {rep.verdict} ({rep.max_drift_px:.2f} px)")

    def on_override_stability(self) -> None:
        if not self.stability:
            return
        try:
            self.stability_record = stability_mod.assert_usable(
                self.stability, override=True, note="GUI override (PLAN §5)")
            self.stability_overridden = True
            self.log(f"stability override logged: {self.stability_record}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"override failed: {exc}")

    # -- Calibrate ----------------------------------------------------------
    # -- the pool plane (3-D) calibration (2026-10-19) ------------------------
    # One quad on the water surface replaces the four bars.  The geometry, the grid and the gates
    # live in ``dog_diving/pool/plane.py``; this is the wiring — one drag gesture on the pane's
    # zoomable canvas (which is padded, so corners can sit outside the frame) and the document build.
    # -- Calibrate canvas gestures (plane only; the 1-D sequence is archived) --
    def on_calib_click(self, event) -> None:
        """Press: place a missing plane corner, or grab a corner/side handle."""
        if self.plane_session is None:
            return
        x, y = self.calib_view.to_image(event.x, event.y)
        if self.plane_session.click(x, y, zoom=self.calib_view.zoom):
            self.calib_canvas.focus_set()          # so the arrow keys reach on_arrow_key
            self._redraw_plane()
            self._sync_plane_widgets()

    def on_calib_drag(self, event) -> None:
        if self.plane_session is None or self.plane_session.dragging is None:
            return
        x, y = self.calib_view.to_image(event.x, event.y)
        self.plane_session.drag_to(x, y)
        self._maybe_snap_edges()
        self._redraw_plane()

    def on_calib_release(self, _event) -> None:
        if self.plane_session is None:
            return
        if self.plane_session.release() is not None:
            self._maybe_snap_edges()
            self._sync_plane_widgets()
            self._redraw_plane()

    def on_new_plane(self, seed: bool = True) -> None:
        """New Plane: make/reset the quad, detect the waterlines, snap to them, arm the toggle.

        Operator order (2026-10-21) — this one button replaces *New Plane*, *Make parallelogram* and
        *Fit to frame*, which were three ways of getting to the same starting shape:

        1. a plane exists already?  **Ask before resetting it** — the quad may be the result of ten
           minutes of dragging, and a stray click must not throw that away;
        2. build the quad (frame fit, so the pane is never empty while the detector runs);
        3. run the waterline detector on the clip's **first 90 frames**;
        4. snap the quad onto the detected waterlines, far edge short / near edge long;
        5. turn the *Snap to Waterline* toggle on, so subsequent edits stay on the lines.

        The detection is cached per clip sha256, so a second New Plane on the same clip is instant
        and only the reset prompt is work.
        """
        if not self.clip:
            self.log("load a clip first")
            return
        if self._plane_in_progress() and not self._confirm_reset_plane():
            self.log("new plane cancelled — the plane on screen is unchanged")
            return
        idx = int(self.calib_frame_idx.get())
        frame = read_frame(self.clip, idx)
        if frame is None:
            self.log(f"could not read frame {idx}")
            return
        far, near, Wd = self._plane_dims_or_default()
        self.pool_frame = frame
        self.plane_session = plane_mod.PlaneSession(
            frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
            length_ft=max(1e-6, far - near), width_ft=Wd, dock_offset_ft=near)
        self.plane_session.fit_to_frame()
        # A new plane supersedes a loaded document: keeping both would let a run use a document that
        # no longer matches what is on screen.
        self.pool_calib = None
        self.pool_calib_path = None
        self._plane_setup_id = None
        self._plane_clips = []
        self.pool_status.configure(text="new plane — not saved for this clip yet")
        # Arm the snap toggle BEFORE the seed lands, so the seed's own edits are already constrained
        # and the operator can see which edges are being held.
        self.snap_on_var.set(True)
        self._sync_snap_btn()
        self._sync_plane_widgets()
        self._redraw_plane(reset_view=True)
        if not seed:
            self.log(f"pool plane started on frame {idx}: drag the corners onto the pool's corners "
                     f"(X = 0 is the dock lip); the canvas extends past the frame")
            return
        cached = self._seed_cache.get(self.clip_sha or "")
        if cached is not None:
            edges, res = cached
            self.log("plane opened from the cached waterline fit for this clip")
            self._apply_seed(edges, res)
            return
        self.log(f"pool plane started on frame {idx}: fitting the detected waterlines — "
                 f"drag from there; the canvas extends past the frame")
        self._start_waterline_detection()

    def _plane_in_progress(self) -> bool:
        """True when there is a plane on screen worth asking about before replacing it."""
        s = self.plane_session
        if s is None:
            return False
        return bool(s.placed or any(p is not None for p in s.corners.values()))

    def _confirm_reset_plane(self) -> bool:
        """Ask before discarding the plane on screen.  ``confirm_reset`` is the seam the tests use."""
        ask = getattr(self, "confirm_reset", None)
        if ask is not None:
            return bool(ask())
        from tkinter import messagebox
        return bool(messagebox.askyesno(
            "New Plane",
            "A plane is already on the canvas.\n\nReset it and fit a new one to the detected "
            "waterlines?",
            parent=self.root, icon="warning", default="no"))

    def on_toggle_snap(self) -> None:
        """The *Snap to Waterline* mode on/off (its own button, operator order 2026-10-21).

        Turning it on snaps immediately: a toggle that only affects the NEXT edit looks broken on a
        plane that is visibly off the lines.
        """
        on = not bool(self.snap_on_var.get())
        self.snap_on_var.set(on)
        self._sync_snap_btn()
        if not on:
            self.log("snap to waterline OFF — corners and edges move freely")
            return
        moved = self._maybe_snap_edges()
        if self.plane_session is not None:
            self._sync_plane_widgets()
            self._redraw_plane()
        held = ", ".join(label for side, label in SNAP_SIDE_LABELS
                         if self.snap_side_vars[side].get())
        self.log(f"snap to waterline ON (holding {held or 'nothing'}) — "
                 + ("edges slid onto the detected lines" if moved
                    else "no detected line has an edge to hold yet"))

    def _sync_snap_btn(self) -> None:
        on = bool(self.snap_on_var.get())
        self.snap_btn.configure(text=f"TOGGLE: Snap to Waterline ({'ON' if on else 'OFF'})")

    def on_make_parallelogram(self) -> None:
        """Rewrite the quad as the parallelogram with the same centre and edge directions.

        No longer a button (New Plane supersedes it, operator order 2026-10-21) but kept as a handler:
        it is the documented recovery from a quad whose edges were dragged separately.
        """
        s = self.plane_session
        if s is None or not s.placed:
            self.log("place the plane's four corners first (New Plane)")
            return
        s.make_parallelogram()
        moved = self._maybe_snap_edges()
        self._sync_plane_widgets()
        self._redraw_plane()
        self.log("plane made a parallelogram — drag from here"
                 + (" (edges held on the detected waterlines)" if moved else ""))

    def _snap_lines(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Detected waterline -> ``(point, direction)``, keyed by the PLANE side it belongs to.

        ``back``/``front`` are the far/near lines as detected.  ``left``/``right`` are the pool's two
        short ends.  Which of them is X = 0 is read off the **quad**, not off the detector: the X = 0
        edge is the one whose corners (``00``/``01``) sit on that side of the image, so the assignment
        survives both the orientation control and :meth:`PlaneSession.flip_x` without a second switch.
        Without that, a flip (which moves no pixel) would have the toggle drag the quad back to where
        it started.
        """
        edges = self.seed_edges
        s = self.plane_session
        if edges is None:
            return {}
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side, name in (("back", "far"), ("front", "near")):
            line = seed_mod.line_of(getattr(edges, name, None))
            if line is not None:
                out[side] = line

        # One line per image end: the detected short end, or the dock line when that end is the one
        # the detector voted for (``dock`` is a copy of the winning ``left``/``right``).
        dock_side = getattr(edges, "dock_side", "unknown")
        dock_line = seed_mod.line_of(getattr(edges, "dock", None))
        end_lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in ("left", "right"):
            line = seed_mod.line_of(getattr(edges, name, None))
            if line is None and dock_side == name:
                line = dock_line
            if line is not None:
                end_lines[name] = line
        if not end_lines:
            return out

        first, other = ("left", "right") if self._x0_end_is_image_left(s) else ("right", "left")
        if first in end_lines:
            out["left"] = end_lines[first]                        # X = 0
        if other in end_lines:
            out["right"] = end_lines[other]                       # X = L
        elif "left" in out and len(end_lines) > 1:
            out["right"] = next(v for k, v in end_lines.items() if k != first)
        return out

    @staticmethod
    def _x0_end_is_image_left(s) -> bool:
        """Which side of the image the quad's X = 0 edge is on.  Mirrors the corner tags exactly."""
        if s is None or not s.placed:
            return True
        lx = 0.5 * (float(s.corners["00"][0]) + float(s.corners["01"][0]))
        rx = 0.5 * (float(s.corners["10"][0]) + float(s.corners["11"][0]))
        return lx <= rx

    def _maybe_snap_edges(self) -> bool:
        """Hold every checked plane edge on its detected waterline, when the toggle says to.

        The detected lines are exactly what the seed was built from, so this is a real constraint on
        a real measurement, not a magnet to the nearest paint: an edge with no detected line is left
        alone rather than snapped somewhere invented.  Two passes, because a corner sits on two edges
        at once (``00`` is on Top and on Left), so the second pass cleans up what the first moved.
        """
        s = self.plane_session
        if s is None or not s.placed or not self.snap_on_var.get():
            return False
        lines = self._snap_lines()
        if not lines:
            return False
        moved = False
        for _pass in range(2):
            for side, _label in SNAP_SIDE_LABELS:
                if not self.snap_side_vars[side].get() or side not in lines:
                    continue
                point, direction = lines[side]
                moved = s.snap_side_to_line(side, point, direction) or moved
        return moved

    def on_plane_fit(self) -> None:
        if self.plane_session is None:
            self.log("start the plane first")
            return
        self.plane_session.fit_to_frame()
        self._sync_plane_widgets()
        self._redraw_plane()

    def on_plane_flip_x(self) -> None:
        """Change only the X numbering: every corner stays exactly where it is."""
        s = self.plane_session
        if s is None or not s.placed:
            self.log("place the plane's four corners first, then flip")
            return
        before = {t: s.corners[t] for t in plane_mod.CORNER_TAGS}
        s.flip_x()
        same = all(s.corners[t] == before[t] for t in plane_mod.CORNER_TAGS)
        self._sync_plane_widgets()
        self._redraw_plane()
        self.log(f"plane flipped left-right: X=0 is now the end at x={s.corners['00'][0]:.0f} px; "
                 f"the quad did not move{' (verified: every corner kept its pixel)' if same else ''}, "
                 f"only the number scale changed")

    # -- automatic placement from the detected waterlines (PLAN §17) ----------
    def _accepted_frames(self) -> list[int]:
        """Frame numbers the pose detector accepted, if a run has happened in this session.

        These are what pick the calm window: upstream's noise map is only valid where nothing is
        moving through the water, and the clip's own detections are the cheapest evidence of that.
        """
        det = getattr(self.outcome, "detect", None) if self.outcome is not None else None
        if det is None:
            return []
        try:
            return list(det.accepted_frames())
        except Exception:                                            # noqa: BLE001
            return []

    def _dock_hint_x(self) -> float | None:
        """Image x where the clip's dog starts — the best evidence of which side the dock is on.

        The dog runs down the dock and jumps *away* from it, so the first accepted tail-set points sit
        on the dock side.  Used for one thing: when X=0 has to be inherited from the quad instead of
        measured from a dock waterline, a mirror-flipped quad is the one wrong answer no gate can see.
        With no detections there is no hint, and S6 says so.
        """
        det = getattr(self.outcome, "detect", None) if self.outcome is not None else None
        if det is None:
            return None
        xs: list[float] = []
        for r in getattr(det, "records", []) or []:
            if r.get("accepted") and r.get("tailset"):
                xs.append(float(r["tailset"][0]))
                if len(xs) >= 3:
                    break
        return float(np.median(xs)) if xs else None

    def on_detect_waterline(self) -> None:
        """Force a fresh waterline detection (the result is cached per clip, so New Plane is free).

        The heavy part (a temporal noise map over ~90 frames) runs on the worker thread; the UI is
        only touched from ``_apply_seed`` on the main thread.  The worker gets a *copy* of the session
        so a drag during the wait cannot be read half-written.
        """
        if not self.clip:
            self.log("load a clip first")
            return
        if self.plane_session is None:
            self.on_new_plane()
            if self.plane_session is None:
                return
        self._start_waterline_detection()

    def _start_waterline_detection(self) -> None:
        """Run the detector on the clip's first frames and hand the result to ``_apply_seed``."""
        s = self.plane_session
        if s is None or not self.clip:
            return
        L, Wd = float(s.length_ft), float(s.width_ft)
        shape = (int(s.frame_shape[0]), int(s.frame_shape[1]))
        snapshot = plane_mod.PlaneSession.from_doc({"plane": s.as_dict()})
        roi = self._seed_roi(snapshot)
        clip = Path(self.clip)
        window = (0, SEED_FRAME_COUNT)
        accepted = self._accepted_frames()
        forced = str(self.dock_side_var.get())
        if forced in ("left", "right"):
            hint = 0.0 if forced == "left" else float(shape[1]) - 1.0
            self.log(f"dock side forced to the {forced} by the pane control — X = 0 will be the "
                     f"{forced} short-end waterline")
        else:
            hint = self._dock_hint_x()
            if hint is None:
                self.log("no detections yet, so the dock side is unknown — set the dock-side control "
                         "or run the clip first; X = 0 will be whichever short end the detector picks")
        self.detect_btn.configure(state="disabled")
        self.log(f"detecting waterlines on {clip.name}: frames {window[0]}-{window[1] - 1} "
                 f"(the clip's opening {SEED_FRAME_COUNT}) — a second or two")

        def work() -> None:
            try:
                edges = waterline_mod.detect_edges(clip, window=window, roi_poly=roi)
                # The opening frames are the asked-for default, not a guarantee: on a clip whose first
                # 10 frames are still moving (the dog is often already in shot) the temporal map only
                # separates the near waterline.  When the long edges are not both there, one wider look
                # at the clip's calm stretch is bought and the operator is told why.
                if not (edges.far.found and edges.near.found):
                    found = ", ".join(edges.found) or "none"
                    self.q.put(("log", f"the first {SEED_FRAME_COUNT} frames gave only: {found} — "
                                       f"widening the waterline window to the clip's calm stretch"))
                    edges = waterline_mod.detect_edges(clip, accepted_frames=accepted, roi_poly=roi)
                res = seed_mod.seed_from_edges(edges, length_ft=L, width_ft=Wd,
                                               frame_shape=shape, current=snapshot,
                                               dock_side=forced if forced in ("left", "right") else None,
                                               dock_hint_x=hint)
                self.q.put(("seed", (edges, res)))
            except Exception as exc:                                 # noqa: BLE001
                self.q.put(("seed_error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.q.put(("seed_done", None))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _seed_roi(session):
        """The session's quad grown 30 % about its centroid, as the detector's search region.

        The top of these frames is sky, roof, banners and arena lights: all high-noise, none of it
        water, and it is what drags the threshold. Growing the quad keeps the pool and the wall
        immediately above it inside the region while excluding the ceiling. A badly placed quad makes
        the region generous rather than restrictive, because the growth is outward.
        """
        if session is None or not session.placed:
            return None
        pts = np.array([session.corners[t] for t in plane_mod.CORNER_TAGS], dtype=float)
        centre = pts.mean(axis=0)
        grown = centre + (pts - centre) * 1.3
        h, w = int(session.frame_shape[0]), int(session.frame_shape[1])
        grown[:, 0] = np.clip(grown[:, 0], 0.0, max(0.0, w - 1.0))
        grown[:, 1] = np.clip(grown[:, 1], 0.0, max(0.0, h - 1.0))
        return grown

    def _apply_seed(self, edges, res) -> None:
        """Main thread: show what was found, and write it only if every gate passed."""
        self.seed_edges = edges
        s = self.plane_session
        gates = " ".join(("warn" if (v.get("pass") and v.get("warn")) else
                          "ok" if v.get("pass") else "FAIL") + f" {k}"
                         for k, v in res.gates.items())
        self.log(f"waterlines: {edges.level} (far {'ok' if edges.far.found else 'no'}, "
                 f"near {'ok' if edges.near.found else 'no'}, "
                 f"dock {'ok' if edges.dock.found else 'no'}, side {edges.dock_side}) · "
                 f"{edges.method} fisher {edges.fisher:.2f} · frames "
                 f"{edges.window[0]}-{edges.window[1]} · {gates}")
        for note in res.notes:
            self.log(f"  seed: {note}")
        for w in edges.warnings:
            self.log(f"  waterline warning: {w}")
        if s is None:
            return
        if res.refused or res.corners is None:
            self.log(f"seed REFUSED — {res.reason}")
            # Even a refused reconstruction leaves real lines on the canvas, so the quad is still slid
            # onto them: "New Plane snaps the plane to the detected waterlines" is the promise, and a
            # refused anchor is a reason not to move the quad projectively, not a reason to ignore the
            # water.
            moved = self._maybe_snap_edges()
            self.log("the quad was left as it was"
                     + (" but its checked edges were slid onto the detected lines" if moved
                        else " and no edge could be snapped")
                     + "; the detected lines stay on the canvas for reference")
            self._sync_plane_widgets()
            self._redraw_plane()
            return
        block = seed_mod.build_block(res, edges, upstream=waterline_mod.UPSTREAM,
                                     file_sha256=waterline_mod.vendored_sha256())
        s.apply_seed(res.corners, block)
        # Far edge short, near edge long (operator order 2026-10-21): the pool is seen down its
        # length, so the far edge must subtend fewer pixels.  Swapping the Y ends keeps every corner
        # where it is and only exchanges which pair is "far", so the operator is not disturbed.
        swapped = s.ensure_far_edge_short()
        # Cache the whole detection against the clip, so New Plane on this clip is instant and the
        # operator is not made to sit through the temporal analysis twice.
        if self.clip_sha:
            self._seed_cache[self.clip_sha] = (edges, res)
        # A seeded plane supersedes a loaded document, exactly like New Plane: keeping both would
        # let a run use a matrix that no longer matches what is on screen.
        self.pool_calib = None
        self.pool_calib_path = None
        self._maybe_snap_edges()
        self._sync_plane_widgets()
        self._redraw_plane()
        if swapped:
            self.log("far edge is now the short one (the seed had the pool the other way round)")
        self.log(f"seed {res.level}: anchor {res.anchor_ft:g} ft · {res.px_per_ft_x:.1f} px/ft along "
                 f"X at the dock edge ({res.px_per_ft_y:.1f} across) · drag a corner or a side to "
                 f"correct it, then Complete")

    def on_plane_dims(self) -> None:
        """Apply the two anchors and the width, then hold the checked edges if snapping is on.

        The labels are the operator's (2026-10-21): *Anchor (near dock)* is the X of the quad's
        DOCK-SIDE edge and *Anchor (opposite dock)* the X of its far edge.  The near anchor is where
        the operator types the distance of the visible pool marks (e.g. 6 ft) and puts the two
        dock-side corners on those marks, instead of reaching for a pool corner outside the frame.
        The distance is still measured from the dock, because the anchor composes with the axis
        rather than moving its origin.
        """
        if self.plane_session is None:
            self.log("start the plane first")
            return
        try:
            far = float(self.plane_len_var.get())
            near = float(self.plane_near_var.get() or 0.0)
            Wd = float(self.plane_wid_var.get())
            if Wd <= 0:
                raise ValueError("width must be positive")
            if near < 0:
                raise ValueError("the near-dock anchor cannot be before the dock (0 ft)")
            if far <= near:
                raise ValueError(f"the anchor opposite the dock ({far:g} ft) must be beyond the "
                                 f"near-dock anchor ({near:g} ft)")
            if far > plane_mod.METRIC_LENGTH_FT:
                raise ValueError(f"the +x axis only runs to {plane_mod.METRIC_LENGTH_FT:g} ft — "
                                 f"the far edge cannot be anchored beyond it")
        except ValueError as exc:
            self.log(f"plane size rejected: {exc}")
            return
        self.plane_session.length_ft = far - near
        self.plane_session.width_ft = Wd
        self.plane_session.dock_offset_ft = near
        self.plane_session.notes.append(
            f"size set: near-dock anchor {near:g} ft, opposite-dock anchor {far:g} ft, width {Wd:g} ft")
        self._maybe_snap_edges()
        self._sync_plane_widgets()
        self._redraw_plane()
        if near > 0:
            self.log(f"the plane's dock-side edge is X = {near:g} ft — put its two corners on the "
                     f"{near:g} ft marks; the grid still runs 0 → "
                     f"{plane_mod.METRIC_LENGTH_FT:g} ft")

    def _plane_dims_or_default(self) -> tuple[float, float, float]:
        """``(far anchor, near anchor, width)`` from the pane, defaulting politely."""
        def num(var, fallback: float, name: str) -> float:
            try:
                return float(var.get())
            except (TypeError, ValueError):
                self.log(f"plane {name} {var.get()!r} is not a number — using {fallback:g} ft")
                return fallback
        far = num(self.plane_len_var, SEED_ANCHOR_FAR_FT, "anchor opposite the dock")
        near = num(self.plane_near_var, SEED_ANCHOR_NEAR_FT, "anchor near the dock")
        Wd = num(self.plane_wid_var, float(plane_mod.DEFAULT_WIDTH_FT), "width")
        if far <= near:
            near = 0.0
        return far, near, Wd

    def _sync_plane_widgets(self) -> None:
        s = self.plane_session
        if s is None:
            self.plane_label.configure(text="no plane yet")
            self._sync_nudge_label()
            return
        n = sum(1 for v in s.corners.values() if v is not None)
        seed_tag = ""
        if s.seed:
            lvl = (s.seed.get("seed") or {}).get("level") or "?"
            seed_tag = f" · seed {lvl} ({s.seed.get('placement', 'auto')})"
        self.plane_label.configure(
            text=(f"anchors {s.dock_offset_ft:g} \u2192 "
                  f"{s.dock_offset_ft + s.length_ft:g} ft · axis 0\u2013"
                  f"{plane_mod.METRIC_LENGTH_FT:g} ft · {s.width_ft:g} ft wide · {n}/4 corners"
                  f"{seed_tag}")
                 + ("" if s.placed else " — click the missing ones"))
        # The two anchors are also the widget state: a loaded document (or a seed) puts its own
        # numbers here, so what is on screen and what will be rebuilt agree.
        try:
            self.plane_near_var.set(f"{s.dock_offset_ft:g}")
            self.plane_len_var.set(f"{s.dock_offset_ft + s.length_ft:g}")
            self.plane_wid_var.set(f"{s.width_ft:g}")
        except Exception:                                            # noqa: BLE001
            pass
        self.calib_prompt.configure(text="pool plane: " + s.prompt)
        self._sync_nudge_label()

    def _redraw_plane(self, reset_view: bool = False) -> None:
        """Show the padded photo once, then keep the plane as a vector overlay on top of it.

        The old path re-rendered ``draw_plane`` (photo + grid + handles, all baked in) on every
        corner drag, which at 16x meant re-resizing a crop for each mouse move AND scaling a 3 px
        datum into a 48 px band.  Now the photo is rendered only when the frame or its padding
        changes; a drag or a nudge just re-draws canvas items.
        """
        s = self.plane_session
        if s is None or self.pool_frame is None:
            return
        base = plane_mod.padded_canvas(self.pool_frame, s.pad)
        # Fit on the PHOTO, not on the padded canvas: 35 % padding each side would otherwise make
        # fit-to-window show the frame at a quarter of the pane and leave the rest black.
        fh, fw = self.pool_frame.shape[:2]
        self.calib_view.content_box = (float(s.pad[0]), float(s.pad[1]),
                                       float(s.pad[0]) + float(fw),
                                       float(s.pad[1]) + float(fh))
        key = (id(s), base.shape)
        if reset_view or key != self._plane_shown_key:
            self._plane_shown_key = key
            self.calib_view.show(base, reset_view=reset_view)
        else:
            self.calib_view.redraw_overlay()
        self._sync_plane_prompt()

    def _draw_plane_overlay(self, view) -> None:
        """The plane, its 0-35 ft grid and the handles, as Tk items over the photo.

        Canvas line widths are **screen** pixels, so a 1 px grid line stays 1 px at 16x zoom
        instead of becoming 16 px.  Handles are drawn at a constant 4 px radius for the same
        reason: they are targets, and a target that grows with the zoom covers the pixels the
        operator is aligning to.
        """
        c = view.canvas
        c.delete("overlay")
        s = self.plane_session
        if s is None:
            return
        z = float(view.zoom or 1.0)
        ox, oy = float(s.pad[0]), float(s.pad[1])

        def P(p) -> tuple[float, float]:
            return ((float(p[0]) + ox) * z, (float(p[1]) + oy) * z)

        # The detected waterlines (PLAN §17), as dashed screen-px vectors: drawn even when the quad is
        # not placed — that is exactly the state in which the operator presses the button, and a
        # refused seed is only diagnosable if the lines it was refused on are visible.
        if self.seed_edges is not None:
            for name, colour in (("far", "#ffcc44"), ("near", "#ff8ce0"), ("dock", "#7cf0ff")):
                e = getattr(self.seed_edges, name, None)
                if e is None or not getattr(e, "found", False) or e.fit is None:
                    continue
                (ax, ay), (bx, by) = e.fit["endpoints"]
                x0, y0 = P((ax, ay))
                x1, y1 = P((bx, by))
                if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1e5:
                    continue
                c.create_line(x0, y0, x1, y1, fill=colour, width=2, dash=(6, 4), tags="overlay")
                if e.points is not None and len(e.points) > 1:
                    step = max(1, len(e.points) // 40)
                    for q in e.points[::step]:
                        qx, qy = P(q)
                        c.create_rectangle(qx - 1.5, qy - 1.5, qx + 1.5, qy + 1.5,
                                           outline=colour, tags="overlay")

        if not s.placed:
            for tag in plane_mod.CORNER_TAGS:
                p = s.corners.get(tag)
                if p is not None:
                    x, y = P(p)
                    c.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#00dcff", width=1,
                                  tags="overlay")
            return

        H = s.homography()
        Wd = float(s.width_ft)
        L = float(s.length_ft)
        off = float(s.dock_offset_ft)

        def seg(X0: float, Y0: float, X1: float, Y1: float, colour: str, width: int,
                dash: tuple[int, int] | None = None) -> None:
            a = plane_mod.project(H, X0, Y0)
            b = plane_mod.project(H, X1, Y1)
            if not (_finite2(a) and _finite2(b)):
                return
            (x0, y0), (x1, y1) = P(a), P(b)
            # Skip absurd endpoints: past the vanishing point the projection flips sign and a
            # "line" would be drawn across the whole canvas, which reads as a plane error.
            lim = 1e5
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) > lim:
                return
            kw = {"fill": colour, "width": width, "tags": "overlay"}
            if dash:
                kw["dash"] = dash
            c.create_line(x0, y0, x1, y1, **kw)

        if bool(self.grid_var.get()):
            minor, major = plane_mod.MINOR_FT, plane_mod.MAJOR_FT
            for i in range(int(round(plane_mod.METRIC_LENGTH_FT / minor)) + 1):
                X = i * minor
                is_major = abs(X / major - round(X / major)) < 1e-9
                seg(X, 0.0, X, Wd, "#78d278" if is_major else "#44804a", 1)
                if is_major:
                    a = plane_mod.project(H, X, 0.0)
                    if _finite2(a):
                        x, y = P(a)
                        if abs(x) < 1e5 and abs(y) < 1e5:
                            c.create_text(x + 3, y - 5, text=f"{X:g}'", fill="#96f096",
                                          anchor="sw", font=("", 8), tags="overlay")
            for j in range(int(round(Wd / minor)) + 1):
                Y = j * minor
                is_major = abs(Y / major - round(Y / major)) < 1e-9
                seg(0.0, Y, plane_mod.METRIC_LENGTH_FT, Y,
                    "#63a063" if is_major else "#456b4f", 1)

        # X = 0 is the dock lip: the datum, and the only line that earns 2 px.  It is drawn even when
        # the near anchor puts the quad's own edge further down-pool.
        seg(0.0, 0.0, 0.0, Wd, "#3c78ff", 2)
        # The two +x anchors: dashed, because they are placement aids, not metric facts.
        seg(off, 0.0, off, Wd, "#ff9f43", 1, dash=(2, 3))
        seg(off + L, 0.0, off + L, Wd, "#e6d24a", 1, dash=(4, 3))
        seg(off, 0.0, off + L, 0.0, "#00c8ff", 1)
        seg(off, Wd, off + L, Wd, "#00c8ff", 1)

        placed_pts = [p for p in s.corners.values() if p is not None]
        cx0 = sum(p[0] for p in placed_pts) / max(1, len(placed_pts))
        cy0 = sum(p[1] for p in placed_pts) / max(1, len(placed_pts))
        for tag in plane_mod.CORNER_TAGS:
            p = s.corners.get(tag)
            if p is None:
                continue
            x, y = P(p)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ffffff", outline="#3c78ff",
                          width=1, tags="overlay")
            # Tag well outside the quad and clear of the 5 ft labels, which sit ~10 px off the line
            right, below = p[0] >= cx0, p[1] >= cy0
            anchor = ("nw" if right and below else "ne" if below else "sw" if right else "se")
            c.create_text(x + (18 if right else -18), y + (20 if below else -20), text=tag,
                          fill="#dfe7ff", anchor=anchor, font=("", 7), tags="overlay")
        for name in plane_mod.SIDE_TAGS:
            m = s.side_midpoint(name)
            if m is None:
                continue
            x, y = P(m)
            c.create_oval(x - 3, y - 3, x + 3, y + 3, outline="#c8c8c8", width=1, tags="overlay")
        for handle in {s.selected, s.dragging}:
            if handle is None:
                continue
            p = (s.corners.get(handle) if handle in plane_mod.CORNER_TAGS
                 else s.side_midpoint(handle))
            if p is None:
                continue
            x, y = P(p)
            c.create_oval(x - 9, y - 9, x + 9, y + 9,
                          outline="#ffee66" if handle == s.selected else "#ffffff",
                          width=2, tags="overlay")

    def _sync_plane_prompt(self) -> None:
        s = self.plane_session
        if s is not None:
            self.calib_prompt.configure(text="pool plane: " + s.prompt)

    def on_build_plane(self) -> None:
        """Complete: turn the aligned plane into a pool-calibration document (status only)."""
        doc = self._build_pool_doc()
        if doc is not None:
            self._show_pool_doc(doc, source="aligned plane")
            self.log("plane complete — Save New Plane makes it stick to this clip")

    def _plane_id(self) -> str:
        """The document's id: the loaded plane's, else one derived from the clip."""
        if self._plane_setup_id:
            return str(self._plane_setup_id)
        stem = (self.clip.stem if self.clip is not None else "pool")
        return f"plane_{stem}"

    def _build_pool_doc(self) -> dict | None:
        """Build the document from the quad as it stands, or log why it could not be built."""
        s = self.plane_session
        if s is None:
            self.log("start the plane first")
            return None
        try:
            return s.build(self._plane_id(), lens=self.pool_lens,
                           notes="water-surface plane (app_gui)")
        except Exception as exc:  # noqa: BLE001 - a bad quad must not kill the pane
            self.log(f"plane document not buildable: {type(exc).__name__}: {exc}")
            return None

    def _show_pool_doc(self, doc: dict, *, source: str, path: str | Path | None = None) -> None:
        """Adopt a pool document and say so in one line.

        The old 12-line gate/mapping dump is gone (operator order 2026-10-21: *"this is too much
        information to display"*).  What decides anything is still evaluated — a plane document's
        delivery gate is P1-P4 — but it is reduced to ``gates PASS`` / ``GATES FAIL`` in the status
        label, with the failing detail in the session log where it can be read on demand.
        """
        gates = doc.get("gates") or pool_calib_mod.validate_gates(doc)
        ok = pool_calib_mod.gates_pass(gates)
        pgates = doc.get("plane_gates") or {}
        pok = all(g["pass"] for g in pgates.values()) if pgates else True
        plane_doc = bool(doc.get("plane"))
        delivery_ok = pok if plane_doc else (ok and pok)

        self.pool_calib = doc
        self.pool_calib_path = None if path is None else Path(path)
        self.pool_status.configure(
            text=f"{doc.get('setup_id')} · route {doc.get('route')} · "
                 f"{'gates PASS' if delivery_ok else 'GATES FAIL'}")
        for name, g in list(pgates.items()) + list(gates.items()):
            if not g.get("pass"):
                self.log(f"  gate {name} FAIL: {g.get('detail')}")
        if plane_doc and gates:
            self.log("  the W gates grade the reconstructed pose — informational on this route: the "
                     "answer is the plane homography, so they can fail while every distance is right")
        if not delivery_ok:
            self.log("this document fails its own gate — any distance it produces is wrong by an "
                     "unmeasured amount; fix the plane before using it")
        mp = doc.get("mapping") or {}
        if mp.get("probe"):
            worst = max(r["roundtrip_error_ft"] for r in mp["probe"])
            at0 = mp.get("px_per_ft_along_at_0")
            self.log(f"  mapping: {'n/a' if at0 is None else f'{at0:.1f} px/ft along X at 0 ft'}, "
                     f"round-trip worst {worst:.1e} ft")
        self.log(f"pool calibration from {source}: route {doc.get('route')}, "
                 f"{'gates pass' if delivery_ok else 'GATES FAIL'}")
        self._redraw_plan()

    def on_save_pool(self) -> None:
        """Build the document and save it against this CLIP — the plane replaces the setup id.

        Operator order (2026-10-21): *"Save pool doc should automatically match planes to a set of
        clips; if an existing plane is selected and saved for another clip, that clip is added to
        the list of accepted clips for which that plane is valid."*  Hence: one file per planE,
        keyed by the clip's sha256, whose ``clips`` list grows whenever the same plane is saved
        while a different clip is open.
        """
        if self.plane_session is None or not self.plane_session.placed:
            self.log("nothing to save — make a plane and place its four corners first")
            return
        if not self.clip:
            self.log("load a clip first: the plane is saved against the clip")
            return
        doc = self._build_pool_doc()
        if doc is None:
            return
        sha = self._clip_sha()
        clip_entry = {"sha256": sha, "name": self.clip.name, "stem": self.clip.stem}
        clips = {c.get("sha256"): dict(c) for c in (self._plane_clips or [])
                 if isinstance(c, dict) and c.get("sha256")}
        clips[sha] = clip_entry
        setup_id = self._plane_id()
        doc["setup_id"] = setup_id
        doc["clips"] = [clips[k] for k in sorted(clips)]
        try:
            path = pool_calib_mod.setup_path(setup_id, getattr(self, "plane_dir", None))
            pool_calib_mod.save(doc, path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"plane save failed: {type(exc).__name__}: {exc}")
            self.pool_status.configure(text="SAVE FAILED — see the log")
            return
        self._plane_setup_id = setup_id
        self._plane_clips = doc["clips"]
        self.pool_calib = doc
        self.pool_calib_path = path
        self._refresh_plane_choices()
        if hasattr(self, "plane_pick_var"):
            self.plane_pick_var.set(setup_id)
        # The one thing the operator asked to see after pressing it.
        self.pool_status.configure(text=f"Plane saved — {len(doc['clips'])} clip(s) · {path.name}")
        self.log(f"plane saved to {path} for {len(doc['clips'])} clip(s): "
                 + ", ".join(c["stem"] for c in doc["clips"]))
        self._redraw_plan()

    def _clip_sha(self) -> str:
        """The clip's sha256, computed once and kept (it is also the plane's key)."""
        if not self.clip_sha and self.clip is not None:
            try:
                self.clip_sha = video_mod.sha256(self.clip)
            except Exception as exc:                                 # noqa: BLE001
                self.log(f"could not hash the clip ({type(exc).__name__}: {exc})")
                self.clip_sha = ""
        return self.clip_sha or ""

    def find_plane_for_clip(self, sha: str | None = None,
                            stem: str | None = None) -> tuple[Path, dict] | None:
        """The saved plane whose clip list contains this clip, or None.

        Matched on sha256 first (a rename cannot break it), then on the clip's stem for documents
        written before clips were recorded.  More than one match is a real ambiguity, so it is
        logged and the most recently written document wins.
        """
        sha = (self._clip_sha() if sha is None else sha) or ""
        stem = (self.clip.stem if stem is None and self.clip is not None else stem) or ""
        hits: list[tuple[Path, dict]] = []
        for path in pool_calib_mod.list_setups(getattr(self, "plane_dir", None)):
            try:
                doc = pool_calib_mod.load(path)
            except Exception:                                        # noqa: BLE001
                continue
            shas = {str(c.get("sha256")) for c in (doc.get("clips") or [])
                    if isinstance(c, dict) and c.get("sha256")}
            stems = {str(c.get("stem")) for c in (doc.get("clips") or [])
                     if isinstance(c, dict) and c.get("stem")}
            if sha and sha in shas:
                hits.append((path, doc))
            elif stem and (stem in stems
                           or doc.get("setup_id") in (stem, f"plane_{stem}")):
                hits.append((path, doc))
        if not hits:
            return None
        if len(hits) > 1:
            self.log(f"{len(hits)} saved planes claim this clip — using the newest: "
                     + ", ".join(p.name for p, _ in hits))
        return max(hits, key=lambda h: h[0].stat().st_mtime)

    def auto_load_plane_for_clip(self) -> bool:
        """Load pool doc, automatic: if a known clip is open, its plane comes with it."""
        if self.clip is None:
            return False
        hit = self.find_plane_for_clip()
        if hit is None:
            self.pool_status.configure(text="no saved plane for this clip")
            self.log("no saved plane for this clip — New Plane, then Save New Plane")
            return False
        path, doc = hit
        self._apply_plane_doc(doc, path)
        return True

    def _apply_plane_doc(self, doc: dict, path: str | Path | None = None) -> None:
        """Adopt a loaded plane document, restoring its editable quad when it has one."""
        self._show_pool_doc(doc, source=str(path or "saved plane"), path=path)
        self._plane_setup_id = doc.get("setup_id")
        self._plane_clips = list(doc.get("clips") or [])
        self._refresh_plane_choices()
        try:
            self.plane_pick_var.set(str(doc.get("setup_id") or "auto"))
        except Exception:                                            # noqa: BLE001
            pass
        try:
            self.plane_session = plane_mod.PlaneSession.from_doc(doc)
        except Exception as exc:  # noqa: BLE001
            self.plane_session = None
            self.log(f"no editable plane in this document ({exc}) — it can still be used as input")
        if self.plane_session is not None:
            if self.clip:
                self.pool_frame = read_frame(self.clip, int(self.calib_frame_idx.get()))
                if self.pool_frame is not None and not self.plane_session.frame_shape[0]:
                    self.plane_session.frame_shape = self.pool_frame.shape[:2]
            self._sync_plane_widgets()
            self._redraw_plane(reset_view=True)
            self.log("plane restored — drag it and Save New Plane again to update this clip's entry")
        n = len(self._plane_clips)
        self.log(f"plane {doc.get('setup_id')} applies to {n} clip(s)"
                 + (f"; this clip is one of them" if n else ""))

    def on_pick_plane(self, _event=None) -> None:
        """Adopt a chosen saved plane for this clip (or go back to ``auto``)."""
        sid = str(self.plane_pick_var.get() or "auto")
        if sid == "auto":
            self.auto_load_plane_for_clip()
            return
        for path in pool_calib_mod.list_setups(self.plane_dir):
            try:
                doc = pool_calib_mod.load(path)
            except Exception:                                        # noqa: BLE001
                continue
            if str(doc.get("setup_id")) == sid:
                self._apply_plane_doc(doc, path)
                self.log(f"plane {sid} adopted for {self.clip.stem if self.clip else 'this clip'} — "
                         f"Save New Plane adds this clip to its list")
                return
        self.log(f"saved plane {sid!r} is no longer on disk — back to auto")
        self.plane_pick_var.set("auto")
        self.auto_load_plane_for_clip()

    def _refresh_plane_choices(self) -> None:
        """The saved-plane list: ``auto`` plus the setup id of every plane on disk."""
        pick = getattr(self, "plane_pick", None)
        if pick is None:
            return
        values = ["auto"]
        for path in pool_calib_mod.list_setups(self.plane_dir):
            try:
                doc = pool_calib_mod.load(path)
            except Exception:                                        # noqa: BLE001
                continue
            sid = str(doc.get("setup_id") or path.stem)
            if sid not in values:
                values.append(sid)
        try:
            pick.configure(values=values)
        except Exception:                                            # noqa: BLE001
            pass
        current = str(self.plane_pick_var.get() or "auto")
        if current != "auto" and current not in values:
            self.plane_pick_var.set("auto")

    def on_clear_pool(self) -> None:
        """Drop the loaded plane/document state.  Kept internal: both Clear buttons were removed
        (operator order 2026-10-21), but a clip change still needs to start from nothing."""
        self.pool_calib = None
        self.pool_calib_path = None
        self.pool_result = None
        self.plane_session = None
        self._plane_setup_id = None
        self._plane_clips = []
        self.pool_status.configure(text="no saved plane for this clip")
        if hasattr(self, "plane_pick_var"):
            self.plane_pick_var.set("auto")
        self._sync_plane_widgets()
        self.log("cleared the pool (3-D) calibration")

    # -- Process ------------------------------------------------------------
    def _model_choices(self) -> list[str]:
        """Every .pt in ``models/``, current selection first — the trajectory fork's behaviour."""
        import glob
        try:
            found = sorted(glob.glob(str(Path(MODELS_DIR) / "*.pt")))
        except Exception:  # noqa: BLE001
            found = []
        cur = str(self.model_var.get()) if getattr(self, "model_var", None) else ""
        return list(dict.fromkeys(([cur] if cur else []) + found))

    # -- backbone scale (n / s), driven by the config registry ----------------
    @staticmethod
    def _scale_choice_for(spec_or_path) -> str:
        """Combobox label for a registry spec (or a checkpoint path), e.g. ``n — yolo26n · dogs v5``.

        An undeclared checkpoint returns :data:`SCALE_CUSTOM_LABEL` — never a parsed guess from the
        file name, because ``yolo26s-pose.pt`` and ``yolo26s-pose_dogs_clahe.pt`` share a backbone
        name and mean completely different keypoints.
        """
        spec = (spec_or_path if hasattr(spec_or_path, "filename")
                else spec_for_path(spec_or_path))
        if spec is None:
            return SCALE_CUSTOM_LABEL
        return f"{spec.scale} — {spec.label}"

    def _scale_specs(self) -> dict[str, object]:
        """Label -> ``ModelSpec`` for every selectable (dog-family) checkpoint, in registry order."""
        return {self._scale_choice_for(spec): spec for spec in dog_specs()}

    def _scale_choices(self) -> list[str]:
        return list(self._scale_specs()) + [SCALE_CUSTOM_LABEL]

    def _scale_status_text(self, model_path) -> str:
        """What the scale box resolved to, with the registry's role — or ``custom``."""
        spec = spec_for_path(model_path)
        if spec is None:
            return "custom checkpoint — not declared in the registry"
        return f"{spec.scale}-scale · {spec.role}"

    def on_scale_pick(self, _event=None) -> None:
        """Backbone choice: move the pose-model file onto that scale's declared checkpoint.

        The file stays the thing that runs (``DetectConfig.model_path``, and therefore the cache
        key); the scale is a selector into the registry, not a second source of truth.
        """
        spec = self._scale_specs().get(self.scale_var.get())
        if spec is None:
            self.log("backbone: custom — the pose model file above decides the checkpoint")
            return
        if not spec.is_file():
            self.scale_status.configure(text=f"{spec.scale}-scale · MISSING")
            self.log(f"backbone {spec.scale}: declared checkpoint is missing ({spec.filename})")
            return
        self.model_var.set(str(spec.path))
        self.cmb_model.configure(values=self._model_choices())
        self.on_model_pick()

    def on_model_pick(self, _event=None) -> None:
        path = Path(str(self.model_var.get()))
        if not path.is_file():
            self.log(f"pose model not found: {path}")
            self.model_status.configure(text="missing")
            return
        self.model_status.configure(text=path.name)
        # keep the backbone box honest: a file pick re-answers "n or s?"
        spec = spec_for_path(path)
        if hasattr(self, "scale_var"):
            self.scale_var.set(self._scale_choice_for(spec))
            self.cmb_scale.configure(values=self._scale_choices())
        if hasattr(self, "scale_status"):
            self.scale_status.configure(text=self._scale_status_text(path))
        self.log(f"pose model: {path.name} — switching checkpoints invalidates warm tracks; the "
                 f"next Run re-runs inference")
        if spec is not None:
            measured = spec.measured_line()
            self.log(f"  registry: {spec.scale}-scale, {spec.role}, layout {spec.layout}"
                     + (f" — measured on the 46-frame set: {measured}" if measured
                        else " — not measured on the labelled set"))
        else:
            self.log("  registry: undeclared checkpoint — its keypoint layout is resolved at load "
                     "and its accuracy has not been measured here")

    def on_model_browse(self) -> None:
        from tkinter import filedialog
        start = Path(str(self.model_var.get())).parent
        p = filedialog.askopenfilename(
            title="Choose a .pt pose checkpoint",
            initialdir=str(start if start.is_dir() else MODELS_DIR),
            filetypes=[("ultralytics checkpoint", "*.pt"), ("all", "*.*")],
        )
        if not p:
            return
        self.model_var.set(p)
        self.cmb_model.configure(values=self._model_choices())
        self.on_model_pick()

    def _model_for(self, cfg: "detect_mod.DetectConfig"):
        """Load (or reuse) the checkpoint for ``cfg``, verifying it has a known keypoint layout."""
        key = str(cfg.model_path)
        model = self._models.get(key)
        if model is None:
            model = detect_mod.load_model(cfg)          # load_model calls verify_checkpoint
            self._models[key] = model
            try:
                layout = detect_mod.verify_checkpoint(model)
                shape, names = list(layout.kpt_shape), dict(layout.names)
            except Exception as exc:  # noqa: BLE001
                layout, shape, names = None, ["?"], {}
                self.q.put(("log", f"MODEL REFUSED: {exc}"))
            if layout is not None:
                self.q.put(("log", f"model OK: {Path(key).name} layout={layout.key} "
                                   f"kpt_shape={shape} names={names}"))
                if layout.tailset_derived:
                    self.q.put(("log", f"WARNING: {layout.key} has no tailset keypoint — the tailset "
                                       f"is derived as {layout.tailset_source}. On Dog Diving footage "
                                       f"this checkpoint sees the person, not the dog."))
        return model

    def on_run(self) -> None:
        if not self.clip:
            self.log("load a clip first")
            return
        if self.stability is None:
            self.log("run the stability check first (PLAN §5 step 1)")
            return
        try:
            stability_mod.assert_usable(self.stability, override=self.stability_overridden,
                                        note="GUI run")
        except Exception as exc:  # noqa: BLE001
            self.log(f"BLOCKED: {exc}")
            return

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.cancel_event = threading.Event()
        self.progress.configure(value=0.0)
        self.progress_label.configure(text="starting…")
        self.contact_mode.configure(
            text=f"contact frame: {self.outcome_frame_spec()}")
        contact_spec = self.contact_spec()
        clip, calib = self.clip, self.calib
        pool_calib, pool_path = self.pool_calib, (
            None if self.pool_calib_path is None else str(self.pool_calib_path))

        min_points = int(self.min_points_var.get())
        self.log(f"minimum accepted points: {min_points}"
                 + ("" if min_points > 0 else "  (ladder off — fixed thresholds)"))

        model_path = Path(str(self.model_var.get()))
        if not model_path.is_file():
            self.log(f"pose model not found: {model_path} — pick one before running")
            return
        self.model_status.configure(text=model_path.name)
        self.log(f"pose model: {model_path.name}")

        def work() -> None:
            cfg = detect_mod.DetectConfig(model_path=model_path, imgsz=DEFAULT_IMGSZ,
                                          conf=DEFAULT_CONF, kp_conf=DEFAULT_KP_CONF)
            model = self._model_for(cfg)
            outcome = run_pipeline(
                clip, calib=calib, pool_calib=pool_calib, pool_calib_path=pool_path,
                cfg=cfg, model=model,
                out_dir=OUT_DIR, contact=contact_spec, draw=True,
                cancel_event=self.cancel_event,
                on_progress=lambda p: self.q.put(("progress", p)),
                min_points=min_points,
            )
            self.q.put(("outcome", outcome))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self.log("run started (worker thread)")

    def contact_spec(self) -> str:
        """The Run button always uses candidate #1 (PLAN §5's preselected pick).

        The operator changes it afterwards in Review, which re-measures without
        re-running inference — that is the whole point of the two-step design.
        """
        return "auto"

    def outcome_frame_spec(self) -> str:
        return "auto (PLAN §5 candidate #1); adjust in Review"

    def on_cancel(self) -> None:
        self.cancel_event.set()
        self.log("cancel requested — this reaches the inference loop only")
        self.cancel_btn.configure(state="disabled")

    def on_preview(self) -> None:
        if not self.clip:
            return
        idx = int(self.calib_frame_idx.get()) if self.review is None else (
            self.review.frame or 0)
        vis = annotated_frame(self.clip, idx, traj=None, contact=None, calib=self.calib,
                              measurement=None, show_grid=bool(self.grid_var.get()),
                              pool_doc=self.pool_calib, pool_result=self.pool_result)
        if vis is None:
            self.log(f"could not read frame {idx}")
            return
        self.process_view.show(vis)
        self.log(f"preview rendered for frame {idx} (no inference)")

    # -- Review -------------------------------------------------------------
    def _install_review(self, outcome: PipelineOutcome) -> None:
        assert outcome.run is not None
        run = outcome.run
        self.outcome = outcome
        # The 3-D route's per-clip result, when one ran.  Read back from the run rather than from
        # anything cached, so the overlay can only ever show a translation this run actually produced.
        self.pool_result = run.get("pool_translation")
        # The adaptive-threshold ladder this run's detection actually used (None when the fixed
        # thresholds applied throughout).  Read back from the run, like the pool result above.
        self.ladder = (run.get("result") or {}).get("detection_ladder")
        if self.ladder:
            for label, value in ladder_rows(self.ladder):
                self.log(f"{label}: {value}")
        if self.pool_result is not None:
            notes = self.pool_result.get("notes") or []
            self.log(f"pool translation: {self.pool_result.get('route')} route, "
                     f"{self.pool_result.get('n_track_points')} track points, "
                     f"{len(notes)} note(s)")
            cc = self.pool_result.get("cross_check") or {}
            if cc.get("compared"):
                self.log(f"cross-check: 3-D {cc['d3d_in']:.2f} in vs 1-D {cc['d1d_in']:.2f} in "
                         f"({cc['diff_in']:+.2f} in, {'AGREE' if cc['agree'] else 'DISAGREE'})")
        self.review = ReviewState(
            clip=Path(run["clip"]), traj=run["trajectory"], contact=run["contact"],
            calib=self.calib, pool_doc=self.pool_calib,
            image_size=(run["video"].width, run["video"].height),
            n_frames=int(run["video"].n_frames),
        )
        # A new run re-derives the entry point from its own detections: carrying the old frame over
        # would silently measure a point this run may not even have.
        self.review.entry_frame = None
        self.review.entry_live = False
        if hasattr(self, "entry_live_var"):
            self.entry_live_var.set(False)
        self.slider.configure(to=max(1, self.review.n_frames - 1))
        if self.review.frame is not None:
            self._syncing = True
            try:
                self.slider.set(int(self.review.frame))
            finally:
                self._syncing = False
        self._redraw_review()
        self._refresh_points()
        self._refresh_results()
        self._redraw_plan()

    def on_nudge(self, delta: int) -> None:
        if self.review is None:
            return
        self.review.nudge(delta)
        self._after_review_change()

    def on_slider(self, value) -> None:
        # tk.Scale.set() fires this callback too, so a programmatic sync would
        # otherwise recurse through _after_review_change -> slider.set -> here.
        if self.review is None or getattr(self, "_syncing", False):
            return
        self.review.choose(int(float(value)), "manual")
        self._after_review_change(redraw=True)


    # -- the entry frame (what the distance is measured at) -------------------
    def on_set_entry(self) -> None:
        """Pin the entry point to the current frame — the operator's 2D->3D choice.

        The frame is snapped to the nearest KEPT detection, so the number always comes from a real
        point; the arc and the drawn trajectory stop there.
        """
        if self.review is None:
            return
        asked = self.review.frame
        if asked is None:
            self.log("no frame selected")
            return
        got = self.review.set_entry_frame(int(asked))
        self.entry_live_var.set(False)
        self.review.entry_live = False
        if got is None:
            self.log("no accepted detection to use as the entry point")
        else:
            self.log(f"entry frame {got}"
                     + ("" if got == int(asked) else f" (nearest detection to frame {asked})")
                     + " — the distance is measured at this point")
        self._after_review_change()

    def on_clear_entry(self) -> None:
        """Back to the PLAN §16.2 rule: the final accepted detection decides the point.

        The manual (dragged) point is cleared as well as the entry frame — "Auto entry" has to mean
        automatic, and a leftover manual point would keep overriding the rule invisibly.
        """
        if self.review is None:
            return
        had_manual = self.review.manual_point is not None
        self.review.set_entry_frame(None)
        self.review.manual_point = None
        self.review.entry_live = False
        self._set_entry_mode(False)
        if hasattr(self, "entry_live_var"):
            self.entry_live_var.set(False)
        self.log("entry frame cleared — measuring the final accepted detection again"
                 + (" (the dragged tailset point was dropped too)" if had_manual else ""))
        self._after_review_change()

    def on_entry_live(self) -> None:
        """Tie the measured point to the frame slider, so stepping frames shows the 3-D result."""
        if self.review is None:
            return
        self.review.entry_live = bool(self.entry_live_var.get())
        if self.review.entry_live:
            self.review.entry_frame = None
            self.log("entry follows the slider — step frames and watch the distance move")
        else:
            got = self.review.set_entry_frame(self.review.frame)
            self.log(f"entry pinned at frame {got}")
        self._after_review_change()

    def on_toggle_marquee(self) -> None:
        """Arm/disarm the group-deselect band (a mode: a click already toggles one point)."""
        self.marquee_mode = not getattr(self, "marquee_mode", False)
        self.marquee_btn.configure(
            text="Deselect points (ARMED — drag a box)" if self.marquee_mode else "Deselect points")
        self.log("drag a box over the points to take out of the trajectory"
                 if self.marquee_mode else "group deselect off")

    def on_toggle_correct(self) -> None:
        """Arm/disarm *Correct Entry*: drag the tailset X onto the real contact point."""
        if self.review is None:
            self.log("run the clip first — there is no trajectory to correct yet")
            return
        self._set_entry_mode(not self.correct_entry_mode)

    def _set_entry_mode(self, on: bool) -> None:
        """Enter/leave the drag mode.  One place, so the button, the cursor and the flag agree."""
        self.correct_entry_mode = bool(on)
        if not on:
            self._entry_drag = None
            self.entry_cursor = None
        self.correct_btn.configure(
            text="Correct Entry (ARMED — drag the X)" if on else "Correct Entry")
        try:
            self.review_canvas.configure(cursor="crosshair" if on else "")
        except Exception:                                            # noqa: BLE001
            pass
        if self.review is not None:
            self.review_view.redraw_overlay()
        self.log("drag the X onto the tailset — the trajectory is cut where you drop it" if on
                 else "entry correction off — the measurement is unchanged")

    @property
    def correct_point_mode(self) -> bool:
        """Back-compat read of :attr:`correct_entry_mode` (the button read "Correct point")."""
        return bool(getattr(self, "correct_entry_mode", False))

    def on_opacity(self, _value=None) -> None:
        """Opacity slider moved: repaint the vector overlay.  The photo is untouched."""
        if self.review is not None:
            self.review_view.redraw_overlay()

    def on_review_motion(self, event) -> None:
        """Pointer over the canvas: while armed, the dot preview follows it and the X goes hot."""
        if self.review is None or not self.correct_entry_mode:
            return
        self.entry_cursor = self.review_view.to_image(event.x, event.y)
        self.review_view.redraw_overlay()

    def on_review_leave(self, _event=None) -> None:
        if self.entry_cursor is None:
            return
        self.entry_cursor = None
        if self.review is not None:
            self.review_view.redraw_overlay()

    def _toggle_frame(self, frame: int) -> bool:
        """Include/exclude one detection, then repaint.  Shared by the canvas and the point list."""
        assert self.review is not None
        now_out = self.review.toggle_point(int(frame))
        arc = self.review.arc()
        rms = arc.get("rms_px")
        self.log(f"frame {int(frame)}: {'EXCLUDED from' if now_out else 'back in'} the trajectory — "
                 f"{arc['n']} point(s) kept, "
                 + (f"arc rms {float(rms):.1f} px" if rms is not None
                    else "too few points for an arc"))
        self._after_review_change()
        return now_out

    def on_review_click(self, event) -> None:
        if self.review is None:
            return
        x, y = self.review_view.to_image(event.x, event.y)
        if getattr(self, "marquee_mode", False):
            # Arm the band.  The rectangle is drawn on the canvas; the points it covers are decided
            # on release, in IMAGE space, so zoom and pan cannot change which points were picked.
            self._band0 = (x, y)
            self._band1 = (x, y)
            self.review_canvas.delete("band")
            self._draw_band()
            return
        if self.correct_entry_mode:
            # Take hold of the mark: it follows the press and the drag, and the release commits
            # (see _finish_entry_drag), so a stray press can be dragged back off the tailset.
            self._entry_drag = (x, y)
            self.review.manual_point = (x, y)
            self.review_view.redraw_overlay()
            return

        # Otherwise: click a detection to take it in or out of the trajectory.  Without a mode to arm,
        # the click has to be forgiving — the grab radius is ~12 *screen* px, so it behaves the same at
        # every zoom instead of demanding a pixel-perfect hit on a 3 px dot.
        hit = self.review.nearest_point(x, y)
        if hit is None:
            self.log("no accepted detections to toggle")
            return
        reach = max(6.0, 12.0 / max(0.2, float(self.review_view.zoom or 1.0)))
        if hit[0] > reach:
            self.log(f"no trajectory point within {reach:.0f} px of the click (nearest "
                     f"{hit[0]:.0f} px) — nothing toggled")
            return
        self._toggle_frame(int(hit[1]["frame"]))

    # -- group deselect: a band, then the points it covers --------------------
    def _draw_band(self) -> None:
        c = self.review_canvas
        c.delete("band")
        if getattr(self, "_band0", None) is None or getattr(self, "_band1", None) is None:
            return
        z = float(self.review_view.zoom or 1.0)
        (x0, y0), (x1, y1) = self._band0, self._band1
        c.create_rectangle(x0 * z, y0 * z, x1 * z, y1 * z,
                           outline="#ffe14d", dash=(4, 3), width=2, tags="band")

    def on_review_drag(self, event) -> None:
        if self.review is None:
            return
        if self._entry_drag is not None:
            # Dragging the tailset: move it with the pointer (the X is a dot while it travels).
            x, y = self.review_view.to_image(event.x, event.y)
            self._entry_drag = (x, y)
            self.entry_cursor = (x, y)
            self.review.manual_point = (x, y)
            self.review_view.redraw_overlay()
            return
        if getattr(self, "_band0", None) is None:
            return
        self._band1 = self.review_view.to_image(event.x, event.y)
        self._draw_band()

    def on_review_release(self, event) -> None:
        """Commit the dragged tailset, or take every detection inside the band out of the trajectory."""
        if self.review is None:
            return
        if self._entry_drag is not None:
            self._finish_entry_drag(event)
            return
        if getattr(self, "_band0", None) is None:
            return
        x0, y0 = self._band0
        x1, y1 = self.review_view.to_image(event.x, event.y)
        self._band0 = self._band1 = None
        self.review_canvas.delete("band")
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        if abs(hi_x - lo_x) < 2.0 and abs(hi_y - lo_y) < 2.0:
            return                                  # a click, not a band: nothing to do
        pts = list(getattr(self.review.traj, "points", None) or [])
        inside = [int(p["frame"]) for p in pts
                  if lo_x <= float(p["x"]) <= hi_x and lo_y <= float(p["y"]) <= hi_y]
        if not inside:
            self.log("no detections inside the box")
            return
        # Group toggle: if every point in the box is already out, the box puts them back — otherwise
        # a second sweep over the same area could only ever add exclusions, with no way back.
        if all(f in self.review.excluded for f in inside):
            self.review.excluded.difference_update(inside)
            verb = "back into"
        else:
            self.review.excluded.update(inside)
            verb = "out of"
        self.log(f"{len(inside)} of {len(pts)} point(s) {verb} the trajectory")
        self._after_review_change()

    def _finish_entry_drag(self, event) -> None:
        """Commit a dragged tailset: the point, and the frame the trajectory is cut at.

        Operator order (2026-10-21): *"override the trajectory and X marker to place the location of
        the tailset"* — so the drop point becomes the measured one AND the trajectory stops at the
        detection nearest it, instead of the two disagreeing about where the dog entered.
        """
        s = self.review
        x, y = self.review_view.to_image(event.x, event.y)
        self._entry_drag = None
        self.entry_cursor = (x, y)
        s.manual_point = (x, y)
        hit = s.nearest_point(x, y)
        cut = None if hit is None else s.set_entry_frame(int(hit[1]["frame"]))
        self._set_entry_mode(False)
        self.log(f"tailset placed at ({x:.1f}, {y:.1f}) px"
                 + (f" — the trajectory and the arc stop at frame {cut}" if cut is not None
                    else " — no detection near it, so the trajectory is unchanged"))
        self._after_review_change()

    # -- the point list (same excluded set as the canvas) ---------------------
    def _refresh_points(self) -> None:
        tree = getattr(self, "pts_tree", None)
        if tree is None:
            return
        rows = self.review.point_rows() if self.review is not None else []
        tree.delete(*tree.get_children())
        for r in rows:
            conf = r.get("kp_conf")
            tree.insert("", "end", iid=str(r["frame"]), values=(
                "— EXCLUDED —" if not r["included"] else "include",
                r["frame"], f"{r['x']:.1f}", f"{r['y']:.1f}",
                "—" if conf is None else f"{float(conf):.2f}",
            ), tags=() if r["included"] else ("out",))
        n_out = len(self.review.excluded) if self.review is not None else 0
        if hasattr(self, "pts_note"):
            self.pts_note.configure(text=(
                f"{len(rows) - n_out} of {len(rows)} point(s) included"
                + (f"   ·   {n_out} deselected" if n_out else "")))
        # keep the row under the frame slider visible, so the list and the picture agree
        if self.review is not None and self.review.frame is not None:
            iid = str(int(self.review.frame))
            if tree.exists(iid):
                tree.see(iid)

    def on_point_click(self, event) -> str | None:
        """A click on the ``kept`` cell toggles that detection; elsewhere it just selects the row."""
        if self.review is None:
            return None
        col = self.pts_tree.identify_column(event.x)
        iid = self.pts_tree.identify_row(event.y)
        if not iid or col != "#1":
            return None
        self._toggle_frame(int(iid))
        return "break"

    def on_point_space(self, _event=None) -> str:
        if self.review is None:
            return "break"
        frames = [int(i) for i in self.pts_tree.selection()]
        if not frames:
            self.log("select one or more points first (Space toggles the selection)")
            return "break"
        for f in frames:
            self.review.toggle_point(f)
        # one repaint for the whole batch, not one per row
        arc = self.review.arc()
        self.log(f"{len(frames)} point(s) toggled — {arc['n']} kept")
        self._after_review_change()
        return "break"

    def on_point_goto(self, event) -> None:
        if self.review is None:
            return
        iid = self.pts_tree.identify_row(event.y)
        if not iid:
            return
        self.review.choose(int(iid), "point-list")
        self._after_review_change()

    def on_exclude_selected(self) -> None:
        if self.review is None:
            return
        frames = [int(i) for i in self.pts_tree.selection()]
        if not frames:
            self.log("select points in the list first, then Exclude selected")
            return
        self.review.excluded.update(frames)
        self.log(f"{len(frames)} point(s) excluded from the trajectory")
        self._after_review_change()

    def on_include_all(self) -> None:
        if self.review is None:
            return
        n = len(self.review.excluded)
        self.review.include_all()
        self.log(f"all points re-included ({n} restored)")
        self._after_review_change()

    def _after_review_change(self, redraw: bool = True) -> None:
        if self.review is None:
            return
        if self.review.frame is not None and not self.correct_point_mode:
            self._syncing = True
            try:
                self.slider.set(int(self.review.frame))
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._syncing = False
        self._redraw_review() if redraw else None
        self._refresh_points()
        self._refresh_results()

    # -- the Review overlay: opacity, zoom-dependent stroke, the entry marker --
    def _opacity(self) -> float:
        """The element opacity as a 0..1 factor, from the slider under the point table."""
        try:
            pct = float(self.entry_opacity.get())
        except Exception:                                            # noqa: BLE001
            pct = OPACITY_DEFAULT_PCT
        return max(0.0, min(1.0, pct / 100.0))

    def _bg_at(self, at) -> tuple[int, int, int]:
        """The frame's own RGB under an image point — the backdrop the alpha blend is computed over."""
        src = getattr(self.review_view, "_src", None)
        if src is None or at is None:
            return (128, 128, 128)
        h, w = int(src.shape[0]), int(src.shape[1])
        x = int(min(max(0, round(float(at[0]))), w - 1))
        y = int(min(max(0, round(float(at[1]))), h - 1))
        r, g, b = src[y, x][:3]
        return int(r), int(g), int(b)

    def _blend(self, colour: str, at=None, alpha: float | None = None) -> str:
        """``colour`` at the element opacity over the frame pixel at ``at``.

        Tk canvas items have no alpha, so transparency is applied as the same arithmetic an alpha
        blend is: ``ink*a + backdrop*(1-a)``.  Sampling the frame under each segment (rather than
        against one flat grey) is what makes 10 % read as *see-through* on water and on deck alike.
        """
        a = self._opacity() if alpha is None else max(0.0, min(1.0, float(alpha)))
        if a >= 0.999:
            return colour
        ir, ig, ib = _hex_rgb(colour)
        br, bg, bb = self._bg_at(at)

        def mix(c: int, d: int) -> int:
            return max(0, min(255, int(round(a * c + (1.0 - a) * d))))

        return f"#{mix(ir, br):02x}{mix(ig, bg):02x}{mix(ib, bb):02x}"

    @staticmethod
    def _review_stroke(k: float) -> tuple[int, tuple[int, int] | None, float]:
        """``(line width, dash, dot radius)`` for a zoom factor ``k = zoom / fit``.

        Operator order (2026-10-21): *"make the line style dynamic and dependent on zoom level,
        currently it obstructs the view of important elements."*  At fit the whole 960x600 frame is
        on an 880-px canvas, so the trajectory, the arc and the X all lie over the dog: a 2 px solid
        stroke there is a bar across the subject.  So at fit it is a 1 px dashed hairline; past 1.5x
        it becomes a closed 2 px stroke and past 4x 3 px, because at those zooms the same width covers
        a small part of the picture and hiding it behind dashes would only hurt reading.
        """
        if k < 1.5:
            return 1, (3, 3), 2.0
        if k < 4.0:
            return 2, (5, 2), 2.5
        return 3, None, 3.5

    def _ink_polyline(self, c, pts, P, colour: str, width: int,
                      dash: tuple[int, int] | None, alpha: float) -> None:
        """Draw a polyline on the canvas, blended per segment below full opacity.

        One item at full opacity (cheap); per-segment items when faded, because the backdrop — and
        so the blended colour — changes along the line.
        """
        if len(pts) < 2:
            return
        if alpha >= 0.999:
            flat: list[float] = []
            for p in pts:
                flat.extend(P(p))
            kw: dict[str, Any] = {"fill": colour, "width": width, "tags": "overlay"}
            if dash:
                kw["dash"] = dash
            c.create_line(*flat, **kw)
            return
        for a, b in zip(pts, pts[1:]):
            (x0, y0), (x1, y1) = P(a), P(b)
            mid = (0.5 * (float(a[0]) + float(b[0])), 0.5 * (float(a[1]) + float(b[1])))
            kw = {"fill": self._blend(colour, mid, alpha), "width": width, "tags": "overlay"}
            if dash:
                kw["dash"] = dash
            c.create_line(x0, y0, x1, y1, **kw)

    def _draw_review_overlay(self, view) -> None:
        """Trajectory, points and the tailset mark as CANVAS items at the operator's opacity.

        Two operator orders meet here (2026-10-21): the transparency slider (every item is drawn
        through :meth:`_blend`) and the zoom-dependent line style (:meth:`_review_stroke`).  Both are
        properties of the *overlay*, not of the photo, so the photo is never re-rendered for them.
        """
        c = view.canvas
        c.delete("overlay")
        r = self.review
        if r is None:
            return
        z = float(view.zoom or 1.0)
        fit = max(1e-6, float(getattr(view, "fit_zoom", 1.0) or 1.0))
        k = max(1.0, z / fit)
        line_w, dash, dot_r = self._review_stroke(k)
        alpha = self._opacity()
        armed = bool(self.correct_entry_mode)

        def P(p) -> tuple[float, float]:
            return (float(p[0]) * z, float(p[1]) * z)

        if alpha > 0.0:
            self._draw_review_grid(c, P, alpha,
                                   (self.review.measurement().get("y_px")
                                    if self.review is not None else None))
            traj_pts = [(float(p["x"]), float(p["y"])) for p in r.draw_points()]
            self._ink_polyline(c, traj_pts, P, "#ffe14d", line_w, dash, alpha)

            arc = r.arc()
            coeffs, span = arc.get("coeffs"), arc.get("frame_range")
            if coeffs and span:
                f0, f1 = int(span[0]), int(span[1])
                curve = []
                for f in range(f0, f1 + 1):
                    q = trajectory_mod.predict(coeffs, f)
                    curve.append((float(q[0]), float(q[1])))
                self._ink_polyline(c, curve, P, "#3cf03c", max(1, line_w - 1), dash, alpha)
            for p in (arc.get("inliers") or []):
                at = (float(p["x"]), float(p["y"]))
                x, y = P(at)
                c.create_oval(x - dot_r, y - dot_r, x + dot_r, y + dot_r,
                              outline=self._blend("#ff5cff", at, alpha),
                              width=1, tags="overlay")
            # Deselected detections stay visible (hollow, struck through): the operator has to be
            # able to find the one they clicked out, which a vanished point would not allow.
            for p in (getattr(r.traj, "points", None) or []):
                if int(p["frame"]) not in r.excluded:
                    continue
                at = (float(p["x"]), float(p["y"]))
                x, y = P(at)
                ink = self._blend("#9aa0a6", at, alpha)
                c.create_oval(x - dot_r, y - dot_r, x + dot_r, y + dot_r,
                              outline=ink, width=1, tags="overlay")
                c.create_line(x - dot_r, y, x + dot_r, y, fill=ink, tags="overlay")

        self._draw_entry_marker(c, P, line_w, dot_r, alpha, armed)
        if armed:
            self._draw_entry_hover(c, P, dot_r)

    def _review_grid(self) -> dict | None:
        """The metric lattice for the Review canvas, cached until the plane document changes.

        Cached because the overlay is re-drawn on every mouse move in *Correct Entry* mode, and the
        lattice is ~55 projected lines: recomputing them per motion event would be pure waste when
        neither the document nor the lens can change mid-drag.
        """
        doc = (self.review.pool_doc if self.review is not None else None) or None
        if not doc or not (doc.get("plane") or {}):
            return None
        block = doc["plane"]
        try:
            length = float(block.get("length_ft") or plane_mod.DEFAULT_LENGTH_FT)
            width = float(block.get("width_ft") or plane_mod.DEFAULT_WIDTH_FT)
        except (TypeError, ValueError):
            return None
        key = (id(doc), round(length, 6), round(width, 6))
        if getattr(self, "_grid_key", None) == key:
            return self._grid_cache
        lens = None
        try:
            lens = pool_calib_mod.lens_from_document(doc)
        except Exception:                                            # noqa: BLE001
            lens = None
        try:
            spec = plane_mod.grid_spec(doc["homography"], length_ft=length, width_ft=width,
                                       lens=lens)
        except Exception:                                            # noqa: BLE001
            spec = None
        self._grid_key, self._grid_cache = key, spec
        return spec

    def _draw_review_grid(self, c, P, alpha: float, fade_y) -> None:
        """The 1-ft lattice as canvas items, at the operator's opacity (operator order 2026-10-21).

        The lattice used to be baked into the photo, which meant the opacity slider could not touch
        it — the one element that covers the most picture was the one element the slider missed.  It
        comes from :func:`PlaneSession`'s own ``grid_spec``, so this canvas lattice and the lattice
        in an exported still are the same lattice.

        ``fade_y`` keeps the earlier rule that the lattice is see-through ABOVE the landing row, so
        the dog on its final frame is not covered by the grid behind it.
        """
        spec = self._review_grid()
        if not spec:
            return
        far = alpha * GRID_FADE_ALPHA

        def seg(a, b, colour: str, width: int, a_here: float) -> None:
            if a_here <= 0.0:
                return
            (x0, y0), (x1, y1) = P(a), P(b)
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1e5:
                return
            mid = (0.5 * (float(a[0]) + float(b[0])), 0.5 * (float(a[1]) + float(b[1])))
            c.create_line(x0, y0, x1, y1, width=width, tags=("overlay", "grid"),
                          fill=self._blend(colour, mid, a_here))

        fy = None if fade_y is None else float(fade_y)
        for ln in spec["lines"]:
            a, b = ln["a"], ln["b"]
            colour = GRID_MAJOR if ln["major"] else GRID_MINOR
            width = 2 if ln["major"] else 1
            if fy is None or abs(float(b[1]) - float(a[1])) < 1e-6:
                seg(a, b, colour, width, far if (fy is not None and float(a[1]) < fy) else alpha)
                continue
            above_a, above_b = float(a[1]) < fy, float(b[1]) < fy
            if above_a == above_b:
                seg(a, b, colour, width, far if above_a else alpha)
                continue
            t = (fy - float(a[1])) / (float(b[1]) - float(a[1]))
            cross = (float(a[0]) + t * (float(b[0]) - float(a[0])), fy)
            if above_a:
                seg(a, cross, colour, width, far)
                seg(cross, b, colour, width, alpha)
            else:
                seg(a, cross, colour, width, alpha)
                seg(cross, b, colour, width, far)
        for lab in spec["labels"]:
            at = lab["at"]
            here = far if (fy is not None and float(at[1]) < fy) else alpha
            if here <= 0.0:
                continue
            x, y = P(at)
            if max(abs(x), abs(y)) > 1e5:
                continue
            c.create_text(x + 3, y - 6, text=lab["text"], anchor="sw", font=("", 8),
                          tags=("overlay", "grid"), fill=self._blend(GRID_LABEL, at, here))
        datum = spec.get("datum")
        if datum:
            seg(datum["a"], datum["b"], GRID_DATUM, 2, alpha)

    def _entry_point(self) -> tuple[float, float] | None:
        """Where the tailset mark sits now, in image px: the measurement point, else the manual one."""
        r = self.review
        if r is None:
            return None
        m = r.measurement()
        if m.get("x_px") is not None and m.get("y_px") is not None:
            return float(m["x_px"]), float(m["y_px"])
        return r.manual_point

    def _entry_at(self, x: float, y: float) -> bool:
        """Is the pointer within the grab radius of the tailset mark?  Screen px, so zoom-proof."""
        cur = self.entry_cursor
        if cur is None:
            return False
        z = max(1e-6, float(self.review_view.zoom or 1.0))
        return float(np.hypot(float(cur[0]) - x, float(cur[1]) - y)) * z <= ENTRY_GRAB_PX

    def _draw_entry_marker(self, c, P, line_w: int, dot_r: float,
                           alpha: float, armed: bool) -> None:
        """The tailset mark: an **X**, or a small dot once the pointer is on it and dragging.

        The X is the one the exported still draws (``annotate``'s tilted cross, same red), so the
        pane and the artifact no longer show two different marks for one quantity.  ``armed`` keeps
        it findable down to 0 % opacity — an invisible handle cannot be grabbed.
        """
        at = self._entry_point()
        if at is None:
            return
        a = max(alpha, 0.35) if armed else alpha
        if a <= 0.0:
            return
        x, y = P(at)
        ink = self._blend(ENTRY_INK, at, a)
        if armed and (self._entry_drag is not None or self._entry_at(*at)):
            rr = max(2.5, float(dot_r) * 1.15)
            c.create_oval(x - rr, y - rr, x + rr, y + rr, fill=ink, outline=ink, tags="overlay")
            return
        s = max(5.0, float(dot_r) * 3.0)          # half-diagonal, screen px
        w = max(1, int(line_w))
        c.create_line(x - s, y - s, x + s, y + s, fill=ink, width=w, tags="overlay")
        c.create_line(x - s, y + s, x + s, y - s, fill=ink, width=w, tags="overlay")
        # One caption, carrying both facts the old baked label carried: which frame the entry is on
        # and where on the plane it lands, in feet.
        cut = self.review.cutoff() if self.review is not None else None
        bits: list[str] = []
        if cut is not None:
            bits.append(f"entry f{cut}")
        meas = self.review.measurement() if self.review is not None else {}
        if meas.get("distance_ft") is not None and meas.get("transverse_ft") is not None:
            bits.append(f"({float(meas['distance_ft']):.1f}, {float(meas['transverse_ft']):.1f}) ft")
        if bits:
            c.create_text(x + s + 4, y - s - 2, anchor="sw", fill=self._blend("#ffdede", at, a),
                          font=("", 8), tags="overlay", text="  ".join(bits))

    def _draw_entry_hover(self, c, P, dot_r: float) -> None:
        """A small dashed dot under the pointer while armed: exactly where the X will land."""
        cur = self.entry_cursor
        if cur is None or self._entry_drag is not None:
            return
        x, y = P(cur)
        rr = max(2.0, float(dot_r) * 0.9)
        c.create_oval(x - rr, y - rr, x + rr, y + rr, dash=(2, 2), width=1, tags="overlay",
                      outline=self._blend(ENTRY_INK, cur, max(0.35, self._opacity())))

    def _redraw_review(self) -> None:
        if self.review is None:
            return
        meas = self.review.measurement()
        # The frame carries the PHOTO only.  Everything the operator can fade or drag — the lattice,
        # the trajectory, the points and the tailset X — is a canvas item now, so the opacity slider
        # and the zoom can reach it (operator order 2026-10-21: the grid did not follow the opacity,
        # and the entry mark was drawn twice: once baked here, once on the canvas).
        vis = annotated_frame(self.review.clip, self.review.frame or 0,
                              traj=None, contact=self.review.contact,
                              calib=self.review.calib, measurement=None,
                              show_grid=False,
                              pool_doc=None, pool_result=None, entry_mark=False,
                              fade_above_y=meas.get("y_px"),
                              arc=None, excluded=())
        self.review_view.show(vis)
        self.readout.configure(text=self.review.readout())
        self._redraw_plan()

    # -- Results ------------------------------------------------------------
    def _refresh_results(self) -> None:
        for iid in self.res_tree.get_children():
            self.res_tree.delete(iid)
        if self.review is not None:
            meas = self.review.measurement()
            rows = measurement_rows(meas)
            for k, v in rows:
                self.res_tree.insert("", "end", values=(k, v))
            self.res_warn.delete("1.0", "end")
            for w in (meas.get("warnings") or []):
                self.res_warn.insert("end", f"- {w}\n")
            if not (meas.get("warnings") or []):
                self.res_warn.insert("end", "(no measurement warnings)\n")
        if self.stability is not None:
            self.res_tree.insert("", "end", values=("stability", self.stability.summary()))
        for label, detail, ok in gate_rows(self.calib):
            self.res_tree.insert("", "end", values=(f"gate {label}", detail))
        if self.calib:
            self.res_tree.insert("", "end", values=(
                "calibration RMS", f"{self.calib['fit']['residual_rms_in']:.3f} in"))
        for label, value in pool_rows(self.pool_result, self.pool_calib):
            self.res_tree.insert("", "end", values=(label, value))
        for label, value in ladder_rows(getattr(self, "ladder", None)):
            self.res_tree.insert("", "end", values=(label, value))
        self._redraw_plan()

    # -- the Plan view: the plane flattened to a rectangle (2026-10-21) ------
    def _redraw_plan(self) -> None:
        """The simplest thing that answers 'where on the pool did it land?'.

        The plane is already a metric rectangle, so the plan view needs no perspective trick: map
        ``(X, Y)`` feet straight onto a rectangle with one equal scale on both axes, and put a dot at
        the landing point.  It lives in the Review pane now (2026-10-21) and sizes itself to its
        slot, so it is drawn at whatever the pane actually gives it.
        """
        c = getattr(self, "plan_canvas", None)
        if c is None:
            return
        c.delete("all")
        plane = ((self.pool_calib or {}).get("plane") or {})
        if not plane:
            c.create_text(12, 12, anchor="nw", fill="#c8d0d8", width=280,
                          text="no plane yet — align one in Calibrate and Save New Plane")
            if hasattr(self, "plan_note"):
                self.plan_note.configure(text="")
            return

        off = float(plane.get("dock_offset_ft") or 0.0)
        L = float(plane.get("metric_length_ft") or 35.0)
        Wd = float(plane.get("width_ft") or 16.0)
        cw = float(c.winfo_width() if c.winfo_width() > 1 else 360)
        ch = float(c.winfo_height() if c.winfo_height() > 1 else 250)
        margin = 34.0
        scale = min((cw - 2 * margin) / max(L, 1e-6), (ch - 2 * margin) / max(Wd, 1e-6))
        ox, oy = margin, ch - margin                     # X = 0 at the left, Y = 0 at the bottom

        def P(X: float, Y: float):
            return ox + X * scale, oy - Y * scale

        # The quad the operator placed covers [off, off + span], not the whole axis: the axis still
        # runs 0 -> 35 ft because that is the range a distance may be reported over.
        span = float(plane.get("length_ft") or L)
        c.create_rectangle(*P(off, 0), *P(min(L, off + span), Wd), outline="#78d278", width=2)
        step = 5 if L > 12 else 2
        for X in range(0, int(L) + 1, step):
            a, b = P(X, 0), P(X, Wd)
            c.create_line(*a, *b, fill="#3c78ff" if X == 0 else "#3f6b47",
                          width=2 if X == 0 else 1)
            c.create_text(a[0], oy + 10, text=f"{X:g}'", fill="#9fd89f", font=("", 7))
        for Y in range(0, int(Wd) + 1, max(2, step)):
            a, b = P(0, Y), P(L, Y)
            c.create_line(*a, *b, fill="#47664f", width=1)
            c.create_text(ox - 5, a[1], text=f"{Y:g}'", fill="#9fd89f", anchor="e", font=("", 7))
        c.create_text(ox + 2, 10, anchor="nw", fill="#7fb2ff", font=("", 7),
                      text="dock edge (X = 0) \u2192")

        meas = self.review.measurement() if self.review is not None else {}
        X, Y = meas.get("distance_ft"), meas.get("transverse_ft")
        if X is None or Y is None:
            c.create_text(ox + 6, oy - 14, anchor="sw", fill="#c8d0d8", width=160,
                          text="no landing point yet")
            if hasattr(self, "plan_note"):
                self.plan_note.configure(text="")
            return
        x, y = P(float(X), float(Y))
        c.create_line(x - 7, y, x + 7, y, fill="#ff4040", width=2)
        c.create_line(x, y - 7, x, y + 7, fill="#ff4040", width=2)
        c.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ff4040", outline="#ffffff")
        c.create_text(x + 8, y - 8, anchor="sw", fill="#ffdede", font=("", 8),
                      text=f"X {float(X):.2f} ft \u00b7 Y {float(Y):.2f} ft")
        if hasattr(self, "plan_note"):
            self.plan_note.configure(
                text=f"{L:g}\u00d7{Wd:g} ft · {scale:.1f} px/ft · {meas.get('display', '')}")

    # -- export -------------------------------------------------------------
    #: Artifacts that must be re-rendered (not copied) when Review has deselections: the pipeline's
    #: own files still describe every detection, and copying them would silently drop the fix.
    REVIEW_EXPORT_KEYS = ("annotated", "still", "result", "report")

    def _review_snapshot(self) -> dict[str, Any] | None:
        """Plain data for a worker thread: what Review currently changed, and the run it edits."""
        if self.review is None or self.outcome is None or self.outcome.run is None:
            return None
        if not self.review.excluded and self.review.cutoff() is None:
            return None
        run = self.outcome.run
        result = run.get("result") or {}
        frame = self.review.frame
        still = (int(frame) if frame is not None
                 else int(getattr(self.review.traj, "last_arc_frame", None) or 0))
        return {
            "clip": Path(run["clip"]), "clip_id": run.get("clip_id"),
            "traj": self.review.traj, "contact": self.review.contact,
            "calib": self.calib, "measurement": self.review.measurement(),
            "arc": self.review.arc(), "excluded": set(self.review.excluded),
            "entry_frame": self.review.cutoff(),
            "still_frame": still, "show_grid": bool(self.grid_var.get()),
            "pool_doc": self.pool_calib if self.pool_result is not None else None,
            "pool_result": self.pool_result,
            "detect": run.get("detect"), "video": run.get("video"),
            "calib_path": (result.get("calibration") or {}).get("path"),
            "detect_config": result.get("config"), "ladder": self.ladder,
            "artifacts": run.get("artifact_names") or {},
            "clip_sha": (result.get("clip") or {}).get("sha256"),
            "clip_name": (result.get("clip") or {}).get("name"),
        }

    def _reviewed_result(self, snap: dict[str, Any]) -> dict:
        """``result.json`` as it stands after the operator's deselections."""
        return report_mod.build_result(
            clip_id=snap["clip_id"], clip_sha=snap["clip_sha"], clip_name=snap["clip_name"],
            video=snap["video"], detect=snap["detect"], traj=snap["traj"], contact=snap["contact"],
            measurement=snap["measurement"], calib=snap["calib"], calib_path=snap["calib_path"],
            detect_config=snap["detect_config"], artifacts=snap["artifacts"],
            still_frame=snap["still_frame"], ladder=snap["ladder"],
            excluded_frames=sorted(snap["excluded"]), arc_after_exclusion=snap["arc"],
            entry_frame=snap.get("entry_frame"),
        )

    def _write_reviewed(self, key: str, dst: Path, snap: dict[str, Any]) -> Path:
        """Render one export with the deselections applied.  Runs in a worker thread."""
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if key == "annotated":
            annotate_mod.write_video(
                snap["clip"], dst, traj=snap["traj"], contact=snap["contact"],
                calib=snap["calib"], measurement=snap["measurement"],
                show_grid=snap["show_grid"], pool_doc=snap["pool_doc"],
                pool_result=snap["pool_result"], arc=snap["arc"],
                excluded=snap["excluded"])
        elif key == "still":
            annotate_mod.write_still(
                snap["clip"], dst, snap["still_frame"], traj=snap["traj"],
                contact=snap["contact"], calib=snap["calib"],
                measurement=snap["measurement"], show_grid=snap["show_grid"],
                pool_doc=snap["pool_doc"], pool_result=snap["pool_result"],
                arc=snap["arc"], excluded=snap["excluded"])
        elif key == "result":
            report_mod.write_result_json(dst, self._reviewed_result(snap))
        elif key == "report":
            report_mod.write_report_txt(dst, self._reviewed_result(snap))
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"{key!r} is not a review-aware export")
        return dst

    def on_export(self, key: str, ext: str) -> None:
        from tkinter import filedialog
        if self.outcome is None or self.outcome.run is None:
            self.log("nothing to export yet")
            return
        src = self.outcome.run["artifacts"].get(key)
        if not src or not Path(src).is_file():
            self.log(f"artifact {key!r} is not present on disk")
            return
        dst = filedialog.asksaveasfilename(
            title=f"Export {key}", defaultextension=ext,
            initialfile=Path(src).name)
        if not dst:
            return

        snap = self._review_snapshot()
        if snap is not None and key in self.REVIEW_EXPORT_KEYS:
            # Re-render instead of copying: the pipeline artifact has no idea about the
            # deselections, and a video/JSON that quietly ignored them would re-report the
            # other dog's point as the jump.
            n = len(snap["excluded"])
            self.log(f"exporting {key} with {n} point(s) deselected — rendering…")
            self.status.configure(text=f"rendering {key} with {n} deselected point(s)")

            def work() -> None:
                try:
                    out = self._write_reviewed(key, Path(dst), snap)
                    self.q.put(("log", f"exported {key} (review applied, {n} deselected) -> {out}"))
                except Exception as exc:  # noqa: BLE001
                    self.q.put(("log", f"export failed: {type(exc).__name__}: {exc}"))

            threading.Thread(target=work, daemon=True).start()
            return

        try:
            shutil.copy2(src, dst)
            self.log(f"exported {Path(src).name} -> {dst}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"export failed: {type(exc).__name__}: {exc}")

    def on_open_outdir(self) -> None:
        if self.outcome is None or self.outcome.run is None:
            return
        import subprocess
        d = self.outcome.run["out_dir"]
        try:
            subprocess.Popen(["explorer", str(Path(d))], shell=False)
        except Exception:  # noqa: BLE001
            self.log(f"output directory: {d}")

    # -- worker -> UI pump --------------------------------------------------
    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "progress":
                    self._on_progress(payload)
                elif kind == "outcome":
                    self._on_outcome(payload)
                elif kind == "stability":
                    self._show_stability(payload)
                elif kind == "log":
                    self.log(str(payload))
                elif kind == "seed":
                    self._apply_seed(*payload)
                elif kind == "seed_error":
                    self.detect_btn.configure(state="normal")
                    self.log(f"waterline detection failed: {payload}")
                elif kind == "seed_done":
                    self.detect_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    def _on_progress(self, p: Progress) -> None:
        self.progress.configure(value=min(1.0, p.fraction))
        self.progress_label.configure(text=p.label() + (f"   [{p.message}]" if p.message else ""))
        if p.stage == "detect" and p.from_cache is not None:
            self.cache_label.configure(
                text="using cached tracks" if p.from_cache else "running inference (cold cache)")

    def _on_outcome(self, outcome: PipelineOutcome) -> None:
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if outcome.error:
            self.log(f"run FAILED: {outcome.error}")
            if outcome.tb:
                self.log(outcome.tb.strip().splitlines()[-1])
            return
        if outcome.cancelled:
            self.progress_label.configure(text="cancelled")
            self.log("run cancelled — no cache written, so this can never be mistaken "
                     "for a complete run")
            return
        self.progress.configure(value=1.0)
        self.progress_label.configure(text="complete")
        if outcome.detect is not None:
            self.cache_label.configure(
                text=("using cached tracks" if outcome.detect.from_cache
                      else "inference complete"))
        self._install_review(outcome)
        self.nb.select(self.tab_review)
        self.log("run complete — Review tab populated")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", type=Path, default=None, help="open this clip at startup")
    args = ap.parse_args(argv)

    import tkinter as tk
    root = tk.Tk()
    App(root, clip=args.clip)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
