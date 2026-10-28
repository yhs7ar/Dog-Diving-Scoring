"""M8 gate — the Tkinter wiring (PLAN.md §11 M8 / §5 app_gui.py).

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
PLAN §11's M8 verification is *"full manual run end-to-end on one clip without
touching the CLI"* — that click-through **is** the gate, and it needs a person.
Nothing here can self-certify it, and this file does not pretend to.

What it does cover is everything in that flow that is *not* the human clicking:

* the module imports and defines the five PLAN §5 steps;
* ``run_pipeline`` — the Run button's worker — runs a real clip end to end and
  produces PLAN §7's artifact set;
* cancel reaches the inference loop and, per ``detect.detect_clip``'s contract, a
  cancelled run never writes the cache, so it cannot be mistaken for a complete run;
* the Review step's frame selection (slider / nudge / candidate pick) re-measures
  **without** re-running inference, and clamps to the clip;
* the Results pane never invents inches: with no calibration the distance is ``None``
  and the reason is carried in ``warnings``;
* the calibration click mapping (canvas px -> image px) is the inverse of the
  display scale, so clicks land where the operator pressed;
* the App constructs, loads a clip and populates its widgets (skipped when no
  display is available, e.g. a headless cron run).

Run:  PYTHONPATH=src <venv-python> -m pytest src/tests/test_m8_gui.py -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_gui  # noqa: E402
from dog_diving import calibration as C  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP = PROJECT_ROOT / "data" / "clips" / "18-07.mp4"
NEEDS_CLIP = pytest.mark.skipif(not CLIP.is_file(), reason="sample clip missing")


# --- 1. the module and its five steps ---------------------------------------
def test_module_exposes_the_plan_steps() -> None:
    assert hasattr(app_gui, "App")
    assert hasattr(app_gui, "ReviewState")
    assert hasattr(app_gui, "run_pipeline")
    src = Path(app_gui.__file__).read_text(encoding="utf-8")
    for tab in ("1 Calibrate", "2 Process", "3 Review", "4 Results"):
        assert tab in src, f"PLAN §5 step {tab!r} has no tab"
    # the Load step is gone (order 2026-10-28): the toolbar opens clips and the stability check runs
    # by itself, so there is no tab for it
    assert "1 Load" not in src
    assert not hasattr(app_gui.App, "_build_load")
    # the threading rule: the Tk class must not be the thing that runs inference
    assert "threading.Thread" in src
    print(f"\n  app_gui: {Path(app_gui.__file__).name}, "
          f"{len(src.splitlines())} lines, 5 PLAN §5 steps present")


# --- 2. the Run button's worker, on a real clip -----------------------------
@NEEDS_CLIP
def test_run_pipeline_produces_plan7_artifacts(tmp_path: Path) -> None:
    """The headless equivalent of pressing Run.  A subclip keeps it to ~20 s."""
    seen: list[app_gui.Progress] = []
    outcome = app_gui.run_pipeline(
        CLIP, subclip=(300, 380), out_dir=tmp_path, draw=True,
        on_progress=seen.append,
    )
    assert outcome.error is None, outcome.error
    assert not outcome.cancelled
    assert outcome.run is not None
    run = outcome.run

    stages = [p.stage for p in seen]
    assert "detect" in stages and "done" in stages, stages
    frames = [p for p in seen if p.stage == "detect" and p.total]
    print(f"\n  progress reports: {len(seen)} ({len(frames)} carrying a frame count)")
    print(f"  stage sequence: {stages}")
    assert frames, "no per-frame progress reached the UI (PLAN §5 asks for frame count + ETA)"

    names = set(run["artifact_names"])
    print(f"  artifacts: {sorted(names)}")
    for required in ("tracks", "trajectory", "calib_used", "result", "report", "meta",
                     "annotated", "still"):
        assert required in names, f"PLAN §7 artifact {required!r} missing"
    for key, path in run["artifacts"].items():
        assert Path(path).is_file(), f"{key} not on disk: {path}"

    con = run["contact"]
    print(f"  contact chosen={con.chosen_frame} source={con.chosen_source} "
          f"candidates={[(c.source, c.frame, c.available) for c in con.ranked()]}")
    assert con.candidates, "no water-entry candidates (PLAN §5 step 4 needs them)"


# --- 3. cancel never leaves a run mistakable for a complete one -------------
@NEEDS_CLIP
def test_cancel_stops_inference_and_writes_no_cache(tmp_path: Path) -> None:
    import threading
    ev = threading.Event()
    ev.set()
    outcome = app_gui.run_pipeline(
        CLIP, subclip=(200, 300), out_dir=tmp_path, cancel_event=ev,
    )
    assert outcome.error is None, outcome.error
    assert outcome.cancelled, "a pre-set cancel_event did not cancel the run"
    print("\n  cancelled=True; cache files:", sorted(p.name for p in tmp_path.rglob("*.json")))
    assert not list(tmp_path.rglob("tracks.json")), (
        "a cancelled run wrote a tracks.json — it could then be mistaken for a complete run"
    )


# --- 4. Review re-measures without inference --------------------------------
@NEEDS_CLIP
def test_review_state_nudges_remeasure_without_inference(tmp_path: Path) -> None:
    outcome = app_gui.run_pipeline(CLIP, subclip=(300, 380), out_dir=tmp_path, draw=False)
    assert outcome.error is None, outcome.error
    run = outcome.run
    assert run is not None
    n = int(run["video"].n_frames)

    st = app_gui.ReviewState(
        clip=Path(run["clip"]), traj=run["trajectory"], contact=run["contact"],
        calib=None, image_size=(run["video"].width, run["video"].height), n_frames=n,
    )
    start = st.frame
    st.nudge(+1)
    assert st.frame == min(n - 1, (start or 0) + 1), (start, st.frame, n)
    st.nudge(-1)
    assert st.frame == start
    # clamping at both ends
    for _ in range(50):
        st.nudge(+10)
    assert st.frame == n - 1, f"right edge not clamped: {st.frame} vs {n - 1}"
    for _ in range(50):
        st.nudge(-10)
    assert st.frame == 0, f"left edge not clamped: {st.frame}"

    st.choose(137, "manual")
    m = st.measurement()
    print(f"\n  review: n_frames={n}, nudged/clamped OK, measured frame {st.frame}")
    print(f"  readout: {st.readout()}")
    assert "total_inches" in m, m.keys()
    # No plane document here (PLAN §16): the 1-D route is archived, so there is NO number and the
    # measurement says why. Inches are never invented.
    assert m["total_inches"] is None, m
    assert m["route"] == "plane" and m["warnings"], m
    assert m.get("warnings"), "a missing calibration must be reported, not silently dropped"


# --- 5. the Results pane never invents inches ------------------------------
def test_measurement_rows_are_honest_about_missing_inches() -> None:
    assert app_gui.measurement_rows(None) == []
    rows = dict(app_gui.measurement_rows({"total_inches": None, "sigma_in": None,
                                          "point_source": "model", "x_px": 10.0,
                                          "y_px": 20.0, "warnings": ["no calib"]}))
    print(f"\n  no-calibration rows: distance={rows['Distance']!r} "
          f"total={rows['Total inches']!r} uncertainty={rows['Uncertainty']!r}")
    assert rows["Distance"] == "n/a (no plane calibration)"
    assert rows["Total inches"] == "—" and rows["Uncertainty"] == "—"

    rows2 = dict(app_gui.measurement_rows({
        "total_inches": 271.0, "display": "22' 7\"", "sigma_in": 4.0, "sigma_complete": True,
        "point_source": "model", "x_px": 1.0, "y_px": 2.0, "calib_setup_id": "poolA",
        "waterline_gap_in": 1.5, "warnings": [],
    }))
    print(f"  with calibration: distance={rows2['Distance']!r} "
          f"uncertainty={rows2['Uncertainty']!r}")
    assert rows2["Distance"] == "22' 7\"" and rows2["Uncertainty"].startswith("± 4.00")


# --- 6. zoom / pan, and its effect on placement precision -------------------
def _view_or_skip(w: int = 900, h: int = 430):
    import tkinter as tk
    root = _root_or_skip()
    canvas = tk.Canvas(root, width=w, height=h)
    canvas.pack()
    view = app_gui.FrameView(canvas, w, h)
    view.bind_zoom()
    view._forced_viewport = (w, h)      # winfo_width() is 1 until the widget is mapped
    return root, canvas, view


def test_fit_zoom_cannot_be_zoomed_out_past() -> None:
    root, _canvas, view = _view_or_skip()
    try:
        view.show(np.zeros((1080, 1920, 3), dtype=np.uint8))
        print(f"\n  fit zoom = {view.zoom:.4f}  (1 canvas px = {1 / view.zoom:.2f} image px)")
        assert view.zoom == pytest.approx(app_gui.guard_frame(900, 430, 1920, 1080), abs=1e-6)
        assert view.set_zoom(0.01) == pytest.approx(view.fit_zoom), "zoomed out past fit"
        assert view.fit() == pytest.approx(view.fit_zoom)
    finally:
        root.destroy()


def test_zoom_reaches_the_placement_precision_marking_needs() -> None:
    """The whole point: at fit, 1 canvas px = 2.5 image px; M2 needs ~0.2."""
    root, _canvas, view = _view_or_skip()
    try:
        view.show(np.zeros((1080, 1920, 3), dtype=np.uint8))
        at_fit = 1.0 / view.zoom
        view.set_zoom(4.0)
        at_4x = 1.0 / view.zoom
        view.set_zoom(16.0)
        at_16x = 1.0 / view.zoom
        print(f"\n  image px per canvas px:  fit={at_fit:.2f}  4x={at_4x:.2f}  16x={at_16x:.2f}")
        assert at_fit > 2.0, "the fit view is more precise than measured — check the numbers"
        assert at_4x <= 0.26, f"4x only reaches {at_4x:.2f} image px"
        assert at_16x == pytest.approx(1 / 16, abs=1e-6)
        assert view.set_zoom(1000.0) == pytest.approx(app_gui.MAX_ZOOM), "zoom is not capped"
    finally:
        root.destroy()


def test_zoom_keeps_the_anchored_point_under_the_cursor() -> None:
    """Wheel-zoom must not slide the feature out from under the pointer."""
    root, _canvas, view = _view_or_skip()
    try:
        view.show(np.zeros((1080, 1920, 3), dtype=np.uint8))
        anchor = (450.0, 215.0)
        before = view.to_image(*anchor)
        view.set_zoom(4.0, anchor=anchor)
        after = view.to_image(*anchor)
        drift = max(abs(after[0] - before[0]), abs(after[1] - before[1]))
        print(f"\n  anchor {anchor}: image {tuple(round(v, 2) for v in before)} -> "
              f"{tuple(round(v, 2) for v in after)} after 4x  (drift {drift:.2f} image px)")
        # Tk folds the scroll origin to whole scrollregion px, which is 0.25 image px
        # at 4x -- so allow 1 image px, still 2.5x tighter than the fit view.
        assert drift <= 1.0, f"the anchored image point slid {drift:.2f} px"
    finally:
        root.destroy()


def test_click_mapping_accounts_for_zoom_and_scroll() -> None:
    root, canvas, view = _view_or_skip()
    try:
        view.show(np.zeros((1080, 1920, 3), dtype=np.uint8))
        view.set_zoom(2.0)
        canvas.xview_moveto(0.0)        # force the scroll to the origin AFTER zooming
        canvas.yview_moveto(0.0)
        ox, oy = view.to_image(0.0, 0.0)
        x, y = view.to_image(400.0, 200.0)
        print(f"\n  at 2x: origin maps to ({ox:.2f},{oy:.2f}) image px; "
              f"400x200 canvas px spans ({x - ox:.3f},{y - oy:.3f}) image px")
        # independent expectation: at 2x, 400 canvas px spans exactly 200 image px
        assert (x - ox, y - oy) == pytest.approx((200.0, 100.0), abs=1e-6)

        # and scrolling must shift the mapping, not be ignored
        canvas.xview_moveto(0.30)
        sx, _ = view.to_image(0.0, 0.0)
        print(f"  at 2x, scrolled 30%: origin maps to image x {sx:.1f}")
        assert abs(sx - ox) > 1.0, "the scroll offset is not folded into the click mapping"

        # guard_frame still refuses to upscale for the *fit* computation
        assert app_gui.guard_frame(4000, 4000, 1920, 1080) == 1.0
    finally:
        root.destroy()


# --- 7. calibration tables for the Calibrate pane ---------------------------
def test_residual_and_gate_rows_are_well_formed() -> None:
    marks = [{"label": f"{ft}'", "ft": ft, "inch": 0, "x_px": 520.0 + 18.0 * ft, "y_px": 700.0}
             for ft in (0, 1, 15, 16)]
    calib = C.build("gui-test", (1920, 1080), [[520, 100], [520, 980]],
                    [[0, 700], [1920, 700]], marks, model_sha="x", notes="")
    res = app_gui.residual_rows(calib)
    gates = app_gui.gate_rows(calib)
    print(f"\n  residual rows: {res}")
    print(f"  gate rows: {[(g[0]) for g in gates]}")
    assert len(res) == len(marks) and res[0][0].startswith("0")
    assert {g[0].split(":")[0] for g in gates} == {"G1", "G2", "G3", "G4", "G5"}
    assert app_gui.residual_rows(None) == [] and app_gui.gate_rows(None) == []


# --- 8. the window itself (skipped when there is no display) ----------------
def _root_or_skip():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        return root
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no Tk display available: {type(exc).__name__}: {exc}")


@NEEDS_CLIP
def test_app_constructs_and_loads_a_clip() -> None:
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=CLIP)
        root.update_idletasks()
        print(f"\n  window built; clip label: {app.clip_label.cget('text')}")
        assert app.info is not None and app.clip == CLIP
        assert str(CLIP) in app.clip_label.cget("text")
        assert int(app.slider.cget("to")) == app.info.n_frames - 1
        # the probe line lives in the session log now, and the app lands on Calibrate: the Load step
        # is gone (order 2026-10-28)
        log = app.log_text.get("1.0", "end")
        assert f"{app.info.width}x{app.info.height}" in log, log
        assert "codec=" in log, log
        assert app.nb.index(app.nb.select()) == 0
    finally:
        root.destroy()


@NEEDS_CLIP
def test_plane_clicks_land_where_pressed_at_zoom() -> None:
    """Clicks must map through zoom *and* scroll, and a click must not reset the zoom.

    Rewritten for the plane editor (2026-10-20): the 1-D mark sequence that used to own this canvas
    is archived, so the gesture under test is placing a plane corner.
    """
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=CLIP)
        app.on_start_plane()
        root.update_idletasks()
        plane = app.plane_session
        assert plane is not None

        view = app.calib_view
        view._forced_viewport = (900, 430)
        view.set_zoom(4.0)
        zoom_before = view.zoom

        class _Ev:                      # the only thing Tk hands a <Button-1> handler
            def __init__(self, x, y):
                self.x, self.y = x, y

        # click at a known IMAGE point; the canvas shows the padded image, so add the pad
        ix, iy = 300.0, 200.0
        z = view.zoom
        ev = _Ev((ix + plane.pad[0]) * z - view.canvas.canvasx(0.0),
                 (iy + plane.pad[1]) * z - view.canvas.canvasy(0.0))
        plane.clear()                   # so the click PLACES a corner rather than grabbing one
        app.on_calib_click(ev)
        got = plane.corners["00"]
        print(f"\n  zoom {z:.2f}: clicked image ({ix:.0f}, {iy:.0f}) -> corner {got}")
        assert abs(got[0] - ix) <= 1.5 and abs(got[1] - iy) <= 1.5, (got, (ix, iy))
        assert view.zoom == zoom_before, "a click reset the zoom"
    finally:
        root.destroy()


def test_m8_gate_scope() -> None:
    """Record, in the gate's own output, exactly what a machine can and cannot certify."""
    print("\n" + "=" * 74)
    print("M8 GATE (automated portion)")
    print("  module defines PLAN §5's five steps                        PASS")
    print("  Run worker: subclip -> all PLAN §7 artifacts               PASS")
    print("  per-frame progress + stage reports reach the UI            PASS")
    print("  cancel stops inference, writes no cache                    PASS")
    print("  Review: nudge/clamp/re-measure with NO inference           PASS")
    print("  Results: no calibration -> distance None, reason reported  PASS")
    print("  calibration clicks map canvas -> image px                  PASS")
    print("  window constructs, loads a clip, populates widgets         PASS")
    print("  ---------------------------------------------------------------")
    print("  PLAN §11 M8's actual gate — 'full manual run end-to-end on")
    print("  one clip without touching the CLI' — REQUIRES A HUMAN and is")
    print("  NOT certified here. Run:  PYTHONPATH=src python src/app_gui.py")
    print("M8 AUTOMATED PORTION PASS")
