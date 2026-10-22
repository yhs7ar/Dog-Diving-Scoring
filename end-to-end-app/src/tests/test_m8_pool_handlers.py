"""M18 / M8 — the pool-plane handlers in ``app_gui.py``, INVOKED through the real App.

Successor to the bars version of this file. The 3-D calibration input is now **one aligned plane**
on the water surface (PLAN §15): the operator drags its corners, the canvas is padded so corners may
sit outside the frame, and the homography that quad defines IS the distance mapping.

Why drive the handlers rather than the module: the module has its own file
(``test_m18_plane.py``). What can only be tested here is that pressing the buttons reaches it — the
defect this file was originally written to catch (``on_load_pool`` called by no test at all).
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_gui  # noqa: E402
from dog_diving import config as cfg_mod  # noqa: E402
from dog_diving.pool import calib as pool_calib_mod  # noqa: E402
from dog_diving.pool import lens as lens_mod  # noqa: E402
from dog_diving.pool import plane as plane_mod  # noqa: E402
from dog_diving.pool import pipeline as pipeline_mod  # noqa: E402
from tests.pool_scene import deck_marks_scene, frame_from_scene  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP = PROJECT_ROOT / "data" / "clips" / "18-07.mp4"
needs_clip = pytest.mark.skipif(not CLIP.is_file(), reason="sample clip missing")


# ===========================================================================
# helpers
# ===========================================================================
class Ev:
    """The only thing Tk ever hands a ``<Button-1>`` handler."""

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


def _canvas_xy(view, ix: float, iy: float) -> tuple[float, float]:
    """Display px -> the canvas px that maps back to them (inverse of ``FrameView.to_image``)."""
    z = view.zoom
    return (ix * z - view.canvas.canvasx(0.0), iy * z - view.canvas.canvasy(0.0))


def _event_at(app, ix: float, iy: float) -> Ev:
    """A press aimed at **image** pixel (ix, iy): the canvas shows the padded image, so add the pad."""
    s = app.plane_session
    pad = s.pad if s is not None else (0.0, 0.0)
    return Ev(*_canvas_xy(app.calib_view, ix + pad[0], iy + pad[1]))


def _root_or_skip():
    import time as _time
    last: Exception | None = None
    for _ in range(3):
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            return root
        except Exception as exc:  # noqa: BLE001
            last = exc
            _time.sleep(1.0)
    pytest.skip(f"no Tk display available after 3 attempts: {type(last).__name__}: {last}")


def calib_snapshot() -> dict[str, str]:
    import hashlib
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(app_gui.CALIB_DIR.glob("*.json"))}


def _restore_calib_dir(snapshot: dict[str, str]) -> None:
    after = calib_snapshot()
    assert after == snapshot, f"the handler drive changed {app_gui.CALIB_DIR}: {snapshot} -> {after}"


def scene_and_frame():
    sc = deck_marks_scene()
    frame = frame_from_scene(sc)
    assert frame.ndim == 3 and frame.shape[2] == 3, frame.shape
    return sc, frame


def scene_lens(sc) -> lens_mod.LensModel:
    return lens_mod.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines")


def corner_pixels(sc, length: float = 25.0) -> dict[str, tuple[float, float]]:
    """The four plane corners on the scene's true water surface, in image px."""
    W = float(sc.width_ft)

    def px(X, Y):
        v = sc.project(X, Y, sc.water_z_ft)
        return (float(v[0]), float(v[1]))

    return {"00": px(0, 0), "10": px(length, 0), "11": px(length, W), "01": px(0, W)}


def _app(root, clip=CLIP):
    app = app_gui.App(root, clip=clip)
    # New Plane now asks before replacing a plane on screen.  Nobody is there to click a modal in a
    # test, so the seam answers "yes" — the same answer the operator gives.
    app.confirm_reset = lambda: True
    root.update_idletasks()
    return app


def _align(app, sc, length: float = 25.0, zoom: float = 2.0, start: bool = True) -> None:
    """Drag the plane's corners onto the scene's true water-surface corners.

    ``start=False`` when the caller has already pressed Start plane — starting twice would build a
    second session and leave the caller inspecting the first one (measured: that mistake reads as a
    641 px "drag error" that is really a stale object).
    """
    if start:
        app.on_new_plane(seed=False)
    v = app.calib_view
    v._forced_viewport = (900, 430)
    app.root.update()
    v.set_zoom(zoom)
    app.root.update()
    for tag, (ix, iy) in corner_pixels(sc, length).items():
        # press ON the handle where it currently is (the pre-fitted quad is nowhere near the true
        # corner), then drag it to the target — which is what an operator does
        cx, cy = app.plane_session.corners[tag]
        app.on_calib_click(_event_at(app, cx, cy))
        assert app.plane_session.dragging == tag, (tag, app.plane_session.dragging)
        app.on_calib_drag(_event_at(app, ix, iy))
        app.on_calib_release(Ev(0.0, 0.0))


def _plane_doc(sc, *, lens=None, length: float = 25.0) -> dict:
    """A gate-passing plane document, built off-screen (for the load/save tests)."""
    W = float(sc.width_ft)
    s = plane_mod.PlaneSession(frame_shape=(sc.image_size[1], sc.image_size[0]),
                               length_ft=length, width_ft=W)
    for tag, p in corner_pixels(sc, length).items():
        s.set_corner(tag, *p)
    return s.build("poolSYN", lens=lens or scene_lens(sc))


# ===========================================================================
# 1. load / clear
# ===========================================================================
def test_the_automatic_load_skips_a_document_of_the_wrong_schema(tmp_path) -> None:
    """Load pool doc is AUTOMATIC now (2026-10-21): a 1-D document on disk must not become the datum.

    The old pane had a file dialog and could reject a document out loud.  There is no dialog any
    more, so the refusal has to happen in the scan: a file that is not a pool document is skipped,
    the clip is reported as having no saved plane, and nothing is adopted.
    """
    root = _root_or_skip()
    app = None
    try:
        app = _app(root)
        app.plane_dir = tmp_path
        app.clip = tmp_path / "unknown.mp4"
        app.clip_sha = "0" * 64
        bad = tmp_path / "poolA.pool.json"
        bad.write_text(json.dumps({"setup_id": "poolA", "marks": [], "fit": {"type": "linear"}}),
                       encoding="utf-8")
        assert app.auto_load_plane_for_clip() is False
        assert app.pool_calib is None, "a 1-D document became the 3-D datum"
        assert "no saved plane for this clip" in app.pool_status.cget("text")
    finally:
        app = None
        gc.collect()
        root.destroy()


@needs_clip
def test_a_saved_plane_is_loaded_automatically_for_its_clip(tmp_path, monkeypatch) -> None:
    """Save plane then reopen the clip: the plane comes back without touching a dialog."""
    root = _root_or_skip()
    before = calib_snapshot()
    sc, frame = scene_and_frame()
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: frame)
    doc = _plane_doc(sc)
    app = None
    try:
        app = _app(root)
        app.plane_dir = tmp_path                 # never write into the operator's CALIB_DIR
        sha = app._clip_sha()
        assert sha, "the clip was not hashed, so a plane could never be matched to it"
        doc["clips"] = [{"sha256": sha, "name": app.clip.name, "stem": app.clip.stem}]
        written = tmp_path / "planeSYN.pool.json"
        pool_calib_mod.save(doc, written)

        assert app.auto_load_plane_for_clip() is True, "the clip's own plane was not found"
        assert app.pool_calib is not None and app.pool_calib["setup_id"] == "poolSYN"
        assert app.pool_calib_path == written
        assert "gates PASS" in app.pool_status.cget("text")
        s = app.plane_session
        assert s is not None, "a plane document did not restore an editable session"
        assert s.corners["00"] == pytest.approx(corner_pixels(sc)["00"])

        app.on_clear_pool()
        assert app.pool_calib is None and app.plane_session is None
        assert app.pool_calib_path is None
        assert app.pool_status.cget("text") == "no saved plane for this clip"
    finally:
        app = None
        gc.collect()
        root.destroy()
    _restore_calib_dir(before)


@needs_clip
def test_a_document_without_a_plane_still_loads_as_input(tmp_path, monkeypatch) -> None:
    root = _root_or_skip()
    sc, frame = scene_and_frame()
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: frame)
    doc = _plane_doc(sc)
    for key in ("plane", "plane_gates", "mapping"):
        doc.pop(key)
    app = None
    try:
        app = _app(root)
        app._apply_plane_doc(doc, tmp_path / "old.pool.json")
        assert app.pool_calib is not None, "a pre-plane document was refused outright"
        assert app.plane_session is None
        assert "no editable plane" in app.log_text.get("1.0", "end")
    finally:
        app = None
        gc.collect()
        root.destroy()


# ===========================================================================
# 2. the plane editor, driven through the pane
# ===========================================================================
@needs_clip
def test_start_plane_aligns_and_builds_a_gate_passing_document(monkeypatch) -> None:
    root = _root_or_skip()
    before = calib_snapshot()
    sc, frame = scene_and_frame()
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: frame)
    app = None
    try:
        app = _app(root)
        app._plane_setup_id = "pool-plane"
        app.on_new_plane(seed=False)
        s = app.plane_session
        assert s is not None, "Start plane created no session"
        assert s.placed, "the plane should open pre-fitted so there is something to drag"
        # Was ``app.session is None`` — the archived 1-D click sequence is gone (2026-10-20), so
        # the surviving "one sequence at a time" guard is that the plane drops any loaded document.
        assert app.pool_calib is None and app.pool_calib_path is None, (
            "a started plane left a previously loaded document live")
        assert "pool plane" in app.calib_prompt.cget("text")

        _align(app, sc, start=False)          # the session was already started above
        assert app.plane_session is s, "Start plane created a second session"
        got = s.corners["00"]
        want = corner_pixels(sc)["00"]
        err = ((got[0] - want[0]) ** 2 + (got[1] - want[1]) ** 2) ** 0.5
        print(f"\n  corner drag landed {err:.2f} image px from the true corner")
        assert err <= 2.5, f"corner drag landed {err:.2f} px from truth (scroll folding ~2 px)"

        app.on_build_plane()
        doc = app.pool_calib
        assert doc is not None, "Build document never reached app.pool_calib"
        assert doc["route"] == "A_correspondences"
        assert doc["plane"]["schema"] == plane_mod.SCHEMA_BLOCK
        print("  P: " + "  ".join(f"{k}={'PASS' if g['pass'] else 'FAIL'}"
                                  for k, g in doc["plane_gates"].items()))
        print("  W: " + "  ".join(f"{k}={'PASS' if g['pass'] else 'FAIL'}"
                                  for k, g in doc["gates"].items()))
        assert all(g["pass"] for g in doc["plane_gates"].values()), doc["plane_gates"]
        # the *delivery* gate is P: the W block grades a pose this route does not use, and with the
        # pane's assumed lens it may fail — the status must still say PASS and the pane must say why
        assert "gates PASS" in app.pool_status.cget("text"), app.pool_status.cget("text")
        assert "informational" in app.pool_gate_text.get("1.0", "end")

        # and the number the tool exists for: distance from the dock edge, from the plane alone.
        # With the pane's ASSUMED lens the distortion mismatch costs ~1 in here (measured 1.03), so
        # the tolerance is the assumed-lens budget; the measured-lens build below shows the route's
        # own error, which is where the accuracy claim lives.
        raw = sc.project(12.0, 8.0, sc.water_z_ft)
        point = (float(raw[0]), float(raw[1]))
        e = pipeline_mod.translate_track([], doc=doc, manual_point=point)["entry"]
        assert e["route"] == "homography"
        assert e["X_ft"] == pytest.approx(12.0, abs=0.2), e["X_ft"]
        app.pool_lens = scene_lens(sc)
        app.on_build_plane()
        exact = pipeline_mod.translate_track([], doc=app.pool_calib, manual_point=point)["entry"]
        assert exact["X_ft"] == pytest.approx(12.0, abs=0.02), exact["X_ft"]
        assert all(g["pass"] for g in app.pool_calib["gates"].values())
        print(f"  entry through the document: X={e['X_ft']:.2f} ft, "
              f"distance {e['distance_in']:.2f} in, pose-vs-plane {e['homography_agreement_ft']:.2e} ft")
    finally:
        app = None
        gc.collect()
        root.destroy()
    _restore_calib_dir(before)


@needs_clip
def test_the_pane_answers_from_the_plane_even_when_the_pose_gates_fail(monkeypatch) -> None:
    """The route's claim, pinned: a measured lens makes W pass; without one they may fail, and the
    distance still comes from the plane."""
    root = _root_or_skip()
    sc, frame = scene_and_frame()
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: frame)
    app = None
    try:
        app = _app(root)
        _align(app, sc)
        app.on_build_plane()
        prior = app.pool_calib
        assert prior["lens"]["source"] == "assumed"
        assert "gates PASS" in app.pool_status.cget("text")

        app.pool_lens = scene_lens(sc)
        app.on_build_plane()
        measured = app.pool_calib
        assert measured["lens"]["source"] == "lines"
        assert all(g["pass"] for g in measured["gates"].values()), measured["gates"]

        raw = sc.project(5.0, 4.0, sc.water_z_ft)
        out = [pipeline_mod.translate_track([], doc=d, manual_point=(float(raw[0]),
                                                                     float(raw[1])))["entry"]
               for d in (prior, measured)]
        print(f"\n  assumed lens: X={out[0]['X_ft']:.3f} ft, W3 "
              f"{'PASS' if prior['gates']['W3']['pass'] else 'FAIL'}")
        print(f"  measured lens: X={out[1]['X_ft']:.3f} ft, W3 "
              f"{'PASS' if measured['gates']['W3']['pass'] else 'FAIL'}")
        for e in out:
            assert e["X_ft"] == pytest.approx(5.0, abs=0.1)
    finally:
        app = None
        gc.collect()
        root.destroy()


@needs_clip
def test_the_canvas_is_padded_so_a_corner_can_be_dragged_outside_the_frame(monkeypatch) -> None:
    root = _root_or_skip()
    sc, frame = scene_and_frame()
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: frame)
    app = None
    try:
        app = _app(root)
        app.on_new_plane(seed=False)
        s = app.plane_session
        assert s.pad[0] > 0 and s.pad[1] > 0
        shown = app.calib_view.image_size
        print(f"\n  frame {frame.shape[1]}x{frame.shape[0]} shown on a padded "
              f"{shown[0]}x{shown[1]} canvas")
        assert shown[0] > frame.shape[1] and shown[1] > frame.shape[0]

        w, h = frame.shape[1], frame.shape[0]
        app.on_calib_click(_event_at(app, *s.corners["11"]))
        assert s.dragging == "11"
        app.on_calib_drag(_event_at(app, w + 200.0, h + 120.0))     # outside the image
        app.on_calib_release(Ev(0.0, 0.0))
        assert s.corners["11"][0] > w and s.corners["11"][1] > h
    finally:
        app = None
        gc.collect()
        root.destroy()


@needs_clip
def test_clicking_places_corners_and_side_drags_move_both_of_its_corners(monkeypatch) -> None:
    root = _root_or_skip()
    sc, frame = scene_and_frame()
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: frame)
    app = None
    try:
        app = _app(root)
        app.on_new_plane(seed=False)
        s = app.plane_session
        app.on_plane_clear()
        assert not s.placed

        pts = corner_pixels(sc)
        for tag in plane_mod.CORNER_TAGS:
            app.on_calib_click(_event_at(app, *pts[tag]))
            assert s.corners[tag] is not None, tag
        assert s.placed, "four clicks did not place the plane"

        before = {t: s.corners[t] for t in ("00", "10")}
        mid = s.side_midpoint("back")
        app.on_calib_click(_event_at(app, *mid))
        assert s.dragging == "back", s.dragging
        app.on_calib_drag(_event_at(app, mid[0], mid[1] + 30.0))
        app.on_calib_release(Ev(0.0, 0.0))
        for t in ("00", "10"):
            # 2.5 px: Tk folds its scroll origin to whole scrollregion pixels (~1-2 canvas px)
            assert s.corners[t][1] == pytest.approx(before[t][1] + 30.0, abs=2.5)
    finally:
        app = None
        gc.collect()
        root.destroy()


# ===========================================================================
# 3. save / reload, and one sequence at a time
# ===========================================================================
@needs_clip
def test_on_save_pool_writes_and_reloads_a_valid_document(tmp_path, monkeypatch) -> None:
    root = _root_or_skip()
    before = calib_snapshot()
    sc, frame = scene_and_frame()
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: frame)
    dest = tmp_path / "pool-plane.pool.json"
    monkeypatch.setattr(pool_calib_mod, "setup_path", lambda setup_id: dest)
    app = None
    try:
        app = _app(root)
        app.setup_id.set("pool-plane")
        app.on_save_pool()                       # nothing built yet -> refused, not a crash
        assert not dest.is_file()

        _align(app, sc)
        app.on_build_plane()
        assert app.pool_calib is not None

        app.on_save_pool()
        assert dest.is_file(), "Save pool doc wrote nothing"
        back = pool_calib_mod.load(dest)
        assert back["schema"] == "dog_diving_pool_calibration/v1"
        assert back["plane"]["schema"] == plane_mod.SCHEMA_BLOCK
        assert len(back["marks"]) == 2, back["marks"]
        assert len(back["plane"]["corners"]) == 4
        print(f"\n  saved {dest.name} ({dest.stat().st_size} B), reloads cleanly")
    finally:
        app = None
        gc.collect()
        root.destroy()
    _restore_calib_dir(before)


@needs_clip
def test_the_pose_model_is_selectable_and_missing_files_are_refused(monkeypatch) -> None:
    """Taken from ``dog-diving-trajectory``: a combobox over models/*.pt plus browse and a cache."""
    root = _root_or_skip()
    app = None
    try:
        app = _app(root)
        choices = app._model_choices()
        assert str(app_gui.detect_mod.MODEL_PATH) in choices, choices
        assert all(c.endswith(".pt") for c in choices), choices
        print(f"\n  model choices ({len(choices)}): "
              f"{[Path(c).name for c in choices][:4]} …")

        # a missing file is refused with a message, not a later crash in the worker thread
        app.model_var.set(str(Path(app_gui.MODELS_DIR) / "does_not_exist.pt"))
        app.on_model_pick()
        assert "not found" in app.log_text.get("1.0", "end")

        # browse lands on an existing checkpoint and is reflected in the combo + status
        target = str(app_gui.detect_mod.MODEL_PATH)
        monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **kw: target)
        app.on_model_browse()
        assert app.model_var.get() == target
        assert Path(app.model_status.cget("text")).name == Path(target).name
        # backbone selector (2026-10-22): n and s are first-class choices, and a scale pick
        # moves the file selection onto that scale's declared checkpoint (the file is what runs).
        n_spec = cfg_mod.default_for_scale("n")
        s_spec = cfg_mod.default_for_scale("s")
        assert app._scale_choice_for(n_spec) in app._scale_choices()
        assert app._scale_choice_for(s_spec) in app._scale_choices()
        app.scale_var.set(app._scale_choice_for(s_spec))
        app.on_scale_pick()
        assert app.model_var.get() == str(s_spec.path), app.model_var.get()
        assert Path(app.model_status.cget("text")).name == s_spec.path.name
        assert app.scale_status.cget("text").startswith("s-scale")
        app.scale_var.set(app._scale_choice_for(n_spec))
        app.on_scale_pick()
        assert app.model_var.get() == str(n_spec.path)
        assert app.scale_status.cget("text").startswith("n-scale")
        # ... and picking a *file* re-answers the scale question instead of leaving it stale
        app.model_var.set(str(s_spec.path))
        app.on_model_pick()
        assert app.scale_var.get() == app._scale_choice_for(s_spec)
        # an undeclared checkpoint is labelled custom, never parsed from its file name
        assert app._scale_choice_for(app_gui.MODELS_DIR / "best.pt") == app_gui.SCALE_CUSTOM_LABEL
        print("  backbone choices:", app._scale_choices())
        assert app._model_for(detect_mod_config(app, target)) is not None
        # ... and the cache returns the same object rather than reloading
        assert app._model_for(detect_mod_config(app, target)) is app._model_for(
            detect_mod_config(app, target))
    finally:
        app = None
        gc.collect()
        root.destroy()


def detect_mod_config(app, path: str):
    from dog_diving import detect as _d
    return _d.DetectConfig(model_path=Path(path))


def test_saving_an_existing_plane_for_another_clip_adds_the_clip(tmp_path, monkeypatch) -> None:
    """Operator rule: one pool = one file, and saving it under another clip ADDS that clip.

    Also the anchor for the whole scheme: the ``setup id`` entry is gone, so the file's name comes
    from the first clip that made it and its ``clips`` list is what decides which clips it serves.
    """
    root = _root_or_skip()
    before = calib_snapshot()
    _sc, frame = scene_and_frame()
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: frame)
    app = None
    try:
        app = _app(root, clip=None)
        app.plane_dir = tmp_path
        app.clip = tmp_path / "clipA.mp4"
        app.clip_sha = "a" * 64
        app.on_new_plane(seed=False)
        assert app.plane_session is not None and app.plane_session.placed
        app.on_save_pool()
        files = sorted(tmp_path.glob("*.pool.json"))
        assert len(files) == 1, files
        doc = pool_calib_mod.load(files[0])
        assert [c["sha256"] for c in doc["clips"]] == ["a" * 64], doc["clips"]
        assert "Plane saved" in app.pool_status.cget("text")

        # the same plane, saved while a DIFFERENT clip is open: the clip joins the list
        app.clip = tmp_path / "clipB.mp4"
        app.clip_sha = "b" * 64
        app.on_save_pool()
        files = sorted(tmp_path.glob("*.pool.json"))
        assert len(files) == 1, f"a second file appeared instead of adding the clip: {files}"
        doc = pool_calib_mod.load(files[0])
        assert sorted(c["sha256"] for c in doc["clips"]) == ["a" * 64, "b" * 64], doc["clips"]
        assert "2 clip(s)" in app.pool_status.cget("text"), app.pool_status.cget("text")
    finally:
        app = None
        gc.collect()
        root.destroy()
    _restore_calib_dir(before)


def test_the_1d_calibration_entry_points_are_gone() -> None:
    """The 1-D mark calibration is archived (2026-10-20) — the plane is the only ground truth."""
    for gone in ("on_start_calib", "on_calib_undo", "on_calib_skip", "on_save_setup",
                 "on_load_setup", "on_load_setup"):
        assert not hasattr(app_gui.App, gone), f"App.{gone} still exists; the archive is incomplete"
    assert hasattr(app_gui.App, "on_new_plane")


def test_the_2026_10_21_calibrate_and_review_controls_are_gone() -> None:
    """Operator order (2026-10-21): Start Plane -> New Plane, and five controls deleted.

    Pinned by name because each one is a plausible re-addition: the setup-id entry (planes are keyed
    by clip now), the Clear plane / Clear buttons, the Undo seed button, the pool gate dump, the
    water-entry candidate table and the whole Plan tab (its rectangle moved into Review).
    """
    for gone in ("on_plane_clear", "on_undo_seed", "on_load_pool", "on_candidate_pick",
                 "_refresh_candidates", "_build_plan", "on_start_plane"):
        assert not hasattr(app_gui.App, gone), f"App.{gone} still exists; the removal is incomplete"
    src = Path(app_gui.__file__).read_text(encoding="utf-8")
    for gone in ("pool_gate_text", "cand_tree", "Undo seed", "Clear plane", "tab_plan",
                 "self.setup_id"):
        assert gone not in src, f"{gone!r} is still in the pane"
    # and the replacements exist
    for new in ("on_new_plane", "on_make_parallelogram", "on_save_pool", "auto_load_plane_for_clip",
                "find_plane_for_clip", "on_set_entry", "on_toggle_marquee", "_draw_review_overlay",
                "_redraw_plan", "plane_near_var", "snap_far_var"):
        assert hasattr(app_gui.App, new) or new in src, f"App.{new} is missing"
