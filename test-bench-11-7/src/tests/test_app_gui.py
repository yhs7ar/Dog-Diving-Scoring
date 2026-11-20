"""M7 (GUI) — headless smoke test: the app constructs, refreshes, and drives
the non-torch flow end-to-end (select -> stage -> suite -> results table)."""
from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

import app_gui                      # noqa: E402
from test_bench import analyze as analyze_mod   # noqa: E402
from test_bench import ground_truth as gt_mod   # noqa: E402
from test_bench import ingest as ingest_mod     # noqa: E402
from test_bench import manifest as manifest_mod  # noqa: E402
from test_bench import suite as suite_mod       # noqa: E402
from dog_diving import trajectory as traj_mod   # noqa: E402

CLIP = REPO / "Our_dog_diving_videos" / "test-clip-video004_right.mp4"
PLANE = REPO / "end-to-end-app" / "planes" / "plane_test-clip-video004_right.pool.json"
SEG01 = REPO / "end-to-end-app" / "out" / "video004_right-00.00.36.812-00.00.38.328-seg01" / "tracks.json"
MODEL_SHA = "bdb7688775ee39201d4a77ef0041c7c3e9eb9202f7a5df287a11c98e80ef82e4"


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no Tk display available")
    r.withdraw()
    yield r
    r.destroy()


def test_app_constructs_headless(root):
    app = app_gui.App(root)
    assert len(app.nb.tabs()) == 7
    assert "Calibrate" in [app.nb.tab(t, "text") for t in app.nb.tabs()]
    assert "Accuracy" in [app.nb.tab(t, "text") for t in app.nb.tabs()]
    tabs = app.nb.tabs()
    assert app.nb.tab(tabs[0], "text") == "Select clips"
    app.on_clips_changed()          # zero-clip refresh must not crash


@pytest.mark.skipif(not CLIP.is_file() or not SEG01.is_file(), reason="no parent artifacts")
def test_non_torch_flow_end_to_end(root, tmp_path):
    app = app_gui.App(root)
    app.root_dir = tmp_path
    (tmp_path / "results").mkdir(exist_ok=True)

    # 1. select the clip (module — the pane is a thin wrapper)
    ingest_mod.ingest_clip(CLIP, tmp_path, clip_id="c1")
    cdir = tmp_path / "c1"

    # 2. stage a plane in the saved-planes folder + detections (seg01 cached) + GT
    setup_id = "test-gui-c1"
    plane_dst = manifest_mod.plane_setup_path(setup_id)
    plane_dst.parent.mkdir(parents=True, exist_ok=True)
    plane = json.loads(PLANE.read_text(encoding="utf-8"))
    plane["setup_id"] = setup_id
    manifest_mod.write_json(plane_dst, plane)
    try:
        records, _ = analyze_mod.load_cached_records(SEG01, model_sha=MODEL_SHA)
        (cdir / "detections.json").write_text(json.dumps(records), encoding="utf-8")
        gt = gt_mod.set_ground_truth(plane, frame=85, x_px=532.9, y_px=311.0, judge_in=96.5)
        gt_mod.write_ground_truth(gt, cdir)
        man = manifest_mod.read_manifest(cdir)
        man["plane"] = {"setup_id": setup_id, "present": True}
        man["ground_truth"] = {"path": "ground_truth.json", "judge_present": True, "gt_in": gt["gt_in"]}
        manifest_mod.write_manifest(man, cdir)

        # 3. the GUI sees the clip and refreshes every pane
        app.on_clips_changed()
        assert app.list_clips() == ["c1"]

        # 4. run a suite via the module, persist, and populate the Results table
        points = traj_mod.assemble(records)
        clips = [{"clip_id": "c1", "points": points, "plane_doc": plane, "gt": gt,
                  "frame_size": [man["meta"]["width"], man["meta"]["height"]]}]
        configs = suite_mod.grid_configs([8, 12], ["last_n", "all"])
        res = suite_mod.run_suite(clips, configs)
        manifest_mod.write_json(tmp_path / "results" / "r1.json",
                                {"run_id": "r1", "config": {}, **res})
        app.last_run = "r1"
        app.results.refresh()
        n_rows = len(configs)  # 2 n_points x 2 selections
        assert len(app.results.tree.get_children()) == n_rows

        # 5. the Analyze pane loads a real frame + trajectory overlay (exercises
        #    FrameView + cv2 + PIL headless)
        app.analyze.clip_var.set("c1")
        app.analyze._on_clip()
        assert app.analyze.man is not None
        assert app.analyze._n_frames > 0
        assert app.analyze.view._src is not None

        # 6. Compare pane loads the active run's runs
        app.compare.load_clip("c1")
        assert len(app.compare.runs) == n_rows
    finally:
        plane_dst.unlink(missing_ok=True)


def test_compare_pane_builds_review_overlays(root):
    app = app_gui.App(root)
    pane = app.compare
    # fake GT + one run with selected detections, a fitted curve and an entry point
    pane.gt = {"x_px": 500.0, "y_px": 300.0, "gt_in": 98.0, "judge_present": False}
    pane.plane_doc = None      # no plane -> grid overlay omitted
    pane.points = []
    run = {
        "method": "arc_2d", "n_points": 4, "selection": "last_n",
        "pixel_error_px": 3.2, "score_error_in": 1.1, "judge_error_in": None,
        "score": {
            "entry_px": [510.0, 305.0],
            "fit": {
                "curve_space": "px",
                "curve": [[500.0, 300.0], [510.0, 305.0]],
                "selected_px": [[500.0, 300.0], [510.0, 305.0]],
            },
        },
    }
    pane.runs = [(run, "#00e5ff")]
    overlays = pane._build_overlays()
    kinds = {o[0] for o in overlays}
    assert {"marker", "dot", "curve", "segment", "label"} <= kinds
    # exactly one red GT marker
    assert sum(1 for o in overlays if o[0] == "marker" and o[2] == "red") == 1
    # the error vector connects GT to the entry point
    segs = [o for o in overlays if o[0] == "segment"]
    assert segs and segs[0][1] == (500.0, 300.0, 510.0, 305.0)
    # filter the run out via a non-matching n_points group -> only the GT marker
    pane.n_group = app_gui.CheckGroup(pane, "n_points", [("99", 99, True)], default_checked=[99])
    assert pane._run_visible(run) is False
    assert {o[0] for o in pane._build_overlays()} == {"marker"}


@pytest.mark.skipif(not PLANE.is_file(), reason="no parent plane doc")
def test_compare_pane_draws_judge_line(root):
    app = app_gui.App(root)
    pane = app.compare
    pane.plane_doc = json.loads(PLANE.read_text(encoding="utf-8"))
    pane.gt = {"x_px": 532.0, "y_px": 311.0, "gt_in": 98.0,
               "judge_present": True, "judge_in": 96.5}
    pane.points = []
    pane.runs = []

    line = pane._judge_line()
    assert line is not None
    a, b = line
    assert all(map(float, (a[0], a[1], b[0], b[1])))     # finite image px
    assert (a[0], a[1]) != (b[0], b[1])                  # a non-degenerate across-pool line

    overlays = pane._build_overlays()
    segs = [o for o in overlays if o[0] == "segment" and o[2] == pane.JUDGE_COLOUR]
    assert len(segs) == 1
    labels = [o for o in overlays if o[0] == "label" and o[2] == pane.JUDGE_COLOUR]
    assert labels and "judge 96.5in" in labels[0][1][2]

    # no judge -> no judge line
    pane.gt = {"x_px": 532.0, "y_px": 311.0, "gt_in": 98.0, "judge_present": False}
    assert pane._judge_line() is None


def test_run_suite_filters_expose_all_axes(root):
    app = app_gui.App(root)
    pane = app.run
    # methods: only implemented ones selectable; reserved slots listed but disabled
    assert pane.method_group.selected() == ["arc_2d", "arc_2d_centerline"]
    assert len(pane.method_group._items) == 4   # arc_2d + arc_2d_centerline + reserved arc_3d/pc_route
    assert set(pane.n_group.selected()) == {4, 6, 8, 12, 20, "all"}
    assert set(pane.sel_group.selected()) == {"last_n", "all", "ransac_inliers"}
    cfgs = suite_mod.grid_configs(pane.n_group.selected(), pane.sel_group.selected(),
                                  methods=pane.method_group.selected())
    assert len(cfgs) == 6 * 3 * 2             # n_points x selections x methods


def test_analyze_marquee_deselects_points_inside_rect(root):
    app = app_gui.App(root)
    pane = app.analyze
    pane._traj = app_gui._Trajectory({"points": [
        {"frame": 0, "x": 100.0, "y": 100.0},
        {"frame": 1, "x": 150.0, "y": 120.0},
        {"frame": 2, "x": 300.0, "y": 300.0},
    ]})
    pane._deselect_mode = True
    pane._plane_doc = None
    pane.clip_var.set("")                     # no clip -> persist short-circuits
    pane._redraw_review = lambda: None        # no frame loaded -> skip canvas render
    pane._on_marquee(90.0, 90.0, 200.0, 200.0)
    assert pane.excluded == {0, 1}            # only points inside the box
    # GT mode / deselection off -> the marquee is a no-op
    pane._deselect_mode = False
    pane._on_marquee(0.0, 0.0, 500.0, 500.0)
    assert pane.excluded == {0, 1}

