"""M7 (GUI) — headless smoke test: the app constructs, refreshes, and drives
the non-torch flow end-to-end (ingest -> stage -> suite -> results table)."""
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
    assert root.title()
    assert len(app.nb.tabs()) == 6
    assert app.list_clips() == []
    app.on_clips_changed()          # zero-clip refresh must not crash


@pytest.mark.skipif(not CLIP.is_file() or not SEG01.is_file(), reason="no parent artifacts")
def test_non_torch_flow_end_to_end(root, tmp_path):
    app = app_gui.App(root)
    app.root_dir = tmp_path
    (tmp_path / "results").mkdir(exist_ok=True)

    # 1. ingest (module — the pane is a thin wrapper already exercised elsewhere)
    ingest_mod.ingest_clip(CLIP, tmp_path, clip_id="c1")
    cdir = tmp_path / "c1"

    # 2. stage plane + detections (seg01 cached) + GT
    plane = json.loads(PLANE.read_text(encoding="utf-8"))
    manifest_mod.write_json(cdir / "plane.json", plane)
    records, _ = analyze_mod.load_cached_records(SEG01, model_sha=MODEL_SHA)
    (cdir / "detections.json").write_text(json.dumps(records), encoding="utf-8")
    gt = gt_mod.set_ground_truth(plane, frame=85, x_px=532.9, y_px=311.0, judge_in=96.5)
    gt_mod.write_ground_truth(gt, cdir)
    man = manifest_mod.read_manifest(cdir)
    man["plane"]["present"] = True
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

    # 5. GT pane loads a real frame (exercises FrameView + cv2 + PIL headless)
    app.gt.refresh_clips()
    assert app.gt.man is not None
    assert app.gt._n_frames > 0
    assert app.gt.view._src is not None

    # 6. Compare pane builds overlay toggles from the persisted run
    app.compare.load_clip("c1")
    assert len(app.compare.run_vars) == n_rows
