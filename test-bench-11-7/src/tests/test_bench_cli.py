"""M7 (CLI) — subcommands on the shared analyzed-clips schema.

Exercises the torch-free path end-to-end: ``select`` -> ``use-plane`` -> ``set-gt``
-> ``run-suite`` produce the same artifacts the modules do, on a temp analyzed-clips
root.  (``analyze`` needs torch and is covered by ``test_bench_detect``.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from cli_run import main  # noqa: E402
from test_bench import manifest as manifest_mod  # noqa: E402

CLIP = REPO / "Our_dog_diving_videos" / "test-clip-video004_right.mp4"
PLANE = REPO / "end-to-end-app" / "planes" / "plane_test-clip-video004_right.pool.json"
SEG01 = REPO / "end-to-end-app" / "out" / "video004_right-00.00.36.812-00.00.38.328-seg01" / "tracks.json"
MODEL_SHA = "bdb7688775ee39201d4a77ef0041c7c3e9eb9202f7a5df287a11c98e80ef82e4"


@pytest.mark.skipif(not CLIP.is_file() or not PLANE.is_file() or not SEG01.is_file(),
                    reason="parent clip/plane/cache not present on this machine")
def test_cli_end_to_end(tmp_path):
    root = tmp_path / "analyzed-clips"
    setup_id = "test-cli-clipA"

    # 1. select the clip
    assert main(["select", str(CLIP), "--root", str(root), "--id", "clipA"]) == 0
    cdir = root / "clipA"
    assert (cdir / "manifest.json").is_file()
    assert not (cdir / "clip.mp4").exists()          # mapped, not copied
    assert manifest_mod.read_manifest(cdir)["source"]["original_path"] == str(CLIP)

    # 2. stage a plane in the saved-planes folder, then assign it (use-plane)
    plane_dst = manifest_mod.plane_setup_path(setup_id)
    plane_dst.parent.mkdir(parents=True, exist_ok=True)
    plane = json.loads(PLANE.read_text(encoding="utf-8"))
    plane["setup_id"] = setup_id
    manifest_mod.write_json(plane_dst, plane)
    try:
        assert main(["use-plane", "clipA", "--setup-id", setup_id, "--root", str(root)]) == 0
        man = manifest_mod.read_manifest(cdir)
        assert man["plane"]["setup_id"] == setup_id

        # 3. detections (from the cached seg01 tracks)
        tracks = json.loads(SEG01.read_text(encoding="utf-8"))
        entry = next(e for e in tracks["entries"].values() if e.get("model_sha") == MODEL_SHA)
        (cdir / "detections.json").write_text(json.dumps(entry["records"]), encoding="utf-8")

        # 4. set-gt (needs the assigned plane)
        assert main(["set-gt", "clipA", "--frame", "85", "--x", "532.8985",
                     "--y", "311.032", "--judge", "96.5", "--root", str(root)]) == 0
        gt = json.loads((cdir / "ground_truth.json").read_text(encoding="utf-8"))
        assert gt["gt_in"] is not None and gt["judge_present"] is True

        # 5. run-suite
        cfg = tmp_path / "suite.json"
        cfg.write_text(json.dumps({"clips": ["clipA"],
                                   "methods": [{"method": "arc_2d", "n_points": "all",
                                                "selection": "all"}]}), encoding="utf-8")
        assert main(["run-suite", "--config", str(cfg), "--root", str(root)]) == 0

        results = list((root / "results").glob("*.json"))
        assert len(results) == 1
        doc = json.loads(results[0].read_text(encoding="utf-8"))
        assert len(doc["rows"]) == 1
        assert "overall" in doc["averages"]
        run = doc["rows"][0]["runs"][0]
        assert run["score"]["status"] in ("ok", "no_fit")
        assert "pixel_error_px" in run
    finally:
        plane_dst.unlink(missing_ok=True)
