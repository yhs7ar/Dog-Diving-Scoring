"""M3b — real YOLO detection on this desktop (venv + vendored checkpoint).

Unlike M3a (cached detections), this exercises the torch path: load the vendored
checkpoint, run ``detect_clip`` over a real clip, and assert records come back in
the parent's fixed shape with at least one tailset detection.  This is the proof
that the install (torch + ultralytics) is *used*, not merely importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]   # test-bench/src
TB = Path(__file__).resolve().parents[2]    # test-bench
REPO = Path(__file__).resolve().parents[3]  # Dog-Diving-Scoring
sys.path.insert(0, str(SRC))

from dog_diving.detect import DetectConfig, detect_clip  # noqa: E402

CLIP = REPO / "Our_dog_diving_videos" / "test-clip-video004_right.mp4"
MODEL = TB / "models" / "yolo26s-pose_dogs_inflated_clahe_720p.pt"

needs_hw = pytest.mark.skipif(
    not CLIP.is_file() or not MODEL.is_file(),
    reason="vendored clip/model not present on this machine",
)


@needs_hw
def test_detect_produces_records():
    cfg = DetectConfig(model_path=MODEL, device="cpu")
    det = detect_clip(CLIP, cfg)
    # one record per frame, in the parent's fixed shape
    for r in det.records:
        assert set(("frame", "nose", "tailset", "box_conf", "track_id", "accepted")) <= set(r)
    assert len(det.records) == det.n_frames_clip
    assert det.n_detections >= 1
    assert det.n_accepted >= 1
    # the checkpoint really is a 2-keypoint dog model (identity gate from the smoke test)
    assert det.model_layout == "dog-2kpt"
