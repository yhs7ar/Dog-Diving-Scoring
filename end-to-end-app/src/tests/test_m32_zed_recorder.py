"""Tests for zed_recorder.py — capture config reconcile + fake-camera seam + governor.

Covers step 2a: the recorder is reconciled to the rig-proven capture standard
(SVGA 1200x720 @ 30 FPS LOSSLESS), exposes a ``ZEDRecorder`` seam with an offline
``FakeZEDRecorder``, and gates capture through a ``ComputeGovernor``.

All tests run offline: no pyzed import, no hardware, no Tk.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving.zed_recorder import (  # noqa: E402
    CAPTURE_COMPRESSION,
    CAPTURE_FPS,
    CAPTURE_HEIGHT,
    CAPTURE_RESOLUTION,
    CAPTURE_WIDTH,
    CLIP_CAM1_SUFFIX,
    CLIP_CAM2_SUFFIX,
    ComputeGovernor,
    DualZEDXLiveBuffer,
    FakeZEDRecorder,
    ZEDRecorder,
    clip_name,
)


# ---------------------------------------------------------------------------
# Config reconcile: the recorder defaults must match the rig-proven standard.
# ---------------------------------------------------------------------------

def test_capture_standard_constants() -> None:
    assert CAPTURE_RESOLUTION == "SVGA"
    assert CAPTURE_FPS == 30
    assert CAPTURE_COMPRESSION == "LOSSLESS"
    assert CAPTURE_WIDTH == 1200
    assert CAPTURE_HEIGHT == 720


def test_recorder_defaults_reconciled() -> None:
    real = DualZEDXLiveBuffer()
    fake = FakeZEDRecorder()
    try:
        assert real.fps == 30
        assert real.chunk_seconds == 10, "step 2b: the plan default is 10 s (was 5)"
        assert real.chunk_frames == real.chunk_seconds * 30
        assert real.DEFAULT_FPS == 30
        assert real.DEFAULT_CHUNK_SECONDS == 10
        assert real.DEFAULT_RESOLUTION == "SVGA"
        assert real.DEFAULT_COMPRESSION == "LOSSLESS"

        assert fake.fps == 30
        assert fake.width == 1200
        assert fake.height == 720
        assert fake.chunk_seconds == 10
        assert fake.DEFAULT_FPS == 30
        assert fake.DEFAULT_CHUNK_SECONDS == 10
    finally:
        real.stop()
        fake.stop()


# ---------------------------------------------------------------------------
# The seam: both backends implement ZEDRecorder.
# ---------------------------------------------------------------------------

def test_both_backends_are_zed_recorders() -> None:
    assert issubclass(DualZEDXLiveBuffer, ZEDRecorder)
    assert issubclass(FakeZEDRecorder, ZEDRecorder)
    assert isinstance(FakeZEDRecorder(), ZEDRecorder)
    assert isinstance(DualZEDXLiveBuffer(), ZEDRecorder)


def test_real_recorder_offline_start_returns_false() -> None:
    # Without pyzed, the hardware backend must refuse to start (not raise).
    rec = DualZEDXLiveBuffer()
    try:
        assert rec.start() is False
        assert rec.running is False
    finally:
        rec.stop()


# ---------------------------------------------------------------------------
# FakeZEDRecorder end-to-end: slot buffer -> save-on-request -> SVO2 + MP4.
# ---------------------------------------------------------------------------

def _run_capture(tmp_path, fps=10, chunk_seconds=0.5) -> FakeZEDRecorder:
    out = tmp_path / "svo"
    clips = tmp_path / "clips"
    rec = FakeZEDRecorder(
        output_dir=out,
        clips_dir=clips,
        fps=fps,
        chunk_seconds=chunk_seconds,
    )
    rec.start()
    return rec


def test_fake_recorder_end_to_end_capture(tmp_path) -> None:
    rec = _run_capture(tmp_path)
    try:
        mp4 = rec.trigger_capture("jump01")
        assert mp4 is not None and mp4.is_file(), "trigger_capture should return the MP4 path"
        assert mp4.parent == tmp_path / "clips"

        # SVO2 metadata stubs for both cameras, with the synthetic capture params.
        meta_paths = [
            tmp_path / "svo" / "jump01_cam1.svo2",
            tmp_path / "svo" / "jump01_cam2.svo2",
        ]
        for p in meta_paths:
            assert p.is_file(), f"missing {p.name}"
            meta = json.loads(p.read_text())
            assert meta["format"] == "svo2"
            assert meta["synthetic"] is True
            assert meta["fps"] == rec.fps
            assert meta["width"] == 1200
            assert meta["height"] == 720
            assert meta["compression"] == "LOSSLESS"
            assert meta["frames"] == rec.chunk_frames

        # MP4 readback: the synthetic frames survived the encode.
        cap = cv2.VideoCapture(str(mp4))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert n == rec.chunk_frames, f"expected {rec.chunk_frames} MP4 frames, got {n}"
    finally:
        rec.stop()


def test_fake_recorder_trigger_without_start_returns_none(tmp_path) -> None:
    rec = FakeZEDRecorder(output_dir=tmp_path / "svo", clips_dir=tmp_path / "clips")
    assert rec.trigger_capture("jump02") is None


# ---------------------------------------------------------------------------
# Step 2b: parameterised duration + true ring (last N seconds across a wrap).
# ---------------------------------------------------------------------------

def test_fake_recorder_parameterised_duration(tmp_path) -> None:
    fps = 10
    seconds = 3
    rec = _run_capture(tmp_path, fps=fps, chunk_seconds=seconds)
    try:
        assert rec.chunk_frames == fps * seconds
        mp4 = rec.trigger_capture("jump_dur")
        assert mp4 is not None and mp4.is_file()

        cap = cv2.VideoCapture(str(mp4))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert n == fps * seconds, f"expected {fps * seconds} MP4 frames, got {n}"

        meta = json.loads((tmp_path / "svo" / "jump_dur_cam1.svo2").read_text())
        assert meta["frames"] == fps * seconds
    finally:
        rec.stop()


def test_fake_recorder_ring_spans_wrap(tmp_path) -> None:
    # chunk_frames = 5: after the ring has wrapped several times, a save must still
    # yield exactly the LAST 5 frames — the pre-ring design returned a shorter window
    # when the request landed just after a slot rollover.
    rec = _run_capture(tmp_path, fps=10, chunk_seconds=0.5)
    try:
        assert rec.chunk_frames == 5
        deadline = time.monotonic() + 2.0
        while rec.frames_written < rec.chunk_frames * 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert rec.frames_written >= rec.chunk_frames * 5, "ring should have wrapped several times"
        assert rec.ring_frames == rec.chunk_frames, "ring should be full (capped) after wrapping"

        mp4 = rec.trigger_capture("jump_wrap")
        assert mp4 is not None and mp4.is_file()
        cap = cv2.VideoCapture(str(mp4))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert n == rec.chunk_frames, f"expected last {rec.chunk_frames} frames, got {n}"

        meta = json.loads((tmp_path / "svo" / "jump_wrap_cam1.svo2").read_text())
        assert meta["frames"] == rec.chunk_frames
    finally:
        rec.stop()


def test_fake_recorder_set_capture_seconds(tmp_path) -> None:
    rec = _run_capture(tmp_path, fps=10, chunk_seconds=0.5)  # chunk_frames = 5
    try:
        assert rec.chunk_frames == 5
        rec.set_capture_seconds(1)
        assert rec.chunk_seconds == 1
        assert rec.chunk_frames == 10  # 1 s * 10 fps

        deadline = time.monotonic() + 2.0
        while rec.ring_frames < rec.chunk_frames and time.monotonic() < deadline:
            time.sleep(0.01)

        mp4 = rec.trigger_capture("jump_resized")
        assert mp4 is not None and mp4.is_file()
        cap = cv2.VideoCapture(str(mp4))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert n == 10, f"resized ring should save 10 frames, got {n}"

        with pytest.raises(ValueError):
            rec.set_capture_seconds(0)
    finally:
        rec.stop()


# ---------------------------------------------------------------------------
# Step 2c: one pipeline clip per camera (cam1 -> "{prefix}_right.mp4",
# cam2 -> "{prefix}_left.mp4"), plus the backward-compatible single-clip return.
# ---------------------------------------------------------------------------

def _mp4_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _mp4_first_frame(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        assert ok, f"failed to read first frame of {path.name}"
        return frame
    finally:
        cap.release()


def test_clip_naming_convention() -> None:
    assert CLIP_CAM1_SUFFIX == "right"
    assert CLIP_CAM2_SUFFIX == "left"
    assert clip_name("jump01", "right") == "jump01_right.mp4"
    assert clip_name("jump01", "left") == "jump01_left.mp4"


def test_fake_recorder_emits_two_named_clips(tmp_path) -> None:
    rec = _run_capture(tmp_path)
    try:
        mp4 = rec.trigger_capture("jump01")
        assert mp4 is not None and mp4.is_file()
        assert mp4.name == "jump01_right.mp4", "primary return must be the cam1 (right) clip"

        right = tmp_path / "clips" / "jump01_right.mp4"
        left = tmp_path / "clips" / "jump01_left.mp4"
        assert right.is_file(), "missing cam1 (right) clip"
        assert left.is_file(), "missing cam2 (left) clip"

        # No bare "{prefix}.mp4" may remain (the naming decision, not a silent one-sided rename).
        assert not (tmp_path / "clips" / "jump01.mp4").exists()

        assert _mp4_frame_count(right) == rec.chunk_frames
        assert _mp4_frame_count(left) == rec.chunk_frames

        # last_saved_mp4 stays the primary (backward compat) and last_saved_clips has the pair.
        assert rec.last_saved_mp4 == right
        assert rec.last_saved_clips == {"right": right, "left": left}

        # SVO2 stubs still written per camera, each with its own frame count.
        c1 = json.loads((tmp_path / "svo" / "jump01_cam1.svo2").read_text())
        c2 = json.loads((tmp_path / "svo" / "jump01_cam2.svo2").read_text())
        assert c1["frames"] == rec.chunk_frames
        assert c2["frames"] == rec.chunk_frames
    finally:
        rec.stop()


def test_fake_recorder_two_clips_have_distinct_content(tmp_path) -> None:
    rec = _run_capture(tmp_path)
    try:
        rec.trigger_capture("jump01")
        right = _mp4_first_frame(tmp_path / "clips" / "jump01_right.mp4")
        left = _mp4_first_frame(tmp_path / "clips" / "jump01_left.mp4")
        # The two clips come from two independent frame streams, so their frames differ
        # (same geometry, same size — but the marker colour distinguishes cam1 from cam2).
        assert right.shape == left.shape
        assert not (right == left).all(), "cam2 clip must not be a copy of cam1"

        # Deterministic check on the synthetic streams (no codec loss): cam1 and cam2 carry
        # the SAME marker position but DIFFERENT colours, so the only differing pixels are
        # the 40x40 marker block (40*40*3 channel values).
        f1 = rec._ring[0]
        f2 = rec._ring2[0]
        assert (f1 != f2).any(), "cam2 ring must differ from cam1 (distinct marker colour)"
        diff_count = int((f1 != f2).sum())
        assert diff_count == 40 * 40 * 3, f"expected only the 40x40 marker to differ, got {diff_count}"
    finally:
        rec.stop()


def test_fake_recorder_ring2_fills_in_lockstep(tmp_path) -> None:
    rec = _run_capture(tmp_path)
    try:
        deadline = time.monotonic() + 2.0
        while (rec.ring_frames < rec.chunk_frames or rec.ring2_frames < rec.chunk_frames) \
                and time.monotonic() < deadline:
            time.sleep(0.01)
        assert rec.ring_frames == rec.chunk_frames
        assert rec.ring2_frames == rec.chunk_frames, "cam2 ring must fill in lock step with cam1"
    finally:
        rec.stop()


def test_fake_recorder_set_capture_seconds_resizes_both_clips(tmp_path) -> None:
    fps = 10
    rec = _run_capture(tmp_path, fps=fps, chunk_seconds=0.5)  # chunk_frames = 5
    try:
        rec.set_capture_seconds(1)  # -> 10 frames
        deadline = time.monotonic() + 2.0
        while rec.ring_frames < rec.chunk_frames and time.monotonic() < deadline:
            time.sleep(0.01)

        mp4 = rec.trigger_capture("jump_resized")
        assert mp4 is not None
        right = tmp_path / "clips" / "jump_resized_right.mp4"
        left = tmp_path / "clips" / "jump_resized_left.mp4"
        assert right.is_file() and left.is_file()
        assert _mp4_frame_count(right) == 10
        assert _mp4_frame_count(left) == 10
        assert rec.last_saved_clips["right"] == right
        assert rec.last_saved_clips["left"] == left
    finally:
        rec.stop()


# ---------------------------------------------------------------------------
# ComputeGovernor: suspension (ref-counted) + graceful psutil-less overload.
# ---------------------------------------------------------------------------

def test_governor_default_allows_capture() -> None:
    g = ComputeGovernor()
    assert g.suspended is False
    assert g.allow_capture() is True


def test_governor_suspend_resume() -> None:
    g = ComputeGovernor()
    g.suspend("detect")
    assert g.suspended is True
    assert g.allow_capture() is False
    g.resume("detect")
    assert g.suspended is False
    assert g.allow_capture() is True


def test_governor_suspend_is_ref_counted() -> None:
    g = ComputeGovernor()
    g.suspend("detect")
    g.suspend("measure")
    g.resume("detect")
    assert g.suspended is True, "one of two holds remains; must still be suspended"
    g.resume("measure")
    assert g.suspended is False
    # resume past zero must not go negative
    g.resume("measure")
    assert g.suspended is False


def test_governor_overload_degrades_without_psutil(monkeypatch) -> None:
    g = ComputeGovernor(max_cpu_percent=10.0)
    monkeypatch.setattr(g, "_psutil", None)
    assert g.cpu_percent() == -1.0
    assert g.overloaded() is False
    assert g.allow_capture() is True


def test_governor_overload_blocks_capture(monkeypatch) -> None:
    g = ComputeGovernor(max_cpu_percent=10.0)

    class _FakePsutil:
        def cpu_percent(self, interval=None):
            return 99.0

    monkeypatch.setattr(g, "_psutil", _FakePsutil())
    assert g.cpu_percent() == 99.0
    assert g.overloaded() is True
    assert g.allow_capture() is False


# ---------------------------------------------------------------------------
# The fake recorder honours governor backpressure (yields while suspended).
# ---------------------------------------------------------------------------

def test_fake_recorder_respects_governor_suspension(tmp_path) -> None:
    governor = ComputeGovernor()
    governor.suspend("scoring")  # blocked before capture even starts
    rec = FakeZEDRecorder(
        output_dir=tmp_path / "svo",
        clips_dir=tmp_path / "clips",
        fps=30,
        chunk_seconds=0.1,
        governor=governor,
    )
    try:
        rec.start()
        assert rec._started.wait(timeout=2.0), "worker thread never started"
        time.sleep(0.3)
        assert rec.frames_written == 0, "suspended governor must block frame writes"

        governor.resume("scoring")
        deadline = time.monotonic() + 2.0
        while rec.frames_written == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert rec.frames_written > 0, "resumed governor must let frames flow"
    finally:
        rec.stop()
