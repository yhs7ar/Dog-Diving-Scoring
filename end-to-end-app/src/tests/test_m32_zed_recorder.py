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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving.zed_recorder import (  # noqa: E402
    CAPTURE_COMPRESSION,
    CAPTURE_FPS,
    CAPTURE_HEIGHT,
    CAPTURE_RESOLUTION,
    CAPTURE_WIDTH,
    ComputeGovernor,
    DualZEDXLiveBuffer,
    FakeZEDRecorder,
    ZEDRecorder,
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
        assert real.chunk_frames == real.chunk_seconds * 30
        assert real.DEFAULT_FPS == 30
        assert real.DEFAULT_RESOLUTION == "SVGA"
        assert real.DEFAULT_COMPRESSION == "LOSSLESS"

        assert fake.fps == 30
        assert fake.width == 1200
        assert fake.height == 720
        assert fake.DEFAULT_FPS == 30
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
