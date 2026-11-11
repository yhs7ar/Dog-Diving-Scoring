"""Tests for the ZED-SDK -> TensorRT engine switch (detect.resolve_runtime_model).

When the ZED X SDK is present (the Jetson rig), a checkpoint's runtime weights file is
its TensorRT ``.engine`` build; everywhere else it is the ``.pt``.  ``resolve_runtime_model``
is the single decision point both ``load_model`` and the GUI route through, and it is
pure — so the whole switch is testable offline (no pyzed, no torch, no GPU, no Tk).

Also covers the on-demand build target (``engine_export_target``) and that
``export_engine`` refuses a missing checkpoint before importing ultralytics.

All tests run offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving.detect import (  # noqa: E402
    HAS_ZED_SDK,
    engine_export_target,
    export_engine,
    resolve_runtime_model,
)


def _checkpoint(tmp_path: Path, name: str = "model.pt", *, make_engine: bool = False) -> Path:
    pt = tmp_path / name
    pt.write_text("dummy")  # existence only; resolve_runtime_model never loads it
    if make_engine:
        pt.with_suffix(".engine").write_text("dummy")
    return pt


def test_has_zed_sdk_is_a_bool() -> None:
    assert isinstance(HAS_ZED_SDK, bool)


def test_zed_present_engine_exists_uses_engine(tmp_path) -> None:
    pt = _checkpoint(tmp_path, make_engine=True)
    path, note = resolve_runtime_model(pt, has_zed_sdk=True)
    assert path == pt.with_suffix(".engine")
    assert note is not None and "TensorRT engine" in note


def test_zed_present_engine_missing_falls_back_to_pt_with_warning(tmp_path) -> None:
    pt = _checkpoint(tmp_path)
    path, note = resolve_runtime_model(pt, has_zed_sdk=True)
    assert path == pt
    assert note is not None and "falling back" in note


def test_no_zed_uses_pt_even_if_engine_exists(tmp_path) -> None:
    pt = _checkpoint(tmp_path, make_engine=True)
    path, note = resolve_runtime_model(pt, has_zed_sdk=False)
    assert path == pt
    assert note is None


def test_no_zed_maps_stray_engine_selection_to_pt(tmp_path) -> None:
    engine = _checkpoint(tmp_path, name="model.engine")
    path, note = resolve_runtime_model(engine, has_zed_sdk=False)
    assert path == engine.with_suffix(".pt")
    assert note is None


def test_accepts_str_path(tmp_path) -> None:
    pt = _checkpoint(tmp_path, make_engine=True)
    path, _ = resolve_runtime_model(str(pt), has_zed_sdk=True)
    assert path == pt.with_suffix(".engine")


# --- on-demand build target --------------------------------------------------
def test_engine_export_target_is_engine_on_zed_with_missing_engine(tmp_path) -> None:
    pt = _checkpoint(tmp_path)
    assert engine_export_target(pt, has_zed_sdk=True) == pt.with_suffix(".engine")


def test_engine_export_target_none_without_zed(tmp_path) -> None:
    pt = _checkpoint(tmp_path)
    assert engine_export_target(pt, has_zed_sdk=False) is None


def test_engine_export_target_none_when_engine_already_built(tmp_path) -> None:
    pt = _checkpoint(tmp_path, make_engine=True)
    assert engine_export_target(pt, has_zed_sdk=True) is None


def test_engine_export_target_none_when_pt_missing(tmp_path) -> None:
    assert engine_export_target(tmp_path / "absent.pt", has_zed_sdk=True) is None


def test_export_engine_missing_checkpoint_raises_before_loading_ultralytics(tmp_path) -> None:
    # The existence check precedes the ultralytics import, so this is fast and needs no GPU.
    with pytest.raises(FileNotFoundError):
        export_engine(tmp_path / "nope.pt")
