"""M28 gate: the model registry — every pose checkpoint declared once, and the n/s scale choice.

Operator request (2026-10-22): *"the user can select whether detection is using YOLO26s weights or
YOLO26n weights alongside the existing selection of the model weights."*  The mechanism is
``config.MODEL_REGISTRY``: one ``ModelSpec`` per checkpoint, carrying its backbone **scale**, the
keypoint **layout** it must resolve to, and how it measured on the 46-frame labelled set.  The GUI's
backbone selector, ``config.MODEL_PATH``, ``eval_frames --scale`` and ``measure_pose_models.py`` all
read it, so there is no second list to drift.

What this asserts
  1. Registry invariants: unique keys and file names, declared scales, and a layout key that
     ``pose_layout`` actually knows (a typo here would be a silently unmapped checkpoint).
  2. Exactly the dog family is scale-selectable, and **one** checkpoint is the default per scale:
     ``n`` is the model of record, ``s`` is the s-scale dog fine-tune — never the stock COCO person
     checkpoint, which shares the ``yolo26s`` name but has no tailset keypoint.
  3. Path resolution: a declared file maps to its spec, an undeclared one to ``None`` (the GUI then
     says "custom"), and bare names resolve inside ``models/``.
  4. Environment selection: ``DD_MODEL_SCALE=s`` moves ``MODEL_PATH`` onto the s checkpoint,
     ``DD_MODEL`` wins over it, and an unusable scale is recorded in ``MODEL_SELECTION_NOTE``
     instead of crashing the import that every entry point performs.
  5. ``measure_pose_models.py`` no longer declares its own list — it reads the registry.
  6. The declared files are the ones on disk (skipped individually when a checkpoint is not
     installed, since ``models/*.pt`` beyond v5 are not in git).

Run:  <venv-python> -m pytest src/tests/test_m28_model_registry.py -s
"""

from __future__ import annotations

import importlib
import gc
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import config as C  # noqa: E402
from dog_diving import pose_layout as PL  # noqa: E402

#: The two dog-family file names the project ships, pinned by name because the registry is what
#: ``MODEL_PATH`` and the GUI resolve through — a rename without updating the spec would leave the
#: backbone selector pointing at nothing.
N_SCALE_MODEL = "yolo26n-pose_dogs_v5.pt"
S_SCALE_MODEL = "yolo26s-pose_dogs_clahe.pt"
COCO_PERSON_MODEL = "yolo26s-pose.pt"


# --- 1. registry invariants --------------------------------------------------
def test_registry_invariants() -> None:
    keys = [s.key for s in C.MODEL_REGISTRY]
    names = [s.filename for s in C.MODEL_REGISTRY]
    assert len(keys) == len(set(keys)), f"duplicate model keys: {keys}"
    assert len(names) == len(set(names)), f"duplicate checkpoint file names: {names}"

    known_layouts = {lay.key for lay in PL.LAYOUTS}
    for spec in C.MODEL_REGISTRY:
        assert spec.scale in C.SCALES, f"{spec.key}: scale {spec.scale!r} not in {C.SCALES}"
        assert spec.family in (C.FAMILY_DOG, C.FAMILY_PERSON), spec.family
        assert spec.layout in known_layouts, (
            f"{spec.key}: layout {spec.layout!r} is not a PoseLayout — known: {sorted(known_layouts)}")
        # the file name carries its backbone, so a human reading models/ can tell them apart
        assert spec.filename.startswith(f"yolo26{spec.scale}-"), (
            f"{spec.key}: file name {spec.filename!r} does not carry its scale {spec.scale!r}")
        print(f"  {spec.describe()}  measured={spec.measured_line() or '(none)'}")


def test_every_declared_checkpoint_is_on_disk_or_skipped() -> None:
    """Declared but absent is a *broken install*, not a test failure: say which file is missing."""
    missing = [s.filename for s in C.MODEL_REGISTRY if not s.is_file()]
    if missing:
        pytest.skip(f"not installed under models/: {missing}")
    for spec in C.MODEL_REGISTRY:
        assert spec.path.parent == C.MODELS_DIR, spec.path


# --- 2. the scale choice -----------------------------------------------------
def test_one_default_per_scale_and_the_s_default_is_the_dog_finetune() -> None:
    n = C.default_for_scale("n")
    s = C.default_for_scale("s")
    assert n.filename == N_SCALE_MODEL, n.filename
    assert s.filename == S_SCALE_MODEL, s.filename
    assert n.scale == "n" and s.scale == "s"
    # the whole point: "s" must not quietly resolve to the 17-keypoint person model
    assert s.family == C.FAMILY_DOG and s.layout == "dog-2kpt"
    assert s.filename != COCO_PERSON_MODEL
    # uppercase / stray whitespace from a shell should still resolve
    assert C.default_for_scale(" S ").filename == S_SCALE_MODEL
    print(f"  scale n -> {n.filename} ({n.role})")
    print(f"  scale s -> {s.filename} ({s.role})  measured {s.measured_line()}")


def test_scale_selectable_set_is_dog_family_only() -> None:
    selectable = {spec.filename for spec in C.dog_specs()}
    assert COCO_PERSON_MODEL not in selectable, (
        "the stock COCO person checkpoint must not be reachable by a scale choice — it has no "
        "tailset keypoint")
    for spec in C.dog_specs():
        assert spec.layout == "dog-2kpt", spec.describe()


def test_unknown_scale_is_refused_with_the_known_ones() -> None:
    with pytest.raises(ValueError) as exc:
        C.default_for_scale("m")
    msg = str(exc.value)
    assert "n" in msg and "s" in msg, msg
    print(f"  refused: {msg}")


# --- 3. path resolution ------------------------------------------------------
def test_spec_for_path_round_trips_and_undeclared_files_are_none() -> None:
    for spec in C.MODEL_REGISTRY:
        assert C.spec_for_path(spec.path) is spec
        assert C.spec_for_path(str(spec.path)) is spec
        assert C.spec_for_path(spec.filename) is spec        # bare name, as DD_MODEL takes it
    assert C.spec_for_path(C.MODELS_DIR / "nope.pt") is None
    assert C.spec_for_path(PROJECT_ROOT / "models" / "best.pt") is None, (
        "best.pt is the vendored fallback, deliberately not a registry entry — the GUI labels it "
        "'custom' rather than claiming a measured role it does not have")
    print("  undeclared checkpoints resolve to None (GUI: 'custom (file choice)')")


def test_resolve_model_path_keeps_dd_model_contract() -> None:
    assert C.resolve_model_path(N_SCALE_MODEL) == C.MODELS_DIR / N_SCALE_MODEL
    abs_path = C.MODELS_DIR / S_SCALE_MODEL
    assert C.resolve_model_path(abs_path) == abs_path


# --- 4. environment selection ------------------------------------------------
def _reload_config(monkeypatch, *, scale: str | None = None, model: str | None = None):
    """Re-import ``config`` under a controlled environment.  Caller must ``finally`` reload."""
    for var, val in (("DD_MODEL_SCALE", scale), ("DD_MODEL", model)):
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, val)
    return importlib.reload(C)


def test_env_scale_selects_that_backbone_and_dd_model_wins(monkeypatch) -> None:
    try:
        cfg = _reload_config(monkeypatch, scale="s")
        assert cfg.MODEL_PATH.name == S_SCALE_MODEL, cfg.MODEL_PATH
        assert cfg.MODEL_SCALE == "s"
        assert cfg.MODEL_SPEC is not None and cfg.MODEL_SPEC.key == "dogs-s-clahe"
        assert cfg.MODEL_SELECTION_NOTE is None
        print(f"  DD_MODEL_SCALE=s -> {cfg.MODEL_PATH.name} ({cfg.MODEL_SCALE}-scale)")

        cfg = _reload_config(monkeypatch, scale="s", model=N_SCALE_MODEL)
        assert cfg.MODEL_PATH.name == N_SCALE_MODEL, "DD_MODEL must beat DD_MODEL_SCALE"
        assert cfg.MODEL_SCALE == "n"
        print(f"  DD_MODEL={N_SCALE_MODEL} beats DD_MODEL_SCALE=s -> {cfg.MODEL_PATH.name}")

        cfg = _reload_config(monkeypatch, scale="m")
        assert cfg.MODEL_PATH.name == N_SCALE_MODEL, "a bad scale keeps the default, never crashes"
        assert cfg.MODEL_SELECTION_NOTE and "m" in cfg.MODEL_SELECTION_NOTE
        print(f"  DD_MODEL_SCALE=m -> kept {cfg.MODEL_PATH.name}; note: {cfg.MODEL_SELECTION_NOTE}")
    finally:
        monkeypatch.delenv("DD_MODEL", raising=False)
        monkeypatch.delenv("DD_MODEL_SCALE", raising=False)
        importlib.reload(C)
    assert C.MODEL_PATH.name == N_SCALE_MODEL or C.MODEL_PATH.is_file()


# --- 5. no second list -------------------------------------------------------
def test_measure_script_is_registry_driven() -> None:
    src = (PROJECT_ROOT / "measure_pose_models.py").read_text(encoding="utf-8")
    assert "MODEL_REGISTRY" in src, "measure_pose_models.py must read config.MODEL_REGISTRY"
    assert "registry_models" in src
    for stale in (N_SCALE_MODEL, S_SCALE_MODEL, COCO_PERSON_MODEL):
        assert stale not in src, (
            f"measure_pose_models.py still names {stale} by hand — the registry is the one "
            f"declaration of a checkpoint")
    print("  measure_pose_models.py declares no checkpoints of its own")


def test_eval_frames_accepts_a_scale() -> None:
    src = (PROJECT_ROOT / "src" / "eval_frames.py").read_text(encoding="utf-8")
    assert "--scale" in src and "default_for_scale" in src


# --- 6. the GUI control ------------------------------------------------------
def _first_clip() -> Path | None:
    clips = sorted((PROJECT_ROOT / "data" / "clips").glob("*.mp4"))
    return clips[0] if clips else None


def test_gui_backbone_selector_switches_n_and_s(tmp_path) -> None:
    """The operator-facing half: the backbone box and the pose-model file agree, both ways."""
    clip = _first_clip()
    if clip is None:
        pytest.skip("no clip under data/clips")
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
    except Exception as exc:  # noqa: BLE001 - no display in this environment
        pytest.skip(f"no Tk display: {type(exc).__name__}: {exc}")

    import app_gui

    app = None
    try:
        app = app_gui.App(root, clip=clip)
        root.update_idletasks()

        n_spec = C.default_for_scale("n")
        s_spec = C.default_for_scale("s")
        assert app._scale_choice_for(n_spec) in app._scale_choices()
        assert app._scale_choice_for(s_spec) in app._scale_choices()
        print("  choices:", app._scale_choices())

        # picking s moves the file selection onto the s checkpoint and says so
        app.scale_var.set(app._scale_choice_for(s_spec))
        app.on_scale_pick()
        assert app.model_var.get() == str(s_spec.path), app.model_var.get()
        assert app.scale_status.cget("text").startswith("s-scale")
        assert "registry" in app.log_text.get("1.0", "end")
        print(f"  backbone s -> {Path(app.model_var.get()).name} "
              f"[{app.scale_status.cget('text')}]")

        app.scale_var.set(app._scale_choice_for(n_spec))
        app.on_scale_pick()
        assert app.model_var.get() == str(n_spec.path)
        assert app.scale_status.cget("text").startswith("n-scale")
        print(f"  backbone n -> {Path(app.model_var.get()).name} "
              f"[{app.scale_status.cget('text')}]")

        # a file pick re-answers the scale question rather than leaving it stale
        app.model_var.set(str(s_spec.path))
        app.on_model_pick()
        assert app.scale_var.get() == app._scale_choice_for(s_spec)

        # an undeclared checkpoint is labelled custom — never parsed from its file name
        custom = tmp_path / "someone_elses_yolo26s.pt"
        custom.write_bytes(b"not really a checkpoint; on_model_pick only stats the file")
        app.model_var.set(str(custom))
        app.on_model_pick()
        assert app.scale_var.get() == app_gui.SCALE_CUSTOM_LABEL
        assert app.scale_status.cget("text").startswith("custom")
        assert "undeclared" in app.log_text.get("1.0", "end")
        print("  undeclared file ->", app.scale_var.get())
    finally:
        app = None
        gc.collect()
        root.destroy()
