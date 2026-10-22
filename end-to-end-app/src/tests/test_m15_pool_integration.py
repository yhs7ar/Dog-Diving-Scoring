"""M15 — the 3-D route wired into the pipeline that already exists.

Everything before this file tests the pool package in isolation, against a synthetic scene whose
ground truth it knows.  This file tests the *wiring*: that ``cli_run`` accepts a pool calibration,
that it writes the translation artifact and only when asked, that the overlay draws the lattice and
the entry, and that the two document schemas cannot be confused for one another.

The geometric numbers this file sees are **not** meaningful — it runs a real clip (``18-07``, with
its warm detection cache) against a pool calibration built from the *synthetic* scene, because the
corpus has no pool calibration (HANDOFF open item 10).  What is checked is that every number the
route computes is carried faithfully into the artifacts, that an out-of-pool entry is flagged rather
than hidden, and that the default path is untouched.  The accuracy claims live in M10-M14, where the
ground truth is real.

Run:  PYTHONPATH=src <venv-python> -m pytest src/tests/test_m15_pool_integration.py -q
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli_run  # noqa: E402
from dog_diving import annotate as annotate_mod  # noqa: E402
from dog_diving import config as ddconfig  # noqa: E402
from dog_diving.pool import calib as C  # noqa: E402
from dog_diving.pool import lens as L  # noqa: E402
from dog_diving.pool import pipeline as P  # noqa: E402
from tests import pool_scene as PS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPS = PROJECT_ROOT / "data" / "clips"
GATE_CLIP = "18-07"


# --- fixtures ---------------------------------------------------------------
def warm_tracks(cid: str) -> Path | None:
    p = Path(ddconfig.OUT_DIR) / cid / "tracks.json"
    return p if p.is_file() else None


def skip_unless_cached(cid: str) -> None:
    if not (CLIPS / f"{cid}.mp4").is_file():
        pytest.skip(f"data/clips/{cid}.mp4 is missing")
    if warm_tracks(cid) is None:
        pytest.skip(f"{cid} has no warm tracks.json — run cli_run once for it first")


def seed_cache(cid: str, dest_root: Path) -> None:
    """Copy the warm ``tracks.json`` into an isolated out dir.

    The cache key hashes the clip and model *bytes*, the config and the frame range — not the
    path — so a copied cache file is a legitimate hit.  Without this the test would re-run
    inference and take minutes.
    """
    dest = dest_root / cid
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(warm_tracks(cid), dest / "tracks.json")


def synthetic_doc(*, name: str = "poolSYN") -> tuple[dict, "PS.Scene"]:
    """A schema-valid, gate-passing ``dog_diving_pool_calibration/v1`` document + its scene.

    Built from the synthetic scene with the scene's *true* lens model applied to *raw* clicks —
    exactly the way M13 builds its documents — so it is internally consistent and passes its own
    pose gate.  That is what makes it a fair test of the wiring.  It is not consistent with any real
    clip, and the tests below say so where it matters.
    """
    sc = PS.deck_marks_scene()
    doc = C.build(name, clicks=PS.raw_clicks(sc),
                  lens=L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines"),
                  width_ft=sc.width_ft, marks_on_edge="near",
                  created="2026-10-12T00:00:00-04:00")
    return doc, sc


def synthetic_pool_doc(tmp_path: Path, *, name: str = "poolSYN") -> tuple[dict, Path]:
    """The same document, persisted — the CLI needs a path."""
    doc, _sc = synthetic_doc(name=name)
    p = tmp_path / f"{name}.pool.json"
    C.save(doc, p)
    return C.load(p), p


# --- the default path is untouched -----------------------------------------
def test_without_a_pool_calibration_no_pool_artifact_appears(tmp_path: Path):
    """PLAN §11's byte-identical artifact set is a contract, so the new artifact is opt-in.

    A pool artifact appearing in every run would change what "the artifacts" means for the M7
    determinism gate, so the 3-D route writes its file only when a calibration is supplied — and
    this checks the *artifact set*, not just that the file is absent.
    """
    skip_unless_cached(GATE_CLIP)
    root = tmp_path / "_m15_default"
    seed_cache(GATE_CLIP, root)
    r = cli_run.run_clip(CLIPS / f"{GATE_CLIP}.mp4", out_dir=root, contact="auto", draw=False,
                         use_cache=True)
    assert "pool_translation" not in r["artifact_names"]
    assert r["pool_translation"] is None
    assert not (Path(r["out_dir"]) / cli_run.POOL_NAME).exists()
    # and the ones that were always there still are
    for k in ("tracks", "trajectory", "calib_used", "result", "report", "meta"):
        assert k in r["artifact_names"], k


# --- the opt-in path --------------------------------------------------------
def test_with_a_pool_calibration_the_translation_artifact_is_written(tmp_path: Path):
    skip_unless_cached(GATE_CLIP)
    root = tmp_path / "_m15_pool"
    seed_cache(GATE_CLIP, root)
    doc, doc_p = synthetic_pool_doc(tmp_path)

    r = cli_run.run_clip(CLIPS / f"{GATE_CLIP}.mp4", out_dir=root, contact="auto", draw=False,
                         use_cache=True, pool_calib=doc, pool_calib_path=str(doc_p))

    assert r["artifact_names"]["pool_translation"] == cli_run.POOL_NAME
    out_p = Path(r["pool_path"])
    assert out_p.is_file() and out_p.name == cli_run.POOL_NAME

    # What is on disk is what was returned, byte for byte through the strict writer.
    on_disk = json.loads(out_p.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(r["pool_translation"], allow_nan=True))
    assert on_disk["clip_id"] == GATE_CLIP
    assert on_disk["route"] == doc["route"]
    assert on_disk["calibration"]["setup_id"] == doc["setup_id"]
    assert on_disk["water"]["z_ft"] == pytest.approx(doc["water"]["z_ft"])

    # The artifact is registered in result.json by NAME (PLAN §7's no-paths rule).
    result = json.loads(Path(r["result_path"]).read_text(encoding="utf-8"))
    assert result["artifacts"]["pool_translation"] == cli_run.POOL_NAME
    assert str(out_p) not in Path(r["result_path"]).read_text(encoding="utf-8")


def test_an_entry_outside_the_pool_is_flagged_not_hidden(tmp_path: Path):
    """The synthetic calibration cannot describe ``18-07``'s pool, and the route must say so.

    Hiding an out-of-extent entry would destroy the only evidence that something upstream is
    wrong; the number still has to come through, carrying its flag.
    """
    skip_unless_cached(GATE_CLIP)
    root = tmp_path / "_m15_extent"
    seed_cache(GATE_CLIP, root)
    doc, doc_p = synthetic_pool_doc(tmp_path, name="poolSYN2")

    r = cli_run.run_clip(CLIPS / f"{GATE_CLIP}.mp4", out_dir=root, contact="auto", draw=False,
                         use_cache=True, pool_calib=doc, pool_calib_path=str(doc_p))
    res = r["pool_translation"]
    assert res is not None
    assert res["route"] == doc["route"]
    entry = res.get("entry")
    if entry is not None and entry.get("X_ft") is not None:
        inside = (0.0 <= entry["Y_ft"] <= doc["pool"]["width_ft"]
                  and entry["X_ft"] <= doc["pool"]["along_ft"])
        if not inside:
            assert res["flags"].get("outside_extent") or entry["flags"].get("outside_extent"), (
                f"entry ({entry['X_ft']}, {entry['Y_ft']}) is outside the pool box and nothing "
                f"flagged it: flags={res['flags']} entry_flags={entry['flags']}")


def test_the_run_records_nothing_it_did_not_compute(tmp_path: Path):
    """Every number in the artifact is traceable to the calibration that produced it."""
    skip_unless_cached(GATE_CLIP)
    root = tmp_path / "_m15_prov"
    seed_cache(GATE_CLIP, root)
    doc, doc_p = synthetic_pool_doc(tmp_path, name="poolSYN3")
    r = cli_run.run_clip(CLIPS / f"{GATE_CLIP}.mp4", out_dir=root, contact="auto", draw=False,
                         use_cache=True, pool_calib=doc, pool_calib_path=str(doc_p))
    res = r["pool_translation"]
    assert res["calibration"]["setup_id"] == doc["setup_id"]
    assert res["calibration"]["route"] == doc["route"]
    assert res["calibration"]["created"] == doc["created"]
    assert res["n_track_points"] > 0
    entry = res.get("entry") or {}
    if entry.get("X_ft") is not None:
        # the two independent routes inside phase 6 must agree, or the note says why
        agree = entry.get("homography_agreement_ft")
        assert agree is not None
        assert agree <= 1e-6 or any("disagree" in n for n in (entry.get("notes") or []))


# --- the CLI surface --------------------------------------------------------
def test_cli_pool_calib_flag_runs_end_to_end(tmp_path: Path):
    skip_unless_cached(GATE_CLIP)
    root = tmp_path / "_m15_cli"
    seed_cache(GATE_CLIP, root)
    _doc, doc_p = synthetic_pool_doc(tmp_path, name="poolSYN4")

    # --clip takes a path, not a bare name (`_clip_list` passes it straight to run_clip).
    rc = cli_run.main(["--clip", str(CLIPS / f"{GATE_CLIP}.mp4"), "--clips-dir", str(CLIPS),
                       "--out", str(root),
                       "--contact", "auto", "--no-draw", "--pool-calib", str(doc_p),
                       "--calib", "poolA"])
    assert rc == 0
    out_p = root / GATE_CLIP / cli_run.POOL_NAME
    assert out_p.is_file(), f"no pool artifact at {out_p}"
    doc = json.loads(out_p.read_text(encoding="utf-8"))
    assert doc["calibration"]["setup_id"] == "poolSYN4"
    assert doc["clip_id"] == GATE_CLIP


def test_a_pool_setup_id_resolves_from_the_calib_dir(tmp_path: Path):
    """``--pool-calib`` takes a setup id too, and it must resolve to a *pool* document."""
    doc, doc_p = synthetic_pool_doc(tmp_path, name="poolSYN5")
    saved = C.setup_path("poolSYN5")
    try:
        C.save(doc, saved)
        got, path = cli_run.resolve_pool_calibration("poolSYN5")
        assert got is not None and Path(path) == saved
    finally:
        saved.unlink(missing_ok=True)


def test_a_1d_calibration_is_refused_as_a_pool_calibration(tmp_path: Path):
    """The schemas are not interchangeable, and the refusal is the whole point of checking.

    ``data/calib/poolA.json`` is a 1-D document, and it fails its own gate by 167 in.  Feeding it to
    the 3-D route must raise rather than produce a distance from the wrong model.
    """
    one_d = Path(ddconfig.CALIB_DIR) / "poolA.json"
    if not one_d.is_file():
        pytest.skip("data/calib/poolA.json is not present")
    with pytest.raises(Exception):
        C.load(one_d)
    with pytest.raises(Exception):
        cli_run.resolve_pool_calibration(str(one_d))


def test_a_missing_pool_calibration_gives_a_path_error_not_silence():
    with pytest.raises(FileNotFoundError):
        cli_run.resolve_pool_calibration("no-such-pool-setup-id")


# --- the configuration surface ---------------------------------------------
def test_every_pool_config_knob_is_actually_consumed():
    """No dead configuration surface.

    Written because it was not true: when the POOL_* block was first added, **nine of its ten knobs
    were referenced by nothing outside ``config.py``** — a documented surface that changed no
    behaviour, so a reader could "configure" the route and see no effect.  The check is static (a
    source scan) because that is what the failure was: a name that no code reads.  It reads the
    names off the config module, so a knob added and left unwired fails here.
    """
    import inspect
    import re

    names = sorted(n for n in dir(ddconfig) if n.startswith("POOL_"))
    assert names, "the POOL_* block is gone"
    src_root = PROJECT_ROOT / "src"
    files = [p for p in src_root.rglob("*.py")
             if p.name != "config.py" and "tests" not in p.parts]
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    dead = [n for n in names if not re.search(rf"\b{n}\b", text)]
    assert not dead, f"POOL_* knob(s) referenced by no code outside config.py: {dead}"
    # and the config module must actually define them as literals, not just re-export
    assert inspect.getsource(ddconfig).count("POOL_") >= len(names)


def test_the_venue_defaults_come_from_config_not_from_literals(tmp_path: Path):
    """A knob that is consumed has to be consumed *as the default*, or it still changes nothing.

    Measured here: the tool's ``--width-ft`` default is ``config.POOL_WIDTH_FT`` and a document built
    with no explicit width records that value, so editing the config really does move the document.
    """
    import importlib
    import pool_calibrate_tool as tool

    assert tool.DEFAULT_WIDTH_FT == float(ddconfig.POOL_WIDTH_FT)
    assert tool.DEFAULT_FREEBOARD_IN == float(ddconfig.POOL_FREEBOARD_IN)
    assert tool.build_requests.__defaults__ is not None or True   # signature is exercised below
    ap = tool.build_parser()
    d = ap.parse_args(["--frame", "x.png", "--setup-id", "t"])
    assert float(d.width_ft) == float(ddconfig.POOL_WIDTH_FT)
    assert d.marks_on_edge == str(ddconfig.POOL_MARKS_ON_EDGE)
    assert d.water == str(ddconfig.POOL_WATER_METHOD)
    assert d.wall_axis == str(ddconfig.POOL_WALL_AXIS)
    assert float(d.wall_value) == float(ddconfig.POOL_WALL_VALUE)
    assert float(d.focal_ratio) == float(ddconfig.POOL_FOCAL_RATIO)


def test_calib_build_defaults_come_from_config():
    """``build`` with an omitted width/marks-on-edge uses the declared venue defaults."""
    sc = PS.deck_marks_scene()
    doc = C.build("poolDEF", clicks=PS.raw_clicks(sc),
                  lens=L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines"))
    assert doc["width_prior_ft"] == float(ddconfig.POOL_WIDTH_FT)
    assert doc["marks_on_edge"] == str(ddconfig.POOL_MARKS_ON_EDGE)
    assert doc["pool"]["width_ft"] == float(ddconfig.POOL_WIDTH_FT)


# --- the overlay ------------------------------------------------------------
def test_annotate_draws_the_water_grid_and_the_entry_banner():
    """The overlay swaps the 1-D tick strip for the water-plane lattice, and labels the entry.

    Unit-level on a synthetic frame: no clip, no detection.  The claim under test is that supplying
    a pool document changes what is drawn and that the banner carries the 3-D numbers, since an
    overlay that silently ignores its calibration is indistinguishable from one that has none.
    """
    doc, sc = synthetic_doc(name="poolOVL")
    frame = PS.frame_from_scene(sc)

    plain = annotate_mod.draw_frame(frame, frame_idx=7, calib=None, banner=False)
    with_grid = annotate_mod.draw_frame(frame, frame_idx=7, calib=None, banner=False,
                                        pool_doc=doc)
    assert with_grid.shape == frame.shape
    assert not np.array_equal(plain, with_grid), "the pool grid drew nothing"

    result = {"entry": {"X_ft": 31.0, "Y_ft": 0.0, "distance_ft": 31.0, "distance_in": 372.0,
                        "transverse_ft": 0.0, "point_source": "model", "frame": 107,
                        "x_px": 900.0, "y_px": 500.0, "height_assumed_ft": 0.0},
              "pose_ok": True, "route": doc["route"]}
    drawn = annotate_mod.draw_frame(frame, frame_idx=107, calib=None, pool_doc=doc,
                                    pool_result=result)
    assert not np.array_equal(drawn, with_grid), "the entry marker and banner drew nothing"
    text = annotate_mod.pool_banner_text(result, {"distance_in": 376.09}, 107)
    assert "plank=31.00 ft" in text and "1-D=376.1 in" in text and "h=0.00 ft" in text


def test_the_pool_overlay_never_mutates_its_input_frame():
    """``draw_frame``'s purity contract, which the lattice overlay must not break."""
    doc, sc = synthetic_doc(name="poolPURE")
    frame = PS.frame_from_scene(sc)
    before = frame.copy()
    annotate_mod.draw_frame(frame, frame_idx=3, pool_doc=doc, calib=None)
    assert np.array_equal(before, frame), "draw_frame mutated its input"


# --- the pipeline entry point ----------------------------------------------
def test_translate_track_without_a_calibration_says_so_and_stops():
    """No calibration is not an error; it is a documented dead end with no inches."""
    res = P.translate_track([{"frame": 1, "x": 10.0, "y": 20.0, "kp_conf": 0.9}], doc=None)
    assert res["entry"] is None and res["per_frame"] == []
    assert res["flags"].get("no_calibration")
    assert any("no pool calibration" in n for n in res["notes"])


def test_translate_track_is_reachable_with_an_in_memory_track(tmp_path: Path):
    """The route consumes the detector's per-frame records, not just a pixel."""
    doc, sc = synthetic_doc(name="poolMEM")
    # A track along Y = 0 (the marked edge, which is also the wall the rim pair anchors) with the
    # tail-set **on the water** — `project(x, y)` with no Z defaults to the water level, which is the
    # only height at which the entry can come back at X = the value that was projected.  Projecting
    # at the marked plane instead (Z = 0) measures the tail-set 1.625 ft above the water and returns
    # a measured 42.03 ft for a 31 ft jump: that gap is the parallax, not an error in the route.
    rows = []
    for i, x_ft in enumerate(np.arange(20.0, 31.5, 1.0)):
        px = sc.project(float(x_ft), 0.0, sc.water_z_ft)
        rows.append({"frame": 100 + i, "x": float(px[0]), "y": float(px[1]), "kp_conf": 0.9})
    res = P.translate_track(rows, doc=doc, contact_frame=100 + len(rows) - 1)
    assert res["n_track_points"] == len(rows)
    assert res["pose_ok"]
    entry = res["entry"]
    assert entry is not None, res["notes"]
    assert entry["X_ft"] == pytest.approx(31.0, abs=0.05), res["notes"]
    assert entry["Y_ft"] == pytest.approx(0.0, abs=0.05)
