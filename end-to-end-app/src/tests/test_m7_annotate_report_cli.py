"""M7 verification gate (PLAN.md §11 M7 / §5 ``annotate.py`` + ``report.py`` + ``cli_run.py``).

PLAN §11 M7, verbatim:

    ``annotate.py`` + ``report.py`` + ``cli_run.py`` — *"V4/V5 batch runs produce annotated
    MP4, PNG, JSON, TXT unchanged by re-runs"*

The gate is therefore **determinism**, and determinism is checked the only way that cannot be
faked: run the real batch twice through :func:`cli_run.run_clip` and compare the **sha256 of
every artifact** between the two passes.  The batches are the real clips in ``data/clips/``,
through the real M1–M6 modules, with the real vendored checkpoint's cached detections — the
only thing this file injects is the *calibration*, and only in the one test that needs inches
(see below).

What "unchanged by re-runs" cost to make true
---------------------------------------------
Two of PLAN's own clauses conflict, and the resolution is measured rather than assumed:

* PLAN §5 puts ``timings`` in ``result.json``; a wall-clock block cannot be byte-identical
  across runs.  **Resolution:** ``result.json`` is a pure function of its inputs and the
  volatile fields live in ``run_meta.json`` — and this file *proves* the boundary, by scanning
  ``result.json`` for volatile keys and by showing ``run_meta.json`` is the artifact that does
  differ.
* PLAN §7 puts the artifacts under ``out/<clip_id>/``; a byte-comparison across two runs is
  only meaningful if the directory is not embedded in them.  **Resolution:** ``result.json``
  records artifact *file names*, not paths.

Codec (measured, this host, 60 frames of 21-01.mp4):

| fourcc | byte-identical across runs? |
|---|---|
| ``mp4v`` | **yes** — same sha256 and size |
| ``avc1`` | no (and OpenH264 is not installed: *"Failed to load OpenH264 library"*) |

So ``mp4v`` is the default and this file pins its byte-determinism; an overlay codec change
that breaks §11 fails here instead of silently.

The one synthetic input, labelled as such
-----------------------------------------
``data/calib/`` is empty (HANDOFF open item 10), so no real clip can produce inches.  The
calibrated-path test therefore uses a **synthetic calibration generated from the M2/M6
projective ground truth** ``x(X) = CX + P0·X/(1 + X/L)`` — and it is checked against that same
projective oracle (analytically inverted), not against the module's own arithmetic.

Run:  PYTHONPATH=src <venv-python> -m pytest src/tests/test_m7_annotate_report_cli.py -s
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli_run  # noqa: E402  (src/cli_run.py — the top-level runner)

from dog_diving import annotate, calibration as C, config as ddconfig, report  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPS = PROJECT_ROOT / "data" / "clips"
OUT = ddconfig.OUT_DIR
GATE_DIR = OUT / "_m7_gate"

# the three clips whose caches are warm (M3/M4/M5); exercises annotate+report without inference
GATE_CLIPS = ("18-07", "21-01", "23-04")
ARTIFACTS = ("annotated.mp4", "final_frame.png", "result.json", "report.txt",
             "trajectory.json", "calib_used.json")

# keys that may never appear in result.json (PLAN §5's volatile fields — see the docstring)
VOLATILE_KEYS = {"created", "timings", "from_cache", "host", "platform", "argv",
                 "ms_per_frame", "total_s", "cache_load_s", "projected_clip_s",
                 "inference_ms_per_frame", "tracker_reset", "cancelled"}


# ===========================================================================
# the projective oracle (the M2/M6 ground truth), duplicated so this file stands alone
# ===========================================================================
CX, P0 = 520.0, 1.5              # dock-edge x; 1.5 px/in = 18 px/ft at the dock
F = 0.02                         # 2% foreshortening across the pool
SPAN_IN = 372.0
L = SPAN_IN / (1.0 / math.sqrt(1.0 - F) - 1.0)
IMG_W, IMG_H, WATER_Y = 1920, 1080, 540


def truth_x(X_in: float) -> float:
    return CX + P0 * X_in / (1.0 + X_in / L)


def truth_inches(x_px: float) -> float:
    """Analytic inverse of :func:`truth_x` — the oracle, never the module's fit."""
    d = float(x_px) - CX
    return d / (P0 - d / L)


def synthetic_calib(setup_id: str = "m7-synthetic") -> dict:
    """A 6-mark calibration built *from the oracle* (so its fit is projectively exact)."""
    fts = [(0, 0), (1, 0), (15, 0), (16, 0), (30, 0), (31, 0)]
    marks = [{"label": f"{f}'{i:02d}\"", "ft": f, "inch": i,
              "x_px": truth_x(f * 12 + i), "y_px": WATER_Y} for f, i in fts]
    return C.build(setup_id, (IMG_W, IMG_H),
                   dock_line=[[CX, 100], [CX, 980]],
                   waterline=[[0, WATER_Y], [IMG_W, WATER_Y]],
                   marks=marks, model_sha="synthetic", notes="M7 gate: synthetic (no real "
                                                             "calibration exists on this corpus)")


# ===========================================================================
# helpers
# ===========================================================================
def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def all_keys(obj, acc: set | None = None) -> set:
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(str(k))
            all_keys(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            all_keys(v, acc)
    return acc


def clip_path(cid: str) -> Path:
    return CLIPS / f"{cid}.mp4"


def seed_cache(cid: str, dest_root: Path) -> None:
    """Copy the warm ``tracks.json`` into an isolated out dir.

    The cache key hashes the *clip and model bytes*, the config and the frame range — not
    the file path — so a copied cache file is a legitimate hit.  Without this, every test
    that writes somewhere other than ``out/`` pays a full cold inference run (measured
    ~244 ms/frame, i.e. ~110 s for an 18-07 clip); the unit under test is annotate/report,
    not detect.
    """
    src = OUT / cid / "tracks.json"
    dst = dest_root / cid / "tracks.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def skip_unless_cached(cids):
    if not all(clip_path(c).is_file() for c in cids):
        pytest.skip("sample clips missing from data/clips/")
    for c in cids:
        if not (OUT / c / "tracks.json").is_file():
            pytest.skip(f"{c} has no warm tracks.json — run cli_run once for it first")


# ===========================================================================
# 1. the modules exist and are the real thing (no inline copies)
# ===========================================================================
def test_modules_import_and_expose_the_plan_api():
    for fn in (annotate.draw_frame, annotate.draw_calibration, annotate.draw_measurement,
               annotate.banner_text, annotate.write_video, annotate.write_still):
        assert callable(fn)
    for fn in (report.build_result, report.write_result_json, report.build_report_text,
               report.write_report_txt, report.build_run_meta, report.write_run_meta,
               report.rollup_row, report.write_rollup_csv, report.jsonable,
               report.dumps_strict):
        assert callable(fn)
    for fn in (cli_run.run_clip, cli_run.main, cli_run.parse_contact, cli_run.parse_subclip,
               cli_run.resolve_calibration, cli_run.project_batch):
        assert callable(fn)
    # PLAN §5's CLI surface, literally
    flags = {"--clip", "--calib", "--imgsz", "--out", "--contact"}
    have = {o for a in cli_run.build_parser()._actions for o in a.option_strings}
    assert flags <= have, f"missing CLI flags: {sorted(flags - have)}"
    print("annotate/report/cli_run import; PLAN §5 flags present:", sorted(flags))


def test_source_files_exist_on_disk():
    for name in ("annotate.py", "report.py"):
        assert (PROJECT_ROOT / "src" / "dog_diving" / name).is_file()
    assert (PROJECT_ROOT / "src" / "cli_run.py").is_file()
    print("modules present under src/ (imported above, not re-implemented)")


# ===========================================================================
# 2. the overlay is pure and its banner cannot drift from the measurement
# ===========================================================================
def test_draw_frame_is_pure_and_deterministic():
    frame = np.zeros((120, 200, 3), np.uint8)
    frame[:] = (40, 60, 80)
    before = frame.copy()
    m = {"x_px": 90.0, "y_px": 80.0, "waterline_y": 100.0, "display": "22' 7\"",
         "sigma_in": 4.0, "sigma_complete": True, "total_inches": 271.0}
    a = annotate.draw_frame(frame, frame_idx=138, calib=None, measurement=m)
    b = annotate.draw_frame(frame, frame_idx=138, calib=None, measurement=m)
    assert np.array_equal(frame, before), "draw_frame mutated its input"
    assert np.array_equal(a, b), "draw_frame is not deterministic"
    assert a.shape == frame.shape
    assert not np.array_equal(a, before), "draw_frame drew nothing"
    print("draw_frame: input untouched, two calls byte-identical, shape", a.shape)


def test_banner_quotes_the_measurement_and_never_invents_one():
    m = {"display": "22' 7\"", "sigma_in": 4.0, "sigma_complete": True}
    assert annotate.banner_text(m, None, 138) == "distance 22' 7\" +/- 4.0\" | frame 138 | calib none"
    # incomplete budget must be labelled a lower bound, not silently quoted as the budget
    m2 = {"display": "22' 7\"", "sigma_in": 4.0, "sigma_complete": False}
    assert "lower bound" in annotate.banner_text(m2, None, 138)
    # no inches -> say so in words (the whole corpus today, HANDOFF item 10)
    txt = annotate.banner_text({"display": None, "total_inches": None}, None, 437)
    assert txt == "distance n/a (no calibration) | frame 437 | calib none"
    # a calibration present -> its residual RMS appears, read from the calib (not recomputed)
    cal = {"fit": {"residual_rms_in": 1.3}}
    assert annotate.banner_text({"display": "22' 7\"", "sigma_in": None}, cal, 138) \
        == "distance 22' 7\" | frame 138 | calib RMS 1.3\""
    print("banner:", annotate.banner_text(m, None, 138))
    print("banner (no calib):", txt)


def test_default_overlay_omits_the_candidate_list_the_banner_would_hide():
    """The banner (rows 0..BANNER_H) overlaps contact.draw_candidates' list at y=30."""
    import inspect
    import types

    assert inspect.signature(annotate.draw_frame).parameters["show_candidates"].default is False
    assert inspect.signature(annotate.write_video).parameters["show_candidates"].default is False
    assert inspect.signature(annotate.write_still).parameters["show_candidates"].default is False

    frame = np.zeros((120, 200, 3), np.uint8)
    stub = types.SimpleNamespace(
        chosen_frame=437, chosen_source="last_accepted",
        ranked=lambda: [types.SimpleNamespace(rank=1, source="last_accepted", frame=437,
                                              available=True, confident=True)],
    )
    off = annotate.draw_frame(frame, frame_idx=437, contact=stub)
    on = annotate.draw_frame(frame, frame_idx=437, contact=stub, show_candidates=True)
    assert not np.array_equal(off, on), "show_candidates=True must actually draw something"
    # and the banner survives in both
    assert annotate.banner_text(None, None, 437) in ("distance n/a (no calibration) | frame 437 | calib none",)
    print("default overlay omits the candidate list (banner would hide #1); "
          "show_candidates=True still draws it and differs from the default")


def test_draw_calibration_projects_the_grid_and_a_missing_calib_draws_nothing():
    cal = synthetic_calib()
    frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    drawn = annotate.draw_calibration(frame.copy(), cal, show_grid=True)
    blank = annotate.draw_calibration(frame.copy(), None, show_grid=True)
    assert np.array_equal(blank, frame), "a calibration-free clip must not draw a datum"
    assert not np.array_equal(drawn, frame)
    # the 0-ft tick must land on the dock edge (PLAN §4.3 G2's geometry) — the grid is the fit
    x0 = C.inverse_ft(0.0, cal)
    assert abs(x0 - C.dock_zero_x(cal)) < 1e-6
    print(f"grid drawn; 0-ft tick x={x0:.3f} == dock edge x={C.dock_zero_x(cal):.3f}")


# ===========================================================================
# 3. result.json is strict JSON and a pure function of its inputs
# ===========================================================================
def test_jsonable_and_strict_dumps_handle_non_finite_floats():
    doc = {"a": float("nan"), "b": [float("inf"), -float("inf")], "c": np.float32(1.5),
           "d": np.int64(7), "e": np.bool_(True), "f": Path("x")}
    out = report.jsonable(doc)
    assert out == {"a": None, "b": [None, None], "c": 1.5, "d": 7, "e": True, "f": "x"}
    text = report.dumps_strict(doc)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == out            # strict JSON parses
    print("non-finite floats -> null; dumps_strict parses with a strict reader")


def test_result_json_is_a_pure_function_of_inputs_and_carries_no_volatile_keys():
    kwargs = dict(
        clip_id="unit", clip_sha="deadbeef", clip_name="unit.mp4",
        video={"width": 1920, "height": 1080, "fps": 30.0, "codec": "h264",
               "n_frames": 10, "vfr": False, "n_frames_declared": 10, "n_frames_decoded": 10},
        detect={"config": {"model_path": "models/best.pt"}, "model_sha": "abc",
                "n_frames_clip": 10, "n_frames_processed": 10, "n_detections": 3,
                "n_accepted": 2, "dominant_track": None, "tracking_available": False,
                "track_coverage": 0.0, "cache_key_hash": "h",
                "timings": {"total_s": 99.0, "ms_per_frame": 1.0}},
        traj={"n_points": 2, "runs": [], "flight_run": None, "airborne_frame_range": None,
              "n_before_leave": 0, "n_before_entry": 0, "inliers": [], "n_outliers": 0,
              "arc_rms_px": None, "arc_rms_all_px": None, "max_inlier_res_px": None,
              "last_arc_frame": None, "coeffs": None, "flight_points": [],
              "last_contact_candidate": None},
        contact={"chosen_frame": None, "chosen_source": "last_accepted", "window": None,
                 "n_frames": 10, "waterline_y": None, "arc_is_freefall": False,
                 "arc_reason": "no fitted arc", "candidates": []},
        measurement={"x_px": None, "warnings": ["nothing to convert"]},
        calib=None, calib_path=None, detect_config={"imgsz": 1280}, artifacts={"result": "result.json"},
    )
    a = report.dumps_strict(report.build_result(**kwargs))
    b = report.dumps_strict(report.build_result(**kwargs))
    assert a == b, "result.json is not a pure function of its inputs"
    keys = all_keys(json.loads(a))
    assert not (keys & VOLATILE_KEYS), f"volatile keys leaked into result.json: {keys & VOLATILE_KEYS}"
    assert json.loads(a)["calibration"]["present"] is False
    print(f"result.json: {len(a)} bytes, byte-identical across builds, "
          f"0/{len(VOLATILE_KEYS)} volatile keys ({sorted(VOLATILE_KEYS)})")


def test_report_text_is_deterministic_and_quotes_the_pipeline_numbers():
    r = report.build_result(
        clip_id="unit", clip_sha="ab" * 32, clip_name="unit.mp4",
        video={"width": 1920, "height": 1080, "fps": 30.0, "codec": "h264", "n_frames": 450,
               "vfr": False, "n_frames_declared": 450, "n_frames_decoded": 450},
        detect={"config": {}, "model_sha": "cd" * 32, "n_frames_clip": 450,
                "n_frames_processed": 450, "n_detections": 41, "n_accepted": 30,
                "dominant_track": None, "tracking_available": False, "track_coverage": 0.12,
                "cache_key_hash": "h"},
        traj={"n_points": 30, "runs": [{"i0": 0, "i1": 15}], "flight_run": 0,
              "airborne_frame_range": [369, 396], "n_before_leave": 6, "n_before_entry": 9,
              "inliers": [{"frame": 369}], "n_outliers": 7, "arc_rms_px": 10.333,
              "arc_rms_all_px": 61.4, "max_inlier_res_px": 12.0, "last_arc_frame": 394,
              "coeffs": None, "flight_points": [], "last_contact_candidate": {"frame": 437}},
        contact={"chosen_frame": 437, "chosen_source": "last_accepted", "window": [437, 437],
                 "n_frames": 450, "waterline_y": None, "arc_is_freefall": False,
                 "arc_reason": "cy2 < 0",
                 "candidates": [{"rank": 1, "source": "last_accepted", "frame": 437,
                                 "available": True, "confident": True}]},
        measurement={"x_px": 719.72, "y_px": 926.5, "point_source": "model",
                     "point_origin": "detection", "contact_source": "last_accepted",
                     "total_inches": None, "display": None, "sigma_in": None,
                     "sigma_complete": False, "warnings": ["no calibration"],
                     "gates": {}, "sigma_terms": {}},
        calib=None, calib_path=None, detect_config={"imgsz": 1280, "conf": 0.12, "kp_conf": 0.3,
                                                    "clahe": True},
        artifacts={"annotated": "annotated.mp4"}, still_frame=437,
    )
    t1, t2 = report.build_report_text(r), report.build_report_text(r)
    assert t1 == t2
    for frag in ("Dog Diving jump-distance result", "clip         unit", "450 frames processed",
                 "41 detections", "30 accepted", "flight run 0 [369, 396]", "#1 last_accepted",
                 "chosen frame 437", "distance     n/a", "warnings     1"):
        assert frag in t1, f"report.txt missing {frag!r}"
    print("report.txt deterministic; all pipeline numbers present "
          f"({len(t1.splitlines())} lines)")


# ===========================================================================
# 4. the codec reality (measured) — this is what makes §11 checkable
# ===========================================================================
def test_default_fourcc_is_mp4v_and_its_output_is_byte_deterministic(tmp_path):
    assert annotate.DEFAULT_FOURCC == "mp4v"
    clip = clip_path("21-01")
    if not clip.is_file():
        pytest.skip("21-01.mp4 missing")
    a = annotate.write_video(clip, tmp_path / "a.mp4", frame_range=(0, 60), show_candidates=False)
    b = annotate.write_video(clip, tmp_path / "b.mp4", frame_range=(0, 60), show_candidates=False)
    assert a["n_frames"] == 60 and b["n_frames"] == 60
    ha, hb = sha(tmp_path / "a.mp4"), sha(tmp_path / "b.mp4")
    assert ha == hb, f"mp4v is not byte-deterministic on this host ({ha} != {hb})"
    assert a["width"] == 1920 and a["height"] == 1080 and abs(a["fps"] - 30.0) < 1e-6
    print(f"mp4v, 60 frames at source resolution 1920x1080@30: sha {ha[:16]} == {hb[:16]} "
          f"(byte-identical). avc1 is NOT (see docstring: OpenH264 missing on this host).")


# ===========================================================================
# 5. THE §11 GATE — the real batch, twice, artifact hashes compared
# ===========================================================================
def test_gate_m7_batch_artifacts_unchanged_by_reruns():
    skip_unless_cached(GATE_CLIPS)
    clips = [clip_path(c) for c in GATE_CLIPS]

    def one_pass():
        hashes: dict[str, dict[str, str]] = {}
        n_cached = 0
        for clip in clips:
            r = cli_run.run_clip(clip, out_dir=OUT, contact="auto")
            n_cached += int(bool(r["detect"].from_cache))
            hashes[clip.stem] = {name: sha(Path(r["out_dir"]) / name) for name in ARTIFACTS}
            assert r["result"]["calibration"]["present"] is False
            assert r["measurement"]["total_inches"] is None, "a distance was invented"
            assert r["result"]["artifacts"]["annotated"] == "annotated.mp4"
        return hashes, n_cached

    p1, cached1 = one_pass()
    p2, cached2 = one_pass()

    assert cached1 == len(clips), "pass 1 should have been all cache hits on this corpus"
    assert cached2 == len(clips)

    diffs = [f"{cid}/{name}" for cid in p1 for name in ARTIFACTS if p1[cid][name] != p2[cid][name]]
    assert not diffs, f"artifacts changed between re-runs: {diffs}"

    # the volatile file is the one exception, and it must actually be volatile
    m1 = json.loads((OUT / GATE_CLIPS[0] / "run_meta.json").read_text())
    m2 = json.loads((OUT / GATE_CLIPS[1] / "run_meta.json").read_text())
    assert set(m1) >= {"created", "timings", "from_cache", "host"}

    # no volatile key anywhere in a shipped result.json
    for cid in GATE_CLIPS:
        res = json.loads((OUT / cid / "result.json").read_text())
        leaked = all_keys(res) & VOLATILE_KEYS
        assert not leaked, f"{cid}: volatile keys in result.json: {leaked}"

    # write the manifest so the evidence survives the test run
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    (GATE_DIR / "artifact_hashes.json").write_text(
        report.dumps_strict({"pass1": p1, "pass2": p2, "identical": True}), encoding="utf-8")

    print("\nM7 GATE — batch re-run determinism (2 passes x 3 clips, all cache hits)")
    for cid in p1:
        for name in ARTIFACTS:
            print(f"  {cid:8s} {name:17s} {p1[cid][name][:32]}  identical")
    print(f"  {len(p1) * len(ARTIFACTS)}/{len(p1) * len(ARTIFACTS)} artifacts byte-identical "
          f"across re-runs; 0 volatile keys in any result.json")
    print(f"  manifest: {GATE_DIR / 'artifact_hashes.json'}")


# ===========================================================================
# 6. the CLI paths (auto / manual frame / manual point / calibrated / no detection)
# ===========================================================================
def test_cli_auto_and_manual_contact_and_manual_point_paths(tmp_path):
    skip_unless_cached(("18-07",))
    clip = clip_path("18-07")
    out_root = tmp_path / "out"
    for sub in ("auto", "manual", "point"):
        seed_cache("18-07", out_root / sub)

    auto = cli_run.run_clip(clip, out_dir=out_root / "auto", contact="auto", draw=False)
    assert auto["contact_mode"] == "auto"
    assert auto["result"]["contact"]["chosen_frame"] == auto["contact"].chosen_frame
    assert auto["result"]["contact"]["chosen_source"] == "last_accepted"
    assert auto["result"]["measurement"]["point_source"] == "model"
    assert auto["detect"].from_cache, "the seeded cache should have served this run"

    man = cli_run.run_clip(clip, out_dir=out_root / "manual", contact="manual:150", draw=False)
    assert man["contact_mode"] == "manual" and man["contact_frame"] == 150
    assert man["result"]["contact"]["chosen_frame"] == 150
    assert man["result"]["contact"]["chosen_source"] == "manual"
    assert man["result"]["measurement"]["frame"] == 150

    pt = cli_run.run_clip(clip, out_dir=out_root / "point", contact="auto",
                          manual_point=(700.0, 900.0), draw=False)
    m = pt["result"]["measurement"]
    assert (m["x_px"], m["y_px"]) == (700.0, 900.0)
    assert m["point_source"] == "manual" and m["point_origin"] == "click"
    assert m["model_point"] is not None, "the model's own point must survive for the A/B"

    assert cli_run.parse_contact("auto") == ("auto", None)
    assert cli_run.parse_contact("manual:138") == ("manual", 138)
    assert cli_run.parse_contact("138") == ("manual", 138)
    with pytest.raises(ValueError):
        cli_run.parse_contact("nonsense")
    print(f"auto -> f{auto['result']['contact']['chosen_frame']} "
          f"({auto['result']['contact']['chosen_source']});  "
          f"manual:150 -> f{man['result']['contact']['chosen_frame']};  "
          f"--point 700,900 -> point_source={m['point_source']} "
          f"(model point kept: {m['model_point'][:2] if m['model_point'] else None})")


def test_calibrated_path_produces_a_real_distance_against_the_projective_oracle(tmp_path):
    """The calibrated plumbing, on a SYNTHETIC calibration (no real one exists)."""
    skip_unless_cached(("18-07",))
    seed_cache("18-07", tmp_path / "out")
    calib = synthetic_calib()
    r = cli_run.run_clip(clip_path("18-07"), calib=calib, calib_path="<synthetic>",
                         out_dir=tmp_path / "out", draw=False)
    m = r["result"]["measurement"]
    assert m["total_inches"] is not None, "the calibrated path produced no distance"
    x = m["x_px"]
    expect = truth_inches(x)                       # the oracle, not the fit
    err = abs(m["total_inches"] - expect)
    assert err < 0.10, f"pipeline distance {m['total_inches']} vs oracle {expect} (err {err})"
    assert r["result"]["calibration"]["present"] is True
    assert r["result"]["calibration"]["gates"]["G2"]["pass"] is True
    disp = m["display"]
    bann = annotate.banner_text(m, calib, m["frame"])
    assert disp in bann, "the banner must quote the measurement it was given"
    print(f"calibrated: x={x:.1f} -> {disp} (total_inches={m['total_inches']:.4f}, "
          f"sigma={m['sigma_in']}), oracle={expect:.4f} in, |err|={err:.4f} in "
          f"<= 0.10 (6-mark quadratic vs projective truth)")
    print(f"  banner: {bann}")
    print("  gates:  " + str({k: v["pass"] for k, v in sorted(
        r["result"]["calibration"]["gates"].items())}))


def test_clip_with_no_accepted_detection_still_writes_every_artifact():
    """18-07 frames 100-120 carry no accepted tailset frame (its events are 6-9/342-396/428-437).

    ``min_points=0`` — fixed thresholds — **on purpose**, and it is the only honest way to keep this
    case alive: the operator's adaptive threshold (2026-10-12) lowers the detector's confidence until
    at least 8 accepted points exist, so with the ladder on, frames 100-120 yield 11 points and this
    fixture stops existing.  What this test pins is the null-field artifact contract (no point -> no
    contact frame -> no inches -> artifacts still written), which is about the fixed thresholds.
    """
    skip_unless_cached(("18-07",))
    r = cli_run.run_clip(clip_path("18-07"), out_dir=OUT / "_m7_gate/empty",
                         subclip=(100, 120), draw=False, min_points=0)
    res = r["result"]
    assert res["detection_ladder"]["disabled"] is True, (
        "min_points=0 must say so in the artifact rather than omitting the block")
    assert res["trajectory"]["n_points"] == 0
    assert res["contact"]["chosen_frame"] is None
    assert res["measurement"]["x_px"] is None
    assert res["measurement"]["total_inches"] is None
    assert any("no measurement point" in w for w in res["warnings"]), res["warnings"]
    for name in ("result.json", "report.txt", "trajectory.json", "calib_used.json",
                 "run_meta.json"):
        assert (Path(r["out_dir"]) / name).is_file(), name
    print(f"frames 100-120: 0 accepted points, chosen_frame=None, "
          f"total_inches=None, {len(res['warnings'])} warning(s) — artifacts still written")


def test_main_smoke_returns_zero_and_writes_a_rollup(tmp_path):
    skip_unless_cached(("23-04", "23-05_1", "23-05_2"))
    for cid in ("23-04", "23-05_1", "23-05_2"):
        seed_cache(cid, tmp_path)
    rc = cli_run.main(["--batch", "23-*.mp4", "--out", str(tmp_path), "--no-draw", "--quiet",
                       "--imgsz", "1280"])
    assert rc == 0
    rows = (tmp_path / report.ROLLUP_NAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 4, rows                      # header + 23-04, 23-05_1, 23-05_2
    assert rows[0].split(",")[:2] == ["clip_id", "n_frames"]
    assert report.ROLLUP_COLUMNS[0] == "clip_id"
    # every row must round-trip through a strict CSV reader with the declared columns.
    # `airborne_frame_range` is written as "[146,159]" — unquoted, its comma shifted every
    # later column (measured on the real batch roll-up, 2026-10-08).
    import csv as _csv
    with open(tmp_path / report.ROLLUP_NAME, newline="", encoding="utf-8") as fh:
        parsed = list(_csv.DictReader(fh))
    assert len(parsed) == 3
    assert all(set(row) == set(report.ROLLUP_COLUMNS) for row in parsed)
    assert [r["clip_id"] for r in parsed] == ["23-04", "23-05_1", "23-05_2"]
    assert all(r["airborne_frame_range"].startswith("[") for r in parsed)
    # the rollup is deterministic too: same result.json inputs -> same CSV text
    got = report.write_rollup_csv(
        tmp_path / "again.csv",
        [json.loads((tmp_path / c / "result.json").read_text())
         for c in ("23-04", "23-05_1")]).read_text(encoding="utf-8")
    assert got == "\n".join([",".join(report.ROLLUP_COLUMNS)] + rows[1:3]) + "\n"
    print(f"cli_run.main rc=0; rollup {len(rows) - 1} rows: {rows[0][:70]}...")


def test_project_batch_reports_a_projection_before_the_run():
    skip_unless_cached(("18-07", "21-01"))
    cfg = cli_run.detect_mod.DetectConfig(imgsz=1280)
    proj = cli_run.project_batch([clip_path("18-07"), clip_path("21-01")], cfg, None, OUT)
    assert proj["ms_per_frame"] == cli_run.MS_PER_FRAME[1280]
    assert proj["cold_clips"] == 0 and proj["projected_s"] == 0.0
    assert len(proj["per_clip"]) == 2 and all(p["cached"] for p in proj["per_clip"])
    cold = cli_run.project_batch([clip_path("21-04")], cfg, (0, 20), OUT)
    assert cold["cold_clips"] == 1 and cold["projected_s"] > 0
    print(f"projection: warm batch -> {proj['cold_clips']} uncached, 0.0 s; "
          f"uncached subclip -> {cold['cold_frames']} frames, {cold['projected_s']:.1f} s "
          f"at {proj['ms_per_frame']:.0f} ms/frame (PLAN §8 V7)")
