"""M5 verification gate (PLAN.md §11 M5 / §5 ``contact.py``).

Gate: PLAN §11 M5 — *"On a clip with a visible splash, the ranked candidates bracket
the expert-picked frame."*

**The literal gate cannot be run on this corpus, and this file says so rather than
pretending.**  Three facts, each checkable from the artefacts in this repo:

* **No expert frame exists.**  ``data/cases/`` is empty and the only ground-truth
  sources named by PLAN §12 item 1 (``Data/Data_Manifest.gsheet``, ``YouTube Ground
  Truth Sources.gdoc``) are ~175-byte Google Drive stubs that do not read as data.
  There is no frame number anywhere in this repo to bracket.
* **No clip's accepted points trace a flight** (HANDOFF open item 9).  Measured on the
  three warm caches: the fitted image-y curvature is **negative on all three**
  (18-07 −0.1680, 21-01 −0.2782, 23-04 −0.0281), so the fitted arc never turns over into
  a descent, and 18-07's per-frame ``dy/dt`` over its "flight" run swings between
  **−169.75 and +62.57 px/frame**.  A free-fall arc can do neither.
* **Nothing is calibrated.**  ``data/calib/`` is empty, so no clip has a waterline row
  and candidate #2 has no input.

So the gate is run against an **external oracle**: a synthetic free-fall jump whose
contact frame is *known by construction* (analytic, not read off the module), with the
splash occlusion PLAN §5's R2 risk describes planted at a known frame.  Same strategy as
M2 (a projective ground truth) and M4 (a synthetic parabola) — the expected answer comes
from algebra, not from the code under test.

The real clips are then pinned as the **negative** evidence: on all three, the honest
output is "candidate #1 only, and here is why the other three cannot be computed".

Run:  PYTHONPATH=src <venv-python> -m pytest src/tests/test_m5_contact.py -s
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import config as ddconfig  # noqa: E402
from dog_diving import contact, detect, trajectory, video  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP = PROJECT_ROOT / "data" / "clips" / "18-07.mp4"
MODEL = PROJECT_ROOT / "models" / "best.pt"
GATE_OUT = PROJECT_ROOT / "out" / "_m5_gate"


# ===========================================================================
# The external oracle: a synthetic free-fall jump with a known contact frame
# ===========================================================================
# Image coordinates, y growing downward.  Gravity is +g on image y; the tailset
# launches upward (v_y0 < 0) from the dock lip and falls into the water.
Y_LIP, VY0, G = 560.0, -18.0, 0.8          # px, px/frame, px/frame^2
X_LIP, VX = 1400.0, -12.0                  # px, px/frame
WATERLINE_Y = 700.0                        # px row of the water surface
L = 100                                    # launch frame
LAST_ON_ARC = 148                          # last detection before the splash occlusion
AFTERMATH = (161, 170)                     # aftermath detections, in the water

# analytic ground truth: the descending-branch root of y(t) = WATERLINE_Y
_TC = (18.0 + np.sqrt(18.0 ** 2 + 4 * 0.5 * G * (WATERLINE_Y - Y_LIP))) / (2 * 0.5 * G)
T_STAR = L + float(_TC)                    # 151.7617...  the "expert-picked" frame
APEX_FRAME = L - VY0 / G                   # 122.5
# the last frame candidate #3's scan reaches: ceil(apex) + max_extrap_frames
SCAN_END = int(np.ceil(APEX_FRAME)) + int(ddconfig.CONTACT_MAX_EXTRAP_FRAMES)      # 213


def _arc(t: float) -> tuple[float, float]:
    tc = float(t) - L
    return X_LIP + VX * tc, Y_LIP + VY0 * tc + 0.5 * G * tc * tc


def _rec(t, x, y, kp=0.9, box=0.20) -> dict:
    return {"frame": int(t), "tailset": [float(x), float(y), float(kp)],
            "nose": [float(x - 60), float(y - 40), float(kp)],
            "box_conf": float(box), "track_id": None, "accepted": True}


def _freefall_records(*, last=LAST_ON_ARC, resume=None, detour=0, box_collapse=(), box=0.20
                      ) -> list[dict]:
    """A synthetic jump: on-arc detections to ``last``, optional gap, optional aftermath.

    ``box`` defaults to 0.20 — a value this checkpoint actually produces (M3 measured
    0.12-0.30 over the full 18-07 clip) — so the box-confidence trigger of PLAN §5 #3 is
    exercised against realistic numbers rather than a convenient one.
    """
    recs = [_rec(t, *_arc(t), box=(0.05 if t in box_collapse else box))
            for t in range(L, last + 1)]
    y, x = _arc(last)[1], _arc(last)[0]
    for t in range(last + 1, last + 1 + detour):        # constant-velocity departure
        y += 80.0
        x -= 3.0
        recs.append(_rec(t, x, y, box=box))
    if resume is not None:
        ax = _arc(last)[0]
        recs += [_rec(t, ax, WATERLINE_Y + 10.0, box=box) for t in range(resume[0], resume[1])]
    return recs


def _result(recs, waterline_y=WATERLINE_Y, cfg=None, n_frames=None):
    traj = trajectory.build_trajectory(recs, clip_id="synth")
    res = contact.contact_candidates(recs, traj, waterline_y=waterline_y, clip_id="synth",
                                     cfg=cfg, n_frames=n_frames)
    return res, traj


@pytest.fixture(scope="module")
def oracle():
    """The clean free-fall oracle whose detections stop 4 frames before contact."""
    recs = _freefall_records(resume=None)
    res, traj = _result(recs)
    print(f"\n--- synthetic oracle: launch f{L}, detections to f{LAST_ON_ARC}, then the "
          f"splash occlusion")
    print(f"analytic contact frame t* = {T_STAR:.4f}   apex = {APEX_FRAME:.1f}")
    print(f"flight run = {traj.airborne_frame_range}  points = {traj.n_points}  "
          f"inliers = {len(traj.inliers)}")
    print(f"fitted y coeffs = {[round(float(v), 6) for v in traj.coeffs['y']]}")
    for c in res.ranked():
        print(f"  #{c.rank} {c.source:18} frame={str(c.frame):>5} available={c.available} "
              f"confident={c.confident}")
        print(f"      {c.reason}")
    print(f"window = {res.window}   chosen = {res.chosen_frame} ({res.chosen_source})")
    return {"records": recs, "result": res, "traj": traj}


# ===========================================================================
# 1. THE GATE — the ranked candidates bracket the known contact frame
# ===========================================================================
def test_ranked_candidates_bracket_the_known_contact_frame(oracle):
    """PLAN §11 M5, run against an oracle whose contact frame is known by construction.

    The claim verified is exactly PLAN's: the ranked candidate *set* spans the true
    water-entry frame.  The window is taken over the *confident* candidates (the module's
    own flag), not over every number it prints.
    """
    res = oracle["result"]
    window = res.window
    print(f"\nbracket check: window={window} must contain t*={T_STAR:.4f}")
    assert window is not None
    assert window[0] <= T_STAR <= window[1], "the ranked candidates do not bracket entry"
    assert res.brackets(T_STAR) and res.brackets(round(T_STAR))
    assert window[1] - window[0] <= 8, f"bracket too wide to be informative: {window}"
    assert res.by_source(contact.S_LAST).frame < T_STAR, "candidate #1 must be EARLY (R2)"
    assert abs(res.by_source(contact.S_ARC).frame - T_STAR) <= 1.0, "candidate #2 misses entry"


def test_extrapolated_crossing_beats_the_last_detection(oracle):
    """The point of candidate #2: it corrects the occlusion bias candidate #1 carries."""
    res = oracle["result"]
    c1 = res.by_source(contact.S_LAST)
    c2 = res.by_source(contact.S_ARC)
    err1, err2 = abs(c1.frame - T_STAR), abs(c2.frame - T_STAR)
    print(f"\ncandidate #1 frame {c1.frame}: error {err1:.2f} frames (early)")
    print(f"candidate #2 frame {c2.frame}: error {err2:.2f} frames  "
          f"(crossing at {c2.evidence['crossing_frame']:.4f}, "
          f"{c2.evidence['extrap_frames']:+.2f} frames past the last observation)")
    assert err1 == pytest.approx(T_STAR - LAST_ON_ARC, abs=0.05)
    assert err2 < err1 / 3.0, "the extrapolation does not improve on the raw last detection"
    assert c2.evidence["crossing_frame"] > c2.evidence["apex_frame"], "not the descending branch"
    assert c2.evidence["apex_frame"] == pytest.approx(APEX_FRAME, abs=0.01)
    assert c2.evidence["extrap_frames"] > 0, "the crossing should lie past the data here"
    # the crossing sits at the waterline row, and lands on the arc
    assert c2.evidence["waterline_y"] == WATERLINE_Y
    assert c2.evidence["y_at_flight_start"] < WATERLINE_Y


def test_confidence_collapse_finds_the_occlusion_start(oracle):
    """PLAN §5 #3: the first frame of the >= 2-frame gap is the candidate."""
    c3 = oracle["result"].by_source(contact.S_COLLAPSE)
    print(f"\ncandidate #3 frame {c3.frame} (the detections stop after {LAST_ON_ARC})")
    assert c3.frame == LAST_ON_ARC + 1
    assert c3.available and c3.confident
    assert c3.evidence["collapse_start"] == LAST_ON_ARC + 1
    assert c3.evidence["run_length"] >= 2
    # the arc really is descending there, or #3 would have refused (see the 18-07 pins)
    assert c3.evidence["slope_at_collapse_px_per_frame"] > c3.evidence["min_descent_px_per_frame"]
    assert "still descending" in c3.reason


def test_velocity_candidate_reports_no_departure_when_the_fit_holds(oracle):
    """A clean arc holds the free-fall slope to the end, so #4 falls back to the last frame."""
    c4 = oracle["result"].by_source(contact.S_VEL)
    print(f"\ncandidate #4 frame {c4.frame} confident={c4.confident} "
          f"({c4.evidence.get('n_pairs')} pairs tested at {c4.evidence['vel_tol_px']} px/frame)")
    assert c4.frame == LAST_ON_ARC
    assert c4.confident is False, "no departure was planted, so #4 must not claim one"
    assert c4.evidence["departure_frame"] is None
    assert "stays within" in c4.reason


def test_velocity_candidate_finds_a_planted_departure():
    """The other half of #4: a planted departure is reported at the right frame."""
    recs = _freefall_records(resume=None, detour=8)
    res, _ = _result(recs)
    c4 = res.by_source(contact.S_VEL)
    print(f"\nplanted departure after frame {LAST_ON_ARC}: got frame {c4.frame}")
    print(f"  observed dy/dt = {c4.evidence['observed_dy_per_frame']:+.3f} vs model "
          f"{c4.evidence['model_dy_per_frame']:+.3f} "
          f"(deviation {c4.evidence['deviation_px_per_frame']:+.3f})")
    assert c4.frame == LAST_ON_ARC
    assert c4.available and c4.confident
    assert c4.evidence["observed_dy_per_frame"] == pytest.approx(80.0)
    assert abs(c4.evidence["deviation_px_per_frame"]) > res.cfg["vel_tol_px"]
    # ...and #2 still finds the crossing, this time from inside the observed data
    c2 = res.by_source(contact.S_ARC)
    assert c2.frame == round(T_STAR)
    assert c2.evidence["extrap_frames"] < 0, "the crossing now lies inside the observed data"


# ===========================================================================
# 2. The screen — every candidate refuses rather than guessing
# ===========================================================================
def test_no_waterline_is_reported_not_guessed():
    """Without a calibration there is no water row: #2 must say so, not invent one."""
    res, _ = _result(_freefall_records(resume=None), waterline_y=None)
    c2 = res.by_source(contact.S_ARC)
    print(f"\nno waterline: #2 available={c2.available}  reason: {c2.reason[:90]}...")
    assert not c2.available and c2.frame is None
    assert "no waterline" in c2.reason and "calib" in c2.reason
    assert c2.evidence["waterline_y"] is None
    # the arc itself is fine here: it is the *input* that is missing
    assert res.arc_is_freefall is True


def test_arc_with_the_wrong_curvature_refuses_all_three_model_candidates():
    """A fitted arc that never turns over cannot produce a water entry.

    This is the shape every warm-cache clip actually has (negative image-y curvature):
    the tailset's image y rises ever faster, so the fit has no minimum and therefore no
    descending branch.  PLAN §5 #2/#3/#4 are each *defined* on a descending arc.
    """
    recs = [_rec(t, X_LIP + VX * (t - L), Y_LIP + 18.0 * (t - L) - 0.4 * (t - L) ** 2)
            for t in range(L, L + 30)]
    res, traj = _result(recs)
    print(f"\nwrong-curvature arc: cy2={traj.coeffs['y'][0]:.4f} "
          f"freefall={res.arc_is_freefall}")
    print(f"  reason: {res.arc_reason[:120]}...")
    assert res.arc_is_freefall is False
    assert res.arc_evidence["cy2"] == pytest.approx(-0.4, abs=1e-6)
    for src in (contact.S_ARC, contact.S_COLLAPSE, contact.S_VEL):
        c = res.by_source(src)
        assert not c.available and c.frame is None, f"#{c.rank} emitted a frame from a bad arc"
        assert "no descending branch" in c.reason
    # candidate #1 never depends on the arc, and is still offered
    assert res.by_source(contact.S_LAST).available
    # and the screen is a documented knob, not a hard-coded refusal
    relaxed, _ = _result(recs, cfg=contact.ContactConfig(freefall_curvature_min=-1.0))
    assert relaxed.arc_is_freefall is True
    assert relaxed.by_source(contact.S_ARC).available


def test_collapse_far_beyond_the_data_is_available_but_not_confident():
    """A crossing predicted long after the last observation is flagged, not trusted."""
    res, _ = _result(_freefall_records(resume=None), waterline_y=6000.0)
    c2 = res.by_source(contact.S_ARC)
    print(f"\nwaterline 6000 px: frame {c2.frame}, extrap {c2.evidence['extrap_frames']:+.1f} "
          f"vs max {c2.evidence['extrap_max']}")
    assert c2.available and c2.frame is not None
    assert c2.confident is False
    assert c2.evidence["extrap_frames"] > c2.evidence["extrap_max"]
    # an unconfident candidate must not drag the bracket open
    assert res.window[1] - res.window[0] <= 8


def test_collapse_uses_keypoint_confidence_but_can_honour_the_box_floor():
    """PLAN §5 #3 says "tailset conf **or** box conf" — why the box half is off by default.

    Box confidence on this checkpoint tops out at 0.25 (M3 measured 0.121/0.141/0.250 over
    the full 18-07 clip), so a box trigger fires almost everywhere.  Three configurations are
    shown: the default (keypoint only), a box floor set *below* this checkpoint's typical
    0.20 (finds the planted dip), and a floor at 0.25 (fires on every frame, and is then
    thrown out by the descent gate at the apex — i.e. PLAN's literal OR is unusable here).
    """
    recs = _freefall_records(last=SCAN_END, resume=None, box_collapse={140, 141}, box=0.20)
    default, _ = _result(recs)
    off = default.by_source(contact.S_COLLAPSE)
    armed, _ = _result(recs, cfg=contact.ContactConfig(box_conf_floor=0.15))
    on = armed.by_source(contact.S_COLLAPSE)
    toohigh, _ = _result(recs, cfg=contact.ContactConfig(box_conf_floor=0.25))
    high = toohigh.by_source(contact.S_COLLAPSE)
    print(f"\nplanted box-conf collapse at 140-141 (box 0.05 there, 0.20 elsewhere; keypoint 0.9):")
    print(f"  default (box floor None): #3 frame={off.frame}  {off.reason[:70]}...")
    print(f"  box_conf_floor=0.15:      #3 frame={on.frame}  {on.reason[:70]}...")
    print(f"  box_conf_floor=0.25:      #3 frame={high.frame}  {high.reason[:70]}...")
    assert off.frame is None, "the keypoint is confident throughout; #3 must not fire by default"
    assert "no 2-frame keypoint-confidence collapse" in off.reason
    assert on.frame == 140 and on.available
    assert on.evidence["box_conf_floor"] == 0.15
    assert high.frame is None and "not descending" in high.reason, (
        "a box floor above this checkpoint's box confidences fires on every frame; the "
        "descent gate is what stops it emitting the apex as a contact"
    )


def test_empty_and_degenerate_inputs():
    res, _ = _result([])
    assert res.chosen_frame is None and res.window is None
    assert all(not c.available for c in res.candidates)
    assert res.arc_is_freefall is False
    assert "no fitted arc" in res.arc_reason

    one = [_rec(5, 10.0, 20.0)]
    r1, t1 = _result(one)
    assert t1.coeffs is None
    assert r1.by_source(contact.S_LAST).frame == 5
    for src in (contact.S_ARC, contact.S_COLLAPSE, contact.S_VEL):
        assert not r1.by_source(src).available


# ===========================================================================
# 3. The operator's controls (PLAN §5: preselect #1, nudge ±1/±5/±10)
# ===========================================================================
def test_preselects_candidate_1_and_scrub_steps(oracle):
    res = oracle["result"]
    assert res.chosen_frame == res.by_source(contact.S_LAST).frame
    assert res.chosen_source == contact.S_LAST
    assert contact.SCRUB_STEPS == (1, 5, 10)


def test_nudge_and_clamping(oracle):
    res200, _ = _result(oracle["records"], n_frames=200)
    print(f"\nnudge: clip length {res200.n_frames}, chosen {res200.chosen_frame}")
    assert contact.nudge(5, -10, 100) == 0
    assert contact.nudge(95, 10, 100) == 99
    assert contact.nudge(50, 5, None) == 55
    assert res200.nudge(-10_000).chosen_frame == 0
    assert res200.nudge(+10_000).chosen_frame == 199
    moved = res200.nudge(+5)
    assert moved.chosen_frame == res200.chosen_frame + 5
    assert moved.chosen_source == res200.chosen_source


def test_choose_records_the_operator_pick(oracle):
    """PLAN §5: the chosen frame *and which candidate it came from* go into result.json."""
    res, _ = _result(oracle["records"], n_frames=200)
    picked = res.choose(round(T_STAR), contact.S_ARC)
    print(f"\npick: frame {picked.chosen_frame} from {picked.chosen_source}")
    assert picked.chosen_frame == round(T_STAR)
    assert picked.chosen_source == contact.S_ARC
    assert picked.by_source(contact.S_ARC).frame == round(T_STAR)
    assert res.choose(10_000, "manual").chosen_frame == 199, "the pick must clamp to the clip"
    # choosing never mutates the candidate set
    assert picked.candidates == res.candidates


# ===========================================================================
# 4. Schema + determinism
# ===========================================================================
def test_schema_and_json_round_trip(oracle):
    res = oracle["result"]
    d = res.as_dict()
    for key in ("clip_id", "candidates", "chosen_frame", "chosen_source", "waterline_y",
                "arc_is_freefall", "arc_reason", "arc_evidence", "window", "n_frames", "cfg"):
        assert key in d, f"missing output {key}"
    assert [c["rank"] for c in d["candidates"]] == [1, 2, 3, 4]
    assert [c["source"] for c in d["candidates"]] == list(contact.ORDER)
    assert all(set(c) >= {"rank", "source", "frame", "reason", "available", "confident",
                          "evidence"} for c in d["candidates"])
    assert set(res.cfg) == {"kp_conf", "collapse_run", "vel_tol_px", "freefall_curvature_min",
                            "min_descent_px_per_frame", "max_extrap_frames", "box_conf_floor"}
    json.dumps(d)                     # result.json is written from exactly this
    assert json.loads(json.dumps(d))["chosen_frame"] == res.chosen_frame


def test_build_is_deterministic(oracle):
    a, _ = _result(list(oracle["records"]))
    b, _ = _result(list(oracle["records"]))
    assert a.as_dict() == b.as_dict()


# ===========================================================================
# 5. The real clips, pinned — the negative evidence
# ===========================================================================
# MEASURED on this host, warm caches (2026-10-08), imgsz=1280, default config:
REAL_PINS = {
    "18-07": {"accepted": 30, "detections": 41, "flight": (369, 396), "c1": 437,
              "cy2": -0.167990, "y_span_px": 502.1, "dy_min": -169.75, "dy_max": 62.57},
    "21-01": {"accepted": 6, "flight": (146, 159), "c1": 159, "cy2": -0.278217,
              "dy_min": -10.46, "dy_max": 29.28},
    "23-04": {"accepted": 17, "flight": (223, 236), "c1": 250, "cy2": -0.028077,
              "dy_min": -1.69, "dy_max": 1.28},
}

# MEASURED 2026-10-08, the same clip at imgsz=1920 (V5's first data point; open item 1).
# 244.0 s wall for 450 frames = 542 ms/frame (1.84 fps) against 244 ms/frame at 1280.
IMGSZ_1920 = {"detections": 92, "accepted": 76, "flight": (246, 286), "cy2": -0.006014,
              "y_span_px": 79.6, "dy_min": -21.96, "dy_max": 32.01, "c1": 446,
              "wall_s": 244.0, "ms_per_frame": 542.0, "fps": 1.84}


@pytest.fixture(scope="module")
def real():
    """Every warm cache, loaded through the same key the modules use.  No inference."""
    cfg = detect.DetectConfig()
    out = {}
    for clip_id in REAL_PINS:
        path = PROJECT_ROOT / "out" / clip_id / "tracks.json"
        clip = PROJECT_ROOT / "data" / "clips" / f"{clip_id}.mp4"
        if not path.is_file():
            pytest.skip(f"missing warm cache {path}")
        doc = detect.load_cache(path)
        kh = detect.key_hash(detect.cache_key(video.sha256(clip), video.sha256(MODEL), cfg, None))
        assert kh in doc.get("entries", {}), (
            f"the default config's key {kh} is not in {path.name}; the M5 gate never runs "
            f"inference — refresh it with the M3 full-clip command instead"
        )
        e = doc["entries"][kh]
        recs = e["records"]
        traj = trajectory.build_trajectory(recs, clip_id=clip_id)
        res = contact.contact_candidates(recs, traj, waterline_y=None, clip_id=clip_id,
                                         n_frames=e.get("n_frames_clip"))
        out[clip_id] = {"entry": e, "records": recs, "traj": traj, "result": res}
    return out


@pytest.mark.parametrize("clip_id", sorted(REAL_PINS))
def test_real_clips_have_no_descending_branch(real, clip_id):
    """PINNED: the fitted arc's image-y curvature is NEGATIVE on all three caches.

    This is the measurement that makes the real half of PLAN §11 M5 un-runnable: PLAN's
    candidates #2/#3/#4 are each defined on a descending arc, and none of these clips has
    one.  A future clip that really does trace a flight fails this test — which is the
    point: that is the signal the gate can finally be run for real.
    """
    pin = REAL_PINS[clip_id]
    r = real[clip_id]
    res, traj = r["result"], r["traj"]
    cy2 = float(traj.coeffs["y"][0]) if traj.coeffs else None
    print(f"\n{clip_id}: accepted={len(traj.points)} flight={traj.airborne_frame_range} "
          f"inliers={len(traj.inliers)}/{len(traj.flight_points)} cy2={cy2!r}")
    assert len(traj.points) == pin["accepted"]
    assert tuple(traj.airborne_frame_range or ()) == pin["flight"], "the flight run moved"
    assert cy2 is not None and cy2 == pytest.approx(pin["cy2"], abs=5e-5)
    assert cy2 < 0
    assert res.arc_is_freefall is False
    assert "no descending branch" in res.arc_reason
    # #1 survives; #2/#3/#4 all refuse, each with its own reason
    assert res.by_source(contact.S_LAST).frame == pin["c1"]
    assert res.by_source(contact.S_ARC).available is False
    assert "no waterline" in res.by_source(contact.S_ARC).reason
    for src in (contact.S_COLLAPSE, contact.S_VEL):
        c = res.by_source(src)
        assert not c.available and c.frame is None
        assert "no descending branch" in c.reason
    assert res.window == [pin["c1"], pin["c1"]]


@pytest.mark.parametrize("clip_id", ["18-07", "21-01", "23-04"])
def test_real_flight_runs_are_not_trajectories(real, clip_id):
    """PINNED: the observed per-frame dy/dt over the "flight" run, which is not a fall.

    18-07 swings from −169.75 to +62.57 px/frame *inside one run*; a free-fall arc's
    per-frame dy is smooth.  23-04's whole run is near-stationary (|dy| <= 1.7 px/frame
    over 6 points) — it cannot be a jump at all.
    """
    r = real[clip_id]
    fp = r["traj"].flight_points
    dys = [(fp[i + 1]["y"] - fp[i]["y"]) / (fp[i + 1]["frame"] - fp[i]["frame"])
           for i in range(len(fp) - 1)]
    pin = REAL_PINS[clip_id]
    print(f"\n{clip_id}: {len(dys)} intervals, dy/dt in [{min(dys):.2f}, {max(dys):.2f}] px/frame")
    assert min(dys) == pytest.approx(pin["dy_min"], abs=0.02)
    assert max(dys) == pytest.approx(pin["dy_max"], abs=0.02)
    if clip_id == "23-04":
        assert max(abs(v) for v in dys) < 2.0, "the run is no longer near-stationary"
    if clip_id == "18-07":
        assert max(dys) - min(dys) > 200.0, "the run no longer looks like noise"


@pytest.fixture(scope="module")
def clip_1920():
    """The imgsz=1920 entry in the same warm cache (written by the M5 measurement run)."""
    path = PROJECT_ROOT / "out" / "18-07" / "tracks.json"
    if not path.is_file():
        pytest.skip(f"missing warm cache {path}")
    doc = detect.load_cache(path)
    cfg = detect.DetectConfig(model_path=MODEL, imgsz=1920)
    kh = detect.key_hash(detect.cache_key(video.sha256(CLIP), video.sha256(MODEL), cfg, None))
    assert kh in doc["entries"], (
        f"the imgsz=1920 entry ({kh}) is not in {path.name}; re-create it with "
        f"detect_clip(clip, DetectConfig(imgsz=1920)) — the 450-frame run costs ~4 min"
    )
    e = doc["entries"][kh]
    recs = e["records"]
    traj = trajectory.build_trajectory(recs, clip_id="18-07")
    return {"entry": e, "traj": traj,
            "result": contact.contact_candidates(recs, traj, waterline_y=None, clip_id="18-07",
                                                 n_frames=e.get("n_frames_clip"))}


def test_imgsz_1920_doubles_the_coverage_and_still_finds_no_arc(clip_1920, real):
    """PINNED: 2.2x the runtime buys 2.2x the detections and no closer to a flight arc.

    This separates open item 1 (``imgsz``) from open item 9 (no clip traces a flight).
    At 1920 the clip yields 92 detections / 76 accepted against 41 / 30 at 1280 — but the
    run that gets selected is a 36-point, **80-px-tall blob** (y span 79.6 px against
    502 px at 1280) whose fitted curvature is -0.006, i.e. still no descent.  Whatever is
    missing from this corpus's detections, it is not resolution.
    """
    e, traj, res = clip_1920["entry"], clip_1920["traj"], clip_1920["result"]
    base = REAL_PINS["18-07"]
    fp = traj.flight_points
    dys = [(fp[i + 1]["y"] - fp[i]["y"]) / (fp[i + 1]["frame"] - fp[i]["frame"])
           for i in range(len(fp) - 1)]
    y_span = max(p["y"] for p in fp) - min(p["y"] for p in fp)
    print(f"\nimgsz 1280 -> 1920, same clip: detections {base['detections']} -> "
          f"{e['n_detections']}, accepted {base['accepted']} -> {e['n_accepted']}")
    print(f"  selected run {traj.airborne_frame_range} ({traj.n_points} pts, "
          f"{len(traj.inliers)} inliers, rms {traj.arc_rms_px:.1f})  y span {y_span:.1f} px "
          f"vs {base['y_span_px']:.1f} px at 1280")
    print(f"  cy2 {base['cy2']:.4f} -> {float(traj.coeffs['y'][0]):.4f}   "
          f"dy/dt in [{min(dys):.2f}, {max(dys):.2f}]")
    print(f"  measured: {e['timings']['total_s']:.1f} s for {e['n_frames_processed']} frames "
          f"= {e['timings']['ms_per_frame']:.0f} ms/frame, {e['timings']['fps']:.2f} fps")
    assert e["n_detections"] == IMGSZ_1920["detections"]
    assert e["n_accepted"] == IMGSZ_1920["accepted"]
    assert tuple(traj.airborne_frame_range or ()) == IMGSZ_1920["flight"]
    assert float(traj.coeffs["y"][0]) == pytest.approx(IMGSZ_1920["cy2"], abs=5e-5)
    assert y_span == pytest.approx(IMGSZ_1920["y_span_px"], abs=0.2)
    assert min(dys) == pytest.approx(IMGSZ_1920["dy_min"], abs=0.02)
    assert max(dys) == pytest.approx(IMGSZ_1920["dy_max"], abs=0.02)
    # more than double the coverage for ~2.2x the time...
    assert e["n_detections"] > 2 * base["detections"]
    assert e["n_accepted"] > 2 * base["accepted"]
    assert 400 < e["timings"]["ms_per_frame"] < 800, "measured throughput moved"
    # ...and the selected run is *flatter* than at 1280, not closer to an arc
    assert y_span < 0.25 * base["y_span_px"]
    assert float(traj.coeffs["y"][0]) < 0
    assert res.arc_is_freefall is False
    assert res.window == [IMGSZ_1920["c1"], IMGSZ_1920["c1"]]


def test_real_clip_candidate_1_discloses_an_aftermath_detection(real):
    """18-07's last accepted point is 41 frames past its arc — the operator must see that."""
    c1 = real["18-07"]["result"].by_source(contact.S_LAST)
    print(f"\n18-07 candidate #1: frame {c1.frame}, "
          f"{c1.evidence['frames_after_last_arc']} frames after the last arc sample "
          f"(f{c1.evidence['last_arc_frame']}), inside_flight_run="
          f"{c1.evidence['inside_flight_run']}")
    assert c1.frame == 437 and c1.evidence["last_arc_frame"] == 396
    assert c1.evidence["inside_flight_run"] is False
    assert c1.evidence["frames_after_last_arc"] == 41
    assert "AFTER the last arc sample" in c1.reason and "aftermath" in c1.reason
    # ...whereas a clip whose last detection IS an arc sample says so instead
    c1b = real["21-01"]["result"].by_source(contact.S_LAST)
    assert c1b.evidence["inside_flight_run"] is True
    assert "inside the fitted flight run" in c1b.reason


# ===========================================================================
# 6. Visual artefact — the GUI's "show all four"
# ===========================================================================
def test_draw_candidates_labels_the_frame(real):
    """The GUI contract needs the four candidate numbers on the frame; prove pixels."""
    import cv2

    GATE_OUT.mkdir(parents=True, exist_ok=True)
    res = real["18-07"]["result"]
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, res.chosen_frame)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read frame {res.chosen_frame}"
    out = contact.draw_candidates(frame, res)
    path = GATE_OUT / f"contact_18-07_frame{res.chosen_frame}.png"
    cv2.imwrite(str(path), out)
    labels = ["#%d %s%s" % (c.rank, c.source, "" if c.available else " (n/a)")
              for c in res.ranked()]
    print(f"\nwrote {path.relative_to(PROJECT_ROOT)}  {labels}")
    assert out.shape == frame.shape and not np.array_equal(out, frame), "overlay drew nothing"
    yellow = ((out[:, :, 0] < 60) & (out[:, :, 1] > 200) & (out[:, :, 2] > 200)).sum()
    assert yellow > 50, "candidate #1 was not drawn"
