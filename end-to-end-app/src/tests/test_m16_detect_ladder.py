"""M16 — the adaptive detection threshold ladder (operator instruction, 2026-10-12).

The feature: *"Dynamically change the threshold so that a minimum of 8 points are identified, with
the threshold being lowered if 8 points can't be found with the current confidence level."*  The
operator settled the four open semantics questions in the same session, and each one has a test
here:

1. both knobs move, `conf` through its ladder first, `kp_conf` only if still short;
2. "a point" is an **accepted keypoint row** — what the trajectory consumes;
3. escalations raise `imgsz` as the final rung before giving up;
4. every point admitted below the base thresholds is marked (`low_conf` + the thresholds used), and
   the rungs tried are recorded in the artifacts.

No inference happens here: the ladder is a pure selection over stored records, and the pass-level
behaviour is tested by monkeypatching :func:`detect.detect_clip`.  That is deliberate — the point of
the design is that a rung needs no inference pass, and a test that needed one would not prove that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli_run  # noqa: E402  (src/ on sys.path, like the other tool tests)
from dog_diving import detect as D  # noqa: E402
from dog_diving import report as R  # noqa: E402
from dog_diving.config import CONF_LADDER, DEFAULT_CONF, DEFAULT_KP_CONF, KP_CONF_LADDER  # noqa: E402

BASE_CONF = float(DEFAULT_CONF)          # 0.12
BASE_KP = float(DEFAULT_KP_CONF)         # 0.30
CLIP = Path(__file__).resolve().parents[2] / "data" / "clips" / "18-07.mp4"


def rec(frame: int, box: float | None, kp: float | None) -> dict:
    """One detect record.  ``kp=None`` means "a box fired but no keypoint was read"."""
    return {
        "frame": int(frame),
        "tailset": None if kp is None else [float(frame), 100.0, float(kp)],
        "nose": None,
        "box_conf": None if box is None else float(box),
        "track_id": None,
        "accepted": False,
    }


def rung(records, min_points=8, **kw) -> tuple[list[dict], D.LadderRung | None, list, bool]:
    return D.select_rung(
        records, min_points=min_points, conf_ladder=CONF_LADDER,
        kp_conf_ladder=KP_CONF_LADDER, coverage_min=0.5, imgsz=1280,
        base_conf=BASE_CONF, base_kp_conf=BASE_KP, **kw)


def fake_result(records: list[dict], *, imgsz: int = 1280, conf: float = BASE_CONF,
                from_cache: bool = True) -> D.DetectResult:
    """A DetectResult shaped enough for the ladder (no model, no inference).

    ``accepted`` is set the way :func:`detect.apply_acceptance` would, so the fake pass behaves like
    a real one (the ladder re-derives it from the thresholds, but the reports and the no-op path read
    the pass's own counts).
    """
    for r in records:
        ts = r.get("tailset")
        r["accepted"] = bool(ts and ts[2] is not None and float(ts[2]) > BASE_KP)
    n_acc = sum(1 for r in records if r["accepted"])
    return D.DetectResult(
        clip_id="fake", clip_path="fake.mp4", clip_sha="0" * 64, model_sha="1" * 64,
        cache_key={"imgsz": imgsz, "conf": conf}, cache_key_hash="deadbeef",
        cache_path="out/fake/tracks.json", frame_range=None, records=records,
        dominant_track=None, tracking_available=False, track_coverage=0.0,
        n_tracked_frames=0, n_dominant_frames=0, n_dominant_with_kp=0,
        n_frames_clip=len(records), n_frames_processed=len(records),
        n_detections=sum(1 for r in records if r["tailset"]), n_accepted=n_acc,
        timings={"total_s": 0.0}, from_cache=from_cache,
    )


# ============================================================================
# 1. a healthy clip is untouched, and costs nothing
# ============================================================================
def test_base_rung_suffices_and_nothing_is_lowered():
    recs = [rec(i, 0.5, 0.9) for i in range(10)]
    sel, got, tried, met = rung(recs)
    assert met and got is not None
    assert (got.conf, got.kp_conf) == (BASE_CONF, BASE_KP)
    assert [r["accepted"] for r in sel] == [True] * 10
    assert not any(r["low_conf"] for r in sel)
    assert len(tried) == 1, "the first rung met the minimum, so no other rung may be listed"


def test_a_clip_with_few_points_but_confident_boxes_is_fixed_by_the_kp_rung_only():
    """kp_conf lowering must be preferred over conf lowering when it is enough (operator's order)."""
    recs = [rec(i, 0.5, 0.25) for i in range(12)]        # boxes fine, keypoints just under 0.30
    sel, got, tried, met = rung(recs)
    assert met and got is not None
    assert got.conf == BASE_CONF, "conf must not be lowered when a kp_conf rung suffices"
    assert got.kp_conf == 0.20
    assert sum(1 for r in sel if r["accepted"]) == 12
    assert all(r["low_conf"] for r in sel if r["accepted"])


# ============================================================================
# 2. conf comes down, and the rung that buys the points is recorded
# ============================================================================
def test_conf_is_lowered_to_the_first_rung_that_reaches_the_minimum():
    recs = [rec(i, 0.5, 0.95) for i in range(3)] + [rec(100 + i, 0.06, 0.95) for i in range(6)]
    sel, got, tried, met = rung(recs)
    assert met and got is not None
    assert got.conf == 0.06 and got.kp_conf == BASE_KP
    assert got.n_accepted == 9 and got.n_detections == 9
    acc = [r for r in sel if r["accepted"]]
    assert sum(1 for r in acc if r["low_conf"]) == 6, "the 6 points the lower rung bought"
    assert [r["frame"] for r in acc if not r["low_conf"]] == [0, 1, 2]
    # every row carries the thresholds it was judged under (operator's provenance requirement)
    assert all(r["conf_used"] == 0.06 and r["kp_conf_used"] == BASE_KP for r in sel)
    # the rungs are listed most-conservative-first, and conf moves before kp_conf
    assert [(t.conf, t.kp_conf) for t in tried][:2] == [(BASE_CONF, BASE_KP), (BASE_CONF, 0.20)]


def test_box_pre_filter_only_tightens_so_a_base_pass_cannot_reach_a_lower_rung():
    """The reason the base pass is swept on the kp ladder only: post-hoc filtering cannot ADD rows."""
    base_pass = [rec(i, 0.5, 0.95) for i in range(3)]     # what a conf=0.12 pass stores
    _, got, tried, met = rung(base_pass)
    # with the full conf ladder offered, the counted points still cannot exceed what the pass stored
    assert max(t.n_accepted for t in tried) == 3
    assert not met


# ============================================================================
# 3. floor reached and still short -> the best rung, flagged
# ============================================================================
def test_below_minimum_reports_the_best_rung_and_is_flagged():
    recs = [rec(0, 0.5, 0.95), rec(1, 0.5, 0.95)]
    sel, got, tried, met = rung(recs)
    assert not met
    assert got is not None and got.n_accepted == 2, "the best rung is reported, not dropped"
    assert sum(1 for r in sel if r["accepted"]) == 2


def test_no_records_at_all_is_not_a_crash():
    sel, got, tried, met = rung([])
    assert sel == [] and not met
    assert got is not None and got.n_accepted == 0, "the rung is still reported, with zero points"
    assert tried and all(t.n_accepted == 0 for t in tried), (
        "every rung is reported even when the clip yielded nothing, so the artifact can show that "
        "the ladder ran and found no points at all")


# ============================================================================
# 4. the pass-level behaviour (monkeypatched detect_clip)
# ============================================================================
def test_adaptive_returns_the_base_rung_without_a_second_pass(monkeypatch):
    calls: list[tuple[int, float, float]] = []

    def fake(clip, cfg=None, **kw):
        calls.append((int(cfg.imgsz), float(cfg.conf), float(cfg.kp_conf)))
        return fake_result([rec(i, 0.5, 0.9) for i in range(12)], imgsz=int(cfg.imgsz),
                           conf=float(cfg.conf))

    monkeypatch.setattr(D, "detect_clip", fake)
    det = D.detect_clip_adaptive("whatever.mp4", D.DetectConfig(), min_points=8)
    assert calls == [(1280, BASE_CONF, BASE_KP)], "a healthy clip must not pay for a floor pass"
    assert det.ladder["chosen"]["conf"] == BASE_CONF
    assert det.ladder["imgsz_escalated"] is False
    assert det.ladder["below_min_points"] is False


def test_adaptive_runs_one_floor_pass_and_finds_the_rung(monkeypatch):
    calls: list[tuple[int, float, float]] = []

    def fake(clip, cfg=None, **kw):
        calls.append((int(cfg.imgsz), float(cfg.conf), float(cfg.kp_conf)))
        if float(cfg.conf) >= BASE_CONF:                     # the base pass
            return fake_result([rec(i, 0.5, 0.95) for i in range(3)], conf=float(cfg.conf))
        recs = [rec(i, 0.5, 0.95) for i in range(3)]
        recs += [rec(100 + i, 0.06, 0.95) for i in range(6)]  # a floor pass sees the rest
        return fake_result(recs, conf=float(cfg.conf), from_cache=False)

    monkeypatch.setattr(D, "detect_clip", fake)
    det = D.detect_clip_adaptive("whatever.mp4", D.DetectConfig(), min_points=8)
    assert calls[0] == (1280, BASE_CONF, BASE_KP)
    assert calls[1] == (1280, min(CONF_LADDER), BASE_KP), (
        "one floor pass, and kp_conf is NOT lowered in a pass (it never reaches the model)")
    assert det.n_accepted == 9
    assert det.ladder["chosen"]["conf"] == 0.06
    assert det.ladder["chosen"]["n_accepted"] == 9
    assert det.ladder["n_low_conf_points"] == 6
    assert det.ladder["imgsz_tried"] == [1280]
    assert {r["conf_used"] for r in det.records if r["tailset"]} == {0.06}
    # the floor pass must not re-try the base-conf rungs: a box_conf filter can only remove rows,
    # so every combination appears exactly once in the artifact
    rungs = [(r["imgsz"], r["conf"], r["kp_conf"]) for r in det.ladder["rungs"]]
    assert len(rungs) == len(set(rungs)), f"duplicated rungs: {rungs}"


def test_adaptive_escalates_imgsz_as_the_final_rung(monkeypatch):
    calls: list[tuple[int, float]] = []

    def fake(clip, cfg=None, **kw):
        calls.append((int(cfg.imgsz), float(cfg.conf)))
        if int(cfg.imgsz) == 1280:
            return fake_result([rec(0, 0.5, 0.95)], imgsz=1280, conf=float(cfg.conf))
        return fake_result([rec(i, 0.5, 0.95) for i in range(12)], imgsz=1920,
                           conf=float(cfg.conf), from_cache=False)

    monkeypatch.setattr(D, "detect_clip", fake)
    det = D.detect_clip_adaptive("whatever.mp4", D.DetectConfig(), min_points=8)
    assert [c[0] for c in calls] == [1280, 1280, 1920], "imgsz is the last rung, after the floor"
    assert det.ladder["imgsz_escalated"] is True
    assert det.ladder["chosen"]["imgsz"] == 1920
    assert det.n_accepted == 12


def test_adaptive_still_short_everywhere_flags_below_minimum(monkeypatch):
    def fake(clip, cfg=None, **kw):
        return fake_result([rec(0, 0.5, 0.95), rec(1, 0.5, 0.95)], imgsz=int(cfg.imgsz),
                           conf=float(cfg.conf))

    monkeypatch.setattr(D, "detect_clip", fake)
    det = D.detect_clip_adaptive("whatever.mp4", D.DetectConfig(), min_points=8)
    assert det.ladder["below_min_points"] is True
    assert det.ladder["chosen"]["n_accepted"] == 2
    assert det.n_accepted == 2


def test_min_points_zero_disables_the_ladder(monkeypatch):
    calls: list[float] = []

    def fake(clip, cfg=None, **kw):
        calls.append(float(cfg.conf))
        return fake_result([rec(i, 0.5, 0.95) for i in range(3)], conf=float(cfg.conf))

    monkeypatch.setattr(D, "detect_clip", fake)
    det = D.detect_clip_adaptive("whatever.mp4", D.DetectConfig(), min_points=0)
    assert calls == [BASE_CONF], "min_points=0 must not run a floor pass"
    assert det.ladder["disabled"] is True
    assert det.ladder["chosen"] is None
    assert det.n_accepted == 3


# ============================================================================
# 5. provenance reaches the artifacts
# ============================================================================
def test_ladder_block_is_absent_on_a_fixed_threshold_run():
    """PLAN §11: a run without the ladder keeps exactly its old key set and warning count."""
    plain = R.build_result(clip_id="c", measurement={"warnings": ["w"]})
    assert "detection_ladder" not in plain
    assert plain["warnings"] == ["w"]
    assert R._ladder_warnings(None) == []


def test_ladder_block_and_warnings_appear_when_the_ladder_ran():
    lad = {"min_points": 8, "base": {"conf": 0.12, "kp_conf": 0.3},
           "chosen": {"imgsz": 1280, "conf": 0.06, "kp_conf": 0.3, "n_accepted": 78,
                      "base_conf": 0.12, "base_kp_conf": 0.3},
           "rungs": [{"imgsz": 1280, "conf": 0.12, "kp_conf": 0.3, "n_accepted": 3}],
           "imgsz_tried": [1280], "imgsz_escalated": False, "below_min_points": False,
           "n_points": 78, "n_low_conf_points": 75, "disabled": False}
    res = R.build_result(clip_id="c", measurement={"warnings": ["w"]}, ladder=lad)
    assert res["detection_ladder"]["n_low_conf_points"] == 75
    assert len(res["warnings"]) == 2
    assert "ADAPTIVE THRESHOLD" in res["warnings"][1]
    assert "75 of 78" in res["warnings"][1]
    text = R.build_report_text(res)
    assert "ladder       min 8 pt -> imgsz=1280 conf=0.06" in text
    # JSON round-trips (the artifact must be writable)
    json.loads(json.dumps(res))


def test_below_minimum_warning_is_its_own_message():
    lad = {"min_points": 8, "chosen": {"imgsz": 1920, "conf": 0.01, "kp_conf": 0.05},
           "below_min_points": True, "n_points": 5, "n_low_conf_points": 0, "disabled": False,
           "rungs": [], "imgsz_tried": [1280, 1920], "imgsz_escalated": True,
           "base": {"conf": 0.12, "kp_conf": 0.3}}
    warns = R._ladder_warnings(lad)
    assert len(warns) == 1 and "below_min_points" in warns[0]
    assert "fewer than 8" in warns[0]


def test_ladder_provenance_reaches_the_trajectory_points_only_when_it_ran():
    """The per-point keys are gated on the ladder having judged the row (PLAN §11 / m4's schema pin).

    ``test_m4_trajectory.py::test_trajectory_schema`` requires a plain run's points to be exactly
    ``{frame, x, y, kp_conf}``; this test requires the ladder's points to carry the thresholds they
    were judged under plus ``low_conf`` — the operator's provenance requirement.
    """
    from dog_diving import trajectory as T
    plain = {"frame": 5, "tailset": [10.0, 20.0, 0.9], "box_conf": 0.5, "track_id": None,
             "accepted": True}
    p = T.assemble([plain])[0]
    assert set(p) == {"frame", "x", "y", "kp_conf"}, p

    judged = dict(plain, conf_used=0.06, kp_conf_used=0.3, low_conf=True)
    q = T.assemble([judged])[0]
    assert set(q) == {"frame", "x", "y", "kp_conf", "box_conf", "conf_used", "kp_conf_used",
                      "low_conf"}, q
    assert q["box_conf"] == 0.5 and q["conf_used"] == 0.06 and q["low_conf"] is True




def test_cli_exposes_min_points():
    args = cli_run.build_parser().parse_args(["--clip", "x.mp4"])
    assert args.min_points == 8
    assert cli_run.build_parser().parse_args(["--clip", "x.mp4", "--min-points", "0"]).min_points == 0


def test_gui_ladder_rows_state_the_choice_and_the_cost():
    app_gui = pytest.importorskip("app_gui")
    lad = {"min_points": 8, "chosen": {"imgsz": 1280, "conf": 0.06, "kp_conf": 0.3,
                                      "base_conf": 0.12, "base_kp_conf": 0.3},
           "n_points": 78, "n_low_conf_points": 75, "rungs": [{}] * 8,
           "imgsz_tried": [1280], "imgsz_escalated": False, "below_min_points": False,
           "disabled": False}
    rows = dict(app_gui.ladder_rows(lad))
    assert "conf=0.06" in rows["threshold ladder"]
    assert "75 low-conf" in rows["ladder points"]
    assert "8 rung(s) tried" in rows["ladder rungs"]
    assert "ladder WARNING" not in rows
    assert "off" in dict(app_gui.ladder_rows({"disabled": True, "min_points": 0}))["threshold ladder"]
    assert app_gui.ladder_rows(None) == []
    assert dict(app_gui.ladder_rows({**lad, "below_min_points": True}))["ladder WARNING"]
