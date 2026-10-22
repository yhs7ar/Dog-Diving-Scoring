"""M14 — the Phase 2 calibration tool: the click sequence, the session, and the headless path.

The tool is deliberately a thin shell over :func:`dog_diving.pool.calib.build`, so these tests check
the things a shell can get wrong: the prompt order, undo/skip bookkeeping, that a completed session
produces a gate-passing document, that the lens-from-clicked-edges route actually recovers a
distortion model, and that the CLI writes a document.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.pool_scene import deck_marks_scene, frame_from_scene, raw_clicks
from dog_diving.pool import calib as C
from dog_diving.pool import lens as L

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pool_calibrate_tool as TOOL  # noqa: E402


# ============================================================================
# The request sequence
# ============================================================================
def test_request_sequence_covers_every_phase_the_plan_needs():
    reqs = TOOL.build_requests([0.0, 1.0, 15.0, 16.0, 30.0, 31.0])
    kinds = [r["kind"] for r in reqs]
    assert kinds.count("mark") == 6
    assert kinds.count("line") == 2 * 2 + 2 * 3          # 2 across + 3 along, 2 ends each
    assert kinds.count("edge") == 2
    assert kinds.count("rim") == 2
    # phase 2 (marks), 3 (both line families), 4 (edge), 5 (rim pair) are all represented
    assert {r["group"] for r in reqs if r["kind"] == "line"} == {"vp_transverse", "vp_along"}
    assert [r["role"] for r in reqs if r["kind"] == "rim"] == ["top", "contact"]


def test_request_sequence_can_drop_the_optional_parts():
    reqs = TOOL.build_requests([0.0, 1.0], n_transverse_lines=1, n_along_lines=1,
                               want_edge=False, want_rim=False)
    assert {r["kind"] for r in reqs} == {"mark", "line"}
    assert not any(r.get("optional") for r in reqs)


def test_extra_line_families_are_marked_optional():
    reqs = TOOL.build_requests([0.0], n_transverse_lines=3, n_along_lines=3)
    optional = [r for r in reqs if r.get("optional")]
    assert len(optional) == 2 * 2, "only the third line in each family is skippable"


# ============================================================================
# The session
# ============================================================================
def _session(sc, **kw):
    reqs = TOOL.build_requests([0.0, 1.0, 15.0, 16.0, 30.0, 31.0], **kw)
    return TOOL.PoolCalibrationSession(frame_shape=(sc.image_size[1], sc.image_size[0]),
                                       requests=reqs, setup_id="poolSYN",
                                       lens=L.LensModel(K=sc.K, dist=sc.dist,
                                                        image_size=sc.image_size,
                                                        source="lines"))


def test_session_walks_the_sequence_and_completes():
    sc = deck_marks_scene()
    s = _session(sc)
    assert "0 ft mark" in s.prompt
    for _ in range(len(s.requests)):
        s.click(500.0, 600.0)
    assert s.done
    assert "complete" in s.prompt
    assert s.click(1.0, 2.0) == "already complete"


def test_session_undo_removes_the_click_it_recorded():
    sc = deck_marks_scene()
    s = _session(sc)
    s.click(1.0, 2.0)
    s.click(3.0, 4.0)
    assert len(s.clicks["marks"]) == 2
    s.undo()
    assert len(s.clicks["marks"]) == 1
    s.undo()
    assert s.clicks["marks"] == []
    assert s.history == []
    s.undo()                                    # undo past the start is a no-op, not a crash
    assert s.history == []


def test_session_undo_clears_a_half_drawn_line_end():
    sc = deck_marks_scene()
    s = _session(sc)
    for _ in range(6):                          # the marks
        s.click(10.0, 10.0)
    s.click(100.0, 200.0)                       # first end of across-line 1
    assert len(s.clicks["vp_transverse"]["lines"]) == 1
    s.undo()
    assert s.clicks["vp_transverse"]["lines"] == []
    assert s.history[-1]["kind"] == "mark"


def test_session_skip_optional_advances_over_the_third_lines():
    sc = deck_marks_scene()
    reqs = TOOL.build_requests([0.0, 1.0])
    s = TOOL.PoolCalibrationSession(frame_shape=(10, 10), requests=reqs, setup_id="t")
    # click everything up to the first optional request
    for r in reqs:
        if r.get("optional"):
            break
        s.click(1.0, 1.0)
    assert s.requests[len(s.history)].get("optional")
    s.skip_optional()
    assert not s.requests[len(s.history)].get("optional") or s.done


def test_session_build_produces_a_gate_passing_document():
    """Drive the whole sequence with the scene's true geometry and require every gate to pass."""
    sc = deck_marks_scene()
    s = _session(sc, want_edge=True, want_rim=True)
    clicks = raw_clicks(sc, wall={"axis": "y", "value": 0.0})
    # feed the clicks in the order the requests ask for them
    order: list[tuple[float, float]] = []
    for m in clicks["marks"]:
        order.append((m["x_px"], m["y_px"]))
    for seg in clicks["vp_transverse"]["lines"] + clicks["vp_along"]["lines"]:
        order += [(seg[0], seg[1]), (seg[2], seg[3])]
    order += [(clicks["edge_line"][0], clicks["edge_line"][1]),
              (clicks["edge_line"][2], clicks["edge_line"][3])]
    order += [(clicks["rim"]["top"][0], clicks["rim"]["top"][1]),
              (clicks["rim"]["contact"][0], clicks["rim"]["contact"][1])]
    assert len(order) == len(s.requests)
    for x, y in order:
        s.click(x, y)
    assert s.done
    doc = s.build()
    assert doc["schema"] == C.SCHEMA
    assert doc["route"] == "A_prime_vps_points"
    assert doc["water"]["z_ft"] == pytest.approx(sc.water_z_ft, abs=1e-6)
    assert C.gates_pass(s.gates()), {k: v["detail"] for k, v in s.gates().items()}


def test_session_refuses_to_build_incomplete_or_anchorless():
    sc = deck_marks_scene()
    s = _session(sc)
    s.click(1.0, 1.0)
    with pytest.raises(ValueError, match="incomplete"):
        s.build()

    reqs = TOOL.build_requests([0.0])
    tiny = TOOL.PoolCalibrationSession(frame_shape=(10, 10), requests=reqs, setup_id="t")
    for _ in range(len(reqs)):
        tiny.click(1.0, 1.0)
    with pytest.raises(ValueError, match="two distance marks"):
        tiny.build()


def test_water_method_cycles_between_the_three_documented_choices():
    sc = deck_marks_scene()
    s = _session(sc)
    s.water_method = "known_freeboard"
    assert s._water_spec()["freeboard_in"] == s.freeboard_in
    s.water_method = "wall_contact"
    assert s._water_spec() == {"method": "wall_contact"}


# ============================================================================
# The lens route the plan names as its fallback
# ============================================================================
def test_lens_from_clicked_edges_refuses_a_chord_it_cannot_learn_from():
    """Two-point clicks are a straight chord, so straightness carries no distortion signal.

    The refusal matters more than the coefficients: an earlier version of this function fitted those
    chords and returned ``k1 = 0.0000`` against a true ``-0.12`` — a confident-looking number that
    meant nothing.
    """
    sc = deck_marks_scene()
    s = _session(sc)
    clicks = raw_clicks(sc)
    s.clicks["vp_transverse"] = clicks["vp_transverse"]
    s.clicks["vp_along"] = clicks["vp_along"]
    s.clicks["edge_line"] = clicks["edge_line"]
    base = L.LensModel(K=sc.K, dist=np.zeros(5), image_size=sc.image_size, source="assumed")
    fitted = TOOL.lens_from_lines(s, base)                 # no frame -> no trace
    assert fitted.source == "assumed"
    assert np.allclose(fitted.dist, 0.0)
    assert any("IDENTIFIABILITY" in n for n in fitted.notes)


def test_lens_from_clicked_edges_never_presents_a_fit_that_did_not_improve(tmp_path: Path):
    """The invariant that keeps this route honest: a moved ``k1`` with an unmoved residual is noise.

    The traced route is exercised end to end on a geometrically exact frame.  Either it recovers the
    distortion, in which case it must have materially reduced the straightness residual, or it
    refuses and returns the prior with the reason.  What it may *never* do is return coefficients
    that leave the residual where it was — that is a fit absorbing tracing error, and on this
    fixture's polyline-rendered sibling it produced ``k1 = -0.074`` against a true ``-0.12``.
    """
    from tests.pool_scene import clean_edge_lines_for, render_world_lines_exact

    sc = deck_marks_scene()
    world_lines = clean_edge_lines_for(sc)
    frame = render_world_lines_exact(sc, world_lines)
    segs = [[*sc.project(*p0), *sc.project(*p1)] for p0, p1 in world_lines]

    s = _session(sc)
    s.clicks["vp_along"] = {"lines": segs[:3]}
    s.clicks["vp_transverse"] = {"lines": segs[3:]}
    base = L.LensModel(K=sc.K, dist=np.zeros(5), image_size=sc.image_size, source="assumed")
    fitted = TOOL.lens_from_lines(s, base, frame=frame)

    assert any("straightness after undistortion" in n for n in fitted.notes), fitted.notes
    if fitted.source == "lines":
        assert fitted.dist[0] == pytest.approx(sc.dist[0], abs=0.05), fitted.notes
    else:
        assert np.allclose(fitted.dist, 0.0)
        assert any("REFUSED" in n for n in fitted.notes), fitted.notes


def test_lens_from_lines_needs_the_lines():
    sc = deck_marks_scene()
    s = _session(sc)
    with pytest.raises(ValueError, match="needs the line clicks"):
        TOOL.lens_from_lines(s, L.placeholder_intrinsics(sc.image_size))


def test_edge_trace_refuses_a_path_that_wanders_off():
    """A march that hits a tangent and runs off is rejected geometrically, not accepted as data."""
    sc = deck_marks_scene()
    frame = np.full((sc.image_size[1], sc.image_size[0], 3), 20, np.uint8)
    # one clean bar across the frame, plus the query line that does NOT follow it
    cv2.line(frame, (100, 300), (1800, 320), (240, 240, 240), 4, cv2.LINE_AA)
    off_line = TOOL._trace_edge(frame, np.array([200.0, 600.0]), np.array([1700.0, 610.0]),
                                radius=6)
    assert off_line is None
    on_line = TOOL._trace_edge(frame, np.array([150.0, 300.5]), np.array([1750.0, 319.5]),
                               radius=8)
    assert on_line is not None and on_line.shape[0] > 20
    # and it followed the bar rather than the chord
    dev = float(np.max(np.abs(on_line[:, 1] - np.interp(on_line[:, 0], [100, 1800], [300, 320]))))
    assert dev < 4.0


# ============================================================================
# Overlay and the CLI
# ============================================================================
def test_overlay_draws_the_clicks_and_never_mutates_the_frame():
    sc = deck_marks_scene()
    s = _session(sc, n_transverse_lines=1, n_along_lines=1)
    frame = frame_from_scene(sc)
    before = frame.copy()
    for _ in range(3):
        s.click(500.0, 500.0)
    out = TOOL.draw_overlay(frame, s)
    assert np.array_equal(frame, before)
    assert out.shape == frame.shape
    assert int((out != before).any(axis=2).sum()) > 0


def test_overlay_grid_toggle_works_once_the_sequence_is_complete(tmp_path: Path):
    sc = deck_marks_scene()
    s = _session(sc, n_transverse_lines=1, n_along_lines=2, want_edge=True, want_rim=True)
    clicks = raw_clicks(sc)
    order: list[tuple[float, float]] = []
    for m in clicks["marks"]:
        order.append((m["x_px"], m["y_px"]))
    for seg in clicks["vp_transverse"]["lines"][:1] + clicks["vp_along"]["lines"][:2]:
        order += [(seg[0], seg[1]), (seg[2], seg[3])]
    order += [(clicks["edge_line"][0], clicks["edge_line"][1]),
              (clicks["edge_line"][2], clicks["edge_line"][3])]
    order += [(clicks["rim"]["top"][0], clicks["rim"]["top"][1]),
              (clicks["rim"]["contact"][0], clicks["rim"]["contact"][1])]
    assert len(order) == len(s.requests), (len(order), len(s.requests))
    for x, y in order:
        s.click(x, y)
    frame = frame_from_scene(sc)
    out = TOOL.draw_overlay(frame, s, lens=s.lens, show_grid=True)
    assert int((out != frame).any(axis=2).sum()) > 2000


def test_cli_headless_path_writes_a_document(tmp_path: Path):
    sc = deck_marks_scene()
    frame_p = tmp_path / "still.png"
    assert cv2.imwrite(str(frame_p), frame_from_scene(sc))
    clicks_p = tmp_path / "clicks.json"
    clicks_p.write_text(json.dumps(raw_clicks(sc)), encoding="utf-8")
    lens_p = tmp_path / "lens.json"
    lens_p.write_text(json.dumps(L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                             source="lines").as_dict()), encoding="utf-8")
    out_p = tmp_path / "poolSYN.pool.json"

    rc = TOOL.main(["--frame", str(frame_p), "--setup-id", "poolSYN", "--clicks", str(clicks_p),
                    "--out", str(out_p), "--print-gates", "--lens-json", str(lens_p)])
    assert rc == 0
    doc = C.load(out_p)
    assert doc["setup_id"] == "poolSYN"
    assert doc["route"] == "A_prime_vps_points"
    assert doc["water"]["z_ft"] == pytest.approx(sc.water_z_ft, abs=1e-4)
    assert doc["lens"]["source"] == "lines"
    assert C.gates_pass(doc["gates"])
    assert doc["marks"][0]["x_px_raw"] == pytest.approx(sc.marks()[0]["x_px"])


def test_cli_with_only_a_focal_prior_pays_a_stated_price(tmp_path: Path):
    """``--lens-source prior`` on raw clicks is an approximation, and its size is measured here.

    The clicks arrive through the lens; with no distortion model they are taken as undistorted.  On
    this scene that moves the water level by 0.7 in (measured: -1.682 ft against a true -1.625 ft).
    The document records ``lens.source = "assumed"`` so a reader can see that, which is the point:
    the cost is bounded and disclosed rather than absent.
    """
    sc = deck_marks_scene()
    frame_p = tmp_path / "still.png"
    assert cv2.imwrite(str(frame_p), frame_from_scene(sc))
    clicks_p = tmp_path / "clicks.json"
    clicks_p.write_text(json.dumps(raw_clicks(sc)), encoding="utf-8")
    out_p = tmp_path / "poolSYN.pool.json"

    rc = TOOL.main(["--frame", str(frame_p), "--setup-id", "poolSYN", "--clicks", str(clicks_p),
                    "--out", str(out_p), "--focal-px", "2112"])
    assert rc == 0
    doc = C.load(out_p)
    assert doc["lens"]["source"] == "assumed"
    err_in = abs(doc["water"]["z_ft"] - sc.water_z_ft) * 12.0
    assert 0.05 < err_in < 2.0, f"the prior's water-level cost measured {err_in:.2f} in"


def test_cli_headless_path_can_fit_the_lens_from_the_clicked_lines(tmp_path: Path, capsys):
    """``--lens-source lines`` runs, records what it did, and refuses to claim more than it can.

    On this synthetic render the trace is not good enough for an accurate ``k1``, and the route says
    so — it returns the prior with a ``REFUSED`` note rather than coefficients it cannot support.  The
    test asserts that contract, plus that the run still completes and the document still gates.
    """
    sc = deck_marks_scene()
    frame_p = tmp_path / "still.png"
    assert cv2.imwrite(str(frame_p), frame_from_scene(sc))
    clicks_p = tmp_path / "clicks.json"
    clicks_p.write_text(json.dumps(raw_clicks(sc)), encoding="utf-8")
    out_p = tmp_path / "poolSYN.pool.json"

    rc = TOOL.main(["--frame", str(frame_p), "--setup-id", "poolSYN", "--clicks", str(clicks_p),
                    "--out", str(out_p), "--lens-source", "lines", "--focal-px", "2112"])
    assert rc == 0
    doc = C.load(out_p)
    assert doc["lens"]["source"] in ("lines", "assumed")
    notes = " ".join(doc["lens"]["notes"])
    assert ("straightness after undistortion" in notes) or ("IDENTIFIABILITY" in notes)
    if doc["lens"]["source"] == "lines":
        assert doc["lens"]["dist"][0] == pytest.approx(sc.dist[0], abs=0.05)
    else:
        assert "REFUSED" in notes or "IDENTIFIABILITY" in notes
    assert C.gates_pass(doc["gates"]), {k: v["detail"] for k, v in doc["gates"].items()}


def test_cli_rejects_a_seed_that_does_not_complete_the_sequence(tmp_path: Path, capsys):
    sc = deck_marks_scene()
    frame_p = tmp_path / "still.png"
    assert cv2.imwrite(str(frame_p), frame_from_scene(sc))
    rc = TOOL.main(["--frame", str(frame_p), "--setup-id", "t", "--seed", "10,10;20,20",
                    "--out", str(tmp_path / "x.json")])
    assert rc == 2
    assert "did not complete" in capsys.readouterr().err


def test_cli_reports_a_missing_file(tmp_path: Path, capsys):
    rc = TOOL.main(["--frame", str(tmp_path / "nope.png"), "--setup-id", "t"])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err
