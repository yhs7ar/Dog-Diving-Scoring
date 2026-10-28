"""The trajectory arc is the FORK's algorithm, not a lookalike (order 2026-10-28).

The authoritative implementation is the fork's own pipeline script::

    C:\\Users\\harri\\Documents\\GitHub\\Dog-Diving-Scoring\\Final_pipeline_w_json.py
    -> fit_parabolic_trajectory(points, frame_width, frame_height, min_points=6)

What is pinned here
-------------------
1. ``trajectory.fit_parabolic_trajectory`` reproduces that function **bit for bit** — the same gates,
   the same ``np.polyfit(x, y, 2)``, the same ±3 px walk from the first point, the same bottom-edge
   contact — checked against a verbatim copy of the reference over random arcs in both directions.
2. The fit runs on the **last twenty** accepted detections only (operator order 2026-10-28), so the
   dog standing on the dock at the start of a clip cannot bend the parabola.
3. Without a frame size the parabola is simply absent, and nothing else about the trajectory changes.

The reference copy below is deliberately an independent transcription (it takes ``(x, y)`` pairs and
does no dict handling) rather than a call into the module under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import trajectory as T  # noqa: E402


def _reference_fit(points, frame_width, frame_height, min_points=6):
    """Verbatim ``Final_pipeline_w_json.py::fit_parabolic_trajectory`` (tuples, no dict branch)."""
    if len(points) < min_points:
        return None, None

    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)

    if np.max(x) - np.min(x) < 30:
        return None, None

    try:
        poly = np.poly1d(np.polyfit(x, y, 2))
    except Exception:  # noqa: BLE001
        return None, None

    dx = x[-1] - x[0]
    step = 3 if dx >= 0 else -3

    curve_points = []
    contact_point = None
    curr_x = int(x[0])

    while 0 <= curr_x < frame_width:
        curr_y = poly(curr_x)
        if curr_y >= frame_height - 1:
            contact_point = (int(curr_x), int(frame_height - 1))
            curve_points.append(contact_point)
            break
        if curr_y >= 0:
            curve_points.append((int(curr_x), int(curr_y)))
        curr_x += step

    return curve_points, contact_point


def _arc(rng, n: int, left_to_right: bool = True):
    """``n`` points on a noisy downward parabola, in one flight direction or the other."""
    span = rng.uniform(0.6, 1.0) * 400.0
    x0 = 120.0 if left_to_right else 840.0
    step = span / max(1, n - 1)
    x = x0 + (np.arange(n, dtype=float) * step if left_to_right
              else -np.arange(n, dtype=float) * step)
    y = 0.0009 * (x - x.mean()) ** 2 + rng.normal(0.0, 1.5, n) + 90.0
    return [(float(a), float(b)) for a, b in zip(x, y)]


@pytest.mark.parametrize("left_to_right", [True, False])
def test_the_parabola_matches_the_fork_pipeline_point_for_point(left_to_right):
    rng = np.random.default_rng(20261028)
    for trial in range(30):
        pts = _arc(rng, int(rng.integers(6, 22)), left_to_right)
        want_curve, want_contact = _reference_fit(pts, 960, 600)
        curve, contact, coeffs = T.fit_parabolic_trajectory(pts, 960, 600)
        assert curve == want_curve, f"trial {trial}: curve differs from the fork"
        assert contact == want_contact, f"trial {trial}: contact differs from the fork"
        assert coeffs is not None and set(coeffs) == {"a", "b", "c"}
        # the coefficients are the ones np.polyfit produced, not a refit
        a, b, c = np.polyfit(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), 2)
        assert coeffs["a"] == pytest.approx(float(a))
        assert coeffs["b"] == pytest.approx(float(b))
        assert coeffs["c"] == pytest.approx(float(c))


def test_the_forks_two_gates_are_enforced():
    rng = np.random.default_rng(5)
    # five points: below min_points=6
    assert T.fit_parabolic_trajectory(_arc(rng, 5), 960, 600) == (None, None, None)
    # six points crammed into < 30 px of x: the span gate, exactly as the fork has it
    tight = [(400.0 + i * 4.0, 300.0 - i * 9.0) for i in range(6)]
    assert T.fit_parabolic_trajectory(tight, 960, 600) == (None, None, None)
    # the same six spread over 40 px do fit
    wide = [(400.0 + i * 8.0, 300.0 - i * 9.0) for i in range(6)]
    curve, _contact, coeffs = T.fit_parabolic_trajectory(wide, 960, 600)
    assert curve and coeffs is not None


def test_dict_points_behave_exactly_like_pairs():
    rng = np.random.default_rng(11)
    pairs = _arc(rng, 14)
    dicts = [{"x": a, "y": b, "frame": i, "kp_conf": 0.9} for i, (a, b) in enumerate(pairs)]
    assert (T.fit_parabolic_trajectory(pairs, 960, 600)
            == T.fit_parabolic_trajectory(dicts, 960, 600))


def test_only_the_last_twenty_detections_are_fitted():
    """Order 2026-10-28: the window is the last 20, so an early dock pose cannot bend the arc."""
    rng = np.random.default_rng(3)
    flight = _arc(rng, 20)                                  # a clean flight, frames 20..39
    dock = [(60.0 + i * 3.0, 520.0 - i * 2.0) for i in range(20)]   # the dog standing, frames 0..19
    points = [{"frame": i, "x": dock[i][0], "y": dock[i][1], "kp_conf": 0.95} for i in range(20)]
    points += [{"frame": 20 + i, "x": p[0], "y": p[1], "kp_conf": 0.95}
               for i, p in enumerate(flight)]

    traj = T.build_trajectory(points, clip_id="window", frame_size=(960, 600))
    assert T.LAST_N_POINTS == 20
    assert len(traj.fit_points) == 20, "the window must be exactly the last twenty"
    assert [p["frame"] for p in traj.fit_points] == list(range(20, 40)), "wrong window"
    assert traj.parabola is not None and traj.curve, traj.fit_note

    # and the fit is the fork's fit over THAT window, not over all forty
    fork_curve, _fork_contact = _reference_fit(flight, 960, 600)
    assert traj.curve[0][:2] == list(fork_curve[0]), "must start at the earliest fitted point"
    fork_by_x = {x: y for x, y in fork_curve}
    for x, y in traj.curve:                                 # same curve wherever both have the x
        if int(x) in fork_by_x:
            assert fork_by_x[int(x)] == int(y), int(x)
    # ...clipped at the ENTRY detection, not walked to the edge of the picture
    assert traj.curve[-1][0] == int(flight[-1][0]), "the arc must end on the entry detection's x"
    assert traj.curve[-1][0] != fork_curve[-1][0], "the arc ran past the entry to the frame edge"
    got_contact = None if traj.contact_point is None else tuple(traj.contact_point)
    assert got_contact == _fork_contact
    assert traj.fit_note.startswith("Final_pipeline_w_json.py:fit_parabolic_trajectory")


def test_the_arc_starts_on_the_first_of_the_window_and_ends_on_the_entry():
    """The two ends are operator requirements (2026-10-28), not frame edges.

    Begin on the earliest of the fitted twenty; end on the ENTRY detection's x.  The fork walks to
    the edge of the picture, which is right for its own drawing and wrong here.
    """
    points = [(300.0 + 20.0 * i, 120.0 + 0.002 * (20.0 * i) ** 2) for i in range(14)]
    curve, _contact, coeffs = T.fit_parabolic_trajectory(points, 960, 600,
                                                        x_end=points[-1][0])
    assert coeffs is not None
    assert curve[0][0] == int(points[0][0]), "must begin on the earliest fitted point"
    assert curve[-1][0] == int(points[-1][0]), "must end on the entry detection's x"
    assert len(curve) < 200, "the walk should stop at the entry, not run to the frame edge"
    # ... and the un-clipped call is still the fork's, for the parity tests
    full, _c, _k = T.fit_parabolic_trajectory(points, 960, 600)
    assert full is not None and len(full) > len(curve)


def test_the_contact_is_where_the_curve_leaves_the_bottom_edge():
    """A steep enough arc does reach the bottom of the picture, and the contact is that crossing."""
    points = [(300.0 + 20.0 * i, 100.0 + 0.05 * (20.0 * i) ** 2) for i in range(14)]
    curve, contact, coeffs = T.fit_parabolic_trajectory(points, 960, 600)
    assert coeffs is not None and curve
    assert contact is not None and contact[1] == 599
    assert contact in curve
    # the walk starts at the first fitted point, so the curve begins there
    assert curve[0][0] == int(points[0][0])


def test_no_frame_size_means_no_parabola_and_nothing_else_changes():
    rng = np.random.default_rng(9)
    points = [{"frame": i, "x": p[0], "y": p[1], "kp_conf": 0.9}
              for i, p in enumerate(_arc(rng, 18))]
    traj = T.build_trajectory(points, clip_id="nosize")
    assert traj.parabola is None and traj.curve == [] and traj.contact_point is None
    assert "frame size unknown" in traj.fit_note
    # the existing trajectory outputs are untouched by the addition
    assert traj.n_points == 18 and traj.coeffs is not None and traj.arc_rms_px is not None
