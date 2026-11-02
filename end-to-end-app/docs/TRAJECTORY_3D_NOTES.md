# The 3-D trajectory route — what was built, and what it measures

`intersection-algo.txt` phases 1-5, applied to this tree. This note exists because the code carries
the *implementation* and the tests carry the *numbers*, and neither one is a good place to say "here
is what this route can and cannot do".

Selected by `config.TRAJ_SPACE` (`"2d"` default, `"3d"`), or per call
(`build_trajectory(space=...)`, `contact_candidates(space=...)`, `translate_track(trajectory_space=...)`,
CLI `--trajectory-space {2d,3d}`).  **Both routes stay in the tree**: the 2-D one is the fork's own
convention, pinned bit-for-bit by `test_m29`, and it draws the Review arc.  Nothing about the 2-D
path changed except that `TrajConfig`/`ContactConfig` grew the flags that select the new one.

## What each phase became

| phase | file | what it does now |
|---|---|---|
| 1 ray primitives | `pool/geometry.py` | `TrajectoryPlane` (a **vertical** plane carried as `n·x = d`), `vertical_plane_from_points` (TLS fit in the horizontal plane, returns `ok`/`few_points`/`no_span`/`degenerate`), `intersect_curve_with_elevation` (where a parametric 3-D quadratic meets `z = z_target`, with `t_all` exposing both roots), `curve_point` |
| 2 curve fitting | `trajectory.py` | `robust_parabola(..., z=...)` — same RANSAC, three quadratics in world feet; `predict3`; `Trajectory` gains `space`, `arc_rms_ft`, `max_inlier_res_ft`, `plane`; `build_trajectory(space="3d")`; `refit(space="3d")`.  `fit_parabolic_trajectory` and `solve_frame_for_y` are **kept and documented as deprecated for measurement** |
| 3 candidates | `contact.py` | on the 3-D route #2 is `arc_waterplane` (the curve against `z = z_water`, not a row), #3 gates on `dZ/dt ≤ -0.02 ft/frame`, #4 compares observed `dZ/dt` to the fit at `0.40 ft/frame`.  `ORDER_3D` is the 3-D ranked four; `ORDER` stays the published 2-D one |
| 4 intersection | `pool/intersection.py` | `entry_coordinate(..., trajectory=...)` solves `Z(t) = water.z_ft`; `x_px` becomes an *output* (the reprojection of the 3-D entry), which is what keeps `cross_check_against_1d` meaningful.  `entry_with_height`/`parallax_bias` are untouched and still serve the single-pixel route |
| 5 orchestration | `pool/pipeline.py` | `track_to_world` unprojects the track (pixels → rays → water-height positions → one vertical plane → 3-D points), then `build_trajectory(space="3d")`; `translate_track(trajectory_space="3d")` reports `world`, `trajectory3d`, `arc_entry`, and — when a contact frame is confirmed — promotes the arc entry to `entry`, keeping the pixel answer as `entry_pixel` with `arc_vs_pixel_in` |

## The switch, where the operator meets it

**Review tab → `scoring method:` → *( ) Trajectory arc (2-D)  ( ) Trajectory plane (3-D)*.**

One control, and it moves three things at once because they all read `ReviewState.space`:

| | Trajectory arc (2-D) — default | Trajectory plane (3-D) |
|---|---|---|
| curve drawn | the image-space parabola (`trajectory.refit` / the fork's `curve`) | the fitted **world** curve projected back to pixels (`geometry.project_curve`) |
| number | `measure.plane_distance` — final detection assumed on the plane, `route="plane"` | `measure.trajectory_plane_distance` — where `Z(t)` meets the water, `route="arc3d"` |
| Results rows | the existing set | adds crossing frame, extrapolation, world-arc RMS, plane RMS |
| exports | the run's own artifacts | re-renders, because the run scored the other method |
| a manual click | answers | refuses with a warning — a click has no track to fit |

The fit is cached per (points, entry frame, document) so the canvas can be redrawn on every mouse
move without re-running RANSAC, and the **scored** curve is the *drawn* curve (both come from
`measure.trajectory_plane_fit`).  Before a run the choice is remembered and applied to it.

CLI equivalent: `--trajectory-space {2d,3d}`.

## Measured, on `tests/pool_scene.py` ground truth (30 fps, water at −1.625 ft)

Truth: a ballistic tail-set, crossing the water at **frame 27.97, X = 17.284 ft (207.4 in)**.

| what | result | where |
|---|---|---|
| the arithmetic, given genuinely 3-D points | **0.000 in** error; fitted `cz2 = −0.017874` against `−g/2 = −0.017874`; plane RMS 0.000 ft; reprojection 8e-13 px | `test_m30::test_the_3d_entry_is_exact_when_the_points_are_really_3d` |
| the single pass on the project's shallow camera (eye 5.6 ft, 17 ft out) | **+110.01 in**; plane RMS 2.598 ft; fitted heights −5.24…+1.24 ft (*below* the water) | `test_m30::test_the_single_pass_bootstrap_cost…` |
| iterating the plane fit (1, 2, 3, 5, 8 passes) | **identical** answer every time (X = 26.451 ft); RMS collapses to 0.000 ft on pass 2 — a fixed point, not a refinement | documented in `track_to_world` |
| the same route on a steep view (eye 60 ft) | plane RMS **0.095 ft — the gate passes** — and the entry is still **−72.8 in** wrong, because the single pass *under*-estimates height | `test_m30::test_a_steep_view_passes_the_plane_gate…` |
| the legacy single-pixel answer at the last detection | +65.1 in (reads long — the documented parallax direction) | `test_m30` prose |

**Conclusion as built.** The intersection arithmetic is exact and the plumbing is complete, but the
single-pass bootstrap it is fed is biased by tens of inches, and the plane-RMS gate only catches the
scattered case — the compressed case passes.  Every `route="arc3d"` record therefore carries
`flags["arc3d_unverified_bootstrap"]` and a note saying so.  It is not a distance; it is a
cross-check on the pixel route.

## Ranked ways to make it a distance (only #1 is cheap)

1. **Pin the free-fall acceleration.**  The bootstrap's only real error is Z.  `Z(t) = z0 + vz·t −
   (g/2)t²` with `g` known from the clip's fps, so the plane's azimuth is the single unknown: search
   it until the reconstructed `cz2` equals `−g/2`.  Additive to `track_to_world`, needs `fps`
   plumbed into the pipeline, no new data.  *Expected* to remove most of the bias — not yet
   measured, so treat that as a hypothesis, not a result.
2. **Stereo disparity (ZED X).**  Real per-frame depth, so no bootstrap at all.  Touches
   `detect.py`/`image_set.py` and the record schema; the largest job of the three and the only one
   that fixes the problem rather than constraining it.
3. **A supplied dog height.**  One operator number, or a per-frame assumption.  Cheapest to write,
   and it reintroduces exactly the parallax assumption the route exists to remove — so it is the
   fallback, not the fix.
