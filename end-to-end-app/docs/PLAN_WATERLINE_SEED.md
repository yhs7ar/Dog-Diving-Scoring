# PLAN — automatic plane placement from waterline detection

**Status:** draft for approval. No code written. To become PLAN.md §17 on approval
(PLAN §14/§15 are already behind HANDOFF's 2026-10-20 orders; HANDOFF is the amendment of record
until this lands).

**Order (verbatim intent, 2026-10-20):** *"I want the plane to be automatically placed by the
Waterline detection algorithm and manually adjusted by the user using the existing controls."*
Upstream: <https://github.com/quintinlevy/Waterline-Detection> (quintinlevy, `main`, created
2026-10-12, no LICENSE file).

## 0. Decisions taken before planning (answers to the 2026-10-20 question set)

| # | decision |
|---|---|
| 1 | **Seed the edges only.** Detection writes the four starting corners; there is no metric solve. The operator finishes with the existing corner drag / side drag / arrow pad. |
| 2 | **Three edges, not one.** Near waterline (Y=W), far waterline (Y=0) and the dock contact (X=0). The algorithm is to be extended in our adapter where it cannot find an edge itself. |
| 3 | **Vendor `detect_waterline.py` unmodified** under `src/third_party/waterline/` with a sha256 manifest; all new logic lives in a thin adapter that imports it. No fork of the upstream file. |
| 4 | **On demand only** — one button. Never on clip load; never silently over an aligned plane. |
| 5 | **Calm window auto-picked** (frames before the first accepted tailset), displayed, editable. |

Two things this plan decides *for* you unless you say otherwise: a failed detection **refuses to
place** and logs the failing numbers (it does not seed "best effort"); and pressing the button
twice is undoable (**Undo seed** restores the pre-seed corners), because the app has no undo today.

## 1. Verified starting state (read from the tree, not from the summaries)

- Shortcut `Dog Diving scorer.lnk` → `run_app.cmd` → `src/app_gui.py`, interpreter
  `C:\Users\harri\dd-venv\Scripts\python.exe` (the project's own `.venv` cannot run from `G:`).
- `src/dog_diving/pool/plane.py` — `PlaneSession`: corners in **image px**, `fit_to_frame()`,
  `click/drag_to/release` (corner or side), `select()` + `nudge()` (survives release; arrow pad
  steps 1 / 10 / 0.1 image px), `build()` → `dog_diving_pool_calibration/v1` route
  `A_correspondences`, `validate_plane()` = **P1–P4** (quad sane, homography conditioned,
  mapping round-trips, foreshortening: fewer px/ft at the far end).
- Metric frame: `X = 0` **on the dock lip**, `+X` down the pool, `Y = 0` on the **far/back** edge,
  `Y = W` near. `config.POOL_LENGTH_FT = 35.0` is a **fixed axis** (`plane.METRIC_LENGTH_FT`); the
  operator entry is an **anchor** (`+x anchor (ft)`), `config.POOL_WIDTH_FT = 16.0` is the width.
- Corner tags `("00","10","11","01")` = (0,0), (L,0), (L,W), (0,W); sides `back/right/front/left`.
  `fit_to_frame()` currently hard-codes the dock on the **left** and both X=const edges vertical.
- App wiring: `calib_frame_idx` + `read_frame(self.clip, idx)`; handlers `on_start_plane`,
  `on_plane_fit`, `on_plane_clear`, `on_plane_dims`, `_draw_plane_overlay` (Tk **vectors**, so line
  widths are screen px at any zoom), `_redraw_plane`, `_sync_plane_widgets`.
- Measurement (unchanged by this plan): `measure.plane_distance` maps the final accepted tail-set
  through `doc["homography"]`; Review draws from the same matrix.
- Detection already runs in-app: `image_set.detect_frames` (plain `predict`, no tracker) via
  `detect.detect_clip`.
- **No human has aligned a plane on real footage yet** (PLAN §15.4). That is why §3 starts with a
  manual reference, not with the seed.

## 2. What the upstream detector gives, and what it does not

`detect_waterline.py` (cv2 + numpy only — both already in `dd-venv`) is a pipeline of standalone
functions: `probe_video`, `resolve_frame_range`, `load_frames`, `compute_noise_maps` /
`compute_noise_map`, `fit_normalization` + `normalize_noise_map`, `threshold_noise_map`,
`clean_mask`, `scan_boundary` / `extract_waterline`, `waterline_profile`, `waterline_stability`,
`separability`, `lighting_drift`, `component_report`, `fit_straight_line`, `draw_*`,
`save_outputs`. Output: water points in **original pixels**, one boundary per scan, plus a Huber
line fit (`slope`, `intercept`, `endpoints`, `rms_residual`, `rms_center`/`rms_ends` for
distortion, `angle_deg`) and confidence material (`method_comparison` Fisher per noise method,
`threshold_used`, `stability`, `glare_check`, `warnings`).

It does **not** give a plane, a scale, or a second/third edge:

- `extract_waterline(..., water_side=...)` returns **one** boundary, chosen globally. `"below"` →
  one point per column (bottom boundary); `"above"` → top boundary; `"left"/"right"` → one point
  per row (a near-vertical edge). It cannot return the far *and* near lines in one pass, and it
  has no notion of "which of these is the dock".
- `fit_straight_line(points, water_side)` already handles the near-vertical case
  (`x = slope*y + intercept` when `water_side in ("left","right")`) and reports the end-vs-centre
  residual split — exactly the two tools the dock edge needs.
- Nothing in it knows about metric feet, and `estimate_water_plane.py` (which does: ZED depth +
  IMU gravity → `n·X = d`) is unreachable from these `.mp4` clips.

**So the work is an adapter that turns one noise mask into three classified, fitted edges**, then
a geometry step that turns three lines into a quad. Neither step is a rewrite of upstream.

## 3. Why three edges are (almost) enough — the geometry, stated

Let the image lines be `L_far` (image of Y=0), `L_near` (Y=W), `L_dock` (X=0). Then:

- `p00 = L_far ∩ L_dock` is the image of **(0,0)** — the metric origin, on the dock lip, at the
  water's inner corner. `p01 = L_near ∩ L_dock` is the image of **(0,W)**.
- `VP_X = L_far ∩ L_near` is the **vanishing point of the X direction** (the two lines are parallel
  in the water plane). This is what makes perspective extrapolation possible at all.
- Remaining freedom: **two** DOF out of a homography's eight after six line constraints —
  (a) **where X = 35 ft lands** (the anchor), and (b) the convergence of the two X=const image
  edges (the Y-direction vanishing point, `VP_Y`), which no observable edge in these clips
  supplies.
- `fit_to_frame()` already makes the affine assumption that (b) is at infinity (both X=const edges
  vertical). The seed keeps that assumption and inherits **L_dock's direction** for the X=L edge,
  which leaves exactly **one** free parameter: the anchor. That parameter already has two
  first-class controls — the `right` side drag and the `+x anchor (ft)` entry — so
  "automatically placed, manually adjusted" is a one-gesture finish, not a re-do.

Construction (no new dependency; 2-D homogeneous maths, numpy):

1. `p00 = L_far ∩ L_dock`, `p01 = L_near ∩ L_dock`, `VP_X = L_far ∩ L_near`.
2. Take the **projective fraction** of `10` along `[p00 → VP_X]` from the quad that is currently in
   the session (i.e. today's `fit_to_frame()` default), and reuse that fraction on the detected
   `L_far`. This transplants the default's foreshortening instead of guessing a frame fraction, so
   the result passes P4 by construction rather than by luck.
3. `11` = `L_near ∩` the line through the new `10` with `L_dock`'s direction.
4. Assign tags by **detected dock side**: dock left → `00` at the left end of `L_far`; dock right →
   mirrored. Tags follow semantics, never image side.

Everything else (P1–P4, `build()`, the document, `measure.plane_distance`, Review) is untouched.

## 4. Architecture

```
src/third_party/waterline/
    detect_waterline.py        copied byte-identical from upstream (commit recorded)
    SHA256SUMS                 sha256 of the file + of this plan's upstream commit
    PROVENANCE.md              repo URL, commit, fetch date, author, "no LICENSE file" note
    upstream_tests/            upstream tests/ copied verbatim, run once as a provenance check

src/dog_diving/waterline.py      NEW  calm window, noise mask, THREE-edge extraction + classification,
                                    confidence, WaterlineEdges result; imports the vendored module
src/dog_diving/pool/seed.py      NEW  lines -> quad (step list in §3), fallback ladder, gates,
                                    SeedResult (lines, level, confidence, refusal reason)
src/app_gui.py                 +    on_detect_waterline, on_undo_seed, seed overlay in
                                    _draw_plane_overlay, log lines, provenance into plane.build()
```

The adapter must not modify the vendored file, so the extensions live in `waterline.py`:

- `top_bottom_boundaries(mask, roi_mask)` — per column, the **first and last** water pixel (upstream's
  `scan_boundary` returns one of them); returns two point sets.
- `runs(points, max_gap_px)` — split a boundary into contiguous runs; a glare blob, a lane rope or a
  lane divider breaks the water mask, and a single global fit would bridge the gap.
- `fit_run(run, water_side)` — upstream `fit_straight_line` per run; keep the longest run whose
  `rms_residual` ≤ 3 px, record `inlier fraction`, `rms_center`/`rms_ends` (distortion signature).
- `classify(far, near, dock_candidates, dock_side)` — far = widest top run, near = widest bottom
  run, dock = the most perpendicular-to-`L_far` vertical run on the dock side (rejecting a run that
  hugs the frame border, per upstream's "boundaries on the ROI edge are crop artefacts" rule).

## 5. Algorithm, end to end

1. **Calm window.** Run/derive the detection results already associated with the clip; the window
   is `[0, first_accepted_tailset − margin)` at the clip's fps, clipped to
   `[60, 150]` frames (≥ 2 s for the ripple period — upstream's own `auto_stability_window`
   rationale; 90-frame windows gave ~0.5 px jitter vs 2.5-5 px at 30). No accepted detection at
   all → first 90 frames. The chosen range is shown in the Calibrate pane and editable (two entry
   boxes), and the log states why it was chosen.
2. **Load** at `scale 0.5`, `step 1`, grey, into one `uint8` stack (150 × 540 × 960 ≈ 78 MB). All
   coordinates are mapped back to original pixels by the adapter (`points / scale`), so the plane's
   px stay comparable with `read_frame`.
3. **Noise map + mask**: `compute_noise_maps`, pick the method with the highest `separability`
   Fisher over `variance/std/detrended/diff` (upstream's own comparison; `detrended` is its
   default and the right prior here), Otsu threshold, `clean_mask` (open 5, close 15).
4. **ROI**: whole frame by default; if a plane is already in the session, offer its quad shrunk by
   a gutter as the ROI — that is the honest use of an aligned plane as a *search region*, not as a
   gate.
5. **Three boundaries**: `top_bottom_boundaries` for far/near; the two row-scans (`water_side`
   `left` and `right`) restricted to the dock side for the dock contact.
6. **Runs + fits + classification** as in §4; per-edge confidence = (inlier fraction, RMS,
   stability std) and a `found` flag.
7. **Dock side**: try both sides; the side with the more confident vertical run wins; the first
   accepted tail-set detections (the dog starts on the dock) are a **cross-check printed in the
   log**, not a gate — the sign of X is too important to rest on a detection that can fail.
8. **Seed** (§3), then `validate_plane` (P1–P4) + the new gates (§6).
9. **Place**: write the four corners into the session, select the anchor side, `_redraw_plane`,
   log one line with the level, each edge's RMS, the dock side, and the anchor used.

## 6. Gates — each must be able to fail, and none may be circular

| gate | test | fail |
|---|---|---|
| **S1** line quality | per found edge: inlier fraction ≥ 0.60, `rms_residual` ≤ 3 px, per-column stability std ≤ 3 px, Fisher of the chosen method ≥ floor set in §10 | that edge is "not found" |
| **S2** mask sanity | water fraction in ROI ∈ [0.05, 0.95]; exactly one dominant high-noise component (else the mask split on glare — upstream warns) | refuse |
| **S3** pairwise geometry | `L_far` and `L_near` not parallel in the image (‖cross‖ > ε) so `VP_X` is finite; `VP_X` outside the near/far band; `L_dock` not parallel to `L_far` | refuse (or drop to a lower level) |
| **S4** quad | P1–P4 on the seeded quad; every corner finite; the water side of `L_far` contains the quad | refuse |
| **S5** area | seeded quad area within 0.3–3.0× the default quad's area (catches a mask that latched onto glare) | refuse |
| **S6** independent re-detect | second, disjoint calm window (e.g. the next 90 frames) must give each edge within **k px** (k = 6 × the first window's stability std, floored at 8 px) | warn + record, do not refuse |

**Circularity rule, written down so it is not re-invented:** comparing the final quad to the line it
was seeded from scores ~0 px by construction — the same class of bug as P4 scoring 1.00 on a
reversed plane. That number may be *displayed* as "seed drift after your edits", never as a gate.
The only independent check on this route is S6 (different frames) and, in the longer run, the
painted ruler (§12).

## 7. UI (Calibrate pane, next to Start plane / Fit to frame / Clear plane)

- **`Detect waterline`** button. Disabled until a clip is loaded. Pressing it: seeds and logs;
  on refusal it logs the failing gate and its measured value, and **leaves the session exactly as
  it was** (today's plane, or none).
- **`Undo seed`** button, enabled only right after a seed: restores the pre-seed corners verbatim
  (one stored tuple, no history machinery).
- Overlay: the three detected lines (clipped to their runs), inlier tick marks, the seed quad drawn
  in the *seed* colour so it is distinguishable from the operator's quad; all via
  `_draw_plane_overlay`'s existing Tk-vector path, so widths stay 1–2 screen px at every zoom and a
  nudge still costs zero image renders.
- Log line, one per detect, fixed shape so it can be diffed across clips:
  `seed L3 | far rms 1.8 inl 0.91 | near rms 2.4 inl 0.74 | dock rms 1.1 inl 0.66 | side right |
   window 0-90 | anchor 25.0 ft (default) | fisher 2.9 (detrended) | gates PASS`.
- No new tab, no new frame-load behaviour, no auto-run on load (§0.4).

## 8. Document and provenance

`plane.build()` gains one block (absent for hand-placed planes, so old documents stay valid):

```
"waterline_seed": {
  "upstream": "quintinlevy/Waterline-Detection@<commit>",
  "file_sha256": "...", "params": {...}, "method": "detrended", "threshold": 118.0,
  "window": [0, 90], "roi": "frame" | "plane",
  "edges": {"far": {...rms, inliers, stability...}, "near": {...}, "dock": {...}},
  "dock_side": "right", "level": "L3", "anchor_ft": 25.0,
  "placement": "auto" | "auto+edited"   # flipped on the first drag/nudge after a seed
}
```

`placement: auto+edited` is set by the first `drag_to`/`nudge` after a seed, so a reviewer can tell
a seeded plane from one the operator moved. `measure.plane_distance` and the P/W gates are not
touched; nothing reads this block except reporting and the log.

## 9. Fallback ladder

| level | edges found | what the operator gets |
|---|---|---|
| **L3** | far + near + dock | all four corners placed; adjust the anchor (side drag or `+x anchor`) and correct any bad edge |
| **L2** | far + dock | near edge transplanted from the default quad by projective fraction; width from the entry |
| **L1** | far + near | dock edge transplanted (default's X=0 direction/fraction); **X=0 is then a default, not a measurement** — flagged in the log |
| **L0** | far only | current `fit_to_frame()` with the far edge snapped onto the detected line; log says "seeded far edge only" |
| **—** | none / S2/S3/S4/S5 | **refuse**: nothing changes, reason logged |

## 10. Verification plan (ordered; nothing below may be skipped)

**Phase 0 — a reference exists first.** You align one plane by hand on one real clip (the app's
current workflow, `18-07` and `video004 … seg01` are the two candidates — different venues, dock on
opposite sides). Store it as `data/calib/`. Without this there is nothing to score a seed against,
and PLAN §15.4's "no human has aligned a plane" stays true.

**Phase 1 — offline measurement, before any UI.** Run the **upstream CLI** (unmodified) on three
clips (`18-07`, `24-00`, `video004 … seg01`) at `--scale 0.5`, 90–150 frames; record for each:
chosen method + Fisher, threshold, line RMS, inlier fraction, stability, glare/drift warnings,
runtime. This is where the Fisher floor (S1) and the scale/step defaults are set from measured
numbers, not guesses. Also: does the far waterline even get found on the indoor arena clips
(dark-on-dark, H.264 noise)? If not, the whole feature is worth reconsidering before any
integration — 30 minutes of work that can cancel the rest.

**Phase 2 — adapter + geometry, offline.** `WaterlineEdges` + `seed()` against the Phase-0
reference: per-corner error in **px**, and the resulting `X` at the reference's entry point in
**inches**, versus (a) the hand-aligned plane and (b) `fit_to_frame`. The seed must beat the default
on both clips; a seed that only equals the default is not worth shipping.

**Phase 3 — UI wiring**, then the same numbers reproduced through the GUI.

**Phase 4 — suite.** `src/tests/test_m21_waterline_seed.py`, ~20 tests:

- geometry: `p00/p01/VP_X` intersection; cross-ratio transplant reproduces the default quad when the
  detected lines *are* the default quad's lines (identity check); degenerate parallel far/near →
  refusal; bow-tie → P1 fail; reversed quad (dock on the right) → P4 still passes and tags are
  correct.
- adapter: `top_bottom_boundaries` on a synthetic mask with a known top/bottom edge (exact y); run
  splitting across a planted gap; a glare blob in the corner must not become an edge; `fit_run`
  rejects a run below the inlier floor.
- window: frames at/after the first accepted tailset are excluded; no detections → first 90;
  requested 30 frames → clamped to ≥ 60.
- gates: each of S1–S5 driven to fail by one planted defect; S6 disagreement warns and does not
  refuse; the circular "seed drift" number is absent from the gate set.
- provenance: `waterline_seed` present after a seed and absent after `Clear plane`; `placement`
  flips on the first nudge; `plane_distance` returns the identical number before and after a seed
  (the seed must not touch measurement).
- regression: `test_m18_plane`, `test_m19_measure`, `test_m20_plane_placement`,
  `test_m8_pool_handlers` unchanged; full suite with `--ignore=src/tests/test_m8_gui_e2e.py`
  (that file cannot run on this host — sympy crash, not a regression).

**Phase 5 — acceptance.**
1. Seeds pass S1–S5 on the clips where the reference exists, and corner error vs the hand-aligned
   plane is ≤ 15 px at every corner (re-measure and set this number in Phase 2).
2. The resulting `X` at the entry point is within 2 in of the hand-aligned plane's `X`, and never
   worse than `fit_to_frame()` on either clip.
3. Zero change to manual behaviour: with no seed, every corner/side/arrow/zoom gesture behaves
   exactly as today.
4. A refusal never mutates the session.

## 11. Risks, with the mitigation each one needs

| risk | why it bites here | mitigation |
|---|---|---|
| **No calm window** | handler walking on the dock and the dog on the dock at frame 0 both raise temporal noise; upstream's own caveat is that motion corrupts the map | dock/wall ROI, window before the first accepted tailset, S6 re-detect, refusal path |
| **Banner skirts hide the water/dock contact** | confirmed in `18-07` and `24-00` (skirt hangs into the water) | that is precisely why L3 is not required; L2/L1 exist and the log names the level |
| **Dark water, dark walls** | the noise contrast between water and structure can vanish indoors; H.264 noise makes static pixels look alive | method chosen by Fisher, manual-threshold control exposed in the pane, S1 gates, Phase 1 can cancel the feature |
| **Near waterline at/below the frame edge** | the near coping is at the bottom of these frames | L2 fallback; the frame padding (35 %) already allows a corner outside the frame, but a *line* found off-frame is not a measurement — flag it |
| **Dock side varies per clip** | `fit_to_frame` assumes dock-left; `video004` is dock-right | side from the dock-edge fit, cross-checked by the first detections; tags, not image sides, carry X |
| **Two-homography style confusion** | a seed that writes `doc["homography"]` from anything other than the four corners would silently change the score | seed writes **corners only**; `build()` stays the single place that makes the matrix |
| **Gate that cannot fail** | the seed-vs-quad comparison is ~0 px by construction | §6 circularity rule + a test asserting that number is not in the gate set |
| **Licence** | upstream has **no LICENSE file** → default "all rights reserved", even for a teammate's repo | `PROVENANCE.md` records author/commit/date; **ask Quintin to add an MIT/Apache LICENSE** before this ships outside the team; vendoring as-is is the interim |
| **Runtime/memory** | 150 × 1920 × 1080 grey ≈ 311 MB, and `compute_noise_maps` builds four float maps | `scale 0.5` + ≤150 frames + method chosen before the full map set is computed; Phase 1 measures wall time and it goes in the log |
| **Provenance of the copy** | this tree has other writers and stale copies | sha256 in `SHA256SUMS` + the same hash in the document block; a mismatch is a visible error, not a footnote |

## 12. Out of scope (explicitly)

- **No metric solve**: X=0 comes from `L_dock`, the anchor stays the operator's/ default's. The
  painted ruler as a scale oracle is a *possible* later phase (it would fix the anchor's last DOF
  from two typed tick values) and would finally give this route the independent check it lacks —
  not in this plan.
- **No ZED route**: `estimate_water_plane.py` (plane `n·X=d` from depth + IMU) is the right answer
  for the capstone rig, and wrong for `.mp4` clips that carry neither depth nor IMU.
- **No changes to** `measure.plane_distance`, Review, the P/W gate meanings, the 1-D archive, the
  detection path, or the vendored detector file.
- **No OCR of the ruler numerals**, no auto-run on clip load, no new tab, no batch/CLI mode.

## 13. Order of work

| step | content | size |
|---|---|---|
| 1 | Phase 0 reference plane + Phase 1 upstream-CLI measurement on 3 clips (go/no-go) | S |
| 2 | vendor + `SHA256SUMS` + `PROVENANCE.md`, run upstream tests once | S |
| 3 | `dog_diving/waterline.py`: window, boundaries, runs, fits, classification | M |
| 4 | `dog_diving/pool/seed.py`: geometry, ladder, gates | M |
| 5 | Phase 2 measurement vs the reference, tune the S1 floors and the anchor fraction | S |
| 6 | GUI: button, undo, overlay, log, `waterline_seed` block | M |
| 7 | `test_m21_waterline_seed.py` + full suite (one run, `--ignore=test_m8_gui_e2e.py`) | M |
| 8 | Phase 5 acceptance table in HANDOFF + fold this into PLAN §17 | S |

Steps 1–2 are the cheap kill-switch: if the far waterline is not reliably found on the indoor arena
clips, nothing after step 2 should be built.
