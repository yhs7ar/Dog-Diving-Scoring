# Test Bench — ROADMAP

A companion application to `end-to-end-app` for **comparing trajectory-scoring
methods against ground truth and a human judge**, built as a self-contained copy
of the parent's detection/plane/scoring machinery with a new orchestration,
annotation, and metrics layer on top. No recording, no long-term flexibility —
it is a bench, not a rig.

Status: **plan (unbuilt).** Nothing under `test-bench/` exists yet except this file.

---

## 1. Purpose

Given a prepared clip (MP4, or SVO2 when the ZED X SDK is installed), the test
bench:

1. ingests the clip into a self-contained `analyzed-clips/` record;
2. confirms pool-plane geometry (re-usable from the parent app's plane route);
3. runs frame-by-frame YOLO pose detection → an accepted tailset trajectory;
4. lets a technician place **ground-truth** (the true tailset∩water intersection)
   and optionally enter the **human judge's** distance;
5. runs an entire **test suite** of scoring methods over the prepared clips;
6. reports, per clip and per method-config, three error metrics — **pixel error**,
   **score error**, **judge error** — plus their averages;
7. shows a Review-style table where any clip can be loaded for visual comparison,
   with toggleable overlays for every test run on that clip.

What it is **not**: no recording, no live ZED capture, no score ledger for
competition day, no adaptive-detection ladder, no long-term schema stability
guarantee. SVO2 is read-only input, and only when the ZED SDK is present.

---

## 2. Locked decisions (from clarification, 2026-11-10)

| Decision | Choice |
|---|---|
| Code reuse | **Copy** the parent's modules verbatim into `test-bench/` (self-contained, like the `dog-diving-trajectory` fork). No cross-app import. |
| Per-clip metrics | Three error columns: **pixel error** (auto entry point vs placed GT point), **score error** (auto distance vs GT-point distance, inches), **judge error** (judge distance vs GT-point distance, inches). |
| v1 scoring method | **2-D image-space parabola** only. The method registry is **modular and standardized**, modeled on `measure.py`'s output contract, so 3-D arc and point-cloud slots drop in later. |
| Interface | **Tk GUI + headless CLI** (mirrors `app_gui.py` + `cli_run.py`). |
| Clip storage | **Map by reference** — `source.original_path` points at the file in place; nothing is copied into `analyzed-clips/`. |

Derived from the metrics decision — the three quantities every scoring result is
compared against:

- **GT point** `(x_px, y_px, frame)` — a pixel the technician places.
- **GT distance** `gt_in` — the GT pixel unprojected through the plane
  homography (the parent's `entry_coordinate` already returns this as `distance_in`).
- **Judge distance** `judge_in` — an optional human-entered number in inches.

`judge_in` is **not** compared to the auto score directly; both the auto score
and the judge are measured against the GT distance. This keeps one reference
(the GT point) and two estimators (auto, judge) — the comparison the bench exists
to make.

---

## 3. What the end-to-end-app already provides (capability assessment)

Everything below exists in `end-to-end-app/` and is copied, not re-implemented.

### 3.1 Detection — `dog_diving/detect.py`
`detect_clip` runs the YOLO pose model with ByteTrack, applies the dominant-track
filter, and emits per-frame records shaped
`{frame, tailset:[x,y,conf], nose:[x,y,conf], box_conf, track_id, accepted}`.
`DetectConfig` carries `model_path` and is part of the cache key, and
`DetectResult.model_sha` records provenance. **Known caveat (from the parent's own
records):** the dominant-track filter can lock onto the handler instead of the dog
— the test bench must surface this rather than fix it, because a scoring bench
must not quietly change the detection pipeline it is grading.

### 3.2 Trajectory — `dog_diving/trajectory.py`
`build_trajectory(recs, ...)` produces the trajectory object with `.points`,
`.coeffs` (2-D parabola), `.airborne_frame_range`, and inlier/outlier splitting
(`ARC_OUTLIER_PX`, `TRAJ_MAX_FRAME_GAP`, `TRAJ_SEED`). `space="3d"` fits world
feet instead of pixels. This is the fit layer the v1 method parameterizes.

### 3.3 Measurement / scoring interface — `dog_diving/measure.py` (the reference)
This is the module the test bench's scoring registry is modeled on. It already
defines **two methods behind one output contract**:

- `plane_distance(...)` → `route="plane"` (final detection assumed on the plane);
- `trajectory_plane_distance(...)` → `route="arc3d"` (3-D world arc crossing).

Both return **the same schema**: `{route, total_inches, display, distance_ft,
transverse_ft, frame, x_px, y_px, point_source, point_origin, warnings,
excluded_frames, entry_override, kp_conf, water_z_ft, homography_agreement_ft,
flags, ...}`. The test bench's `ScoreResult` (below) is a superset of this — the
"standardized connection" the user asked for, so a new method only has to emit
this shape.

### 3.4 Plane / world mapping — `dog_diving/pool/`
The plane document (`dog_diving_pool_calibration/v1`, `route: A_correspondences`)
carries a `homography` (3×3, pixel → `(X,Y)` feet), `plane.corners`, `water.z_ft`,
`lens`, `extent`, `pose`. `intersection.entry_coordinate(pose, water, px,
homography=H, ...)` returns `{distance_in, X_ft, Y_ft, x_px, y_px, flags,
notes}` — exactly the pixel→inches conversion the GT distance needs, and the
single source of truth the metrics must use (never a re-derived pose; the
parent's documented two-homographies bug).

### 3.5 Model selection — `dog_diving/config.py`
`MODEL_REGISTRY` (`ModelSpec` dataclass), `resolve_model_path`, `default_for_scale`,
`MODEL_PATH`/`MODEL_SPEC`/`MODEL_SCALE`, and the `DD_MODEL` / `DD_MODEL_SCALE`
env selectors. The test bench reuses this unchanged so "re-detect with a different
YOLO model" is one `ModelSpec` switch, and the registry already records how each
checkpoint measured.

### 3.6 Camera metadata (MP4) — `dog_diving/video.py` + `result.json`
The parent already probes and persists the exact MP4 metadata the bench needs in
`result.json`'s `clip` block: `width, height, fps, codec, n_frames, vfr,
declared_vs_decoded, sha256`. The bench's `meta.json` copies this block.

### 3.7 Review UI pattern — `src/app_gui.py`
`ReviewState` (space switch, `measurement()`, `arc()`, `toggle_point()`,
`manual_point`, `excluded`), the Review canvas, and the results tree are the
template for the comparison pane. The frame canvas + zoom/pan + overlay drawing
(`annotate.py`, and the `image-measurement-ui` zoom work) transfer directly.

### 3.8 SVO2 — `dog_diving/zed_recorder.py` / `third_party/waterline/zed_stub.py`
SVO2 read is only available when the ZED SDK is present; `zed_stub` is the
no-SDK fallback. The bench treats SVO2 as an optional input path that raises a
clean "SDK absent" notice otherwise.

### 3.9 Not copied / intentionally dropped

Trimmed at M1 (2026-11-10) after building the import graph, so every module the
bench keeps still imports. Dropped — nothing the bench keeps imports them, and
they serve features the bench excludes:

  * `zed_recorder.py` — recording (the bench has no recording).
  * `fusion.py`, `fused_trajectory.py`, `pc_datum.py`, `pc_distance.py`,
    `pc_trajectory.py`, `pc_water_plane.py` — the ZED point-cloud route
    (deferred; needs the ZED SDK. §8 reserves its registry slot).
  * `ledger.py`, `report.py` — the parent's score ledger / report export
    (the bench keeps its own results + metrics table).
  * `calibration_capture.py`, `stability.py` — calibration-capture / stability CLIs.
  * `measure_final.py`, `measure_new.py`, `measure.py.bak` — superseded drafts of
    `measure.py` (removed so there is exactly one canonical scoring module).

Kept verbatim, contrary to an earlier draft of this section:

  * `calibration.py` and `contact.py` — the import graph shows `annotate.py`
    imports `contact`, and both `annotate.py` and `distance.py` import
    `calibration`; they are load-bearing, not dead.
  * `measure.py` and `detect.py` are kept **unmodified** — no "1-D mark route"
    deletion, no ladder rewrite. The bench pins behaviour through `config.py`
    and the registry, never by editing a copied module.

---

## 4. Repository layout

```
test-bench/
├── ROADMAP.md                     # this file
├── run_app.cmd  /  run_app.bash   # launcher (owns interpreter path, per parent)
├── models/                        # copied .pt checkpoints + PROVENANCE
├── planes/                        # saved plane geometry (shared, like parent)
├── analyzed-clips/                # self-contained clip records (see §5)
│   ├── <clip_id>/
│   │   ├── clip.mp4               # the copied clip (or clip.svo2)
│   │   ├── manifest.json          # record of record (see §5.1); plane.setup_id
│   │   │                          #   points at planes/<id>.pool.json
│   │   ├── detections.json        # frame-by-frame records (parent tracks.json shape)
│   │   ├── trajectory.json        # built trajectory (parent trajectory.json shape)
│   │   ├── ground_truth.json      # {frame, x_px, y_px, gt_in, judge_in}
│   │   └── meta.json              # {width, height, fps, codec, ...}
│   └── results/
│       └── <run_id>.json          # one suite run's per-clip rows + aggregates
│                                  # (global: a run spans clips, so rows live here)
└── src/
    ├── app_gui.py                 # NEW — test-bench GUI
    ├── cli_run.py                 # NEW — headless select/calibrate/analyze/run-suite
    ├── pool_calibrate_tool.py     # COPIED — the parent's semi-manual plane calibration (cv2)
    ├── calibrate_tab.py           # EXTRACTED — the parent's Calibrate tab (Tk), embedded
    ├── test_bench/                # NEW package
    │   ├── __init__.py
    │   ├── manifest.py            # analyzed-clips record schema + read/write
    │   ├── ingest.py              # mp4/svo2 -> analyzed-clips (copy + probe)
    │   ├── ground_truth.py        # GT point + judge entry, pixel->inch
    │   ├── meta.py                # MP4/SVO2 camera-metadata probe
    │   ├── scoring/
    │   │   ├── __init__.py
    │   │   ├── registry.py        # method + axis registry, ScoreResult contract
    │   │   └── arc_2d.py          # v1: 2-D image-space parabola
    │   ├── suite.py               # axis-grid generator + runner
    │   ├── metrics.py             # pixel/score/judge error + averages
    │   └── compare.py             # comparison view-model (overlays per run)
    ├── dog_diving/                # COPIED verbatim (trimmed, see §3.9) from end-to-end-app
    │   ├── detect.py  trajectory.py  video.py  measure.py  distance.py
    │   ├── waterline.py  annotate.py  pose_layout.py  config.py  image_set.py
    │   ├── calibration.py  contact.py  replay.py
    │   └── pool/  (calib, geometry, intersection, lens, pipeline, plane,
    │               homography, surface, vanishing, grid, seed)
    ├── third_party/               # COPIED verbatim — vendored waterline detector
    │   └── waterline/  (detect_waterline.py, detect_waterline_lite.py,
    │                    estimate_water_plane.py, zed_stub.py, zed_real.py, ...)
    └── tests/                     # ported pins + new bench tests
```
```
Copying rule: the `dog_diving/` and `third_party/` subtrees are **byte-identical**
to the parent's for the modules listed (verified by `diff`, as the trajectory
fork did), so the parent's tests continue to describe them. Only
`src/test_bench/`, `app_gui.py`, and `cli_run.py` are new.

---

## 5. Data model

### 5.1 `manifest.json` — the record of record

```json
{
  "schema": "test_bench.manifest/1",
  "clip_id": "video004_right-seg01",
  "source": { "original_path": "...", "sha256": "...", "kind": "mp4" },
  "meta": { "width": 960, "height": 600, "fps": 60.0, "codec": "mpeg4",
            "n_frames": 92, "vfr": false, "declared_vs_decoded": [92, 92] },
  "plane": { "path": "plane.json", "setup_id": "...", "present": true },
  "detections": { "path": "detections.json", "model": "yolo26s-pose_dogs_inflated_clahe_720p.pt",
                  "model_sha": "...", "n_accepted": 33, "config": { "imgsz": 1280, "conf": 0.12, "kp_conf": 0.30 } },
  "trajectory": { "path": "trajectory.json" },
  "ground_truth": { "path": "ground_truth.json",
                    "frame": 85, "x_px": 532.9, "y_px": 311.0,
                    "gt_in": 98.0, "judge_in": 96.5, "judge_present": true },
  "results": [ "results/<run_id>.json" ]
}
```

### 5.2 `ground_truth.json`

```json
{ "schema": "test_bench.ground_truth/1",
  "frame": 85, "x_px": 532.9, "y_px": 311.0,
  "gt_in": 98.0, "gt_in_source": "plane_homography",
  "judge_in": 96.5, "judge_present": true, "updated": "..." }
```

`gt_in` is **always recomputed** from `(x_px, y_px)` through the plane document's
own `homography` (via `intersection.entry_coordinate`) — never stored as a
free-standing number — so re-drawing the plane re-derives the GT distance. The
stored `gt_in` is a cache, not a source of truth.

### 5.3 Scoring result — the standardized contract (`ScoreResult`)

Modeled on `measure.py`'s output; every method emits this one shape:

```json
{
  "method": "arc_2d",
  "route": "arc_2d",
  "config": { "n_points": 8, "selection": "last_n" },
  "status": "ok",                    // ok | no_calibration | few_points | no_fit | no_crossing | ...
  "entry_px": [532.9, 311.0],        // scored water-entry point (image space)
  "entry_frame": 85.0,
  "distance_in": 98.02,              // world distance via plane homography
  "distance_ft": 8.168,
  "x_px": 532.9, "y_px": 311.0,      // measure.py aliases kept for parity
  "warnings": [],
  "fit": { "coeffs": {...}, "rms_px": 7.68, "n_points_used": 8 }
}
```

A new method (3-D arc, point-cloud) is **one function** conforming to this
contract, registered in `registry.py`. No changes to the suite runner, the
metrics, or the table.

### 5.4 Suite config and results

```json
// suite request (CLI arg file or GUI "Run suite" panel)
{
  "clips": ["<clip_id>", ...],        // or "all prepared"
  "methods": [
    { "method": "arc_2d", "n_points": 4,  "selection": "last_n" },
    { "method": "arc_2d", "n_points": 8,  "selection": "last_n" },
    { "method": "arc_2d", "n_points": 12, "selection": "all" },
    { "method": "arc_2d", "n_points": 12, "selection": "ransac_inliers" }
  ]
}
```

```json
// results/<run_id>.json — per clip row + aggregates
{
  "run_id": "2026-11-10T12-00-00",
  "config": { ...the request... },
  "rows": [
    {
      "clip_id": "video004_right-seg01",
      "gt": { "frame": 85, "x_px": 532.9, "y_px": 311.0, "gt_in": 98.0,
              "judge_in": 96.5, "judge_present": true },
      "runs": [
        { "method": "arc_2d", "n_points": 4, "selection": "last_n",
          "pixel_error_px": 3.2, "score_error_in": 1.1, "judge_error_in": 1.5,
          "score": { ...ScoreResult... } },
        ...
      ]
    }
  ],
  "averages": {
    "overall": { "pixel_error_px": 4.1, "score_error_in": 1.9, "judge_error_in": 2.2 },
    "per_method": { "arc_2d": { "pixel_error_px": 4.1, "score_error_in": 1.9 } }
  }
}
```

### 5.5 Metric definitions (the bench's deliverable)

For a clip with GT point `g` and judge `j`, and a scored result `s`:

- **pixel error** `= ‖ s.entry_px − (g.x_px, g.y_px) ‖₂` (pixels)
- **score error** `= | s.distance_in − g.gt_in |` (inches)
- **judge error** `= | j.judge_in − g.gt_in |` (inches; `null` when no judge)

Average accuracy = the **mean** of each metric over the clips included in the run
(overall and per-method). Clips with `judge_present = false` contribute to the
pixel and score averages but are excluded from the judge average — and are still
fully comparable (auto vs GT), per the spec.

---

## 6. Workflows

### 6.1 Select clips (mp4 / svo2)
1. Choose a clip (`.mp4`, `.mov`, or `.svo2` when the ZED SDK is present).
2. Probe metadata (resolution, fps, codec, frame count) → `meta.json`.
3. Record the clip **by reference** — `manifest.json`'s `source.original_path`
   maps to the file in place (no copy into the repo).
4. Write a fresh `manifest.json` with `source.sha256` and an empty
   plane/ground-truth/result state.

### 6.2 Calibrate, then analyze
A clip starts with **no plane**.  The first step is the standard Calibration —
the parent's `pool_calibrate_tool.py` (semi-manual click of the distance marks
and waterline), which writes the `dog_diving_pool_calibration/v1` document to
`planes/<setup-id>.pool.json`.  The clip's manifest records `plane.setup_id`;
the plane document itself lives in `planes/`, never next to the clip.

1. `calibrate <clip_id> --setup-id <id>` (or **Calibrate plane…** in the GUI).
2. Frame-by-frame YOLO detection → `detections.json` (parent `tracks.json` shape,
   with `model` + `model_sha` provenance). **Re-runnable** with a different
   registered model.
3. Build the trajectory → `trajectory.json`.
4. Measure through the assigned plane (route `plane`).

### 6.3 Ground truth + judge
1. Scrub to the entry frame; click the true tailset∩water intersection
   (zoomable canvas — the parent's `image-measurement-ui` zoom, because
   fit-to-window quantises at ~2.5 image px/canvas px).
2. `ground_truth.py` recomputes `gt_in` through the plane homography.
3. Optionally enter the judge's distance (inches); stored as `judge_in`.
4. Both are **re-placeable** at any time.

### 6.4 Re-do operations (each is an idempotent overwrite)
- **Re-calibrate / re-assign plane** → re-run the parent's calibration tool
  (`pool_calibrate_tool.py`) to write a new `planes/<setup-id>.pool.json`;
  recompute every `gt_in` and mark affected `results/` stale.
- **Re-detect** with a different model (`ModelSpec` switch) → new
  `detections.json` + `trajectory.json`; invalidate `results/`.
- **Re-place GT / judge** → new `ground_truth.json`; invalidate `results/`.

Invalidation rule: `results/<run_id>.json` carries the `detections.model_sha`,
`plane.setup_id`, and `ground_truth.updated` it was computed from, so a stale
row is detected on load and flagged rather than silently shown.

### 6.5 Run a test suite
1. Select prepared clips (or all).
2. Define method configs — an explicit list or a grid over the three axes
   (§6.6).
3. `suite.py` runs each (method-config × clip), emitting a `ScoreResult` per
   cell via the registry.
4. `metrics.py` computes per-clip errors and averages → `results/<run_id>.json`.

### 6.6 The three scoring axes (v1)

| Axis | Values |
|---|---|
| `method` | `arc_2d` (2-D image-space parabola) — v1 only; `arc_3d`, `pc_route` reserved |
| `n_points` | cap on accepted points fed to the fit: `4, 6, 8, 12, 16, 20, all` |
| `selection` | `last_n` (last N by frame), `all` (no cap), `ransac_inliers`, `top_kp_conf_n`, `flight_run_only` |

The v1 `arc_2d` method: take the accepted tailset points, apply the selection
rule, cap to `n_points`, fit `x(t)`/`y(t)` quadratics via `trajectory.py`, then
locate the water entry and map it through the plane homography to `distance_in`,
reporting `entry_px`. The precise water-crossing rule for the 2-D fit (extrapolate
to the plane's waterline in image space vs. unproject along the fitted curve) is
an open sub-decision flagged in §9 — the registry contract is agnostic to it.

---

## 7. GUI (mirrors the parent's Review table, plus overlays)

Tabs:

1. **Ingest** — pick mp4/svo2, show metadata, copy into `analyzed-clips/`.
2. **Analyze** — plane confirmation (parent plane UI), model selector
   (`ModelSpec` registry), detect, build trajectory, status.
3. **Ground truth** — scrub, place GT point (zoom/pan canvas), enter judge
   distance; re-place buttons.
4. **Run suite** — clip multi-select, method-config grid builder, run, live
   progress.
5. **Results table** — one row per clip per method-config:
   `clip | method | n_points | selection | pixel error | score error | judge
   error | status`. Averages bar at the top (overall + per method). Selecting a
   row loads the clip into the **Compare** pane.
6. **Compare** (Review-style) — frame canvas with the plane grid, the GT marker,
   the judge marker, and **one toggleable overlay per test run** (fitted curve +
   entry point), so any subset of methods can be overlaid simultaneously. Same
   zoom/pan + image-coordinate readout as the parent Review.

---

## 8. CLI

`python cli_run.py` subcommands (headless, for batch and CI):

- `select --clip <path> [--id <id>]` — copy a clip into `analyzed-clips/`
- `calibrate <clip_id> --setup-id <id> [--frame-index N]` — semi-manually place
  the plane (the parent's `pool_calibrate_tool.py`), saved to `planes/<id>.pool.json`
- `use-plane <clip_id> --setup-id <id>` — assign an already-calibrated plane
- `analyze <clip_id> [--model <spec>]` — detect + trajectory + measure
- `set-gt <clip_id> --frame N --x X --y Y [--judge IN]`
- `run-suite --config <suite.json>`
- `report --run <run_id> [--clip <id>]` (text table + averages)

All subcommands read/write the same `analyzed-clips/` schema the GUI uses, so the
two are interchangeable.

---

## 9. Open sub-decisions (to resolve at build time, not blockers)

1. **2-D arc water-crossing rule** — RESOLVED (M5, 2026-11-10): the entry is the
   fitted curve's **terminal point** (evaluated at the last selected detection's
   x), mapped through the plane document's own `homography`. The 2-D fit carries
   no Z, so it cannot itself locate the water plane; extrapolating the fitted
   curve to a waterline is a follow-up that adds a `crossing` config key without
   touching the registry. (Recorded in `arc_2d.py`'s docstring.)
2. **SVO2 frame decoding** — confirm the ZED SDK read path (left-camera stream,
   fps) matches the MP4 path's metadata probe; keep behind the SDK-presence check.
3. **Results invalidation granularity** — full-run invalidation on any re-do vs.
   per-cell recompute. (Recommended: full-run; benches re-run whole suites.)

## 10. Milestones

1. **M1 — skeleton + copy.** Create the tree, copy the `dog_diving/` subtree
   byte-identical, copy `models/`, wire the launcher. Verify imports + port the
   parent's smoke test.
2. **M2 — ingest + manifest.** mp4 metadata probe, clip copy, `manifest.json`,
   `meta.json`. Verify sha256 + metadata match the parent's `result.json` block.
3. **M3 — analyze.** Plane load + detect + trajectory reusing the parent modules;
   `detections.json`/`trajectory.json`/`plane.json` writes. Verify with a parent
   artifact's known numbers (e.g. the `video004_right-seg01` 98.02 in run).
4. **M4 — ground truth + judge.** Zoomable GT placement, `gt_in` via homography,
   judge entry, re-place/re-draw/re-detect invalidation. Verify `gt_in` equals a
   hand-computed `entry_coordinate` value.
5. **M5 — scoring registry + arc_2d.** `ScoreResult` contract, `registry.py`,
   `arc_2d` with the three axes. Verify the method reproduces the parent's
   `plane_distance` number at the equivalent config.
6. **M6 — suite + metrics.** Grid generator, runner, pixel/score/judge errors,
   averages. Verify averages against a hand-computed fixture.
7. **M7 — GUI + CLI.** Ingest/Analyze/GT/Run-suite/Results/Compare panes; CLI
   subcommands. Verify GUI↔CLI schema interop.
8. **M8 — tests.** Ported pins + new bench tests (manifest round-trip, metric
   arithmetic, method reproduction, invalidation, judge-optional averaging).

## 11. Testing strategy

- **Reproduction pins:** the copied modules keep the parent's pinned numbers
  (e.g. `plane_distance` == 98.015… in on the known artifact), so a copy that
  drifts fails loudly.
- **Metric arithmetic:** hand-computed fixtures for pixel/score/judge error and
  averages (including a judge-absent clip).
- **Contract conformance:** every registered method's `ScoreResult` validates
  against one schema (mirrors the parent's one-schema rule).
- **Invalidation:** changing plane/model/GT marks results stale; loading a stale
  row flags it.
- **GUI:** port the parent's `app_gui` test harness pattern (`test_m26`/`test_m31`)
  for the Compare pane and the results table.
