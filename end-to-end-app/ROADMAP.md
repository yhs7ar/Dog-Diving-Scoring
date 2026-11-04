# ROADMAP — End-to-End App vs. the Five-Stage Scoring Plan

> **Canonical progress ledger:** `C:\Users\harri\agent-workspace\ROADMAP.md`.
> This file is the *static* design assessment + dependency rationale. The workspace copy is what the
> overnight agents read and update; edit progress there, not here.

**Scope:** this document maps the tentative five-stage plan (Calibration → Clipping →
Detection → Intersection/Review → Saving) onto what already exists in
`end-to-end-app/`, and calls out what still has to be designed. It is meant to be reused
across sessions: every plan item gets a **status**, the **exact files** that already back it,
the **planned files** that will hold the missing work, and the **resources that are not yet
identified** (APIs, models, dependencies, design decisions).

**Repo root (verified):** `C:\Users\harri\Documents\GitHub\Dog-Diving-Scoring`
**Writable scope:** `end-to-end-app/` only — everything else in the clone is read-only reference.
**Interpreter:** `run_app.cmd` is the source of truth (`DD_PY` points at a local venv; CPU only).

> Status key: **DONE** = shipped and tested · **PARTIAL** = some form present, real gaps remain ·
> **GAP** = absent, must be designed · **N/A** = plan item does not apply to this build.

---

## 0. TL;DR status matrix

| # | Plan item | Status | Where it lives today |
|---|---|---|---|
| 1a | Capture frames for calibration (separate option) | **GAP** | — (zed_recorder has no calibration mode) |
| 1b | Calibrate pool plane — **auto** (stereo point cloud + waterline) | **PARTIAL** | waterline seed on mono MP4 exists; no point cloud |
| 1b | Calibrate pool plane — **manual** (2-D adjustments) | **DONE** | `pool/plane.py`, Calibrate pane |
| 1c | Establish distance from dock (+X, 0 ft on confirmed plane) | **DONE** | `pool/intersection.py`, `measure.py`, fixed 35 ft axis |
| 2a | Record video continuously (compute-aware rolling) | **PARTIAL** | `zed_recorder.py` (5 s chunk buffer only) |
| 2b | Save last N seconds as two SVO2 clips | **PARTIAL** | `trigger_capture` (fixed 5 s, not configurable) |
| 2c | Convert to MP4 (both right cameras) | **PARTIAL** | one MP4 (cam1 LEFT view) only |
| 2d | Fuse clips / take point cloud | **GAP** | — (`DEPTH_MODE.NONE`, no fusion) |
| 3a | Detect tailset with trained model | **DONE** | `detect.py`; model switch to `trained_models/…720p` decided (see §3a) |
| 3b | Last-20 detections → trajectory | **DONE** | `trajectory.py` (`LAST_N_POINTS = 20`) |
| 4a | Intersection: arc/plane × point cloud | **PARTIAL** | two scoring methods, but **no point cloud** |
| 4b | Display distance / edit point / Replay / Accept | **PARTIAL** | distance + manual entry exist; Replay/Accept do not |
| 4c | Replay at 0.25× with frame nav | **GAP** | Review is a still-frame scrubber only |
| 5a | Save score to CSV with dog name | **GAP** | — (only `batch_rollup.csv`, a batch artifact) |

The single biggest hole across the whole plan: **nothing uses stereo depth or a point cloud.**
The entire "clip → point cloud → plane/intersection" spine of the plan is the one part of this
codebase that has never been built — the app today is a **mono** pipeline (one MP4 per jump, one
camera's pixels, homography on a hand/waterline-placed quad). The ZED X pair is captured only as a
raw 5-second rolling buffer, with depth mode explicitly **disabled**.

---

## 1. Calibration (pool plane)

### 1a. Capture frames for calibration — **GAP**

Plan: a separate operator option that, once tripods are set up, captures **two SVO2 frames**
for calibration (one per camera), to be fused later.

- **Exists:** nothing. `zed_recorder.py` has exactly one capture path
  (`trigger_capture(prefix)`) that dumps the *live-jump* rolling buffer; there is no
  calibration-capture entry point and no still-frame capture.
- **Gaps to design:**
  - A "capture calibration frames" button/option distinct from "Score Live Jump".
  - Saving a *calibration* SVO2 pair to a separate location (e.g. `data/calib_svo/`), tagged
    with the tripod setup, not the jump timestamp.
  - Whether calibration frames are grabbed as a still pair or a short burst (the plan says two
    frames, but a short burst gives temporal averaging for the specular water).
- **Planned files:** extend `src/dog_diving/zed_recorder.py` (a `trigger_calibration()` method);
  wire a button in `src/app_gui.py` Calibrate pane.
- **Unidentified:** ZED SDK still-capture API surface (`sl.Camera.grab` once vs. the rolling
  loop); whether calibration needs `depth_mode` ON (it does, for the point cloud in 1b).

### 1b. Perform calibration of pool plane

#### Automatic — **PARTIAL** (the waterline half exists; the stereo half does not)

Plan: combine the two SVO2 frames into a **single point cloud**, then pair that with the
waterline algorithm to auto-detect the waterline edges.

- **Exists (waterline auto-seed):** `src/dog_diving/waterline.py` (three-edge adapter over the
  vendored `src/third_party/waterline/`) + `src/dog_diving/pool/seed.py` (lines → quad). Wired to
  the **"Detect waterline"** button in the Calibrate pane (`on_detect_waterline`,
  `_start_waterline_detection`, `_apply_seed`). It builds a *temporal noise map* on a **mono MP4**
  clip (calm-window frames), fits far/near waterlines + dock edge, and seeds the plane's four
  corners. Gates S1–S6, fallback ladder L3→L0, refusal path. Documented in
  `docs/PLAN_WATERLINE_SEED.md`.
- **Does NOT exist:** the point cloud. There is no stereo fusion, no `retrieve_measure`, no
  RANSAC plane fit on 3-D points. The waterline seed is 2-D image geometry, not "point cloud +
  waterline".
- **Gaps to design:**
  - ZED two-camera extrinsics (the two ZED X are separate cameras; fusing their depth needs the
    inter-camera transform from `sl.CalibrationParameters` / `CameraInformation`).
  - A `retrieve_measure`-based point cloud from the calibration SVO2 pair.
  - A robust plane fit on the fused cloud (water is specular/deep-weak; the dock lip and pool
    walls are the reliable structure — see `dock-diving-scoring` skill: "stereo depth on the water
    surface is the weak link … fit planes and accumulate").
  - The interface between that 3-D plane and the existing `pool/plane.py` quad.
- **Planned files:** `src/dog_diving/pool/stereo.py` (new — fusion + plane fit), or extend
  `pool/calib.py`; a `stereo_point_cloud` block in the calibration document.
- **Unidentified:** ZED SDK fusion/extrinsics API; whether one Orin Nano can run two ZED X in
  `ULTRA` depth at 60 fps (the compute-budget question from §2a applies here too).

#### Manual — **DONE**

Plan: present the tentative pool plane; operator adjusts it with the updated 2-D system.

- **Exists:** `src/dog_diving/pool/plane.py` (`PlaneSession`: corner drag, side drag, arrow-pad
  nudge, `flip_x`, `validate_plane` gates P1–P4); zoomable `FrameView` canvas in `app_gui.py`
  (wheel zoom, Fit/1×–16×, cursor readout in image px — the 0.2 px precision requirement from the
  domain skill); "Detect waterline" seeds the corners and the operator finishes by hand. Fully
  tested (`test_m18_plane`, `test_m20_plane_placement`, `test_m22_plane_tools`,
  `test_m24_plane_calibrate_review`, `test_m25_calibrate_pane`).

### 1c. Establish distance from dock — **DONE**

Plan: use the user-confirmed plane's "0 ft" as the metric zero; +X is the distance from the dock.

- **Exists:** `config.POOL_LENGTH_FT = 35.0` is a **fixed metric axis** (+X from dock lip to 35 ft);
  the dock lip is `X = 0` in the plane; `pool/intersection.py:entry_coordinate` and
  `measure.plane_distance` map a pixel (or world point) through `doc["homography"]` and report
  `X` in inches. This is exactly the plan's "user-confirmed plane 0 ft".

---

## 2. Clipping

### 2a. Record video — **PARTIAL**

Plan: competition video rolls continuously; recording pauses only while saving/processing for
scoring, to protect the Jetson Orin Nano Super from overload; the program tracks its own compute
usage per stage.

- **Exists:** `src/dog_diving/zed_recorder.py` `DualZEDXLiveBuffer` — a **5-second chunked
  rolling buffer**: two alternating slots, each camera writes a lossless SVO2 to a temp slot, and
  the completed chunk is committed only when `save_requested`. Both cameras opened at 720p/60,
  `DEPTH_MODE.NONE`.
- **Gaps vs. the plan:**
  - It is **not** "always recording except during save/process" — it always records 5 s chunks and
    swaps slots; there is no pause-during-scoring.
  - **No compute-load tracking.** Nothing monitors Orin Nano utilization, no backpressure that
    suspends capture while `detect`/`measure` run, no per-stage compute budget. The plan's whole
    "protect the Orin from overload" mechanism is undesigned.
  - No notion of "competition start" — capture just starts at app launch.
- **Planned files:** a recorder state machine in `zed_recorder.py` (or a new
  `src/dog_diving/capture.py`) plus a compute governor; likely `psutil`-based load watch on the
  Orin.
- **Unidentified:** Orin Nano measured CPU/GPU headroom for (a) dual-ZED capture, (b) YOLO
  inference, (c) depth fusion — the numbers that decide whether capture and scoring can overlap
  at all, and whether depth mode can run during capture.

### 2b. Select / save last [10] seconds as two SVO2 clips — **PARTIAL**

Plan: a button saves the last N seconds (default 10, user-changeable) as two individual SVO2
clips; use ReplaySorcery as a baseline for the rolling-buffer pattern.

- **Exists:** `trigger_capture(prefix)` freezes the current 5 s chunk and copies
  `{prefix}_cam1.svo2` + `{prefix}_cam2.svo2` to `data/svo_recordings/`.
- **Gaps:**
  - Duration is **hard-coded 5 s** (`chunk_seconds=5`); not the plan's default 10 s and not
    user-adjustable. "Last N seconds" is also chunk-aligned — a press just after a slot rollover
    yields a shorter effective window.
  - **DECIDED (operator):** keep `zed_recorder.py` as a **custom edit** — never vendor or port
    ReplaySorcery (it is a Linux *screen* recorder with no ZED/SVO2 support). Reference
    ReplaySorcery's README **as documentation of intended behaviour only**: always-on ring buffer,
    save-last-N-on-demand, and (from §2a) backpressure so the recorder yields to scoring compute.
    The native ZED implementation is `sl.RecordingParameters`, which the custom edit already uses.
- **Planned files:** `zed_recorder.py` — parameterise `chunk_seconds`, add a user-facing
  duration setting, and make the buffer a true ring (save the last N seconds across slot
  boundaries) rather than "the chunk I happen to be in".
- **Unidentified:** whether lossless SVO2 at 60 fps × 2 cams is sustainable on the Orin's storage
  (write bandwidth) — the ring-buffer sizing depends on it.

### 2c. Convert to MP4 — **PARTIAL**

Plan: both **right-camera** MP4s are saved for later display.

- **Exists:** `_commit_jump_data` writes **one** MP4 (`{prefix}.mp4`) from `recent_frames_cam1`,
  which holds **cam1's LEFT view** (`sl.VIEW.LEFT`). No cam2 MP4, no right-view MP4.
- **Gaps:** need per-camera (and per-view, if the plan really means the right *view* of each
  camera) MP4 export; decide naming (`_cam1.mp4`, `_cam2.mp4`); decide whether MP4 export happens
  at capture time or is deferred (the plan implies capture-time, to keep the rolling buffer from
  being overwritten).
- **Planned files:** `zed_recorder.py` `_commit_jump_data` + the app's clip-load path
  (`load_clip`) which today assumes a single MP4.
- **Unidentified:** SVO2 → MP4 conversion path for **cam2** (the code only ever calls
  `retrieve_image` on cam1; cam2's frames are never pulled out of its SVO2 into an array).

### 2d. Fuse clips and take point cloud — **GAP**

Plan: both SVO2 files are combined and processed to align point-cloud data.

- **Exists:** nothing. `_init_cam` sets `depth_mode = sl.DEPTH_MODE.NONE`, so no depth is ever
  captured; there is no `retrieve_measure`, no disparity, no point cloud, no fusion, and no
  inter-camera extrinsics. The SVO2 pair is saved but never read back or fused.
- **This is the linchpin gap.** It blocks 1b (auto calibration from a point cloud) **and** 4a
  (trajectory-plane × point-cloud intersection).
- **Planned files:** `src/dog_diving/pool/stereo.py` (new) — fusion + point-cloud generation +
  plane fit; a `point_cloud` artifact per jump under `out/<clip_id>/`.
- **Unidentified:** ZED SDK `sl.CameraInformation` / `sl.CalibrationParameters` for the
  two-camera transform; depth-mode choice (NEURAL vs ULTRA) and its Orin cost; the exact
  downstream format the intersection step (4a) will consume.

---

## 3. Detection (trajectory plane)

### 3a. Detect tailset with the trained model — **DONE** (but a different model than the plan names)

Plan: detect the tailset in MP4 frames using
`trained_models/dog_pose_s_2kpt_inflated_clahe_50epochs_720p` weights and the existing detection
scripts.

- **Exists:** `src/dog_diving/detect.py` — YOLO-pose wrapper (CLAHE on LAB-L, ByteTrack,
  keypoint-confidence gating, adaptive threshold ladder, `tracks.json` cache). Fully tested
  (`test_m3`-family, `test_m16_detect_ladder`, `test_m28_model_registry`).
- **The model mismatch — flag this explicitly:**
  - The app's **model of record** is `models/yolo26n-pose_dogs_v5.pt` (declared in
    `config.MODEL_REGISTRY`, measured 46/46 detected, tailset median 8.18 px).
  - The plan's model, `trained_models/dog_pose_s_2kpt_inflated_clahe_50epochs_720p`, **exists on
    disk** at the repo root (with `weights/best.pt` and `weights/last.pt`, plus `args.yaml` /
    `results.csv`) but is **not** in `MODEL_REGISTRY`, is **not** in `models/`, and is **not**
    wired anywhere. It is a newer run (the repo's `inflate_dataset.py` / `resume_training.py` were
    touched Nov 1) and has **not been measured** on the labelled still set.
  - **DECIDED (operator):** switch detection to
    `trained_models/dog_pose_s_2kpt_inflated_clahe_50epochs_720p/weights/best.pt`. Register it in
    `MODEL_REGISTRY` (resolve the weights inside `models/` so the registry's `MODELS_DIR`
    resolution holds), and A/B it against v5 on the 46-frame labelled still set **as the acceptance
    gate** (domain rule: gate on the app's clip, never on `results.csv`). v5 stays the recorded
    control, not the default.

### 3b. Use detections to calculate trajectory (last 20 high-confidence) — **DONE**

Plan: the last 20 high-confidence detections compute the trajectory arc and plane.

- **Exists:** `src/dog_diving/trajectory.py` — `LAST_N_POINTS = 20` (operator order 2026-10-28),
  robust RANSAC parabola in the **time** domain, temporal-run segmentation, and a **3-D** route
  (`robust_parabola(..., z=...)`, world feet). The 3-D route is selected by `config.TRAJ_SPACE`
  and the Review pane's "scoring method" radio.

---

## 4. Intersection and review

### 4a. Automatic intersection/distance (arc or plane × point cloud) — **PARTIAL**

Plan: default to **trajectory plane × point cloud**; the intersection is auto-detected from the
trajectory and the point cloud. Manual: the operator may pick the trajectory *arc* to generate the
intersection instead, with the arc shown/hidden mutually exclusively, and may include/exclude
detections (trajectory auto-recalculated).

- **Exists (the two scoring *methods*, mono):**
  - "Trajectory arc (2-D)" → `measure.plane_distance` — **final detected tailset assumed on the
    plane**, mapped through `doc["homography"]`; `route="plane"`. This is the *scoring rule* (the
    last accepted detection), **not** an arc intersection.
  - "Trajectory plane (3-D)" → `measure.trajectory_plane_distance` — world-space arc extrapolated
    to `Z = z_water`; `route="arc3d"`.
  - Mutual-exclusive toggle: the Review pane "scoring method" radio (`on_trajectory_space`,
    `ReviewState.set_space`) switches the drawn curve and the number together.
  - Include/exclude detections + refit: `ReviewState.excluded`, `toggle_point`, `include_all`,
    `trajectory.refit` (click a marker to drop it, arc refits).
- **Gaps vs. the plan:**
  - **Neither method uses a point cloud.** The 3-D route is built from a **mono bootstrap**
    (`pool/pipeline.py:track_to_world` — pixels → rays → one vertical plane → 3-D points), *not*
    from stereo depth. `docs/TRAJECTORY_3D_NOTES.md` is explicit: the bootstrap is biased **±tens
    of inches** and every `route="arc3d"` result carries `arc3d_unverified_bootstrap`. The same
    note ranks "stereo disparity (ZED X)" as the *only* fix that removes the bias rather than
    constraining it.
  - So "trajectory plane × point cloud" (the plan's default) is **not implemented**; the current
    "trajectory plane" is a mono stand-in that must be replaced by the §2d stereo path.
- **Planned files:** rework `measure.trajectory_plane_fit` / `pool/pipeline.track_to_world` to
  consume the fused point cloud instead of the mono bootstrap; add a
  `trajectory_plane × point_cloud` intersection to `pool/intersection.py`.
- **Unidentified:** the exact intersection algorithm on a point cloud (trajectory-plane fit vs
  cloud vs. arc-on-cloud) — the plan leaves "trajectory arc AND point cloud OR trajectory plane
  AND point cloud" as an open design choice.

### 4b. Display estimated distance; edit; Replay; Accept — **PARTIAL**

Plan: auto-display the distance with the frame nearest the intersection time; the operator may
edit the intersection point/line, then **Replay** or **Accept**.

- **Exists:** Results tab (`_build_results`, `measurement_rows`, `_refresh_results`) shows the
  distance in feet/inches + provenance. Manual entry-point override exists (`on_set_entry`,
  `on_entry_live`, `on_clear_entry`, `on_nudge` — click/drag the entry point/line in Review).
- **Gaps:** **Replay** and **Accept** do not exist anywhere (no replay button, no accept/save
  flow — see §5).
- **Planned files:** add "Replay" (delegates to §4c) and "Accept" (delegates to §5a) controls to
  the Review/Results pane in `app_gui.py`.

### 4c. Replay the jump at 0.25× with frame navigation — **GAP**

Plan: replay at 0.25× with restart / pause / play / jump-to-first / jump-to-closest-intersection /
step frame-by-frame.

- **Exists:** only a **still-frame scrubber** — the Review pane's slider (`on_slider`), ±1 frame
  nudge (`on_nudge`), and jump-to-last-detection. No video playback, no 0.25× slow-mo, no
  play/pause/restart, no jump-to-closest-intersection.
- **Planned files:** a playback controller in `app_gui.py` (Tk `after`-loop over
  `video.iter_frames` at 0.25×, or cv2 `CAP_PROP_POS_FRAMES` seek). Reuse
  `video.py:iter_frames` / `read_frame`.
- **Unidentified:** none — a pure UI/playback task, but decide whether replay runs off the MP4 or
  re-reads the SVO2 (MP4 is simpler and already produced in §2c).

---

## 5. Saving jumps

### 5a. Save jump distance to CSV with dog name — **GAP**

Plan: after Accept, show the score with a field for a new dog name; previously entered names
appear in a selectable list on the left; the score is saved to a CSV navigable from within the app.

- **Exists:** `report.py` writes `result.json`, `report.txt`, `run_meta.json`, and a
  `batch_rollup.csv` — but the rollup is a **validation-batch artifact** (per-clip rows), **not** a
  score ledger, and has no dog-name field and no "Accept" trigger.
- **Gaps:** the entire accept→name→persist flow: a "dog name" entry field, a remembered-names
  list (persisted where?), an append-to-CSV of `{dog, distance, clip, timestamp, ...}`, and an
  in-app way to open the CSV.
- **Planned files:** a small `src/dog_diving/ledger.py` (new) for the CSV schema + append, plus
  controls in `app_gui.py` (`_build_results` / a new "save" block); a persisted names list
  (`data/dog_names.json` or similar).
- **Unidentified:** where the ledger lives (inside `end-to-end-app/data/` vs. an operator-chosen
  path); whether "Excel" means a real `.xlsx` (needs `openpyxl` — not currently a dependency) or a
  CSV that Excel opens (the plan says "CSV" in the body, "Excel" in the title — **decide**). The
  current `requirements.txt` has no spreadsheet library.

---

## 6. Cross-cutting gaps and risks (read before starting any stage)

1. **The point cloud is the missing spine.** §1b-auto, §2d, and §4a all depend on it, and none of
   it exists. The app is mono end-to-end today. Building stereo (two-camera extrinsics + depth +
   fusion) is the largest single piece of new work and should be sequenced first, because the
   waterline auto-seed (which *is* built) becomes a cross-check rather than a replacement.
2. **Model switch decided.** Detection will use
   `trained_models/dog_pose_s_2kpt_inflated_clahe_50epochs_720p/weights/best.pt`. It is on disk but
   unregistered and unmeasured — register it in `MODEL_REGISTRY` and A/B against
   `models/yolo26n-pose_dogs_v5.pt` as the acceptance gate.
3. **Orin compute budget is unmeasured.** The plan's central constraint (don't overload the
   Nano; the program tracks its own compute) has no design and no numbers yet. Dual-ZED capture +
   depth + YOLO + fusion on one Orin Nano is the thing to de-risk first — it may force
   depth-mode-off during live capture (which is exactly what the current code does) and depth
   computed later from SVO2.
4. **ReplaySorcery = behaviour spec, not code.** `zed_recorder.py` stays a custom edit;
   ReplaySorcery's README documents the intended behaviour (ring buffer, save-last-N,
   yield-to-scoring backpressure). Nothing from it is vendored.
5. **`zed_recorder.py` is an untested skeleton.** No test file references it (grep confirms it is
   imported only by `app_gui.py`), it has an offline stub mode, and it runs `depth_mode=NONE`. Any
   §2/§1b work should add a test seam (inject a fake camera) before building on it.
6. **Two scoring methods already exist and must not be silently replaced.** The Review pane's
   arc/plane toggle and the `route="plane"` vs `route="arc3d"` distinction are load-bearing and
   pinned by tests (`test_m30_trajectory_plane_3d`, `test_m31_review_scoring_method`). The plan's
   "trajectory plane × point cloud" should *replace the mono bootstrap* inside
   `track_to_world`, not introduce a third method.
7. **The datum rule is settled and must survive.** Measurement = "the last accepted tailset
   detection is the point of intersection with the plane; its +X is the distance." Any new
   intersection/point-cloud work must keep that rule as the default, with the 3-D arc as the
   alternate (see `dock-diving-scoring` skill and `measure.py` module docstring).

---

## 7. Resources not yet identified (consolidated)

- **ZED SDK APIs** (all unverified on this host): multi-camera extrinsics
  (`sl.CalibrationParameters`, `sl.CameraInformation`), `retrieve_measure` point cloud, depth-mode
  cost (NEURAL/ULTRA), SVO2 read-back and per-camera frame retrieval for cam2.
- **Orin Nano measured throughput** for dual capture + depth + inference (the §2a governor's
  input).
- **Model:** `trained_models/dog_pose_s_2kpt_inflated_clahe_50epochs_720p/weights/best.pt` — the
  selected model; register in `MODEL_REGISTRY` and measure on the labelled still set (46 frames).
- **ReplaySorcery** (`github.com/matanui159/ReplaySorcery`) — documentation source for recorder
  behaviour only (ring buffer, save-last-N, backpressure); no code vendored.
- **Point-cloud → plane / intersection algorithm** (the open choice in §4a).
- **Ledger format** (CSV vs `.xlsx`; needs `openpyxl` if the latter) and dog-names persistence
  path.

---

## 8. Build order — one substage at a time (dependency-ordered)

Order below is the recommended linear sequence for a DeepSeek agent that does **one stage or one
substage per session**. "Depends on" names the completed substage that must land first; "off-path"
marks work that does not block the stereo/geometry spine and could be pulled earlier if visible
progress is wanted. The **critical path** (what the whole plan hangs on) is
`2a → 2b → 2d → 1b-auto → 4a`.

| # | Do | Depends on | Notes |
|---|---|---|---|
| 1 | **3a** — register + switch to the `…720p` pose model | — (independent) | small, measurable: A/B vs v5 on the 46-frame still set. De-risks detection before anything builds on it |
| 2 | **2a** — continuous ZED capture with **depth ON** + compute governor | — | recorder keeps recording except while saving/processing; ReplaySorcery documents the behaviour |
| 3 | **2b** — "save last N seconds" as a true ring → two SVO2 clips | 2a | parameterise duration (default 10 s), real ring across slot boundaries |
| 4 | **2d** — fuse SVO2 pair → point cloud | 2b | **linchpin**; two-camera extrinsics + `retrieve_measure`. Unlocks 1b-auto and 4a |
| 5 | **1a** — calibration-capture mode | 2a, 2b | thin variant of the capture path; gives real data to test 1b-auto |
| 6 | **1b-auto** — point cloud → water-plane fit → seed `PlaneSession` | 2d | manual 1b already done; this makes calibration automatic |
| 7 | **4a** — re-point `track_to_world` at the fused cloud; intersection = plane×cloud (default) / arc×cloud | 2d, 3b | replaces the mono bootstrap; keeps the settled "final detection on plane" rule as default |
| 8 | **2c** — emit both right-camera MP4s | 2b | off-path; needed for replay and live detection display |
| 9 | **4c** — replay player (0.25× + frame nav) | 2c | off-path; Review is a still scrubber today |
| 10 | **4b** — Replay / Accept wiring | 4c, 4a | Replay → 4c; Accept → 5a |
| 11 | **5a** — score ledger CSV + dog names | 4b | off-path; completes the accept→persist flow |

**Why this order.** The model switch (step 1) is the only independent, high-value quick win — it
also satisfies the operator's "switch to the `…720p` model" flag. Everything after is the two
halves of the stereo build: the **geometry spine** (2a→2b→2d→1b-auto→4a) and the **display/UX
spine** (2c→4c→4b→5a). The geometry spine is the risk and the dependency — the point cloud (2d) is
the single gate that 1b-auto and 4a both wait on, and if dual-ZED depth on the Orin Nano turns out
not to hold at 60 fps, that fact needs to surface at step 2–4, before any of the consumers are
built.

**Risk note (state it, don't hide it).** If step 2–4 stalls on Orin depth throughput, the app is
still complete on the **mono** path (calibrate → process → review → results already works), so the
UX spine (8–11) can be finished first and the stereo spine swapped in afterward. The order above
front-loads the risky stereo work on purpose; it is not the only viable order.
