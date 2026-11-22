# Test-bench 11-7 — clip map plan (Camera0_Right / Camera1_Right planes)

Status: **plan, not executed.** Updated 2026-11-20 with the operator's five decisions (below).

Scope: `test-bench-11-7/` only. `test-bench/` is read-only for this whole plan.

Source of truth for the clip list: `C:\Users\harri\Videos\Dog-Diving-Videos\paw_scores_w_videos.csv`
Videos live in place at `...\Dog-Diving-Videos\Documents\Captstone-Repo\end-to-end-app\data\clips_11-7\`
(ingest is by reference — nothing is copied into the bench).

---

## 1. What was measured before writing this plan

| Fact | Number |
|---|---|
| CSV data rows | **73** jumps across 9 dogs (lida ford 18, joule 17, ICE 9, lucian 7, sapper 6, lydia 5, surfer 5, blue 4, sable 2) |
| Rows **with** a judge score (ingested) | **58** → **116 clips** (58 `camera1` + 58 `camera0`) |
| Rows **without** a score (excluded) | **15** (12 `N/A`, 2 `missing score`, 1 `~~~`) → 30 clips never ingested; he keeps them for future YOLO training |
| Videos referenced / existence | every referenced video exists; 0 duplicates, 0 missing |
| CSV column → camera | `Video_Left` = `camera1`, `Video_Right` = `camera0` (verified name↔path on all 73 rows, pairs share the timestamp) |
| Distance parse | `19'10"` → 238 in; only the 58 parseable rows proceed |
| mp4 on disk, unreferenced | **22** — incl. 16 from **other sessions/dates** (11-04 … 11-16), 2 `_empty`, 2 unscored (191348, 191425). Untouched — candidate future training data. |
| Clip geometry | every clip 960×600, 30 fps, 150 frames, 5.0 s |
| Rig static across the session? | yes. Phase-correlation of frame 0 vs a mid-session clip: camera0 max **1.28 px**, camera1 max **1.00 px**; all clips *after the first* agree within **0.1 px**. Only `190234` (first clip) sits ~1.2 px off. ⇒ **one plane per camera is valid**; calibrate on a mid-session clip. |
| Detection cost (model of record, imgsz 1280, CPU) | **~42 s / clip** ⇒ **≈81 min** for 116 clips |
| Accepted tailset rows, fixed thresholds (conf 0.12 / kp_conf 0.30) — 8-clip sample | camera0: **27, 15, 0, 9, 11** · camera1: **12, 1, 5** ⇒ **some clips fail at fixed thresholds**; those are logged and excluded (not rescued) |
| `test-bench` vs `test-bench-11-7` source | byte-identical, 81 files each, 0 diffs ⇒ clearing 11-7's artifacts loses no code |
| Artifacts in 11-7 | `analyzed-clips` 66 files / 3.4 MB (16 clip dirs, 20 suite runs) · `out` 3 files · `planes` 6 `.pool.json` (5× 1920×1080, 1× 960×600 old session) · `data/active_plane.json` (stale Jetson path) |
| Git | `test-bench-11-7/` is **untracked** ⇒ any delete is unrecoverable without a backup |

---

## 2. Target end state

```
test-bench-11-7/
  analyzed-clips/
    analyzed_log.csv                    detection status per jump (see §2.1) — the "exclude" record
    <clip_id>/                          ×116  manifest.json, meta.json, detections.json,
                                              trajectory.json; ground_truth.json on the "ok" clips
  planes/
    (old .pool.json files stay until he builds the new ones)        ← kept, per decision #2
    camera0_11-7.pool.json        new, 960×600, all 58 camera0 clips
    camera1_11-7.pool.json        new, 960×600, all 58 camera1 clips
  data/clip_index.json                  CSV mirror for the 58 rows: dog, score_raw, judge_in, paths
  data/active_plane.json                GUI picker pointer (cleared of its stale Jetson path)
  out/                                  scratch: acceptance, audit tables, report
  results/                              suite runs
  src/, models/, ROADMAP.md, ACCURACY_TAB_*.md, run_app.*   unchanged
```

`clip_id` = `<HHMMSS>_<camera>_<side>[_sapper]`, e.g. `190234_camera0`. Keeps the camera tag
(that is what selects the plane), sorts chronologically, unique across all 58 rows.

**Plane mapping is data, not code.** `manifest.plane.setup_id` already resolves to
`planes/<setup_id>.pool.json`; every camera0 clip carries `camera0_11-7`, every camera1 clip
`camera1_11-7`. The planned `camera0_camera1_11-7` / `camera1_camera1_11-7` planes slot in later
with zero refactor.

### 2.1 What is saved onto each clip

- **Dog name + judge score (automatic, at ingest)** — `manifest.json` gains a top-level `labels`
  block, and `ground_truth.json` carries the judge score in the bench's own field:
  ```json
  // manifest.json
  "labels": { "dog": "joule", "judge_score_in": 130.0, "score_raw": "10'10\"",
              "csv_line": 31, "camera": "camera0" }
  // ground_truth.json (written at ingest, before GT is placed)
  { "schema": "test_bench.ground_truth/1",
    "frame": null, "x_px": null, "y_px": null, "gt_in": null,
    "judge_in": 130.0, "judge_present": true,
    "reference": { "dog": "joule", "score_raw": "10'10\"", "csv_line": 31 },
    "updated": "..." }
  ```
- **Ground truth (hand-clicked, decision #1)** — for each clip whose detection **succeeded**, he
  places the true tailset∩water entry pixel in the GUI's zoomable canvas; `gt_in` is then derived
  through the plane's homography (the bench's native path). Failed clips get **no** GT.

### 2.2 `analyzed_log.csv` — the failure/exclusion record (decision #3)

Path: `C:\Users\harri\Documents\GitHub\Dog-Diving-Scoring\test-bench-11-7\analyzed-clips\analyzed_log.csv`
One row per jump (58 rows), one status cell per camera feed:

| `jump_time` | `dog` | `judge_score_in` | `score_raw` | `Camera0_Left` | `Camera0_Right` | `Camera1_Left` | `Camera1_Right` |
|---|---|---|---|---|---|---|---|

Each camera cell is `ok` or `failed:<reason>`. **`ok` = ≥ 4 accepted tailset rows AND `arc_2d`
produces a water-entry** (reason ∈ `no_detection` = 0 accepted, `few_points` = 1–3 accepted,
`no_fit`, `no_crossing`). The Left cells stay empty until those planes/clips exist. **A failed cell
⇒ that clip is excluded from the Run-suite and Accuracy windows.** (The existing GUI filter already
drops any clip without `ground_truth.json`; the status in the manifest + this log make the
exclusion explicit and auditable.)

---

## 3. Decisions — all settled

1. **Ground truth = hand-clicked pixel.** He clicks the entry pixel for every clip with successful
   detections; `gt_in` is derived from that pixel through the plane homography. The judge score is
   still auto-saved as `judge_in` (turn-2 requirement). Both fields then live in
   `ground_truth.json` exactly as ROADMAP §5.2 intends.
2. **Keep the old planes until the new ones exist.** `planes/` is **not** cleared; the new
   960×600 planes are added alongside. (The stale `data/active_plane.json` pointer is still
   cleared.) ⇒ the 960×600 size guard (D6) is now **mandatory**, since 1920×1080 planes stay in
   the picker next to the new ones.
3. **No adaptive ladder.** Detection runs at fixed thresholds (`detect_clip` as-is). A clip that
   fails is logged in `analyzed_log.csv` (per-camera cell = `failed:<reason>`) and excluded from
   the testing windows. No re-run rescue.
4. **He builds the two planes** in the Calibrate tab / `pool_calibrate_tool`. I provide the
   mid-session reference clip per camera + the gate/size checks; I do not pre-place marks.
5. **Unscored rows skipped entirely** (never ingested); he may reuse them (and the 22 unreferenced
   videos) as future YOLO training data.

**D6 — plane-size guard (mandatory).** `manifest.clip_plane` must refuse a plane whose `image_size`
≠ the clip's `(width, height)`, with `DD_ALLOW_PLANE_SIZE_MISMATCH=1` as the only override. This is
what makes decision #2 safe.

---

## 4. Phases, with acceptance gates

### P0 — Back up, then verify (5 min)
1. Zip `analyzed-clips/`, `planes/`, `out/`, `data/` → `C:\Users\harri\agent-workspace\backups\test-bench-11-7-preclear-2026-11-20.zip` (~3.8 MB).
2. Re-open the zip and count entries vs on-disk (66 + 6 + 3 + 1 = **76 files**).
3. sha256 manifest of the **whole `test-bench/` tree** → `agent-workspace\backups\test-bench-manifest-pre.txt`.
Gate: zip 76/76, archive test passes, `test-bench` manifest recorded.

### P1 — Clear artifacts, keep planes (2 min)
Delete `analyzed-clips/*`, `out/*`, `data/active_plane.json`. **Leave `planes/` intact** (decision
#2). `models/`, `src/`, docs, `run_app.*` untouched.
Gate: `analyzed-clips/` and `out/` empty, `planes/` still has its 6 files, `data/` has only
`clip_index.json` after P2; fresh sha256 manifest of `test-bench/` is byte-identical to P0's;
`git status --short` still shows only `?? test-bench-11-7/`.

### P2 — Index the CSV, ingest 116 clips by reference (10 min)
Driver (new, `src/bench_11_7.py`, mirroring `cli_run.py`):
1. `index` — parse CSV, keep the 58 parseable rows; assert 116 videos exist / path basename ==
   filename; write `data/clip_index.json` (the explicit 116-file allow-list — keeps the 22
   non-CSV videos and the 15 unscored rows out).
2. `ingest` — per clip `ingest_clip(path, root, clip_id=..., preserve_existing=False)`; write
   `labels` (dog, judge_score_in, score_raw, csv_line, camera); write the judge record in
   `ground_truth.json` (via new `ground_truth.from_score()`); set `manifest.plane.setup_id` by
   camera tag (guarded by D6).
Gate: **116** clip dirs; 116 manifests with `labels.dog` non-empty and `labels.judge_score_in`
matching a re-parse of the CSV (diff, not eyeball); camera counts 58/58; 116 `ground_truth.json`
with `judge_present = true`; 0 clips pointing at a plane whose `image_size` ≠ 960×600; 3
spot-checked manifests' `meta` = 960/600/30/150.

### P3 — Detect + trajectory, write the status log (background, ≈81 min)
Fixed thresholds (`detect_clip`). Resumable: skip a clip whose `detections.json` exists with the
same `model_sha`. For each clip: run detect → build trajectory → attempt the arc fit, classify
`ok` / `failed:<reason>` → write the cell in `analyzed_log.csv` and `manifest.detections.status`.
Gate: 116/116 `detections.json`; `analyzed_log.csv` has 58 rows × 8 columns, every camera cell
`ok` or `failed:<reason>`; the failed count is stated in the report (measured expectation: at least
the sampled zero/near-zero clips fail).

### P4 — He builds the two planes (his clicks; ~20–40 min)
Reference clips (mid-session, avoids the 1.2 px first-clip offset): `192148_camera0`,
`192148_camera1`. Planes land at `planes/camera0_11-7.pool.json` and
`planes/camera1_11-7.pool.json`, both 960×600.
Gate: W1/W2/W4 and P1/P3 pass; px/ft along + across recorded (~20–26 px/ft along, ~1 px/ft across
— transverse stays coarse, limiting lateral placement, not distance); marks round-trip to their
labels; `plane_coverage.py` on 5 sample clips per camera shows high, stable `pool_in_quad`.

### P5 — Measure the ok clips (needs planes) (5 min)
For every `ok` clip, run the measure path through its plane; write the result. Failed clips are
skipped (already excluded).
Gate: every `ok` clip has `score.distance_in`; failed clips have none and are listed.

### P6 — He hand-clicks GT on the ok clips (his clicks; needs planes)
For each `ok` clip, place the entry pixel → `gt_in` derived through the homography. Failed clips
get no GT (and therefore stay out of the windows). `analyzed_log.csv` may gain a `gt_placed` marker
per cell.
Gate: `ok` clips each have `ground_truth.json` with `gt_in` non-null; failed clips have none.

### P7 — Suite + accuracy numbers (10 min)
Run-suite over the ok clips (grid: `n_points` 4/8/12 × `selection` last_n / all / ransac_inliers).
Deliver: score error vs GT (in + %), judge error vs GT, score-vs-judge, ±2/±4/±6 shares, split
per camera and per dog; **cross-camera agreement** — for the 58 pairs compare camera0 vs camera1
measured distance (a camera whose plane is off shows as a systematic bias).
Gate: `results/<run_id>.json`, `out/audit_11-7.csv`, `out/REPORT_11-7.md`; all 58 jumps accounted
for (58×2 = 116 clips, no silent drops).

### P8 — Document and stop
`BENCH_11-7.md`: the 116 clips, the two setup ids, the `labels`/`judge`/GT storage, the
`analyzed_log.csv` contract, the measured acceptance/failure distribution, how the Left planes join
later.

---

## 5. Risks and how each is handled

| Risk | Handling |
|---|---|
| A 1920×1080 plane used on a 960×600 clip → 2× distance error | D6 guard in `manifest.clip_plane`; old 1920 planes stay only because the guard blocks them |
| Other-session videos swept in by a glob | ingest from the CSV's explicit 116-file allow-list, never `glob(*)` |
| The 15 unscored rows accidentally ingested | not in the allow-list; P2 gate asserts exactly 116 / 58+58 |
| Failed clips silently scored or shown | `analyzed_log.csv` cell = `failed:<reason>`; no GT → excluded by the existing filter; P3 gate counts failures |
| Wrong plane↔camera pairing (CSV "Left" is `camera1`) | assignment keyed on the filename's camera tag; P2 gate asserts 58/58 |
| ~81-min detection job dying with the session | background, resumable per clip, log in `out/` |
| Unrecoverable delete (untracked tree) | P0 backup + count check before P1 |
| Dog/score provenance drifting from the CSV | single `data/clip_index.json`; P2 gate re-parses and diffs `judge_score_in` |
| GT clicked before planes exist | P6 is ordered after P4; `gt_in` needs the homography, so the tool simply cannot derive it earlier |

---

## 6. Deliverables

1. This plan.
2. Backed-up bench with `analyzed-clips`/`out` cleared, `planes/` kept.
3. 116 registered clips, each carrying `labels` (dog, judge score) + `ground_truth.json`
   (`judge_in`, then `gt_in` after his clicks), and `data/clip_index.json`.
4. `analyzed-clips/analyzed_log.csv` — per-jump, per-camera ok/failed record.
5. `planes/camera0_11-7.pool.json`, `planes/camera1_11-7.pool.json` (his, 960×600,
   gate-passing).
6. `out/acceptance_11-7.csv`, `out/audit_11-7.csv`, `out/REPORT_11-7.md`, `results/<run_id>.json`.
7. `BENCH_11-7.md`.

---

## 7. Settled follow-ups

**U1 — the "failed" bar (decided).** `ok` iff the clip has **≥ 4 accepted tailset rows** AND
`arc_2d` produces a water-entry (`distance_in` non-null); otherwise `failed:<reason>`
(`no_detection` = 0 accepted, `few_points` = 1–3 accepted / < 4 after selection, `no_fit`,
`no_crossing`). ≥ 4 is deliberately stricter than `arc_2d.MIN_POINTS` (3) — the operator's call.

**U2 — `analyzed_log.csv` layout (decided).** Columns fixed as
`jump_time, dog, judge_score_in, score_raw, Camera0_Left, Camera0_Right, Camera1_Left, Camera1_Right`
(Left cells blank until those feeds exist). This header is the contract other stages read.
