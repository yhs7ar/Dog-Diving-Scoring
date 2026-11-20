# Accuracy Tab — Implementation Plan

**Repo:** `C:\Users\harri\Documents\GitHub\Dog-Diving-Scoring\test-bench`
**Status:** Approved plan, not yet implemented.
**Date:** 2026-11-16

Read-only reporting tab that adds the project proposal's *simple* distance-accuracy
metrics on top of the bench's existing per-clip results. No re-detection, no re-run.

---

## 1. Scope

**In scope**

- Mean and median absolute score error, in **inches and percent**
- Percentage of jumps within **±2 / ±4 / ±6 in**
- Judge-vs-ground-truth error, in **inches and percent**
- **Inches per pixel** per clip, in two forms:
  - **nominal** — derived from the user-confirmed distance marks on the pool plane
  - **local at entry** — derived from the homography at the scored entry point
- Method-config selector + clip multi-select: visualise one method across all
  selected clips in the current run.

**Out of scope (deliberate)**

| Item | Why |
|---|---|
| Frame error vs expert contact frame | not requested; "simple metrics only" |
| OKS / keypoint MAE | not requested |
| Calibration RMSE over anchor points | lives in parent calibrate tool, not the bench |
| Confidence-interval calibration (≥90 % coverage) | scoring methods emit no uncertainty yet |
| FPS / latency | not recorded in results |
| Generalisation slices (dog / lighting / splash) | manifest has no structured fields for these |
| CSV export | not requested |
| GT population of the 12 empty clips | user will place GT later; the tab picks it up automatically |

**Data source:** the already-saved run JSON under `analyzed-clips/results/*.json`
(default: newest), plus each clip's `planes/<setup_id>.pool.json` and
`manifest.json` `meta`. Strictly read-only — this tab never writes results.

---

## 2. Metric definitions

For a clip with GT record `g` (`test_bench.ground_truth/1`) and scored cell `s`:

| Metric | Formula | Source |
|---|---|---|
| `score_error_in` | `|s.distance_in − g.gt_in|` | existing `metrics.score_error_in` |
| `score_error_pct` | `100 · score_error_in / g.gt_in` | **new** |
| `judge_error_in` | `|g.judge_in − g.gt_in|` | existing `metrics.judge_error_in` |
| `judge_error_pct` | `100 · judge_error_in / g.gt_in` | **new** |
| `nominal_in_per_px` | `|total_in₂ − total_in₁| ÷ ‖px₂ − px₁‖₂` over the two farthest user marks | **new** |
| `local_in_per_px_at_entry` | `12 ÷ (px per foot)` at the entry point, along the jump (X) axis | **new** |

**Percent error is `err/GT × 100`, not the existing `accuracy_pct`.** The existing
`accuracy_pct = 100·(1 − err/GT)` reads ~95 % even at ±6 in on a 120 in jump,
which makes the primary and fallback thresholds indistinguishable. Report the
error percentage, and keep `accuracy_pct` untouched for backwards compatibility.

**Nominal in/px.** Marks in the plane document carry `ft`, `inch`, `x_px`, `y_px`
(see `planes/11_7_bad_*.pool.json`: 7′ at x≈1090, 12′ at x≈876). Take the pair of
marks farthest apart in real inches, convert the spacing to pixels, divide.
Worked example for clip `11_7_stabilized`: 60 in over 214 px = **0.28 in/px**.

**Local in/px at entry.** The plane homography maps world feet `(X, Y)` to pixels.
Invert it at the scored `entry_px` to recover `(X, Y)`, then finite-difference one
foot along `X`:

```
px_per_ft = ‖H(X+1, Y) − H(X, Y)‖₂
in_per_px = 12 / px_per_ft
```

This is the honest number at the measurement point; it varies across the frame
because of perspective.

---

## 3. New module — `src/test_bench/accuracy.py`

Pure functions, no I/O except reading passed-in dicts, so it is fully testable
headless. `metrics.py` stays frozen to protect the existing green suite.

```python
def median(values: Iterable[float | None]) -> float | None
def score_error_pct(score: dict, gt: dict) -> float | None
def judge_error_pct(gt: dict) -> float | None
def nominal_in_per_px(plane_doc: dict) -> float | None
def local_in_per_px_at_entry(plane_doc: dict, entry_px) -> float | None
def bin_counts(errors_in: Iterable[float], bins: tuple[int, ...] = (2, 4, 6))
    -> dict[int, dict[str, float]]          # {2: {"count": n, "pct": p}, ...}
def aggregate(rows: list[dict], method_cfg: dict, clip_ids: list[str] | None = None)
    -> dict                                # per-clip table + overall summary
```

`aggregate` returns:

```python
{
    "clips": [ {clip_id, n_points_in_per_px, in_per_px_nominal, in_per_px_local,
                score_error_in, score_error_pct, judge_error_in, judge_error_pct,
                within_2, within_4, within_6} , ... ],
    "summary": {n_clips, n_with_judge,
                mean_in, median_in, mean_pct, median_pct,
                bins: {2: {count, pct}, 4: {...}, 6: {...}},
                judge_mean_in, judge_median_in, judge_mean_pct},
}
```

### Edge cases

| Case | Behaviour |
|---|---|
| Clip has no `ground_truth.json` | row shows in/px only; excluded from error, bin and judge stats |
| Clip has GT but no judge | contributes to score and bin stats; judge aggregate covers only judge-present clips |
| Plane has fewer than 2 marks | `nominal_in_per_px = None`, rendered as `—` |
| Selected method config has no run cell for a clip | `local_in_per_px = None`, rendered as `—` |
| Degenerate homography / `entry_px` missing | `local_in_per_px = None` |
| Non-positive `gt_in` | percent metrics `None` (undefined), inch metrics still shown |
| Selected run is stale | banner shown, computed via `suite.is_stale` |
| Plane `image_size` ≠ clip `meta` resolution | warning row: in/px assumes native-resolution plane |

---

## 4. New pane — `AccuracyPane` in `src/app_gui.py`

Registered as a seventh tab immediately after **Compare**
(`app_gui.py:1188-1194`), and refreshed from `on_clips_changed`
(`app_gui.py:1204-1209`), matching the existing pane contract
(`IngestPane`, `CalibratePane`, `AnalyzePane`, `RunSuitePane`, `ResultsPane`,
`ComparePane`).

Layout, top to bottom:

1. **Selector row**
   - Run dropdown — every `results/*.json`, labelled timestamp, default newest.
   - Method-config dropdown — the run's `config` entries (method, n_points, selection).
   - Clip multi-select — default all clips with GT; empty selection means all.
2. **Per-clip table** — clip · nominal in/px · local in/px @entry ·
   score err (in) · score err (%) · judge err (in) · judge err (%) ·
   ±2 / ±4 / ±6 pass marks.
3. **Summary footer** — n clips · mean (in, %) · median (in, %) ·
   % within ±2 / ±4 / ±6 · judge mean (in, %).
4. **Stale banner** — visible only when `suite.is_stale` is true for the selection.

Column widths and fonts follow `ResultsPane` (`app_gui.py:801-833`) so the two
panes read as siblings.

---

## 5. Tests — `src/tests/test_bench_accuracy.py`

| Test | Assertion |
|---|---|
| `median` | odd list, even list, empty, list containing `None` |
| `score_error_pct` / `judge_error_pct` | correct value; `None` on missing GT, missing judge, non-positive GT |
| `nominal_in_per_px` | hand-computed fixture (7′ and 12′ marks, known px gap); `<2` marks → `None`; identical marks → `None` |
| `local_in_per_px_at_entry` | synthetic homography with uniform 0.25 in/px → 0.25 ± 1e-9; singular homography → `None` |
| `bin_counts` | exact counts and percentages for a known error list incl. ties at the boundary |
| `aggregate` | synthetic 3-clip run, one without judge, one without GT → correct mean/median/bins/judge stats and correct clip exclusion |

---

## 6. Verification gates

1. `pytest src/tests/test_bench_accuracy.py` — all new tests pass.
2. `pytest src/tests/test_bench_suite.py src/tests/test_bench_ground_truth.py src/tests/test_bench_scoring.py` — still **32 passed** (baseline verified 2026-11-16).
3. Manual smoke via `run_app.cmd`: open **Accuracy**, select run
   `2026-11-13T16-17-56`, method `arc_2d_centerline`:
   - 3 GT clips appear
   - `11_7_stabilized` nominal in/px ≈ **0.28**
   - aggregate score error ≈ **6.45 in**
   - judge columns populated for the 3 judge-present clips

---

## 7. Increment sequence

Each step must leave the repo test-green before the next begins.

| # | Step | Done when |
|---|---|---|
| 1 | `accuracy.py` scalar metrics (`median`, pct helpers, `bin_counts`) + tests | step-1 tests pass, 32 baseline still pass |
| 2 | `accuracy.py` in/px functions (`nominal_in_per_px`, `local_in_per_px_at_entry`) + tests | fixtures reproduce 0.28 and 0.25 cases |
| 3 | `accuracy.py` `aggregate` + tests | synthetic-run test passes |
| 4 | `AccuracyPane` with selectors, table, summary, stale banner | app launches, tab renders against the real run |
| 5 | GUI wiring (`on_clips_changed` refresh) + resolution/plane warning | smoke checklist in §6 all true |
| 6 | Doc note in `ROADMAP.md` §5.5 pointing at the new metrics | roadmap no longer claims mean-only is the deliverable |

**Constraints for every increment**

- Never commit or push — leave changes in the working tree for review.
- Do not modify `metrics.py`, `suite.py` or existing scoring methods.
- Do not touch `analyzed-clips/` data or add GT records.
- Prefer `patch` over rewriting `app_gui.py`; if a patch fails twice, re-read the
  region rather than retrying blind.
