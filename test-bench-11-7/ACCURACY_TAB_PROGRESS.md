# Accuracy Tab — Progress Log

One entry per increment run. Plan: `ACCURACY_TAB_PLAN.md` §7.

---

## 2026-11-16 — Step 1 complete

- **Step:** 1 — `src/test_bench/accuracy.py` scalar metrics (`median`, `score_error_pct`,
  `judge_error_pct`, `bin_counts`) + tests.
- **Files changed:**
  - `src/test_bench/accuracy.py` (new — module docstring documents `err/GT × 100` vs
    `metrics.score_accuracy_pct`; `median` drops `None`; pct helpers return `None` for
    missing/None/non-positive `gt_in`; `bin_counts` bins are inclusive at the edge, returns
    `{2: {count, pct}, 4: …, 6: …}` with `pct 0.0` on empty input)
  - `src/tests/test_bench_accuracy.py` (new — 9 tests: median odd/even/empty/`None`,
    score/judge pct values + all `None` cases, the err/GT-not-accuracy guard, bin counts
    exact counts/pcts, boundary ties, empty/`None`-only, custom bins + negative inputs)
  - `ACCURACY_TAB_PROGRESS.md` (new, this file)
- **Pytest:**
  - `src/tests/test_bench_accuracy.py` → **9 passed**
  - baseline `test_bench_suite.py + test_bench_ground_truth.py + test_bench_scoring.py` → **32 passed**
- **Not verified by this step:** no in/px, no `aggregate`, no GUI — expected (steps 2–4).
- **Remaining:** steps 2–6. **Next:** Step 2 — `nominal_in_per_px` + `local_in_per_px_at_entry`
  (+ tests; fixtures must reproduce 0.28 in/px from the 7′/12′ marks and 0.25 in/px from a
  synthetic uniform homography).
- **Constraints honoured:** nothing committed; `metrics.py`, `suite.py`, `scoring/*`,
  `analyzed-clips/`, `planes/`, results JSON all untouched.

---

## 2026-11-16 — Step 2 complete

- **Step:** 2 — `nominal_in_per_px` + `local_in_per_px_at_entry` (+ tests).
- **Files changed:**
  - `src/test_bench/accuracy.py` (added `math`/`numpy` imports; `nominal_in_per_px` uses
    the two marks farthest apart in real inches and divides the inch span by the pixel
    distance — 0.28 in/px reproduced from the 7′/12′ marks; `local_in_per_px_at_entry`
    inverts `H` at `entry_px`, finite-differences 1 ft along +X, returns `12 / px_per_ft`
    — 0.25 in/px on a synthetic uniform 48 px/ft homography; both return `None` on
    missing/coincident/singular inputs)
  - `src/tests/test_bench_accuracy.py` (8 new tests: nominal hand-computed 0.4, the
    plan's 7′/12′ ≈ 0.28, farthest-pair selection over three marks, <2 / identical
    marks → None; local uniform 0.25 at two entries, singular H → None, missing
    homography/entry → None)
- **Pytest:**
  - `src/tests/test_bench_accuracy.py` → **17 passed**
  - baseline `test_bench_suite.py + test_bench_ground_truth.py + test_bench_scoring.py` → **32 passed**
- **Remaining:** steps 3–6. **Next:** Step 3 — `aggregate` over run results.
- **Constraints honoured:** nothing committed; `metrics.py`, `suite.py`, `scoring/*`,
  `analyzed-clips/`, `planes/`, results JSON all untouched.

---

## 2026-11-16 — Step 3 complete

- **Step:** 3 — `accuracy.aggregate` (+ tests).
- **Files changed:**
  - `src/test_bench/accuracy.py` (added `aggregate` + `_find_run`; `aggregate(rows,
    method_cfg, clip_ids=None)` walks a run's `rows`, selects each clip's cell matching
    `{method, n_points, selection}` (string-wise n_points so int `4` matches `"4"` and
    `"all"` matches `"all"`), and emits the per-clip table + `summary`. Each row may carry
    a `plane_doc` (the GUI loads it; the function is pure). Per-clip: `clip_id`, `n_points`,
    `in_per_px_nominal`, `in_per_px_local` (via `score.entry_px`), `score_error_in/pct`,
    `judge_error_in/pct`, `within_2/4/6` booleans on the **score** error. `summary`:
    `n_clips`, `n_with_judge`, `mean_in`, `median_in`, `mean_pct`, `median_pct`, `bins`
    (`bin_counts` over score errors), `judge_mean_in`, `judge_median_in`, `judge_mean_pct`.
    Clips with no GT / no matching cell still appear (in/px only) and are excluded from all
    error/bin/judge stats.)
  - `src/tests/test_bench_accuracy.py` (3 new tests: full 3-clip run with one judge-less and
    one GT-less clip → mean/median/bins/judge stats + per-clip exclusions; clip filter /
    missing-config cell; `"all"` n_points matching)
  - `ACCURACY_TAB_PROGRESS.md` (this entry)
- **Pytest:**
  - `src/tests/test_bench_accuracy.py` → **20 passed**
  - baseline `test_bench_suite.py + test_bench_ground_truth.py + test_bench_scoring.py` → **32 passed**
- **Remaining:** steps 4–6. **Next:** Step 4 — `AccuracyPane` (selectors, table, summary,
  stale banner) in `src/app_gui.py`, registered as a seventh tab after Compare and rendered
  against the real run `2026-11-13T16-17-56`.
- **Constraints honoured:** nothing committed; `metrics.py`, `suite.py`, `scoring/*`,
  `analyzed-clips/`, `planes/`, results JSON all untouched.

---

## 2026-11-16 — Step 4 complete

- **Step:** 4 — `AccuracyPane` (selectors, table, summary, stale banner) in
  `src/app_gui.py`, registered as a seventh tab after Compare and refreshed from
  `on_clips_changed`, rendered headless against the real run
  `2026-11-13T16-17-56`.
- **Files changed:**
  - `src/app_gui.py` (`from test_bench import accuracy as accuracy_mod`; new
    `AccuracyPane` class before `App`: run dropdown (newest-by-mtime default,
    18 results), method-config dropdown (the run's `config.methods` entries),
    extended-select clip list (default all GT clips, empty = all), a 10-column
    `ttk.Treeview` (clip · nominal in/px · local in/px @entry · score err in/% ·
    judge err in/% · ±2/±4/±6), a mean/median/within/judge summary footer, and a
    red stale banner shown only when `suite.is_stale` is true for any selected
    clip — staleness read from the on-disk `ground_truth.json`, not the run's
    embedded GT); `App.__init__` registers the pane + `on_clips_changed` calls
    `self.accuracy.refresh()`)
  - `src/tests/test_app_gui.py` (`test_app_constructs_headless`: tab count 6 → 7,
    added an `Accuracy` tab assertion — the intentional 7th-tab addition)
  - `ACCURACY_TAB_PROGRESS.md` (this entry)
- **Pytest:**
  - `src/tests/test_bench_accuracy.py` → **20 passed**
  - baseline `test_bench_suite.py + test_bench_ground_truth.py + test_bench_scoring.py` → **32 passed**
  - `src/tests/test_app_gui.py` → **5 passed, 1 failed** — the failure is the
    pre-existing known-failing `test_compare_pane_builds_review_overlays`
    (`_build_overlays()` returns only `marker` for a fitted-curve run), unrelated
    to this step and documented in the skill.
- **Smoke (headless `App` construct + refresh):** run `2026-11-13T16-17-56`
  auto-selected; 6 method-config labels; 3 clips all selected; 3 tree rows;
  `11_7_stabilized` nominal in/px = **0.28** (matches plan §6); summary renders
  mean/median/within/judge; stale banner unmapped (run not stale). `12_8` shows
  empty score/local columns for the n=4 cell (its n=4 score has
  `entry_px: null`), judge columns still populated — correct per plan §3.
- **Remaining:** steps 5–6. **Next:** Step 5 — resolution/plane warning row
  (`plane.image_size` vs clip `meta` resolution) + verify the §6 smoke checklist
  (select `arc_2d_centerline`, 3 GT clips, 0.28 in/px, aggregate ≈ 6.45 in,
  judge columns populated).
- **Constraints honoured:** nothing committed; `metrics.py`, `suite.py`,
  `scoring/*`, `analyzed-clips/`, `planes/`, results JSON all untouched.

---

## 2026-11-16 — Step 5 complete

- **Step:** 5 — resolution/plane warning row + verified §6 smoke checklist.
- **Files changed:**
  - `src/app_gui.py` (AccuracyPane: added `res_lbl` amber warning label; `_render`
    compares each clip's plane `image_size` `[w,h]` against its manifest
    `meta.width`/`height` and, on mismatch, shows "in/px assumes native-resolution
    plane: <clip ids>"; `_clear` and `_render` both `pack_forget()` it so it clears
    on every render. The `on_clips_changed` → `accuracy.refresh()` wiring was landed
    in step 4 and is confirmed present at `app_gui.py:1411`.)
  - `ACCURACY_TAB_PROGRESS.md` (this entry)
- **Pytest:**
  - `src/tests/test_bench_accuracy.py` → **20 passed**
  - baseline `test_bench_suite.py + test_bench_ground_truth.py + test_bench_scoring.py` → **32 passed**
  - `src/tests/test_app_gui.py` → **5 passed, 1 failed** (the pre-existing known
    `test_compare_pane_builds_review_overlays`, unrelated to this step)
- **§6 smoke (headless App, run `2026-11-13T16-17-56`, method `arc_2d_centerline`):**
  - 3 GT clips appear (11_7 / 12_8 / 16_7), all selected ✓
  - `11_7_stabilized` nominal in/px = **0.28** ✓
  - judge columns populated for all 3 judge-present clips ✓ (19.23 / 12.35 / 24.98 in)
  - resolution warning not fired on the real run (all planes 1920×1080 match their
    clips) ✓; mismatch branch verified headless via monkeypatch — names the 960-wide
    clip, omits the matching one.
  - **⚠ spec discrepancy:** the plan's "aggregate score error ≈ 6.45 in" is the
    **all-configs mean** (6.4485 in over 17 cells), not a value any single per-config
    cell can show. Per-config means: n=4 → 2.04 (n=2; 12_8 n=4 has `entry_px: null`),
    n=6 → 6.06, n=8 → 8.38, n=12 → 7.32, n=20/all → 6.71. Flagging for step 6: the
    ROADMAP §5.5 note must not claim a per-config "6.45 in".
- **Remaining:** step 6. **Next:** Step 6 — ROADMAP.md §5.5 doc note pointing at the
  new metrics.
- **Constraints honoured:** nothing committed; `metrics.py`, `suite.py`, `scoring/*`,
  `analyzed-clips/`, `planes/`, results JSON all untouched.

---

## 2026-11-16 — Step 6 complete

- **Step:** 6 — Doc note in `ROADMAP.md` §5.5 pointing at the new metrics.
- **Files changed:**
  - `ROADMAP.md` (added a blockquote note immediately after §5.5's mean-only "Average
    accuracy" paragraph and before the `---` at the section boundary. It states the
    run-file `averages` are mean-only, points at `src/test_bench/accuracy.py` +
    `AccuracyPane` in `src/app_gui.py` for the full distance-accuracy set — median+mean
    score error in inches and percent, ±2/±4/±6 in share, judge-vs-GT error in inches and
    percent, nominal and local in/px — and clarifies percent error is `err/GT × 100`, not
    `metrics.score_accuracy_pct`. It names `ACCURACY_TAB_PLAN.md` / `ACCURACY_TAB_PROGRESS.md`
    as plan of record / build log. It also records the §6 spec-discrepancy: the plan's
    "≈ 6.45 in" is the **all-configs mean** (6.4485 in over 17 cells), not any single
    per-config value (per-config means range 2.04 → 8.38 in).)
  - `ACCURACY_TAB_PROGRESS.md` (this entry)
- **Pytest:**
  - `src/tests/test_bench_accuracy.py` → **20 passed**
  - baseline `test_bench_suite.py + test_bench_ground_truth.py + test_bench_scoring.py` → **32 passed**
  - (combined run: **52 passed** in 12.59s)
- **§6 smoke re-verified (headless App, run `2026-11-13T16-17-56`, method
  `arc_2d_centerline`, n=4 default config):**
  - 3 GT clips appear (11_7 / 12_8 / 16_7) ✓
  - `11_7_stabilized` nominal in/px = **0.28** ✓
  - judge columns populated for all 3 judge-present clips ✓ (19.23 / 12.35 / 24.98 in)
  - "aggregate score error ≈ 6.45 in" — confirmed to be the **all-configs mean**
    (6.4485 over 17 cells); the n=4 config's mean is 2.04 in. Documented in the ROADMAP note.
- **ALL SIX STEPS COMPLETE.** Full §6 smoke checklist holds (with the single 6.45-in
  discrepancy now documented in `ROADMAP.md`). No further work in scope.
- **Constraints honoured:** nothing committed; `metrics.py`, `suite.py`, `scoring/*`,
  `analyzed-clips/`, `planes/`, results JSON all untouched.

