# Dock Diving Scoring End-to-End

Automated dock diving distance scoring from video: the dog's tail-set is detected with a
fine-tuned YOLO pose model, and the distance is read off a **plane aligned on the water surface** —
`X` in feet from the dock lip, through the calibration document's own homography.

This repository is the app plus everything needed to run it. It is not the training data, the
augmentation/training scripts, or the annotated corpus.

## What it does

1. **Load** a clip.
2. **Calibrate** — align the water-surface plane (a quad on the pool, `X = 0` on the dock lip) by
   dragging its corners and sides, or let the **waterline detector place it** (below).
3. **Process** — per-frame pose detection → tail-set track.
4. **Review / Results** — the measurement rule is: *the last accepted tail-set detection is the point
   where the dog meets the plane; its `X` is the reported distance* (PLAN §16.2).

## Requirements

Python 3.11+, plus:

```
pip install -r requirements.txt
```

`tkinter` ships with CPython on Windows. `opencv-python` and `numpy` are also all the vendored
waterline detector needs.

## Running

```
run_app.cmd                                   # no clip; use "Open clip..." in the GUI
run_app.cmd --clip data\clips\video004_right-00.00.36.812-00.00.38.328-seg01.mp4
```

`run_app.cmd` is the single source of truth for the interpreter: set `DD_PY` to override it. The
project's own `.venv` cannot be used from a cloud-synced drive, which is why the launcher takes an
explicit path.

`DD_MODEL` selects a checkpoint inside `models/` (default `yolo26n-pose_dogs_v5.pt`); the path is part
of the detection cache key, so switching it forces a re-run.

`DD_MODEL_SCALE=n|s` selects the same thing by **backbone scale** through `config.MODEL_REGISTRY` (one
declared entry per checkpoint: scale, keypoint layout, role, and its measured accuracy). On the
Process tab the **backbone** selector beside *pose model* does the same job, and picking a file moves
the scale box — the file is always what runs, the scale is a way of choosing it. Only the dog-family
checkpoints are reachable by scale; the stock COCO person model stays a file-level choice because it
has no tailset keypoint. `DD_MODEL` beats `DD_MODEL_SCALE`, and `python src/eval_frames.py --scale s`
scores a scale's checkpoint without naming its file.

## Automatic plane placement from the waterline detector (PLAN §17)

Press **Detect waterline** in the Calibrate pane. Nothing runs automatically, on clip load or
otherwise, and a refused seed leaves the plane exactly as it was.

What it does: a temporal **noise map** over a calm window separates water (flickers) from structure
(dock, wall, coping — static), the mask's boundaries are fitted as lines, and the lines are converted
into the plane's four corners:

| edge | plane line | where it comes from |
|---|---|---|
| `far`  | `Y = 0` | water against the far wall (per-column top boundary) |
| `near` | `Y = W` | water against the near wall (per-column bottom boundary) |
| `dock` | `X = 0` | water against the dock face (per-row boundary) |

The metric origin is their intersection; the width comes from `near ∩ dock`; the X scale comes from
where the far waterline runs out of shot. **There is no metric solve** — the operator finishes with
the existing corner drag / side drag / arrow pad, and **Undo seed** restores the previous corners.

Honest limits, measured on the included clip:

* the **dock edge is the weak one** — banner skirts hang into the water and a jittery mask often
  cannot resolve the water/dock contact. Seed level `L1` (far + near, dock inherited) is the normal
  outcome, and gate **S6** reports that `X = 0` is then a default, not a measurement;
* with no run yet, the pane does not know which side the dock is on — set the **dock side** control,
  or run the clip first so the first detections can supply it;
* a clip with no calm stretch before the jump, or with dark-on-dark water, may be refused outright.
  That is the intended behaviour: the log names the failing gate and its measured value.

The detector is vendored **unmodified** from <https://github.com/quintinlevy/Waterline-Detection>
(commit `e0c4126`) under `src/third_party/waterline/`, with `SHA256SUMS` and `PROVENANCE.md`. The
adapter (`src/dog_diving/waterline.py`) reuses its functions and adds the three-edge extension; the
vendored file is never edited. Every seeded plane's calibration document records the upstream commit
and the file's sha256.

## Review and placement tools

Added 2026-10-21, all of them on the existing controls rather than beside them:

- **Flip X** (Calibrate) reverses the plane's X direction in place, at any time — seeded or hand-placed.
  A plane whose `X = 0` sits on the wrong end measures every distance *towards* the dock, and no gate
  can see it (P4's foreshortening is symmetric under the mirror).
- **Review grid** is blended above the landing row and left at full strength below it, so the dog on
  its final frame is not covered by the lattice behind it.
- **Click a detection in Review** to take it in or out of the fitted arc (hollow marker = excluded).
  The measurement is unaffected — PLAN §16.2 takes the final detection, not the arc — the arc is refit
  from what remains.
- **Trajectory strokes** now follow the plane overlay's weights (1-2 px, small handles) instead of the
  vendored reference's 4 px polyline, which covered the dog it was annotating.
- **6 Plan** tab: the aligned plane flattened to a rectangle at one equal scale, with the landing point
  marked in feet. A prototype — the plane *is* a metric rectangle, so no perspective trick is needed.

## Layout

```
run_app.cmd                 launcher (interpreter selection)
src/app_gui.py              the Tk app
src/cli_run.py              headless equivalent
src/dog_diving/               pipeline: video, detect, trajectory, measure, annotate, pool/ (plane, seed,
                            homography, calib), waterline.py (three-edge adapter)
src/third_party/waterline/  vendored upstream detector (unmodified) + provenance
src/tests/                  test suite (pytest, `pythonpath = src`)
models/                     YOLO pose checkpoints + provenance
data/clips/                 the one clip this repo ships
docs/PLAN_WATERLINE_SEED.md design note for the automatic placement feature
```

## Tests

```
python -m pytest src/tests -q --ignore=src/tests/test_m8_gui_e2e.py
```

`test_m8_gui_e2e.py` cannot run on every host (a sympy import crash before any test body); the rest
of the suite, including the Tk pane tests, does.

## Licence

No licence is granted for this repository. `src/third_party/waterline/` is upstream code that ships
without a licence file; see its `PROVENANCE.md` before redistributing.
