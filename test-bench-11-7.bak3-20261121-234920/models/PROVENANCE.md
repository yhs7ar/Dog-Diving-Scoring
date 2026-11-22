# `models/best.pt` — provenance

Vendored copy of the authoritative 2-keypoint dog-pose checkpoint. Do not edit.

## Identity (verified)

| Field | Value |
|---|---|
| Source repo | `https://github.com/yhs7ar/Dog-Diving-Scoring` (public) |
| Commit (HEAD) | `2451f332bece14f84facc38a8854e03f8f46611a` |
| Commit date | 2026-10-01 22:17:05 -0400 |
| Commit subject | `New model with trajectory drawing and some of our data added` |
| Path in repo | `trained_models/dog_pose_s_2kpt_50epochs_720p_finetune/weights/best.pt` |
| **git blob SHA** | **`a9b79981e28fed2688d1dcdd432811391f8d5e58`** |
| SHA-256 of file | `32eb3110d7067960e4edd82b25eedf0a849e9798b8ab4ea141fb0ad3ca5df599` |
| Size | 21 712 238 bytes (21.7 MB) |
| Vendored on | 2026-10-07 |

The git blob SHA matches the value specified in `PLAN.md` §1.1 (`a9b79981…`), so this
is the checkpoint the plan was written against.

## Model facts

- Task: pose (Ultralytics YOLO26s-pose architecture; `dog_pose_s_2kpt_50epochs_720p_finetune`).
- `kpt_shape = [2, 3]` — **2 keypoints**, each `(x, y, conf)`.
- Keypoint mapping: **`0 = Nose`**, **`1 = Tailset`**. (`TAILSET_INDEX = 1`.)
- `names = {0: 'dog'}` — single class.
- Fine-tuned from `trained_models/dog_pose_s_2kpt_30epochs_720p_basedata`, which in turn
  derives its 2-kpt labels from the original 24-kpt `dog_pose_720p_50epochs` model.

`src/tests/test_m0_smoke.py` hard-asserts both `kpt_shape == [2, 3]` and
`names == {0: 'dog'}` so a wrong checkpoint fails loudly rather than silently
producing nonsense distances.

## How this copy was obtained

1. Partial-clone artifact from an earlier aborted clone was discarded.
2. Fresh **shallow + blob-filtered + sparse** clone into `reference/Dog-Diving-Scoring/`:

   ```
   git clone --depth=1 --filter=blob:none --sparse \
       https://github.com/yhs7ar/Dog-Diving-Scoring.git reference/Dog-Diving-Scoring
   git sparse-checkout set --no-cone \
       '/*.py' '/*.txt' '/*.md' '/*.yaml' '/*.json' \
       '/weights/*.pt' '/trained_models/**/weights/*.pt' '/trained_models/**/args.yaml' \
       '/Initial Dog diving videos/*.mp4' \
       '/dog-pose/*.yaml' '/dog-pose-2kpt/*.yaml' '/unscored_Pose_9-30-2026/*.yaml'
   ```
3. Verified the blob SHA against `PLAN.md` §1.1, then copied to `models/best.pt`
   and re-verified the SHA-256 of the copy.

## Reproduce / re-verify

```bash
cd reference/Dog-Diving-Scoring
git hash-object trained_models/dog_pose_s_2kpt_50epochs_720p_finetune/weights/best.pt
# expect: a9b79981e28fed2688d1dcdd432811391f8d5e58

sha256sum ../../models/best.pt
# expect: 32eb3110d7067960e4edd82b25eedf0a849e9798b8ab4ea141fb0ad3ca5df599
```

## Upstream status

`reference/Dog-Diving-Scoring` is **read-only**. Nothing is ever pushed to
`origin` (the upstream `yhs7ar/Dog-Diving-Scoring`). The reference scripts are kept
for behaviour preservation only — they are never executed directly, because
`draw_tail_trajectory.py` hard-codes a non-existent `I:\Capstone\…` path and
`device=0` (no CUDA on this machine).

## Status: superseded for the trajectory (2026-10-19)

This file is **no longer the model of record**. `PLAN.md` §12 item 3 is resolved: the trajectory
runs `models/yolo26n-pose_dogs_v5.pt`, selected by `config.MODEL_PATH` (override with `DD_MODEL`;
`best.pt` is only the fallback when the configured file is missing).

Measured on the same 46-frame labelled still set, imgsz 1280 / conf 0.12 / kp_conf 0.30:

| checkpoint | detected | tailset median px | within 20 % of box diag |
|---|---|---|---|
| `yolo26n-pose_dogs_v5.pt` (in use) | 46/46 | 8.18 | 89.1 % |
| `yolo26n-pose_dogs.pt` (local v1) | 45/46 | 10.10 | 68.9 % |
| **this file, `best.pt`** | **0/46** | — | 0 % |

At `conf 0.01` this file does fire (45/46), but its tailset lands **324 px** from ground truth
(4.4 % within 20 %), so those detections are not the dog. The identity facts above are unchanged and
still verified — the file is kept for provenance and for `DD_MODEL=best.pt` reproductions.
