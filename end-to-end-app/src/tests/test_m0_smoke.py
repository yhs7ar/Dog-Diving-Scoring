"""M0 verification gate (PLAN.md S11 M0 / S8 V1).

Gate: models/best.pt loads, kpt_shape == [2, 3], one-frame inference returns a
tailset point (keypoint index 1), using the reference recipe's defaults.

Recipe preserved from reference/Dog-Diving-Scoring/draw_tail_trajectory.py:
    CLAHE on L of LAB, clipLimit=2.5, tileGridSize=(8, 8)
    model.track(imgsz=..., conf=0.12, max_det=1, tracker='bytetrack.yaml',
                persist=True, verbose=False)
    accept tailset when keypoint confidence > 0.3
    TAILSET_INDEX = 1
Two deliberate deviations from the reference script (PLAN S1.1):
    device='cpu' (no CUDA on this machine), and no I:\\Capstone\\... paths.

FRAME SELECTION (fixed 2026-10-07 — this gate previously failed on a sampling
artefact, not on the model). The tailset is only confidently detected while the
dog is airborne/visible, which is a small fraction of a clip: measured on
18-07.mp4 at imgsz=1280, only ~11% of frames yield any detection. The original
gate sampled 6 frames uniformly at 5/20/35/50/65/80 % of the clip and asserted
one of them cleared kp_conf, so it tested the dice, not the model — it missed a
window whose peak confidence is 0.978. The gate now SCANS the clip every
SCAN_STRIDE frames and asserts at least one frame clears KP_CONF. The threshold
is unchanged; only the frame-selection was made non-arbitrary.

Run:  <venv-python> -m pytest src/tests/test_m0_smoke.py -s
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The model *actually used*, not a hard-coded vendored name: this gate exists to fail loudly when
# the checkpoint's kpt_shape/names are wrong, so it has to be looking at config's choice
# (``models/yolo26n-pose_dogs_v5.pt`` since 2026-10-19; ``DD_MODEL`` overrides).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_diving.config import MODEL_PATH  # noqa: E402
# The clip is the one this tree ships (``data/clips``); the gate used to reach into the
# read-only ``reference/Dog-Diving-Scoring`` clone, which is not part of the tree.
CLIP_DIR = PROJECT_ROOT / "data" / "clips"

# --- reference recipe defaults (PLAN S6 config surface) ---------------------
IMGSZ = 1280          # PLAN S6 default; repo's own value was 1920
CONF = 0.12           # repo value
KP_CONF = 0.30        # repo value, tailset acceptance
TRACKER = "bytetrack.yaml"
DEVICE = "cpu"        # forced: no NVIDIA GPU on this machine
CLAHE_CLIP = 2.5
CLAHE_GRID = (8, 8)
TAILSET_INDEX = 1
NOSE_INDEX = 0
SCAN_STRIDE = 3       # scan every Nth frame (see FRAME SELECTION above)

KPT_SHAPE = [2, 3]    # 2 keypoints x (x, y, conf)


def clahe_lab_l(frame: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB, as draw_tail_trajectory.py does."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_enh = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_enh, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def pick_clip() -> Path:
    if not CLIP_DIR.is_dir():
        raise FileNotFoundError(f"reference clips missing: {CLIP_DIR}")
    clips = sorted(p for p in CLIP_DIR.glob("*.mp4") if "Slow Mo" not in p.name)
    if not clips:
        clips = sorted(CLIP_DIR.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"no .mp4 clips in {CLIP_DIR}")
    return clips[0]


def main() -> int:
    from ultralytics import YOLO

    print(f"python      : {sys.version.split()[0]}")
    print(f"model path  : {MODEL_PATH}")
    assert MODEL_PATH.is_file(), f"missing vendored model: {MODEL_PATH}"
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"model size  : {size_mb:.1f} MB")

    t0 = time.perf_counter()
    model = YOLO(str(MODEL_PATH))
    print(f"load time   : {time.perf_counter() - t0:.2f}s")

    kpt_shape = list(model.model.kpt_shape)
    names = model.model.names
    print(f"kpt_shape   : {kpt_shape}")
    print(f"names       : {names}")
    assert kpt_shape == KPT_SHAPE, f"kpt_shape {kpt_shape} != {KPT_SHAPE}"
    assert names == {0: "dog"}, f"unexpected class names: {names}"
    print("checkpoint identity: kpt_shape + names PASS\n")

    clip = pick_clip()
    cap = cv2.VideoCapture(str(clip))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"clip        : {clip.name}  {w}x{h} @ {fps:.2f} fps, {n_frames} frames")

    # Scan the whole clip; the airborne window is a small fraction of frames.
    stride = max(1, SCAN_STRIDE)
    n_scanned = n_det = n_above = 0
    per_frame_ms: list[float] = []
    best = None
    print(f"\nscanning every {stride} frame(s): imgsz={IMGSZ} conf={CONF} kp_conf>{KP_CONF}")
    print(f"\n{'frame':>6} {'tailset x,y':>18} {'kp_conf':>8} {'box_conf':>9}  ms")

    t_scan = time.perf_counter()
    cap = cv2.VideoCapture(str(clip))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            n_scanned += 1
            enhanced = clahe_lab_l(frame)
            t1 = time.perf_counter()
            res = model.track(
                source=enhanced, imgsz=IMGSZ, conf=CONF, max_det=1,
                tracker=TRACKER, persist=True, device=DEVICE, verbose=False,
            )
            dt_ms = (time.perf_counter() - t1) * 1000
            per_frame_ms.append(dt_ms)
            r = res[0]
            kp = getattr(r.keypoints, "data", None)
            xy = None if kp is None or len(kp) == 0 else kp[0].tolist()
            if xy is None:
                idx += 1
                continue
            n_det += 1
            bconf = float(r.boxes.conf[0]) if (r.boxes is not None and len(r.boxes)) else float("nan")
            tx, ty, tc = xy[TAILSET_INDEX]
            nx, ny, nc = xy[NOSE_INDEX]
            accepted = tc > KP_CONF
            if accepted:
                n_above += 1
            print(f"{idx:>6} {f'{tx:.0f},{ty:.0f}':>18} {tc:>8.3f} {bconf:>9.3f}  {dt_ms:.0f}"
                  f"{'   <- ACCEPTED' if accepted else ''}")
            if accepted and (best is None or tc > best[3]):
                best = (idx, tx, ty, tc, nx, ny, nc, dt_ms)
        idx += 1
    cap.release()

    scan_s = time.perf_counter() - t_scan
    med_ms = float(np.median(per_frame_ms)) if per_frame_ms else float("nan")
    print(f"\nscanned {n_scanned} frames in {scan_s:.1f}s "
          f"| detections {n_det} ({100 * n_det / max(n_scanned, 1):.1f}%) "
          f"| tailset>{KP_CONF}: {n_above}")

    assert best is not None, (
        f"no frame among {n_scanned} scanned (every {stride}) produced a tailset "
        f"keypoint above kp_conf={KP_CONF}; gate FAILED"
    )
    idx, tx, ty, tc, nx, ny, nc, dt_ms = best
    print("\nM0 GATE:")
    print(f"  kpt_shape        == {KPT_SHAPE}          PASS")
    print(f"  names            == {{0: 'dog'}}    PASS")
    print(f"  tailset point    frame {idx}: ({tx:.1f}, {ty:.1f}) conf={tc:.3f}  PASS")
    print(f"  nose point       frame {idx}: ({nx:.1f}, {ny:.1f}) conf={nc:.3f}")
    print(f"  accepted frames  {n_above}/{n_scanned} scanned  "
          f"(detection rate {100 * n_det / max(n_scanned, 1):.1f}%)")
    print(f"  cpu inference    imgsz={IMGSZ}: median {med_ms:.0f} ms/frame "
          f"({1000 / med_ms:.2f} fps)")
    print(f"  whole-clip ETA   {n_frames} frames @ {med_ms / 1000:.3f} s/frame "
          f"= {n_frames * med_ms / 1000 / 60:.1f} min")
    print("M0 PASS")
    return 0


def test_m0_gate() -> None:
    """pytest entry point — same checks as main()."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
