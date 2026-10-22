"""Score the pose model on a labelled frame set (CVAT pose export) — keypoint error, not vibes.

    python src/eval_frames.py --frames "<dir or .zip>" [--model <pt> | --scale n|s] [--imgsz 1280]
                              [--conf .12] [--kp-conf .30] [--out DIR] [--overlays N]

``--scale`` is the registry route (``config.default_for_scale``): it scores that backbone's declared
dog checkpoint without the caller having to remember its file name.  ``--model`` still wins, for
checkpoints that are not declared in ``config.MODEL_REGISTRY``.

For every frame with ground-truth keypoints it reports the tailset and nose error in pixels
(mean / median / p90 / max), the same error normalised by the dog box diagonal, and the share of
predictions inside 5 % / 10 % / 20 % of that diagonal.  Then it builds the trajectory from the
accepted points exactly as the app does, so the trajectory is exercised on this data too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from dog_diving import config as C
from dog_diving import detect as D
from dog_diving import image_set as IS
from dog_diving import trajectory as T


def _stats(err: np.ndarray) -> dict:
    if err.size == 0:
        return {"n": 0}
    return {"n": int(err.size), "mean": round(float(err.mean()), 2),
            "median": round(float(np.median(err)), 2),
            "p90": round(float(np.percentile(err, 90)), 2), "max": round(float(err.max()), 2)}


def evaluate(fs: IS.FrameSet, cfg, model, out_dir: Path, overlays: int = 6) -> dict:
    det = IS.detect_frames(fs, cfg, model=model)
    traj = T.build_trajectory(det, clip_id=fs.root.name)

    tail_e, nose_e, tail_n, nose_n, diag = [], [], [], [], []
    hit5 = hit10 = hit20 = 0
    rows = []
    for r in det.records:
        gt = r.get("gt")
        if not gt or "tailset" not in gt or r.get("tailset") is None:
            continue
        w, h = gt["box"][2], gt["box"][3]
        d = float(np.hypot(w, h))
        if d <= 1:
            continue
        tx = float(np.hypot(r["tailset"][0] - gt["tailset"][0], r["tailset"][1] - gt["tailset"][1]))
        nx = float(np.hypot(r["nose"][0] - gt["nose"][0], r["nose"][1] - gt["nose"][1]))
        tail_e.append(tx); nose_e.append(nx); diag.append(d)
        tail_n.append(tx / d); nose_n.append(nx / d)
        hit5 += tx / d <= 0.05
        hit10 += tx / d <= 0.10
        hit20 += tx / d <= 0.20
        rows.append({"frame": r["frame"], "tailset_err_px": round(tx, 1), "nose_err_px": round(nx, 1),
                     "box_diag_px": round(d, 1), "tailset_err_rel": round(tx / d, 4),
                     "kp_conf": round(float(r["tailset"][2]), 3)})

    n_match = len(tail_e)
    res = {
        "frames": fs.n_frames,
        "frames_with_gt": len(fs.labels),
        "frames_with_detection": det.n_detections,
        "frames_accepted": det.n_accepted,
        "frames_scored": n_match,
        "tailset_err_px": _stats(np.array(tail_e, float)),
        "nose_err_px": _stats(np.array(nose_e, float)),
        "tailset_rel_err": _stats(np.array(tail_n, float)),
        "nose_rel_err": _stats(np.array(nose_n, float)),
        "tailset_within_5pct": round(hit5 / n_match, 3) if n_match else None,
        "tailset_within_10pct": round(hit10 / n_match, 3) if n_match else None,
        "tailset_within_20pct": round(hit20 / n_match, 3) if n_match else None,
        "per_frame": rows,
        "trajectory": {"points": traj.n_points, "runs": len(traj.runs),
                       "airborne": traj.airborne_frame_range,
                       "arc_rms_px": None if traj.arc_rms_px is None else round(traj.arc_rms_px, 2),
                       "inliers": len(traj.inliers), "outliers": traj.n_outliers},
        "model": str(cfg.model_path),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(res, indent=2))

    # overlays: GT green, prediction red, for the N frames with the largest tailset error
    ranked = sorted(rows, key=lambda r: -r["tailset_err_px"])[:overlays]
    by_frame = {r["frame"]: r for r in det.records}
    for k, rr in enumerate(ranked):
        i = fs.numbers.index(rr["frame"])
        img = fs.read(i)
        gt = by_frame[rr["frame"]]["gt"]
        pred = by_frame[rr["frame"]]["tailset"]
        x0, y0, bw, bh = gt["box"]
        cv2.rectangle(img, (int(x0 - bw / 2), int(y0 - bh / 2)),
                      (int(x0 + bw / 2), int(y0 + bh / 2)), (0, 255, 0), 2)
        for name, col in (("nose", (0, 200, 0)), ("tailset", (0, 255, 0))):
            gx, gy = gt[name]
            cv2.circle(img, (int(gx), int(gy)), 6, col, 2)
        cv2.drawMarker(img, (int(pred[0]), int(pred[1])), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.line(img, (int(gt["tailset"][0]), int(gt["tailset"][1])),
                 (int(pred[0]), int(pred[1])), (255, 0, 255), 1)
        cv2.putText(img, f"f{rr['frame']} tail_err {rr['tailset_err_px']:.0f}px "
                         f"({100 * rr['tailset_err_rel']:.1f}% of box)", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, f"f{rr['frame']} tail_err {rr['tailset_err_px']:.0f}px "
                         f"({100 * rr['tailset_err_rel']:.1f}% of box)", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"err_{k:02d}_f{rr['frame']}.png"), img)

    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="image directory or the CVAT .zip")
    ap.add_argument("--model", default=None,
                    help="checkpoint path or file name inside models/ "
                         "(default: config.MODEL_PATH — DD_MODEL / DD_MODEL_SCALE aware)")
    ap.add_argument("--scale", choices=C.SCALES, default=None,
                    help="score this backbone's declared dog checkpoint (registry); --model wins")
    ap.add_argument("--imgsz", type=int, default=C.DEFAULT_IMGSZ)
    ap.add_argument("--conf", type=float, default=C.DEFAULT_CONF)
    ap.add_argument("--kp-conf", type=float, default=C.DEFAULT_KP_CONF)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overlays", type=int, default=6)
    args = ap.parse_args(argv)

    if args.model:
        model_path = C.resolve_model_path(args.model)
    elif args.scale:
        model_path = C.default_for_scale(args.scale).path
    else:
        model_path = Path(C.MODEL_PATH)

    fs = IS.FrameSet.open(args.frames)
    cfg = D.DetectConfig(model_path=model_path, imgsz=args.imgsz, conf=args.conf,
                         kp_conf=args.kp_conf, clahe=not args.no_clahe)
    model = D.load_model(cfg)
    spec = C.spec_for_path(model_path)
    print(f"frames {fs.n_frames} from {fs.root}  ({len(fs.labels)} with ground truth)", flush=True)
    print(f"model  {model_path.name}  ({spec.scale}-scale, {spec.role})" if spec is not None
          else f"model  {model_path.name}  (undeclared checkpoint)", flush=True)

    out_dir = args.out or (C.OUT_DIR / f"{Path(args.frames).stem}_eval")
    res = evaluate(fs, cfg, model, out_dir, overlays=args.overlays)
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=2))
    print(f"\nartifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
