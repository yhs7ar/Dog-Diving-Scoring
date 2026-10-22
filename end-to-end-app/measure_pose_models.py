"""Ad-hoc A/B/C: run the shipped clip through every pose checkpoint the registry declares.

Not part of the suite — a one-off measurement so a claim like "the s-scale fine-tune runs" comes
with numbers (detections, accepted points, ms/frame, projected full-clip time, the resolved
keypoint layout) instead of a shrug.  Accuracy belongs in ``eval_frames.py`` against the labelled
still set; this is throughput and "does the pipeline accept it at all".

The checkpoint list is **not declared here** (2026-10-22).  It is ``config.MODEL_REGISTRY`` — the
same registry ``config.MODEL_PATH``, the GUI's backbone selector and ``eval_frames --scale`` read —
so adding or replacing a checkpoint is one ``ModelSpec`` in ``config.py`` and this script follows.
That also removed the drift the old hardcoded list invited: it named files by hand, so a checkpoint
could be selectable in the GUI and absent from the A/B, or worse, listed under a stale label.

Run:  <venv-python> measure_pose_models.py [--imgsz 1280] [--scale n|s] [--models a.pt b.pt]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dog_diving import detect as D  # noqa: E402
from dog_diving.config import MODEL_REGISTRY, SCALES, spec_for_path  # noqa: E402

CLIP = ROOT / "data" / "clips" / "video004_right-00.00.36.812-00.00.38.328-seg01.mp4"


def registry_models(scale: str | None = None) -> list[tuple[str, Path]]:
    """``(label, checkpoint)`` for every declared checkpoint, model-of-record first.

    Registry order is kept: the n-scale model of record leads, the alternative(s) follow, and the
    stock COCO person checkpoint comes last because it is the case that needs a derived tailset —
    it has no dog class, so on Dog Diving footage it tracks the handler.
    """
    return [
        (f"{spec.label} ({spec.scale}-scale, {spec.role})", spec.path)
        for spec in MODEL_REGISTRY
        if scale is None or spec.scale == str(scale)
    ]


def run_one(label: str, path: Path, imgsz: int, out: Path) -> None:
    if not path.is_file():
        print(f"\n{label}: MISSING {path}")
        return
    cfg = D.DetectConfig(model_path=path, imgsz=imgsz)
    t0 = time.perf_counter()
    res = D.detect_clip(CLIP, cfg, out_dir=out, use_cache=False)
    wall = time.perf_counter() - t0
    with_kp = sum(1 for r in res.records if r.get("tailset"))
    boxes = sum(1 for r in res.records if r.get("box_conf") is not None)
    spec = spec_for_path(path)
    print(f"\n{label}")
    print(f"  checkpoint     : {path.name}")
    if spec is not None:
        measured = spec.measured_line()
        print(f"  registry       : {spec.scale}-scale {spec.role}"
              + (f" — labelled-set {measured}" if measured else " — not measured on the labelled set"))
    print(f"  layout         : {res.model_layout}  tailset_derived={res.tailset_derived}")
    print(f"  frames         : {res.n_frames_processed}  wall {wall:.1f}s  "
          f"({res.timings['ms_per_frame']:.0f} ms/frame)")
    print(f"  projected      : {res.timings['projected_clip_s']:.1f}s for the whole clip")
    print(f"  boxes          : {boxes}  tailset points {with_kp}  accepted {res.n_accepted}")
    confs = [round(r["tailset"][2], 3) for r in res.records if r.get("tailset")]
    if confs:
        print(f"  tailset conf   : max {max(confs)}  median {sorted(confs)[len(confs) // 2]}")
    else:
        print("  tailset conf   : (no tailset point in any frame)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--scale", choices=SCALES, default=None,
                    help="measure only this backbone scale (the registry's dog-family split)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="override the checkpoint list (paths or names inside models/) — for "
                         "checkpoints that are not (yet) declared in config.MODEL_REGISTRY")
    args = ap.parse_args(argv)

    if args.models:
        models = [(Path(m).name, Path(m) if Path(m).is_absolute() else ROOT / "models" / m)
                  for m in args.models]
    else:
        models = registry_models(args.scale)
    if not models:
        print(f"nothing to measure: the registry declares no checkpoint for scale {args.scale!r}")
        return 2

    out = Path(tempfile.mkdtemp(prefix="dd-layout-ab-"))
    print(f"clip   : {CLIP.name}")
    print(f"imgsz  : {args.imgsz}   out: {out}")
    print(f"models : {len(models)} from config.MODEL_REGISTRY"
          + (f" (scale {args.scale})" if args.scale else ""))
    for label, path in models:
        run_one(label, path, args.imgsz, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
