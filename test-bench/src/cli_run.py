"""Test-bench headless CLI — ``ingest | analyze | set-gt | run-suite`` (M7).

Everything reads/writes the same ``analyzed-clips/`` schema the GUI uses, so the
two are interchangeable.  ``analyze`` is the only subcommand that needs torch
(it imports ``dog_diving.detect`` lazily); ``ingest``, ``set-gt`` and
``run-suite`` run torch-free.

Usage::

    python cli_run.py ingest   <clip.mp4> [--root DIR] [--id ID]
    python cli_run.py analyze  <clip_id>  [--root DIR] [--model PATH]
    python cli_run.py set-gt   <clip_id>  --frame N --x X --y Y [--judge IN] [--root DIR]
    python cli_run.py run-suite --config <suite.json> [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_bench import ground_truth as gt_mod  # noqa: E402
from test_bench import ingest as ingest_mod  # noqa: E402
from test_bench import manifest as man_mod  # noqa: E402
from test_bench import suite as suite_mod  # noqa: E402
from dog_diving import trajectory as traj_mod  # noqa: E402

DEFAULT_ROOT = SRC.parent / "analyzed-clips"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def _load_prepared(root: Path, clip_id: str) -> dict:
    cdir = man_mod.clip_dir(root, clip_id)
    man = man_mod.read_manifest(cdir)
    records = man_mod.read_json(cdir / "detections.json")
    points = traj_mod.assemble(records)
    plane = (man_mod.read_json(cdir / "plane.json")
             if (cdir / "plane.json").is_file() else None)
    gt = (man_mod.read_json(cdir / "ground_truth.json")
          if (cdir / "ground_truth.json").is_file() else {"gt_in": None, "judge_present": False})
    meta = man["meta"]
    return {"clip_id": clip_id, "points": points, "plane_doc": plane, "gt": gt,
            "frame_size": [int(meta["width"]), int(meta["height"])]}


def cmd_ingest(args) -> int:
    man = ingest_mod.ingest_clip(args.clip, args.root, clip_id=args.id)
    print(json.dumps({"clip_id": man["clip_id"], "dir": str(man_mod.clip_dir(args.root, man["clip_id"])),
                      "kind": man["source"]["kind"], "sha256": man["source"]["sha256"]}))
    return 0


def cmd_analyze(args) -> int:
    from test_bench import analyze as analyze_mod          # torch-free
    from dog_diving import detect as detect_mod           # lazy: needs ultralytics
    from dog_diving import measure as measure_mod

    cdir = man_mod.clip_dir(args.root, args.clip_id)
    man = man_mod.read_manifest(cdir)
    if man["source"]["kind"] != "mp4":
        raise SystemExit("analyze of SVO2 needs the ZED X SDK (not wired)")
    clip_file = cdir / man["source"]["clip_file"]
    cfg = detect_mod.DetectConfig(model_path=Path(args.model)) if args.model else detect_mod.DetectConfig()
    det = detect_mod.detect_clip(clip_file, cfg)

    man_mod.write_json(cdir / "detections.json", det.records)
    w, h = int(man["meta"]["width"]), int(man["meta"]["height"])
    traj = analyze_mod.build_trajectory(det.records, clip_id=args.clip_id, frame_size=(w, h))
    man_mod.write_json(cdir / "trajectory.json", traj.as_dict())
    man["detections"] = {"path": "detections.json", "model_sha": det.model_sha,
                         "n_detections": det.n_detections, "n_accepted": det.n_accepted}
    man["trajectory"] = {"path": "trajectory.json", "n_points": traj.n_points}

    meas = None
    if (cdir / "plane.json").is_file():
        meas = measure_mod.plane_distance(traj, man_mod.read_json(cdir / "plane.json"))
        man["plane"]["present"] = True
    man_mod.write_manifest(man, cdir)
    print(json.dumps({"clip_id": args.clip_id, "n_detections": det.n_detections,
                      "n_accepted": det.n_accepted,
                      "measurement": None if meas is None else
                      {k: meas.get(k) for k in ("total_inches", "x_px", "y_px", "frame")}}))
    return 0


def cmd_set_gt(args) -> int:
    cdir = man_mod.clip_dir(args.root, args.clip_id)
    plane_p = cdir / "plane.json"
    if not plane_p.is_file():
        raise SystemExit(f"no plane.json in {cdir}; assign a plane before placing GT")
    gt = gt_mod.set_ground_truth(man_mod.read_json(plane_p), frame=args.frame,
                                 x_px=args.x, y_px=args.y, judge_in=args.judge,
                                 updated=_now())
    gt_mod.write_ground_truth(gt, cdir)
    man = man_mod.read_manifest(cdir)
    man["ground_truth"] = {"path": "ground_truth.json",
                           "judge_present": gt["judge_present"], "gt_in": gt["gt_in"]}
    man_mod.write_manifest(man, cdir)
    print(json.dumps({"clip_id": args.clip_id, "gt_in": gt["gt_in"],
                      "judge_present": gt["judge_present"]}))
    return 0


def cmd_run_suite(args) -> int:
    cfg = man_mod.read_json(args.config)
    root = Path(args.root)
    clips_cfg = cfg.get("clips", "all")
    if clips_cfg == "all":
        ids = sorted(p.name for p in root.iterdir()
                     if p.is_dir() and (p / "detections.json").is_file())
    else:
        ids = list(clips_cfg)
    clips = [_load_prepared(root, cid) for cid in ids]
    prov = {}
    for cid in ids:
        cdir = root / cid
        man = man_mod.read_manifest(cdir)
        plane = man_mod.read_json(cdir / "plane.json") if (cdir / "plane.json").is_file() else None
        gt = man_mod.read_json(cdir / "ground_truth.json") if (cdir / "ground_truth.json").is_file() else None
        prov[cid] = suite_mod.clip_provenance(man, plane, gt)
    res = suite_mod.run_suite(clips, cfg["methods"])
    run_id = _now_compact()
    out = root / "results" / f"{run_id}.json"
    man_mod.write_json(out, {"run_id": run_id, "config": cfg, "provenance": prov, **res})
    print(json.dumps({"run_id": run_id, "out": str(out), "n_clips": len(res["rows"]),
                      "averages": res["averages"]["overall"]}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli_run.py", description="Dog-diving test bench (headless).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="copy a clip into analyzed-clips/")
    pi.add_argument("clip")
    pi.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pi.add_argument("--id", default=None)
    pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("analyze", help="detect + trajectory + measure a prepared clip")
    pa.add_argument("clip_id")
    pa.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pa.add_argument("--model", default=None)
    pa.set_defaults(func=cmd_analyze)

    pg = sub.add_parser("set-gt", help="place the ground-truth point + optional judge score")
    pg.add_argument("clip_id")
    pg.add_argument("--frame", type=int, required=True)
    pg.add_argument("--x", type=float, required=True)
    pg.add_argument("--y", type=float, required=True)
    pg.add_argument("--judge", type=float, default=None)
    pg.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pg.set_defaults(func=cmd_set_gt)

    ps = sub.add_parser("run-suite", help="run a method grid over prepared clips")
    ps.add_argument("--config", required=True)
    ps.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ps.set_defaults(func=cmd_run_suite)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
