"""Test-bench headless CLI — ``select | calibrate | analyze | set-gt | run-suite`` (M7).

Everything reads/writes the same ``analyzed-clips/`` schema the GUI uses, so the
two are interchangeable.  ``analyze`` is the only subcommand that needs torch
(it imports ``dog_diving.detect`` lazily); ``select``, ``calibrate``, ``set-gt``
and ``run-suite`` run torch-free.

Usage::

    python cli_run.py select    <clip.mp4> [--root DIR] [--id ID]
    python cli_run.py calibrate <clip_id>  --setup-id ID [--frame-index N] [--root DIR]
    python cli_run.py analyze   <clip_id>  [--root DIR] [--model PATH]
    python cli_run.py set-gt    <clip_id>  --frame N --x X --y Y [--judge IN] [--root DIR]
    python cli_run.py run-suite --config <suite.json> [--root DIR]

``calibrate`` launches the parent's ``pool_calibrate_tool`` (semi-manual plane
placement) on the clip's first frame; the saved plane lands in
``test-bench/planes/<setup-id>.pool.json`` and is assigned to the clip.
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
    plane = man_mod.clip_plane(man)
    gt = (man_mod.read_json(cdir / "ground_truth.json")
          if (cdir / "ground_truth.json").is_file() else {"gt_in": None, "judge_present": False})
    meta = man["meta"]
    return {"clip_id": clip_id, "points": points, "plane_doc": plane, "gt": gt,
            "frame_size": [int(meta["width"]), int(meta["height"])]}


def cmd_select(args) -> int:
    man = ingest_mod.ingest_clip(args.clip, args.root, clip_id=args.id)
    print(json.dumps({"clip_id": man["clip_id"], "dir": str(man_mod.clip_dir(args.root, man["clip_id"])),
                      "kind": man["source"]["kind"], "sha256": man["source"]["sha256"]}))
    return 0


def cmd_calibrate(args) -> int:
    import pool_calibrate_tool as calib_tool
    cdir = man_mod.clip_dir(args.root, args.clip_id)
    man = man_mod.read_manifest(cdir)
    clip_file = man_mod.clip_video_path(man, cdir)
    if clip_file is None:
        raise SystemExit(f"clip file not found for {args.clip_id} "
                         f"(original_path: {man.get('source', {}).get('original_path')!r})")
    out_path = man_mod.plane_setup_path(args.setup_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rc = calib_tool.main(["--clip", str(clip_file), "--frame-index", str(args.frame_index),
                          "--setup-id", args.setup_id, "--out", str(out_path)])
    if rc != 0:
        return rc
    if not out_path.is_file():
        print("calibration did not produce a plane file", file=sys.stderr)
        return 2
    man["plane"] = {"setup_id": args.setup_id, "present": True}
    man_mod.write_manifest(man, cdir)
    print(json.dumps({"clip_id": args.clip_id, "plane": args.setup_id,
                      "plane_path": str(out_path)}))
    return 0


def cmd_use_plane(args) -> int:
    cdir = man_mod.clip_dir(args.root, args.clip_id)
    man = man_mod.read_manifest(cdir)
    if not man_mod.plane_setup_path(args.setup_id).is_file():
        raise SystemExit(f"plane {args.setup_id!r} not found in the planes folder")
    man["plane"] = {"setup_id": args.setup_id, "present": True}
    man_mod.write_manifest(man, cdir)
    print(json.dumps({"clip_id": args.clip_id, "plane": args.setup_id}))
    return 0


def cmd_analyze(args) -> int:
    from test_bench import analyze as analyze_mod          # torch-free
    from dog_diving import detect as detect_mod           # lazy: needs ultralytics
    from dog_diving import measure as measure_mod

    cdir = man_mod.clip_dir(args.root, args.clip_id)
    man = man_mod.read_manifest(cdir)
    if man["source"]["kind"] != "mp4":
        raise SystemExit("analyze of SVO2 needs the ZED X SDK (not wired)")
    clip_file = man_mod.clip_video_path(man, cdir)
    if clip_file is None:
        raise SystemExit(f"clip file not found for {args.clip_id} "
                         f"(original_path: {man.get('source', {}).get('original_path')!r})")
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
    plane = man_mod.clip_plane(man)
    if plane is not None:
        meas = measure_mod.plane_distance(traj, plane)
        man["plane"]["present"] = True
    man_mod.write_manifest(man, cdir)
    print(json.dumps({"clip_id": args.clip_id, "n_detections": det.n_detections,
                      "n_accepted": det.n_accepted,
                      "measurement": None if meas is None else
                      {k: meas.get(k) for k in ("total_inches", "x_px", "y_px", "frame")}}))
    return 0


def cmd_set_gt(args) -> int:
    cdir = man_mod.clip_dir(args.root, args.clip_id)
    man = man_mod.read_manifest(cdir)
    plane = man_mod.clip_plane(man)
    if plane is None:
        raise SystemExit(f"no plane assigned to {args.clip_id}; run 'calibrate' first")
    gt = gt_mod.set_ground_truth(plane, frame=args.frame,
                                 x_px=args.x, y_px=args.y, judge_in=args.judge,
                                 updated=_now())
    gt_mod.write_ground_truth(gt, cdir)
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
        plane = man_mod.clip_plane(man)
        gt = man_mod.read_json(cdir / "ground_truth.json") if (cdir / "ground_truth.json").is_file() else None
        prov[cid] = suite_mod.clip_provenance(man, plane, gt)
    res = suite_mod.run_suite(clips, cfg["methods"])
    run_id = _now_compact()
    out = root / "results" / f"{run_id}.json"
    man_mod.write_json(out, {"run_id": run_id, "config": cfg, "provenance": prov, **res})
    print(json.dumps({"run_id": run_id, "out": str(out), "n_clips": len(res["rows"]),
                      "averages": res["averages"]["overall"]}))
    return 0


def cmd_deselect(args) -> int:
    from test_bench import analyze as analyze_mod  # torch-free
    cdir = man_mod.clip_dir(args.root, args.clip_id)
    man = man_mod.read_manifest(cdir)
    frames = sorted({int(f) for f in (man.get("excluded_frames") or [])} |
                    {int(f) for f in args.frames})
    analyze_mod.deselect_detections(cdir, args.clip_id, frames)
    print(json.dumps({"clip_id": args.clip_id,
                      "excluded_frames": frames}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli_run.py", description="Dog-diving test bench (headless).")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("select", help="copy a clip into analyzed-clips/")
    ps.add_argument("clip")
    ps.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ps.add_argument("--id", default=None)
    ps.set_defaults(func=cmd_select)

    pc = sub.add_parser("calibrate", help="semi-manually place the pool plane (saves to planes/)")
    pc.add_argument("clip_id")
    pc.add_argument("--setup-id", required=True)
    pc.add_argument("--frame-index", type=int, default=0)
    pc.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pc.set_defaults(func=cmd_calibrate)

    pu = sub.add_parser("use-plane", help="assign an already-calibrated plane (from planes/) to a clip")
    pu.add_argument("clip_id")
    pu.add_argument("--setup-id", required=True)
    pu.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pu.set_defaults(func=cmd_use_plane)

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

    pd = sub.add_parser("deselect", help="mark detection frames as faulty (excluded from scoring)")
    pd.add_argument("clip_id")
    pd.add_argument("--frames", type=int, nargs="+", required=True,
                    help="frame numbers of the detections to deselect")
    pd.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pd.set_defaults(func=cmd_deselect)

    pr = sub.add_parser("run-suite", help="run a method grid over prepared clips")
    pr.add_argument("--config", required=True)
    pr.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    pr.set_defaults(func=cmd_run_suite)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
