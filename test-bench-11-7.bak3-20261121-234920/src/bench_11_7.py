"""Test-bench 11-7 batch driver.

Maps ``paw_scores_w_videos.csv`` into ``test-bench-11-7/analyzed-clips/`` for the
two cameras.  One plane per camera (``camera0_11-7`` /
``camera1_11-7``) is assigned to every clip of that camera by reference.

Subcommands (run from the repo root, ``python src/bench_11_7.py <cmd>``)::

    index    parse the CSV -> data/clip_index.json   (58 scored rows -> 116 clips)
    ingest   register every clip by reference + labels + judge record + plane id
    detect   fixed-threshold detect + trajectory for all 116 -> analyzed_log.csv
    apply-planes  assign the latest camera0/camera1 planes to every clip (idempotent)
    score    (after planes applied) arc_2d classify + audit -> out/audit_11-7.csv

``index``/``ingest``/``detect`` are torch-free except ``detect`` (loads ultralytics
lazily).  ``score`` needs the two 960x600 planes to have been placed in ``planes/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dog_diving import config as _config  # noqa: E402
from test_bench import ground_truth as _gt  # noqa: E402
from test_bench import ingest as _ingest  # noqa: E402
from test_bench import manifest as _manifest  # noqa: E402

PROJECT_ROOT = _config.PROJECT_ROOT           # test-bench-11-7/
ANALYZED_ROOT = PROJECT_ROOT / "analyzed-clips"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"
PLANES_DIR = PROJECT_ROOT / "planes"

def _get_csv_path() -> Path:
    local_csv = Path(r"C:\Users\harri\Videos\Dog-Diving-Videos\paw_scores_w_videos.csv")
    if local_csv.is_file():
        return local_csv
    jetson_csv = PROJECT_ROOT / "paw_scores_w_videos.csv"
    if jetson_csv.is_file():
        return jetson_csv
    return Path("/home/dogdiving/Desktop/paw_scores_w_videos.csv")

def _resolve_video_path(video_name: str, fallback_path: str) -> Path:
    if Path.home().name == "harri" or Path("C:/Users/harri").exists():
        base = Path(r"C:\Users\harri\Videos\Dog-Diving-Videos\Documents\Captstone-Repo\end-to-end-app\data\clips_11-7")
        cand = base / video_name
        if cand.is_file():
            return cand
    else:
        base = Path("/home/dogdiving/Documents/Captstone-Repo/end-to-end-app/data/clips")
        jetson_name = video_name
        if "camera1" in jetson_name:
            jetson_name = jetson_name.replace("camera1_", "").replace("_right.mp4", "_left.mp4")
        elif "camera0" in jetson_name:
            jetson_name = jetson_name.replace("camera0_", "")
        cand = base / jetson_name
        if cand.is_file():
            return cand
        for b in [base, Path("/home/dogdiving/Documents/Captstone-Repo/end-to-end-app/data/clips_11-7")]:
            if (b / video_name).is_file():
                return b / video_name
    fb = Path(fallback_path)
    if fb.is_file():
        return fb
    return Path(video_name)

CSV_PATH = _get_csv_path()

INDEX_SCHEMA = "test_bench_11_7.clip_index/1"
INDEX_PATH = DATA_DIR / "clip_index.json"
LOG_PATH = ANALYZED_ROOT / "analyzed_log.csv"

LOG_COLUMNS = ["jump_time", "dog", "judge_score_in", "score_raw",
               "Camera0", "Camera1"]

PLANE_SUFFIX = "11-7"
MIN_OK_POINTS = 4                            # operator decision U1: ok needs >= 4 accepted


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #
_DIST_RE = re.compile(r"^\s*(\d+)\s*'\s*(\d+)\s*[\"']*\s*$")


def parse_distance(raw: str) -> int | None:
    """``19'10\"`` -> 238 (inches), else None."""
    m = _DIST_RE.match((raw or "").strip())
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def parse_filename(name: str) -> dict[str, Any]:
    """``live_jump_19691231_191517_camera0_sapper.mp4`` -> parts."""
    stem = Path(name).stem
    if stem.startswith("live_jump_19691231_"):
        stem = stem[len("live_jump_19691231_"):]
    parts = stem.split("_")
    time = parts[0]
    ci = next(i for i, p in enumerate(parts) if p.startswith("camera"))
    cam = parts[ci]                         # camera0 / camera1
    side = parts[ci + 1]                    # camera1 / camera0
    extra = "_".join(parts[ci + 2:])        # '' or 'sapper' (etc.)
    return {"time": time, "cam": cam, "side": side, "tag": f"{cam}_{side}",
            "extra": extra}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_index() -> dict[str, Any]:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
def cmd_index(args) -> int:
    clips: list[dict[str, Any]] = []
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        for line, row in enumerate(csv.DictReader(fh), start=2):
            raw = (row["Distance"] or "").strip()
            inches = parse_distance(raw)
            if inches is None:
                continue  # 15 unscored rows excluded (operator decision)
            dog = (row["Name"] or "").strip()
            for col in ("Video_Camera1", "Video_Camera0"):
                name = (row[col] or "").strip()
                raw_pth = row["Camera1_Path" if col == "Video_Camera1" else "Camera0_Path"].strip()
                if not name or not raw_pth:
                    raise SystemExit(f"row {line} {col}: empty name/path")
                pth = _resolve_video_path(name, raw_pth)
                if not pth.is_file():
                    raise SystemExit(f"row {line}: path not a file: {pth}")
                p = parse_filename(name)
                cid = p["time"] + "_" + p["tag"] + (f"_{p['extra']}" if p["extra"] else "")
                clips.append({
                    "clip_id": cid,
                    "dog": dog,
                    "judge_score_in": inches,
                    "score_raw": raw,
                    "csv_line": line,
                    "jump_time": p["time"],
                    "camera": p["tag"],          # camera0 / camera1
                    "cam": p["cam"],
                    "side": p["side"],
                    "video_name": name,
                    "path": str(pth),
                    "setup_id": f"{p['tag']}_{PLANE_SUFFIX}",
                })

    by_cam: dict[str, int] = {}
    for c in clips:
        by_cam[c["camera"]] = by_cam.get(c["camera"], 0) + 1
    doc = {"schema": INDEX_SCHEMA, "csv_path": str(CSV_PATH),
           "n_clips": len(clips), "clips": clips}
    _manifest.write_json(INDEX_PATH, doc)
    print(json.dumps({"n_clips": len(clips), "by_camera": by_cam,
                      "index": str(INDEX_PATH)}))
    return 0


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def cmd_ingest(args) -> int:
    doc = _load_index()
    clips = doc["clips"]
    n = 0
    for c in clips:
        man = _ingest.ingest_clip(c["path"], ANALYZED_ROOT, clip_id=c["clip_id"],
                                  preserve_existing=False)
        # CSV provenance + plane assignment (data, not code: one setup id per camera).
        man["labels"] = {"dog": c["dog"], "judge_score_in": c["judge_score_in"],
                         "score_raw": c["score_raw"], "csv_line": c["csv_line"],
                         "camera": c["camera"], "jump_time": c["jump_time"]}
        man["plane"] = {"setup_id": c["setup_id"], "present": False}
        _manifest.write_manifest(man, ANALYZED_ROOT / c["clip_id"])
        # judge-only ground truth; gt_in is filled when the operator clicks the entry.
        _gt.write_ground_truth(
            _gt.from_score(c["judge_score_in"], dog=c["dog"], score_raw=c["score_raw"],
                           csv_line=c["csv_line"], updated=_now()),
            ANALYZED_ROOT / c["clip_id"])
        n += 1
    print(json.dumps({"ingested": n, "expected": len(clips)}))
    return 0


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #
def cmd_detect(args) -> int:
    from dog_diving import detect as _detect        # lazy: ultralytics
    from dog_diving import trajectory as _traj

    doc = _load_index()
    clips = doc["clips"]
    cfg = _detect.DetectConfig()                     # model of record, fixed thresholds
    statuses: dict[str, str] = {}
    done = skipped = 0
    for i, c in enumerate(clips, 1):
        cdir = ANALYZED_ROOT / c["clip_id"]
        det_path = cdir / "detections.json"
        traj_path = cdir / "trajectory.json"
        if det_path.is_file() and traj_path.is_file():
            man = _manifest.read_manifest(cdir)
            statuses[c["clip_id"]] = (man.get("detections") or {}).get("status", "detected")
            skipped += 1
            continue
        man = _manifest.read_manifest(cdir)
        clip_file = _manifest.clip_video_path(man, cdir)
        if clip_file is None:
            raise SystemExit(f"clip file missing for {c['clip_id']}")
        det = _detect.detect_clip(clip_file, cfg, out_dir=OUT_DIR, use_cache=True)
        w, h = int(man["meta"]["width"]), int(man["meta"]["height"])
        traj = _traj.build_trajectory(det.records, clip_id=c["clip_id"], frame_size=(w, h))
        _manifest.write_json(det_path, det.records)
        _manifest.write_json(traj_path, traj.as_dict())

        n_acc = det.n_accepted
        if n_acc == 0:
            status = "failed:no_detection"
        elif n_acc < MIN_OK_POINTS:
            status = "failed:few_points"
        else:
            status = "detected"                      # finalised by `score` once planes exist
        man["detections"] = {"path": "detections.json", "model_sha": det.model_sha,
                             "n_detections": det.n_detections, "n_accepted": n_acc,
                             "status": status}
        man["trajectory"] = {"path": "trajectory.json", "n_points": traj.n_points}
        _manifest.write_manifest(man, cdir)
        statuses[c["clip_id"]] = status
        done += 1
        print(f"[{i}/{len(clips)}] {c['clip_id']} accepted={n_acc} -> {status}", flush=True)

    _write_log(clips, statuses)
    print(json.dumps({"done": done, "skipped": skipped, "log": str(LOG_PATH)}))
    return 0


# --------------------------------------------------------------------------- #
# apply-planes
# --------------------------------------------------------------------------- #
def _latest_plane_setup_id(tag: str) -> str | None:
    """The most recently written ``<tag>_*.pool.json`` in planes/ (or None).

    Returns the setup id **without** the ``.pool`` suffix (the plane document's
    ``setup_id``), matching how :func:`manifest.plane_setup_path` normalises ids.
    """
    cands = sorted(PLANES_DIR.glob(f"{tag}_*.pool.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    sid = cands[-1].stem
    if sid.endswith(".pool"):
        sid = sid[:-len(".pool")]
    return sid


def cmd_apply_planes(args) -> int:
    doc = _load_index()
    clips = doc["clips"]
    tags = sorted({c["camera"] for c in clips})
    mapping: dict[str, str] = {}
    for tag in tags:
        sid = _latest_plane_setup_id(tag)
        if sid is None:
            raise SystemExit(f"no {tag}_*.pool.json in planes/ — build the plane first")
        plane = _manifest.load_plane(sid)
        if plane is None:
            raise SystemExit(f"plane {sid!r} failed to load")
        clips_of_tag = [c for c in clips if c["camera"] == tag]
        man = _manifest.read_manifest(ANALYZED_ROOT / clips_of_tag[0]["clip_id"])
        mm = _manifest.plane_size_mismatch(man, plane)
        if mm:
            raise SystemExit(f"plane {sid!r} does not fit the clips: {mm}")
        mapping[tag] = sid
    for c in clips:
        c["setup_id"] = mapping[c["camera"]]
    _manifest.write_json(INDEX_PATH, doc)
    for c in clips:
        cdir = ANALYZED_ROOT / c["clip_id"]
        man = _manifest.read_manifest(cdir)
        man["plane"] = {"setup_id": mapping[c["camera"]], "present": True}
        _manifest.write_manifest(man, cdir)
    print(json.dumps({"applied": mapping, "n_clips": len(clips)}))
    return 0


# --------------------------------------------------------------------------- #
# score (needs planes)
# --------------------------------------------------------------------------- #
def cmd_score(args) -> int:
    from dog_diving import trajectory as _traj
    from test_bench.scoring import registry as _registry

    doc = _load_index()
    clips = doc["clips"]

    # fail fast: every clip's plane must already resolve
    missing = []
    for c in clips:
        man = _manifest.read_manifest(ANALYZED_ROOT / c["clip_id"])
        if _manifest.load_plane((man.get("plane") or {}).get("setup_id")) is None:
            missing.append(c["clip_id"])
    if missing:
        raise SystemExit(f"missing planes for {len(missing)} clips "
                         f"(e.g. {missing[:5]}) — run 'apply-planes' first")

    statuses: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for c in clips:
        cdir = ANALYZED_ROOT / c["clip_id"]
        man = _manifest.read_manifest(cdir)
        det_path = cdir / "detections.json"
        records = json.loads(det_path.read_text(encoding="utf-8"))
        n_acc = (man.get("detections") or {}).get("n_accepted", 0)
        distance = None
        if n_acc == 0:
            status = "failed:no_detection"
        elif n_acc < MIN_OK_POINTS:
            status = "failed:few_points"
        else:
            plane = _manifest.clip_plane(man)          # size-guarded
            if plane is None:
                status = "failed:no_calibration"
            else:
                w, h = int(man["meta"]["width"]), int(man["meta"]["height"])
                points = _traj.assemble(records)
                score = _registry.run(
                    "arc_2d", points, plane,
                    {"n_points": "all", "selection": "all", "frame_size": [w, h]})
                s = score.get("status")
                distance = score.get("distance_in")
                # a jump distance must land in the pool: 0 <= X <= pool length
                if s != "ok":
                    status = f"failed:{s}"
                elif distance is None or not (0.0 <= float(distance) <= _config.POOL_LENGTH_FT * 12.0):
                    status = "failed:out_of_extent"
                else:
                    status = "ok"
        statuses[c["clip_id"]] = status
        rows.append({"jump_time": c["jump_time"], "dog": c["dog"], "camera": c["camera"],
                     "judge_score_in": c["judge_score_in"], "clip_id": c["clip_id"],
                     "status": status, "distance_in": distance})

    _write_log(clips, statuses)
    _write_audit(clips, rows)
    print(json.dumps({"n_ok": sum(1 for r in rows if r["status"] == "ok"),
                      "n_failed": sum(1 for r in rows if r["status"] != "ok"),
                      "audit": str(OUT_DIR / "audit_11-7.csv")}))
    return 0


# --------------------------------------------------------------------------- #
# shared outputs
# --------------------------------------------------------------------------- #
def _write_log(clips: list[dict], statuses: dict[str, str]) -> None:
    jumps: dict[str, dict[str, Any]] = {}
    for c in clips:
        j = jumps.setdefault(c["jump_time"], {
            "jump_time": c["jump_time"], "dog": c["dog"],
            "judge_score_in": c["judge_score_in"], "score_raw": c["score_raw"],
            "Camera0": "", "Camera1": ""})
        col = f"Camera{c['cam'][-1]}"   # camera0 -> Camera0
        j[col] = statuses[c["clip_id"]]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        wr.writeheader()
        for jt in sorted(jumps):
            wr.writerow(jumps[jt])


def _write_audit(clips: list[dict], rows: list[dict]) -> None:
    import statistics
    ok = [r for r in rows if r["status"] == "ok"]
    errs = [abs(r["distance_in"] - r["judge_score_in"]) for r in ok
            if r["distance_in"] is not None]
    by_dog: dict[str, list[float]] = {}
    by_cam: dict[str, list[float]] = {}
    for r in ok:
        if r["distance_in"] is not None:
            e = abs(r["distance_in"] - r["judge_score_in"])
            by_dog.setdefault(r["dog"], []).append(e)
            by_cam.setdefault(r["camera"], []).append(e)

    def _bins(xs):
        out = {}
        for b in (2, 4, 6):
            n = sum(1 for v in xs if v <= b)
            out[b] = {"count": n, "pct": round(100.0 * n / len(xs), 1) if xs else 0.0}
        return out

    # cross-camera agreement per jump (both feeds must be ok)
    by_jump: dict[str, dict[str, float]] = {}
    for r in ok:
        by_jump.setdefault(r["jump_time"], {})[r["camera"]] = r["distance_in"]
    cross = [abs(v["camera0"] - v["camera1"])
             for v in by_jump.values()
             if "camera0" in v and "camera1" in v
             and v["camera0"] is not None and v["camera1"] is not None]

    out = OUT_DIR / "audit_11-7.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["jump_time", "dog", "camera", "judge_score_in", "distance_in",
                     "error_in", "status"])
        for r in rows:
            e = (round(abs(r["distance_in"] - r["judge_score_in"]), 2)
                 if r["distance_in"] is not None else "")
            wr.writerow([r["jump_time"], r["dog"], r["camera"], r["judge_score_in"],
                         "" if r["distance_in"] is None else round(r["distance_in"], 2),
                         e, r["status"]])

    rep = OUT_DIR / "REPORT_11-7.md"
    lines = ["# Test-bench 11-7 — audit", "", f"ok clips: {len(ok)} / {len(rows)}", ""]
    if errs:
        lines += [f"- score-vs-judge error (in): mean {statistics.mean(errs):.2f}, "
                  f"median {statistics.median(errs):.2f}, ±2/±4/±6 = {_bins(errs)}"]
    if cross:
        lines += [f"- cross-camera |camera0 − camera1| (in): n={len(cross)}, "
                  f"median {statistics.median(cross):.2f}, max {max(cross):.2f}"]
    lines += ["", "## per camera (score-vs-judge, mean±median in)",
              "```"] + [f"{c}: n={len(v)}  mean={statistics.mean(v):.2f}  median={statistics.median(v):.2f}"
                        for c, v in sorted(by_cam.items())] + ["```", "",
              "## per dog", "```"] + [f"{d}: n={len(v)}  mean={statistics.mean(v):.2f}  median={statistics.median(v):.2f}"
                                       for d, v in sorted(by_dog.items())] + ["```"]
    rep.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench_11_7.py", description="Test-bench 11-7 batch driver.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("index", cmd_index), ("ingest", cmd_ingest),
                     ("detect", cmd_detect), ("apply-planes", cmd_apply_planes),
                     ("score", cmd_score)):
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
