"""cli_run — headless single-clip + batch runner (M7 deliverable, PLAN.md §5/§7/§11).

PLAN §5 ``cli_run.py``, verbatim:

    Headless path used by the validation batch (and by CI-ish regression runs):
    ``--clip``, ``--calib``, ``--imgsz``, ``--out``, ``--contact manual:<frame>|auto``.
    Produces the same artifacts as the GUI minus interaction.

PLAN §7's per-clip artifact set is written here, in full::

    out/<clip_id>/
      tracks.json        raw per-frame detections (cache)          <- written by detect.py
      trajectory.json    arc points, inliers, parabola, QA
      calib_used.json    copy of the calibration actually applied
      result.json        full provenance + distance + uncertainty
      annotated.mp4      overlay video
      final_frame.png    still at the chosen contact frame with callouts
      report.txt         human summary

plus one file PLAN does not list: ``run_meta.json``, which holds the wall-clock/cache/
host fields that PLAN §5 assigns to ``result.json`` but that PLAN §11's *"unchanged by
re-runs"* guarantee cannot admit (see :mod:`dog_diving.report`).  ``result.json`` carries
**artifact file names only** — never the output directory — for the same reason: run 2
of the §11 gate writes to a different directory and must still be byte-identical to
run 1.

Honest defaults on this corpus
------------------------------
``data/calib/`` is empty (HANDOFF open item 10), so every clip here runs with
``calib=None``: the pixel-domain fields (x, y, the sigma_kp window, the chosen frame)
are real output and the inches are ``null`` with the reason in ``warnings``.  A distance
is never invented to make a batch look complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# allow `python src/cli_run.py` as well as `import cli_run` with PYTHONPATH=src
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dog_diving import annotate as annotate_mod  # noqa: E402
from dog_diving import calibration as calibration_mod  # noqa: E402
from dog_diving import contact as contact_mod  # noqa: E402
from dog_diving import detect as detect_mod  # noqa: E402
from dog_diving import distance as distance_mod  # noqa: E402
from dog_diving import measure as measure_mod  # noqa: E402
from dog_diving import report as report_mod  # noqa: E402
from dog_diving import trajectory as trajectory_mod  # noqa: E402
from dog_diving import video as video_mod  # noqa: E402
from dog_diving.pool import calib as pool_calib_mod  # noqa: E402
from dog_diving.pool import pipeline as pool_pipeline_mod  # noqa: E402
from dog_diving.config import (  # noqa: E402
    CALIB_DIR,
    CLIP_DIR,
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_KP_CONF,
    MIN_ACCEPTED_POINTS,
    OUT_DIR,
    POOL_EXTENT_MARGIN_FT,
)

# The 3-D route's own artifact.  Separate from ``result.json`` on purpose: PLAN §11 requires the
# existing artifact set to be byte-identical across re-runs, and a new file that only appears when a
# pool calibration is supplied leaves that guarantee untouched.
POOL_NAME = "pool_translation.json"

# Measured on this host at imgsz=1280 (M3, 450 frames of 18-07) and 1920 (M5, same clip).
# Used only to *project* a batch before starting it (PLAN §8 V7); the per-clip timings in
# run_meta.json replace it with the real number as soon as one clip has run.
MS_PER_FRAME = {1280: 244.0, 1920: 542.0}


# --- argument parsing helpers ----------------------------------------------
def parse_contact(spec: str | int | None) -> tuple[str, int | None]:
    """PLAN §5's ``--contact manual:<frame>|auto`` -> ``(mode, frame)``.

    ``auto`` uses the preselected candidate (PLAN §5: "#1 last accepted arc sample");
    ``manual:<f>`` is the operator's frame override (D4).  A bare integer is accepted
    as manual too, because that is what a human types.
    """
    if spec is None:
        return "auto", None
    s = str(spec).strip().lower()
    if s in ("", "auto"):
        return "auto", None
    if s.startswith("manual:"):
        s = s.split(":", 1)[1].strip()
        if not s:
            raise ValueError("--contact manual: needs a frame number, e.g. manual:138")
    if s.lstrip("+-").isdigit():
        return "manual", int(s)
    raise ValueError(f"unrecognised --contact {spec!r}: expected 'auto' or 'manual:<frame>'")


def parse_subclip(spec: str | None) -> tuple[int, int] | None:
    """``START:END`` -> the half-open ``(start, stop)`` detect.py's cache key expects."""
    if not spec:
        return None
    a, _, b = str(spec).partition(":")
    start, stop = int(a or 0), int(b)
    if stop <= start:
        raise ValueError(f"--subclip must be start<stop, got {spec!r}")
    return start, stop


def parse_point(spec: str | None) -> tuple[float, float] | None:
    if not spec:
        return None
    x, _, y = str(spec).partition(",")
    return float(x), float(y)


def resolve_calibration(spec: str | Path | None) -> tuple[dict | None, str | None]:
    """``--calib`` may be a JSON path or a setup id under ``data/calib/``.

    Returns ``(calib_or_None, path_or_None)``.  A missing calibration is *not* an
    error here: it is the honest state of this corpus and the run still produces
    every artifact except the inches.
    """
    if spec is None:
        return None, None
    p = Path(spec)
    if not p.is_file():
        p = calibration_mod.setup_path(str(spec))
    if not p.is_file():
        raise FileNotFoundError(f"no calibration at {spec!r} (also tried {p})")
    return calibration_mod.load(p), str(p)


def resolve_pool_calibration(spec: str | Path | None) -> tuple[dict | None, str | None]:
    """``--pool-calib`` — the 3-D route's document, by path or setup id.

    Deliberately a *separate* resolver with a separate schema check, because the two documents are
    not interchangeable: feeding a 1-D calibration to the 3-D route (or the reverse) would produce a
    number rather than an error, and one of the 1-D documents on this machine fails its own gate by
    167 in.  ``pool_calib_mod.load`` refuses anything that is not ``dog_diving_pool_calibration/v1``.
    """
    if spec is None:
        return None, None
    p = Path(spec)
    if not p.is_file():
        p = pool_calib_mod.setup_path(str(spec))
    if not p.is_file():
        raise FileNotFoundError(f"no pool calibration at {spec!r} (also tried {p})")
    return pool_calib_mod.load(p), str(p)


# --- the projection (PLAN §8 V7) -------------------------------------------
def _cached(clip: Path, cfg: detect_mod.DetectConfig, frame_range, out_dir) -> bool:
    try:
        clip_sha = video_mod.sha256(clip)
        model_sha = video_mod.sha256(Path(cfg.model_path))
    except OSError:
        return False
    key = detect_mod.cache_key(clip_sha, model_sha, cfg, frame_range)
    doc = detect_mod.load_cache(detect_mod.tracks_path(clip.stem, out_dir))
    return detect_mod.key_hash(key) in (doc.get("entries") or {})


def project_batch(clips: Sequence[Path], cfg: detect_mod.DetectConfig, frame_range,
                  out_dir) -> dict[str, Any]:
    """Project a batch's runtime from the measured ms/frame *before* starting it.

    Returns ``{ms_per_frame, cold_clips, cold_frames, projected_s, per_clip}``.  Only
    the cold clips are projected; cache hits cost decode+draw only.
    """
    ms = MS_PER_FRAME.get(int(cfg.imgsz), MS_PER_FRAME[DEFAULT_IMGSZ])
    per: list[dict[str, Any]] = []
    cold_frames = 0
    for clip in clips:
        hit = _cached(clip, cfg, frame_range, out_dir)
        try:
            n = int(video_mod.probe(clip, count_frames=False).n_frames)
        except Exception:  # noqa: BLE001
            n = 0
        if not hit:
            cold_frames += n
        per.append({"clip": clip.name, "cached": hit, "n_frames": n})
    return {
        "ms_per_frame": ms,
        "cold_clips": sum(1 for p in per if not p["cached"]),
        "cold_frames": cold_frames,
        "projected_s": cold_frames * ms / 1000.0,
        "per_clip": per,
    }


# --- one clip ---------------------------------------------------------------
def run_clip(
    clip: str | Path,
    *,
    calib: dict | None = None,
    calib_path: str | None = None,
    detect_cfg: detect_mod.DetectConfig | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    kp_conf: float = DEFAULT_KP_CONF,
    out_dir: str | Path | None = None,
    contact: str | int | None = "auto",
    manual_point: Sequence[float] | None = None,
    subclip: str | tuple[int, int] | None = None,
    use_cache: bool = True,
    model: Any | None = None,
    draw: bool = True,
    fourcc: str = annotate_mod.DEFAULT_FOURCC,
    show_grid: bool = True,
    show_traj_conf: bool = True,
    progress: Callable[[str, dict], None] | None = None,
    pool_calib: dict | None = None,
    pool_calib_path: str | None = None,
    pool_height_ft: float = 0.0,
    argv: Sequence[str] | None = None,
    min_points: int | None = None,
) -> dict[str, Any]:
    """Run the whole headless pipeline for one clip and write PLAN §7's artifacts."""
    clip = Path(clip)
    if not clip.is_file():
        raise FileNotFoundError(f"no such clip: {clip}")
    frame_range = parse_subclip(subclip) if isinstance(subclip, str) else subclip
    if isinstance(subclip, tuple):
        frame_range = (int(subclip[0]), int(subclip[1]))
    cfg = detect_cfg or detect_mod.DetectConfig(imgsz=int(imgsz), conf=float(conf),
                                                kp_conf=float(kp_conf))
    cdir = (Path(out_dir) if out_dir is not None else OUT_DIR) / clip.stem
    cdir.mkdir(parents=True, exist_ok=True)
    report_stage: Callable[[str], None] = lambda s: progress and progress(s, {})

    # 1. detect (or load the cache) -----------------------------------------
    # PLAN §5's fixed thresholds plus the operator's adaptive ladder (2026-10-12): when the base
    # rung cannot yield `min_points` accepted rows the threshold is lowered until it can, and every
    # point that bought is marked `low_conf` and disclosed in the artifacts (see
    # `detect.detect_clip_adaptive`).  `min_points=0` restores the fixed-threshold behaviour.
    report_stage("detect")
    det = detect_mod.detect_clip_adaptive(clip, cfg, frame_range=frame_range, out_dir=out_dir,
                                          use_cache=use_cache, model=model,
                                          min_points=min_points)
    ladder = det.ladder
    info = video_mod.probe(clip, count_frames=False)

    # 2. trajectory ---------------------------------------------------------
    report_stage("trajectory")
    traj = trajectory_mod.build_trajectory(det, clip_id=clip.stem)

    # 3. contact candidates + the operator's frame -------------------------
    report_stage("contact")
    wy = distance_mod.waterline_y(calib) if calib else None
    con = contact_mod.contact_candidates(det, traj=traj, waterline_y=wy, clip_id=clip.stem)
    mode, frame = parse_contact(contact)
    chosen = con.choose(int(frame), "manual") if (mode == "manual" and frame is not None) else con

    # 4. measure ------------------------------------------------------------
    # PLAN §16: with a plane calibration the measurement is the FINAL detected tail-set, assumed to
    # be the intersection with the plane; its +X is the reported distance.  The 1-D mark route is
    # archived and no longer produces a number here.
    report_stage("measure")
    plane_mode = bool((pool_calib or {}).get("plane"))
    if plane_mode:
        meas = measure_mod.plane_distance(traj, pool_calib, manual_point=manual_point)
        last = measure_mod.final_point(traj)
        if last is not None and manual_point is None:
            chosen = con.choose(int(last["frame"]), "final-detection")
        elif manual_point is not None:
            chosen = con.choose(int(chosen.chosen_frame or 0), "manual")
    else:
        meas = distance_mod.measure_contact(chosen, traj, calib, manual_point=manual_point,
                                            clip_id=clip.stem,
                                            image_size=(info.width, info.height))

    # 4b. the 3-D translation (pool-translation-plan.txt phases 1-6) ----------
    # Only when a pool calibration is supplied, so the default run stays byte-identical (PLAN §11).
    pool_res: dict[str, Any] | None = None
    pool_path: Path | None = None
    if pool_calib is not None:
        report_stage("pool_translate")
        # In plane mode the entry IS the measured point: hand translate_track that point rather
        # than a contact frame it would have to re-select.
        entry_point = (manual_point if manual_point is not None
                       else (None if meas.get("x_px") is None
                             else (meas["x_px"], meas["y_px"])))
        pool_res = pool_pipeline_mod.translate_track(
            det.records, doc=pool_calib,
            contact_frame=None if plane_mode else chosen.chosen_frame,
            manual_point=entry_point, height_ft=float(pool_height_ft),
            legacy_calib=calib, clip_id=clip.stem, margin_ft=float(POOL_EXTENT_MARGIN_FT))
        pool_path = cdir / POOL_NAME
        pool_path.write_text(report_mod.dumps_strict(pool_res), encoding="utf-8")

    # 5. artifacts ----------------------------------------------------------
    report_stage("annotate")
    traj_p = cdir / "trajectory.json"
    traj_p.write_text(report_mod.dumps_strict(traj.as_dict()), encoding="utf-8")
    calib_p = cdir / "calib_used.json"
    calib_p.write_text(report_mod.dumps_strict(
        calib if calib else {"present": False, "reason": "no calibration supplied"}),
        encoding="utf-8")

    still_frame = (chosen.chosen_frame if chosen.chosen_frame is not None
                   else (traj.last_arc_frame if traj.last_arc_frame is not None else 0))
    written: dict[str, Any] = {}
    if draw:
        pool_doc_for_draw = pool_res is not None and pool_calib or None
        written["annotated"] = annotate_mod.write_video(
            clip, cdir / "annotated.mp4", traj=traj, contact=chosen, calib=calib,
            measurement=meas, frame_range=frame_range, fourcc=fourcc, show_grid=show_grid,
            show_traj_conf=show_traj_conf, pool_doc=pool_doc_for_draw)
        written["still"] = annotate_mod.write_still(
            clip, cdir / "final_frame.png", int(still_frame), traj=traj, contact=chosen,
            calib=calib, measurement=meas, show_grid=show_grid,
            show_traj_conf=show_traj_conf, pool_doc=pool_doc_for_draw)

    # result.json carries file NAMES only, so two runs into different output
    # directories are byte-identical (PLAN §11).
    names = {
        "tracks": Path(det.cache_path).name,
        "trajectory": traj_p.name,
        "calib_used": calib_p.name,
        "annotated": "annotated.mp4" if draw else None,
        "still": "final_frame.png" if draw else None,
        "result": report_mod.RESULT_NAME,
        "report": report_mod.TEXT_NAME,
        "meta": report_mod.META_NAME,
        "pool_translation": POOL_NAME if pool_path is not None else None,
    }
    artifacts = {k: v for k, v in names.items() if v}

    result = report_mod.build_result(
        clip_id=clip.stem, clip_sha=det.clip_sha, clip_name=clip.name, video=info,
        detect=det, traj=traj, contact=chosen, measurement=meas, calib=calib,
        calib_path=calib_path, detect_config=cfg.as_dict(), artifacts=artifacts,
        still_frame=int(still_frame), ladder=ladder,
    )
    result_p = report_mod.write_result_json(cdir / report_mod.RESULT_NAME, result)
    report_mod.write_report_txt(cdir / report_mod.TEXT_NAME, result)
    meta = report_mod.build_run_meta(
        clip_id=clip.stem, timings=det.timings, from_cache=det.from_cache,
        argv=list(argv) if argv is not None else None,
        extra={"imgsz": int(cfg.imgsz), "conf": float(cfg.conf), "kp_conf": float(cfg.kp_conf),
               "min_points": min_points, "ladder": ladder,
               "contact": str(contact), "subclip": list(frame_range) if frame_range else None,
               "draw": bool(draw)},
    )
    meta_p = report_mod.write_run_meta(cdir / report_mod.META_NAME, meta)

    abs_paths = {k: str(cdir / v) for k, v in artifacts.items()}
    return {
        "clip_id": clip.stem, "clip": str(clip), "out_dir": str(cdir),
        "artifacts": abs_paths, "artifact_names": artifacts,
        "result": result, "result_path": str(result_p), "meta": meta,
        "meta_path": str(meta_p), "measurement": meas, "trajectory": traj,
        "contact": chosen, "detect": det, "video": info, "written": written,
        "contact_mode": mode, "contact_frame": frame,
        "pool_translation": pool_res, "pool_path": None if pool_path is None else str(pool_path),
    }


# --- the CLI ----------------------------------------------------------------
def _clip_list(args: argparse.Namespace) -> list[Path]:
    if args.clip:
        return [Path(args.clip)]
    base = Path(args.clips_dir)
    if args.batch:
        return sorted(base.glob(args.batch))
    if args.all:
        return sorted(base.glob("*.mp4"))
    return []


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Headless dock-diving scorer (M7).")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--clip", type=Path, help="one clip to process")
    src.add_argument("--batch", metavar="GLOB", help="glob under --clips-dir (e.g. '*.mp4')")
    src.add_argument("--all", action="store_true", help="every .mp4 under --clips-dir")
    ap.add_argument("--clips-dir", type=Path, default=CLIP_DIR)
    ap.add_argument("--calib", default=None, help="calibration JSON path or setup id")
    ap.add_argument("--pool-calib", default=None,
                    help="pool-translation calibration JSON path or setup id (the 3-D route)")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--kp-conf", type=float, default=DEFAULT_KP_CONF)
    ap.add_argument("--min-points", type=int, default=MIN_ACCEPTED_POINTS,
                    help="lower the detection threshold until at least this many accepted rows "
                         "exist (operator instruction 2026-10-12; 0 = fixed thresholds)")
    ap.add_argument("--out", type=Path, default=None, help=f"output root (default {OUT_DIR})")
    ap.add_argument("--contact", default="auto", help="'auto' or 'manual:<frame>'")
    ap.add_argument("--point", default=None, help="manual measurement point 'X,Y' (PLAN §5 override)")
    ap.add_argument("--subclip", default=None, help="frames 'START:END' (PLAN §5's --subclip)")
    ap.add_argument("--no-cache", action="store_true", help="ignore + do not write tracks.json")
    ap.add_argument("--no-draw", action="store_true", help="skip the MP4/PNG (JSON/TXT only)")
    ap.add_argument("--no-traj-conf", action="store_true",
                    help="omit the per-point confidence labels on the yellow trajectory polyline")
    ap.add_argument("--fourcc", default=annotate_mod.DEFAULT_FOURCC)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clips = _clip_list(args)
    if not clips:
        print("no clips selected: pass --clip, --batch GLOB or --all", file=sys.stderr)
        return 2

    calib, calib_path = resolve_calibration(args.calib)
    pool_calib, pool_calib_path = resolve_pool_calibration(args.pool_calib)
    frame_range = parse_subclip(args.subclip)
    manual_point = parse_point(args.point)
    cfg = detect_mod.DetectConfig(imgsz=int(args.imgsz), conf=float(args.conf),
                                  kp_conf=float(args.kp_conf))
    out_root = args.out or OUT_DIR

    if not args.no_cache and len(clips) > 1:
        proj = project_batch(clips, cfg, frame_range, out_root)
        print(f"batch projection (PLAN §8 V7): {len(clips)} clip(s), "
              f"{proj['cold_clips']} uncached [{proj['cold_frames']} frames], "
              f"{proj['ms_per_frame']:.0f} ms/frame measured -> "
              f"~{proj['projected_s'] / 60.0:.1f} min inference "
              f"(annotation adds ~15 ms/frame)")

    results: list[dict] = []
    failed: list[str] = []
    for clip in clips:
        try:
            r = run_clip(clip, calib=calib, calib_path=calib_path, detect_cfg=cfg,
                         out_dir=out_root, contact=args.contact, manual_point=manual_point,
                         subclip=frame_range, use_cache=not args.no_cache, draw=not args.no_draw,
                         fourcc=args.fourcc, show_traj_conf=not args.no_traj_conf, argv=argv,
                         pool_calib=pool_calib, pool_calib_path=pool_calib_path,
                         min_points=(None if args.min_points is None
                                     else max(int(args.min_points), 0)))
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill a batch
            failed.append(f"{clip.name}: {type(exc).__name__}: {exc}")
            if not args.quiet:
                print(f"FAILED {clip.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results.append(r["result"])
        if not args.quiet:
            m, c = r["measurement"], r["result"]["contact"]
            dist = m.get("display") or "n/a (no calibration)"
            print(f"{r['clip_id']:10s} frames={r['video'].n_frames:4d} "
                  f"accepted={r['detect'].n_accepted:3d} contact f{c.get('chosen_frame')} "
                  f"({c.get('chosen_source')}) distance={dist} "
                  f"warnings={len(m.get('warnings') or [])}"
                  f"{' [cached]' if r['detect'].from_cache else ''}")

    if len(results) > 1:
        csv_p = report_mod.write_rollup_csv(Path(out_root) / report_mod.ROLLUP_NAME, results)
        if not args.quiet:
            print(f"roll-up: {csv_p}")

    if failed:
        print(f"{len(failed)} clip(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
