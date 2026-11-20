"""Recover placed ground-truth pixels from ``results/`` run snapshots.

A suite run records each clip's ``gt`` document verbatim, so a ``results/*.json``
is a snapshot of a ground truth that was placed and then lost from
``ground_truth.json`` (the 11-7 batch shipped with ``x_px: null`` on all 116
clips on every machine).  This rebuilds the per-clip record from the newest
snapshot, re-deriving ``gt_in`` through the clip's plane the way
:func:`test_bench.ground_truth.set_ground_truth` always does.

Dry run by default; ``--apply`` writes ``ground_truth.json`` and refreshes the
manifest's ``ground_truth`` block.  Idempotent: a clip that already has a placed
pixel is never touched.

    python src/recover_gt_from_results.py             # report only
    python src/recover_gt_from_results.py --apply     # write the records
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

from test_bench import ground_truth as gt_mod   # noqa: E402
from test_bench import manifest as manifest_mod  # noqa: E402

DEFAULT_ROOT = SRC.parent / "analyzed-clips"

#: A derived gt_in this far from the snapshot's is reported as a mismatch (in).
GT_TOLERANCE_IN = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshots(results_dir: Path) -> dict[str, dict]:
    """clip_id -> the newest run-document ``gt`` snapshot that carries a pixel."""
    best: dict[str, tuple[str, dict]] = {}
    if not results_dir.is_dir():
        return {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run_id = str(doc.get("run_id") or path.stem)
        for row in doc.get("rows") or []:
            cid = row.get("clip_id")
            gt = row.get("gt") or {}
            x, y = gt.get("x_px"), gt.get("y_px")
            if not cid or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            prev = best.get(cid)
            if prev is None or run_id > prev[0]:
                best[cid] = (run_id, gt)
    return {cid: gt for cid, (_rid, gt) in best.items()}


def _placed(gt: dict | None) -> bool:
    if not isinstance(gt, dict):
        return False
    return isinstance(gt.get("x_px"), (int, float)) and isinstance(gt.get("y_px"), (int, float))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="analyzed-clips root")
    ap.add_argument("--apply", action="store_true", help="write ground_truth.json files")
    args = ap.parse_args(argv)

    root = Path(args.root)
    snaps = snapshots(root / "results")
    print(f"{len(snaps)} clip(s) with a GT pixel in results/ snapshots\n")

    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    drift: list[tuple[str, float, float]] = []

    for cid in sorted(snaps):
        cdir = root / cid
        if not (cdir / "manifest.json").is_file():
            skipped.append((cid, "no clip directory"))
            continue
        man = manifest_mod.read_manifest(cdir)
        existing: dict = {}
        gtp = cdir / "ground_truth.json"
        if gtp.is_file():
            try:
                existing = json.loads(gtp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
        if _placed(existing):
            skipped.append((cid, "GT already placed"))
            continue

        snap = snaps[cid]
        plane = manifest_mod.plane_resolution(man)
        if plane["doc"] is None:
            failures.append((cid, plane["error"] or "no plane resolved"))
            continue
        frame = int(snap.get("frame") or 0)
        judge = existing.get("judge_in")
        if judge is None:
            judge = snap.get("judge_in")
        try:
            gt = gt_mod.set_ground_truth(plane["doc"], frame=frame,
                                         x_px=float(snap["x_px"]), y_px=float(snap["y_px"]),
                                         judge_in=judge, updated=_now())
        except ValueError as e:
            failures.append((cid, str(e)))
            continue
        # Keep the CSV provenance the manifest/ingest recorded.
        ref = existing.get("reference") or (man.get("labels") or {})
        if ref:
            gt["reference"] = {k: ref.get(k) for k in ("dog", "score_raw", "csv_line")}
        snap_in = snap.get("gt_in")
        if isinstance(snap_in, (int, float)) and gt.get("gt_in") is not None:
            if abs(float(snap_in) - float(gt["gt_in"])) > GT_TOLERANCE_IN:
                drift.append((cid, float(snap_in), float(gt["gt_in"])))

        print(f"{cid}: frame {frame}  ({snap['x_px']:.2f}, {snap['y_px']:.2f}) px"
              f"  -> gt_in {gt.get('gt_in') if gt.get('gt_in') is None else round(gt['gt_in'], 2)}"
              f"  plane {plane['setup_id']} [{plane['source']}]"
              f"  (snapshot said {snap_in if snap_in is None else round(float(snap_in), 2)} in)")
        written.append(cid)
        if args.apply:
            gt_mod.write_ground_truth(gt, cdir)
            man["ground_truth"] = {"path": "ground_truth.json",
                                   "judge_present": gt["judge_present"],
                                   "gt_in": gt.get("gt_in")}
            manifest_mod.write_manifest(man, cdir)

    print(f"\n{'recovered' if args.apply else 'would recover'}: {len(written)}")
    if drift:
        print("gt_in differs from the snapshot (plane or pixel changed):")
        for cid, was, now in drift:
            print(f"  {cid}: snapshot {was:.2f} in -> derived {now:.2f} in")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for cid, why in skipped:
            print(f"  {cid}: {why}")
    if failures:
        print(f"failed {len(failures)}:")
        for cid, why in failures:
            print(f"  {cid}: {why}")
        return 1
    if not args.apply and written:
        print("\nre-run with --apply to write these records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
