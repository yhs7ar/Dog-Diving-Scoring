"""pc_score.py — headless point-cloud scoring entry point (auto-datum convenience follow-on).

The roadmap's point-cloud cluster (12/13/14/1b-auto/1b-lip) is geometrically complete: a real rig
run can go SVO2 -> NEURAL depth -> fused cloud -> water plane -> auto-located lip -> ``pc_distance``
with **no operator-supplied datum**.  The only missing piece was a single command that wires the two
proven ends together — :func:`dog_diving.pc_datum.locate_dock_lip` (finds the lip in the cloud) and
:func:`dog_diving.pc_distance.pc_distance` (scores against it).  This CLI is that command.

The chain, each refusal reported honestly (never a placeholder number):

1. Resolve the cloud to fit the water/dock against — ``--cloud <fused.npy>`` if you have a fused or
   image-shaped cloud, else every ``frame_*.npy`` under ``--frames-dir`` is merged
   (:func:`~dog_diving.pc_trajectory.merge_frames_cloud`).
2. :func:`~dog_diving.pc_datum.locate_dock_lip` -> the dock lip ``datum`` (water plane fit, dock-top
   horizontal-plane fit, measured pool direction, hull-edge lip).  A refusal (``no_water_plane`` /
   ``no_dock`` / ``no_pool_side`` / …) exits non-zero with the failing gate's detail — the datum is
   never invented.
3. :func:`~dog_diving.pc_distance.pc_distance` with ``lip=<located datum>`` and the *same* water
   height the locate step settled on, so the two halves agree on the water level by construction.

Run on the rig (after ``generate_point_cloud.py`` has produced the per-frame clouds, and the
detector has produced accepted tail-set pixels):

    python src/pc_score.py --frames-dir <stem>_frames --detections detections.json

``detections.json`` is a JSON list of ``{"frame": i, "x": px, "y": px}`` objects **or**
``[frame, x, y]`` triples (both are the ``points_at_detections`` input shape).  The pixels are in
the *reference* camera's frame — the same camera ``frame_*.npy`` was generated from.

Exit code 0 = a distance was produced; non-zero = a step refused (with the reason on stderr and in
the JSON ``status``/``detail``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# allow `python src/pc_score.py` as well as `import pc_score` with PYTHONPATH=src
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from dog_diving import pc_datum as _pc_datum  # noqa: E402
from dog_diving import pc_distance as _pc_distance  # noqa: E402
from dog_diving.pc_trajectory import (  # noqa: E402
    ToWorld,
    merge_frames_cloud,
    zed_to_world,
)

#: The result JSON the CLI writes when ``--out`` is given.
RESULT_NAME = "pc_score.json"


def _load_detections(path: Path) -> list[Any]:
    """Load a detections JSON list; the entries' *shape* is validated downstream (dict or triple)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"detections JSON must be a list, got {type(data).__name__}")
    return data


def _resolve_cloud(cloud: Any, frames_dir: Path) -> np.ndarray:
    """``--cloud`` path/array, or the merge of every ``frame_*.npy`` under ``frames_dir``."""
    if cloud is None:
        return merge_frames_cloud(frames_dir)
    if isinstance(cloud, (str, Path)):
        return np.load(str(cloud))
    return np.asarray(cloud, dtype=float)


def run_pc_score(
    frames_dir: str | Path,
    detections: Any,
    *,
    cloud: Any = None,
    water_z_ft: float | None = None,
    to_world: ToWorld = zed_to_world,
) -> dict[str, Any]:
    """Chain ``locate_dock_lip`` -> ``datum`` -> ``pc_distance`` (the auto-datum scoring route).

    ``frames_dir`` holds the per-frame ``frame_*.npy`` clouds; ``detections`` is a path to a JSON
    list or an iterable of ``{frame, x, y}`` dicts / ``(frame, x, y)`` triples; ``cloud`` is an
    optional fused/merged cloud (path or array) for the water/dock fit — when ``None`` the
    ``frame_*.npy`` files are merged.  ``water_z_ft`` supplied here skips the water fit.

    Returns ``{status, detail, lip, distance, water_z_ft, water_fit, dock_z_ft, pool_dir}`` where
    ``lip`` is the full ``locate_dock_lip`` result and ``distance`` the full ``pc_distance`` result.
    ``status`` is ``"ok"`` only when *both* halves answered; a locate refusal leaves ``distance``
    as ``None``.  The result is JSON-serialisable (no numpy arrays leak through).
    """
    frames_dir = Path(frames_dir)
    cloud_arr = _resolve_cloud(cloud, frames_dir)
    dets = _load_detections(Path(detections)) if isinstance(detections, (str, Path)) else list(detections)

    result: dict[str, Any] = {
        "status": None, "detail": "", "lip": None, "distance": None,
        "water_z_ft": None, "water_fit": None, "dock_z_ft": None, "pool_dir": None,
    }

    located = _pc_datum.locate_dock_lip(cloud_arr, water_z_ft=water_z_ft, to_world=to_world)
    result["lip"] = located
    result["water_z_ft"] = located.get("water_z_ft")
    result["water_fit"] = located.get("water_fit")
    result["dock_z_ft"] = located.get("dock_z_ft")
    result["pool_dir"] = located.get("pool_dir")
    if located["status"] != "ok":
        result["status"] = located["status"]
        result["detail"] = located["detail"]
        return result

    # The datum is located *and* the water height is settled: hand both to the distance route so the
    # crossing uses exactly the water level the lip was measured against (no second, independent fit).
    dist = _pc_distance.pc_distance(
        frames_dir, dets, lip=located["datum"], water_z_ft=located["water_z_ft"], to_world=to_world)
    result["distance"] = dist
    result["status"] = dist["status"]
    result["detail"] = dist["detail"]
    return result


def _print_summary(result: dict[str, Any]) -> None:
    lip = result["lip"] or {}
    dist = result["distance"] or {}
    print(f"status      : {result['status']}")
    if result["status"] != "ok":
        print(f"detail      : {result['detail']}")
        return
    d = lip.get("datum") or {}
    print(f"water plane : Z = {result['water_z_ft']:+.2f} ft "
          f"({(lip.get('water_fit') or {}).get('detail', '')})")
    print(f"dock top    : Z = {result['dock_z_ft']:+.2f} ft")
    print(f"pool dir    : [{result['pool_dir'][0]:+.3f}, {result['pool_dir'][1]:+.3f}]")
    print(f"lip corners : a={d.get('a')} b={d.get('b')}  (length {d.get('length_ft', 0):.2f} ft)")
    if dist.get("status") == "ok":
        print(f"distance    : {dist.get('total_inches', 0):.2f} in = "
              f"{dist.get('distance_ft', 0):.2f} ft (crossing frame {dist.get('frame')})")
        print(f"lip distance: {dist.get('lip_distance_in', 0):.2f} in "
              f"(signed {dist.get('lip_signed_ft', 0):+.2f} ft)")
    else:
        print(f"distance    : n/a ({dist.get('status')}: {dist.get('detail')})")
    if dist.get("warnings"):
        for w in dist["warnings"]:
            print(f"warning     : {w}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Point-cloud scoring: auto-locate the dock lip in the cloud, then score the "
                    "trajectory against it (no operator-supplied datum).")
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="directory of per-frame frame_*.npy clouds (from generate_point_cloud.py)")
    ap.add_argument("--detections", type=Path, required=True,
                    help="JSON list of {frame,x,y} objects or [frame,x,y] triples (accepted tail-set pixels)")
    ap.add_argument("--cloud", type=Path, default=None,
                    help="a fused/merged (N,3) or (H,W,3) .npy cloud for the water/dock fit "
                         "(default: merge every frame_*.npy under --frames-dir)")
    ap.add_argument("--water-z-ft", type=float, default=None,
                    help="supply the water height in world feet (skips the water-plane fit)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"write the full result JSON here (default: {RESULT_NAME} not written, "
                         "summary only)")
    ap.add_argument("--json", action="store_true",
                    help="print the full result as single-line JSON to stdout")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.frames_dir).is_dir():
        print(f"no such frames dir: {args.frames_dir}", file=sys.stderr)
        return 2
    if not Path(args.detections).is_file():
        print(f"no such detections file: {args.detections}", file=sys.stderr)
        return 2

    try:
        result = run_pc_score(args.frames_dir, args.detections, cloud=args.cloud,
                              water_z_ft=args.water_z_ft)
    except Exception as exc:  # noqa: BLE001 — a clean refusal, not a traceback
        print(f"pc_score FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.out is not None:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"result JSON : {args.out}", file=sys.stderr)

    if args.json:
        print(json.dumps(result))
    else:
        _print_summary(result)

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
