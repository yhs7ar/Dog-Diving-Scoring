"""stability.py — dock-edge drift / camera-pan detection.

PLAN.md §3 lists this module in the tree and §5 specifies it:

    ### ``stability.py``
    - Sample N frames; take the calibration line endpoints and re-locate the dock
      edge by re-running the same edge search used in calibration; compute drift
      in px.
    - Also global-motion check via ``cv2.phaseCorrelate`` on a static region (a
      pool corner).
    - Report: ``max_drift_px``, ``verdict: stable | marginal | unusable``.
      ``unusable`` blocks measurement (with an override, logged).

It did not exist in the tree (built 2026-10-08, scheduled run).  It is the one
hard dependency of **M8's step 1** (``app_gui.py`` *Load*: "run the stability
check; block with a clear message if ``unusable``"), and PLAN §8 **V6** is its
gate ("run the detection on a deliberately panned clip and confirm the system
refuses/refuses-with-warning") — a gate an unattended run *can* execute, unlike
M8's own manual click-through.

Why the datum is perishable (and why this module exists)
--------------------------------------------------------
``dock-diving-scoring``: "position (pixels, or a 3D line) … perishable — any
camera movement invalidates it".  A wrong datum biases *every* score in a
session in the same direction — a silent, correlated error — so the staleness
watchdog ships before any detector does.

Two independent cues
--------------------
1. **Coherent shift** — ``cv2.phaseCorrelate`` over a grid of tiles, reduced by
   a robust consensus (median, then an inlier re-median).  A pan is a coherent,
   whole-frame displacement: the tiles agree on one shift.  PLAN says "a static
   region (a pool corner)"; the inlier consensus is the prior-free way to the
   same number without knowing where a corner is, and it is what *rejects* the
   tiles the dog / splash / water occupy.  This cue drives the verdict.
2. **Dock-edge re-location** — re-running the **same** edge search the
   calibrator uses (``calibration.refine_dock_edge``, PLAN §5) on the sampled
   frames and measuring how far the re-located line moves from its own
   consensus.  Only available with a calibration (``data/calib/`` is empty on
   this corpus, HANDOFF item 10).  It is reported **with its measured noise
   floor** and may only escalate the verdict *beyond that floor*.

Two measured realities that fix the design (2026-10-08, this host, 18-07.mp4)
-----------------------------------------------------------------------------
**(a) Compare *adjacent* samples, never frame 0 vs frame N.**  Frame 0 vs frame
359 measured a bogus 44.87 px "shift" with a phase-correlation response of
**0.031** and per-tile disagreement of 221 px — because these corpus clips are
**compilations**: 18-07 has a hard cut at frame 16 (adjacent-frame
``mean|absdiff|`` **22.51** against a p99 of **2.18** and a median of 0.69).
The camera is static — every adjacent pair measures **0.01-0.19 px** at response
0.6-0.99.  So the estimator accumulates increments between *temporally adjacent*
samples and closes the running segment whenever a pair is rejected, which makes a
shot change a *rejectable pair* instead of a fake camera move.

**(b) The verdict cannot be carried by the dock-edge cue.**  On this corpus the
dock edge is a broad, low-contrast ramp (peak mean-profile gradient ~10/255),
so re-locating it frame-to-frame is *not* sub-pixel — measured ~9 px spread /
~2.1 px std (HANDOFF, M2-tool section).  Used alone it would false-trigger on a
perfectly static camera.  The coherent-shift cue is precise on the same frames
(known translation recovered to <0.03 px), so it carries the verdict and the
edge cue is corroboration gated by its own floor.

Thresholds (justified, not picked)
----------------------------------
The measured local scale for this corpus is **0.667 in/px** near the dock and
0.693 in/px far out (M6 ``distance.local_in_per_px``; HANDOFF M6 finding 1), so

* ``STABLE_MAX_PX = 1.0``  -> ~0.7 in of datum bias, inside the Dog Diving 1 in
  graduation;
* ``UNUSABLE_PX  = 3.0``   -> ~2.0 in of datum bias — that alone consumes the
  capstone's entire ±2 in criterion, i.e. it *is* the block condition.

Fail-closed: if **no** pair yields a usable estimate the verdict is ``unusable``
with ``estimate_available=False`` and the reason stated — the module refuses
rather than reporting "stable" on no evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

# allow `python src/dog_diving/stability.py` as well as `import dog_diving.stability`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import calibration as calibration_mod  # noqa: E402
from dog_diving import video as video_mod  # noqa: E402

# --- thresholds (see the module docstring for the 0.667-0.693 in/px derivation) ---
STABLE_MAX_PX = 1.0    # ~0.7 in of datum bias
UNUSABLE_PX = 3.0      # ~2.0 in of datum bias == the capstone's whole +/-2 in budget

VERDICTS = ("stable", "marginal", "unusable")


class StabilityError(RuntimeError):
    """Raised by :func:`assert_usable` when the datum is unusable and not overridden."""


@dataclass(frozen=True)
class StabilityConfig:
    """All knobs in one place (house style: one frozen dataclass per module)."""

    n_samples: int = 12          # frames sampled across the clip (PLAN §5 "sample N frames")
    work_width: int = 960        # FFT working scale; estimates are converted BACK
                                 # to source pixels before they are reported
    tile: int = 160              # px tile for the coherent-shift consensus
    min_tile_std: float = 3.0    # a flat tile carries no phase information
    min_tiles: int = 3           # below this a pair is not estimable
    min_inlier_frac: float = 0.5  # >= this share of tiles must agree on ONE shift
    inlier_px: float = 3.0       # ... where "agree" means within this (work px)
    hann: bool = True            # window each tile before the FFT (edge wraparound)
    edge_radius: int = 25        # the SAME radius calibration.refine_dock_edge uses
    edge_noise_floor: float = 2.0  # ~2.1 px measured std; below this the edge cue is noise
    stable_px: float = STABLE_MAX_PX
    unusable_px: float = UNUSABLE_PX

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT = StabilityConfig()


# --- sampling ---------------------------------------------------------------
def sample_indices(n_frames: int, n_samples: int) -> list[int]:
    """Evenly spaced frame indices over ``[0, n_frames)``, always incl. 0 and the last.

    Deterministic (not a random draw) so the verdict is reproducible — the same
    guarantee M7's ``result.json`` carries.
    """
    if n_frames <= 0:
        return []
    k = max(1, min(int(n_samples), int(n_frames)))
    if k == 1:
        return [0]
    return sorted({int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)})


# --- the coherent-shift cue -------------------------------------------------
def _to_gray_work(frame: np.ndarray, work_width: int) -> tuple[np.ndarray, float]:
    """(float32 gray at work scale, scale) where ``scale = work_w / native_w``."""
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    scale = 1.0
    if work_width and w > work_width:
        scale = work_width / float(w)
        g = cv2.resize(g, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return g.astype(np.float32), scale


def _tile_shifts(ref: np.ndarray, cur: np.ndarray, cfg: StabilityConfig
                 ) -> list[tuple[float, float, float]]:
    """Per-tile ``(dx, dy, response)`` in WORK pixels, textured tiles only."""
    h, w = ref.shape
    out: list[tuple[float, float, float]] = []
    for y0 in range(0, h, cfg.tile):
        for x0 in range(0, w, cfg.tile):
            y1, x1 = min(h, y0 + cfg.tile), min(w, x0 + cfg.tile)
            a, b = ref[y0:y1, x0:x1], cur[y0:y1, x0:x1]
            if a.shape[0] < 32 or a.shape[1] < 32:
                continue
            if float(a.std()) < cfg.min_tile_std:
                continue  # flat tile -> phaseCorrelate returns noise
            win = (cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
                   if cfg.hann else None)
            try:
                (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
            except cv2.error:  # pragma: no cover - cv2 raises on degenerate input
                continue
            out.append((float(dx), float(dy), float(resp)))
    return out


def _pair_estimate(ref: np.ndarray, cur: np.ndarray, cfg: StabilityConfig
                   ) -> dict[str, Any]:
    """Robust coherent shift (WORK px) for one frame pair, with a quality verdict.

    The consensus test is what makes this usable at all: a *pan* moves every tile
    the same way, while a scene cut or a large moving subject scatters the tiles.
    Requiring a majority of tiles to agree on ONE shift keeps the two apart
    (measured: a static adjacent pair agrees to 0.45 px, a cut pair to 221 px).
    """
    tiles = _tile_shifts(ref, cur, cfg)
    res: dict[str, Any] = {"available": False, "tiles": len(tiles), "reason": ""}
    if len(tiles) < cfg.min_tiles:
        res["reason"] = f"only {len(tiles)} textured tile(s)"
        return res
    arr = np.asarray([[t[0], t[1]] for t in tiles], dtype=float)
    med = np.median(arr, axis=0)
    dev = np.linalg.norm(arr - med, axis=1)
    mask = dev <= cfg.inlier_px
    n_in = int(mask.sum())
    frac = n_in / len(tiles)
    res.update({"inlier_frac": frac, "inliers": n_in,
                "response_median": float(np.median([t[2] for t in tiles]))})
    if n_in < cfg.min_tiles or frac < cfg.min_inlier_frac:
        res["reason"] = (f"tiles do not agree on one shift "
                         f"({n_in}/{len(tiles)} within {cfg.inlier_px:.0f} px)")
        return res
    sub = arr[mask]
    med2 = np.median(sub, axis=0)            # trimmed re-median -> sub-pixel precision
    res.update({
        "available": True,
        "dx": float(med2[0]), "dy": float(med2[1]),
        "magnitude": float(np.hypot(med2[0], med2[1])),
        "tile_agreement_px": float(np.max(np.linalg.norm(sub - med2, axis=1))),
    })
    return res


def _line_x_at(p1: Sequence[float], p2: Sequence[float], y: float) -> float:
    if abs(p2[1] - p1[1]) < 1e-9:
        return float(p1[0])
    return float(p1[0] + (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1]))


def _line_y_at(p1: Sequence[float], p2: Sequence[float], x: float) -> float:
    if abs(p2[0] - p1[0]) < 1e-9:
        return float(p1[1])
    return float(p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0]))


def _line_drift(frames: Sequence[np.ndarray], line: Sequence[Sequence[float]], *,
                axis: str, radius: int) -> tuple[float, float, list[float]] | None:
    """Re-locate ``line`` on every frame -> ``(drift_max, noise_std, positions)``.

    Drift is relative to the *consensus* (median of the re-located positions), so
    a static camera reads ~0 and a moved camera reads the displacement; ``noise``
    is the spread of the same positions, i.e. the cue's own floor on this footage.
    """
    if not line or len(line) != 2:
        return None
    p1 = (float(line[0][0]), float(line[0][1]))
    p2 = (float(line[1][0]), float(line[1][1]))
    ref_fn = (calibration_mod.refine_dock_edge if axis == "x"
              else calibration_mod.refine_waterline)
    mid = (p1[1] + p2[1]) / 2.0 if axis == "x" else (p1[0] + p2[0]) / 2.0
    vals = []
    for fr in frames:
        q1, q2 = ref_fn(fr, p1, p2, radius=radius)
        vals.append(_line_x_at(q1, q2, mid) if axis == "x" else _line_y_at(q1, q2, mid))
    if len(vals) < 2:
        return None
    vals = np.asarray(vals, dtype=float)
    med = float(np.median(vals))
    return float(np.max(np.abs(vals - med))), float(np.std(vals)), [float(v) for v in vals]


# --- the report -------------------------------------------------------------
@dataclass
class StabilityReport:
    """PLAN §5's ``max_drift_px`` + ``verdict``, plus the evidence behind them."""

    verdict: str
    max_drift_px: float
    blocks_measurement: bool
    estimate_available: bool
    n_frames: int
    n_samples: int
    sample_frames: list[int]
    global_shift_px: float = 0.0          # worst cumulative within-segment drift, SOURCE px
    global_shift_xy: tuple[float, float] = (0.0, 0.0)
    max_pair_shift_px: float = 0.0        # worst single adjacent-pair shift
    segments: int = 1
    rejected_pairs: int = 0
    edge_drift_px: float | None = None    # dock-edge re-location, SOURCE px
    edge_noise_px: float | None = None    # this cue's own floor on these frames
    waterline_drift_px: float | None = None
    drift_in: float | None = None         # inches, when a calibration gives a scale
    thresholds: dict[str, float] = field(default_factory=dict)
    per_pair: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.verdict != "unusable"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usable"] = self.usable
        return d

    def summary(self) -> str:
        bits = [f"verdict={self.verdict}",
                f"max_drift={self.max_drift_px:.2f} px",
                f"cumulative={self.global_shift_px:.2f} px"]
        if not self.estimate_available:
            bits.append("NO ESTIMATE")
        if self.edge_drift_px is not None:
            bits.append(f"dock_edge={self.edge_drift_px:.2f} px "
                        f"(floor {self.edge_noise_px:.2f})")
        if self.waterline_drift_px is not None:
            bits.append(f"waterline={self.waterline_drift_px:.2f} px")
        if self.drift_in is not None:
            bits.append(f"~{self.drift_in:.2f} in")
        bits.append(f"samples={self.n_samples}/{self.n_frames} segments={self.segments}")
        return "  ".join(bits)


# --- the estimator ----------------------------------------------------------
def analyze_frames(
    frames: Sequence[np.ndarray],
    *,
    calib: dict | None = None,
    cfg: StabilityConfig = DEFAULT,
    frame_indices: Sequence[int] | None = None,
) -> StabilityReport:
    """PLAN §5's stability check over an explicit frame sequence.

    Split out from :func:`analyze` so the whole thing is drivable from a test with
    synthesized frames whose displacement is known by construction — the external
    oracle PLAN §8 V6 needs.  Increments are measured between **adjacent** frames
    (see the module docstring, measured reality (a)) and accumulated within a
    segment; a rejected pair closes the segment.
    """
    if len(frames) < 2:
        raise ValueError("stability needs at least 2 frames (the first is the reference)")
    idx = list(frame_indices) if frame_indices is not None else list(range(len(frames)))
    if len(idx) != len(frames):
        raise ValueError("frame_indices must label every frame")

    grays, scales = zip(*(_to_gray_work(f, cfg.work_width) for f in frames))
    scales = [1.0 if not s else s for s in scales]

    per_pair: list[dict[str, Any]] = []
    acc = np.zeros(2)
    worst_cum = 0.0
    worst_cum_xy = (0.0, 0.0)
    worst_pair = 0.0
    rejected = 0
    segments = 1

    for i in range(1, len(grays)):
        est = _pair_estimate(grays[i - 1], grays[i], cfg)
        rec = {"from": idx[i - 1], "to": idx[i],
               "tiles": est.get("tiles"), "response": est.get("response_median")}
        if not est.get("available"):
            rejected += 1
            segments += 1
            acc = np.zeros(2)                       # a break: do not accumulate across it
            rec.update({"available": False, "reason": est.get("reason"),
                        "magnitude": None})
            per_pair.append(rec)
            continue
        dx, dy = est["dx"] / scales[i], est["dy"] / scales[i]   # -> SOURCE px
        mag = float(np.hypot(dx, dy))
        acc += (dx, dy)
        cum = float(np.hypot(acc[0], acc[1]))
        if cum > worst_cum:
            worst_cum, worst_cum_xy = cum, (float(acc[0]), float(acc[1]))
        worst_pair = max(worst_pair, mag)
        rec.update({"available": True, "dx": dx, "dy": dy, "magnitude": mag,
                    "cumulative": cum, "inlier_frac": est.get("inlier_frac"),
                    "agreement_px": est.get("tile_agreement_px") / scales[i]})
        per_pair.append(rec)

    usable_pairs = sum(1 for p in per_pair if p.get("available"))
    estimate_available = usable_pairs > 0
    reasons: list[str] = []
    warnings: list[str] = []

    score = worst_cum
    if not estimate_available:
        reasons.append("no usable motion estimate: every sampled frame pair "
                       "disagreed on a single shift (low texture / scene "
                       "discontinuity) — refusing rather than reporting 'stable'")
    else:
        reasons.append(f"cumulative coherent drift {worst_cum:.2f} px over "
                       f"{usable_pairs} adjacent pair(s) in {segments} segment(s) "
                       f"(worst single pair {worst_pair:.2f} px)")
    if rejected:
        warnings.append(f"{rejected} sampled pair(s) rejected as discontinuous "
                        "(a shot change, or a pair that cannot be correlated) — "
                        "if these are shot changes at different camera positions, "
                        "one calibration cannot cover the clip (HANDOFF item 11)")

    edge_drift = edge_noise = water_drift = None
    if calib:
        d = _line_drift(frames, calib.get("dock_line") or [], axis="x",
                        radius=cfg.edge_radius)
        if d:
            edge_drift, edge_noise = d[0], d[1]
        w = _line_drift(frames, calib.get("waterline") or [], axis="y",
                        radius=cfg.edge_radius)
        if w:
            water_drift = w[0]

    if edge_drift is not None and edge_noise is not None:
        floor = max(cfg.edge_noise_floor, edge_noise)
        excess = max(0.0, edge_drift - floor)
        if excess > score:
            score = excess
            reasons.append(f"dock-edge re-location moved {edge_drift:.2f} px, "
                           f"beyond its {edge_noise:.2f} px floor -> the stored "
                           f"datum no longer sits on the edge")
        else:
            reasons.append(f"dock-edge re-location {edge_drift:.2f} px is within "
                           f"its {floor:.2f} px floor (the edge is a low-contrast "
                           f"ramp on this footage)")
    else:
        warnings.append("no calibration: the dock-edge re-location cue is unavailable "
                        "(data/calib/ is empty) — verdict rests on the coherent-shift cue")
    if water_drift is not None:
        warnings.append(f"waterline moved {water_drift:.2f} px — that is the water "
                        "LEVEL, not the camera; reported, not scored")

    if not estimate_available or score > cfg.unusable_px:
        verdict = "unusable"
    elif score > cfg.stable_px:
        verdict = "marginal"
    else:
        verdict = "stable"

    drift_in = None
    if calib:
        try:
            from dog_diving import distance as distance_mod
            drift_in = float(distance_mod.local_in_per_px(
                calibration_mod.dock_zero_x(calib), calib)) * score
        except Exception as exc:  # noqa: BLE001 - a scale is a convenience, not the gate
            warnings.append(f"could not convert drift to inches: {type(exc).__name__}: {exc}")

    max_drift = max([worst_cum] + ([edge_drift] if edge_drift is not None else []))
    return StabilityReport(
        verdict=verdict,
        max_drift_px=float(max_drift),
        blocks_measurement=(verdict == "unusable"),
        estimate_available=estimate_available,
        n_frames=len(frames),
        n_samples=len(frames),
        sample_frames=idx,
        global_shift_px=float(worst_cum),
        global_shift_xy=(float(worst_cum_xy[0]), float(worst_cum_xy[1])),
        max_pair_shift_px=float(worst_pair),
        segments=segments,
        rejected_pairs=rejected,
        edge_drift_px=(float(edge_drift) if edge_drift is not None else None),
        edge_noise_px=(float(edge_noise) if edge_noise is not None else None),
        waterline_drift_px=(float(water_drift) if water_drift is not None else None),
        drift_in=drift_in,
        thresholds={"stable_px": cfg.stable_px, "unusable_px": cfg.unusable_px},
        per_pair=per_pair,
        reasons=reasons,
        warnings=warnings,
    )


def analyze(
    clip: str | Path,
    *,
    calib: dict | None = None,
    cfg: StabilityConfig = DEFAULT,
    frame_range: tuple[int, int] | None = None,
    progress: Callable[[str, dict], None] | None = None,
) -> StabilityReport:
    """PLAN §5's check on a clip: sample ``cfg.n_samples`` frames and analyze them.

    Decoding goes through :mod:`dog_diving.video`, so the VFR/MOV traps M1 found
    apply here too (sample indices are placed on the *measured* frame count).
    """
    clip = Path(clip)
    info = video_mod.probe(clip, count_frames=False)
    n = int(info.n_frames)
    idx = sample_indices(n, cfg.n_samples)
    if frame_range:
        a, b = int(frame_range[0]), int(frame_range[1])
        idx = [a + i for i in sample_indices(max(2, b - a), cfg.n_samples) if a + i < b]
    if len(idx) < 2:
        raise StabilityError(f"clip {clip.name} is too short for a stability check")
    if progress:
        progress("stability", {"clip": clip.name, "n_frames": n, "samples": len(idx)})

    wanted = set(idx)
    frames: list[np.ndarray] = []
    got: list[int] = []
    for i, fr in video_mod.iter_frames(clip, start=min(idx), stop=(max(idx) + 1)):
        if i in wanted:
            frames.append(fr)
            got.append(i)
    if len(frames) < 2:
        raise StabilityError(f"could not decode 2+ sample frames from {clip}")

    rep = analyze_frames(frames, calib=calib, cfg=cfg, frame_indices=got)
    rep.n_frames = n
    return rep


# --- the block/override contract (PLAN §5: "blocks measurement, override logged") ---
def assert_usable(report: StabilityReport, *, override: bool = False,
                  note: str = "") -> dict[str, Any]:
    """Raise unless the datum is usable, or the caller explicitly overrides.

    Returns the override record to log in ``result.json``/``run_meta.json`` — an
    override that leaves no trace is exactly the silent correlated error the
    watchdog exists to prevent.
    """
    rec = {"verdict": report.verdict, "overridden": False, "note": note,
           "max_drift_px": report.max_drift_px,
           "estimate_available": report.estimate_available}
    if report.verdict == "unusable":
        if not override:
            raise StabilityError(
                f"unstable camera: {report.verdict} ({report.max_drift_px:.2f} px; "
                f"{report.summary()}). Re-calibrate, or override explicitly."
            )
        rec["overridden"] = True
    return rec


# --- CLI --------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Dock-edge drift / camera-pan check (PLAN §5/§8 V6).")
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--calib", default=None, help="calibration JSON path or setup id")
    ap.add_argument("--samples", type=int, default=DEFAULT.n_samples)
    ap.add_argument("--stable-px", type=float, default=DEFAULT.stable_px)
    ap.add_argument("--unusable-px", type=float, default=DEFAULT.unusable_px)
    ap.add_argument("--json", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calib = None
    if args.calib:
        p = Path(args.calib)
        if not p.is_file():
            p = calibration_mod.setup_path(str(args.calib))
        calib = calibration_mod.load(p)
    cfg = StabilityConfig(n_samples=args.samples, stable_px=args.stable_px,
                          unusable_px=args.unusable_px)
    rep = analyze(args.clip, calib=calib, cfg=cfg)
    if args.json:
        print(json.dumps(rep.as_dict(), indent=2))
    else:
        print(rep.summary())
        for r in rep.reasons:
            print(f"  - {r}")
        for w in rep.warnings:
            print(f"  ! {w}")
    return 0 if rep.usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
