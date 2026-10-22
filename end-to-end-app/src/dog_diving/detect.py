"""Detection — YOLO pose wrapper, per-frame keypoints + confidences, cached tracks.

M3 deliverable (PLAN.md §5 ``detect.py``).

What this module is for
-----------------------
Turn a clip into a *per-frame record* the rest of the pipeline can consume::

    {frame, tailset:[x, y, conf], nose:[x, y, conf], box_conf, track_id, accepted}

``accepted`` is the only judgement made here, and it is deliberately narrow: the
tailset keypoint is present, its **keypoint** confidence clears ``kp_conf``, and
the detection belongs to the dominant track for the clip.  Everything downstream
(trajectory, contact, distance) consumes ``records`` and re-reads the raw
confidences — this module never throws a low-confidence tailset away, it only
labels it.

The reference recipe, preserved
-------------------------------
``reference/Dog-Diving-Scoring/draw_tail_trajectory.py`` / ``commands.txt``::

    CLAHE on L of LAB, clipLimit=2.5, tileGridSize=(8, 8)
    model.track(source=enhanced, imgsz=..., conf=0.12, max_det=1,
                tracker="bytetrack.yaml", persist=True, verbose=False)
    accept the tailset keypoint when its keypoint confidence > 0.3
    TAILSET_INDEX = 1

Two deviations, both from PLAN §1.1/§6: ``device="cpu"`` (no CUDA on this host)
and no hard-coded ``I:\\Capstone\\...`` paths.

Gate on KEYPOINT confidence, not box confidence (measured 2026-10-07)
--------------------------------------------------------------------
On the vendored fine-tune the **box** confidence is badly miscalibrated — 0.12 to
0.30 on frames where the tailset keypoint reports 0.98.  Any accept/reject rule
written against box confidence throws away the best frames in the clip, so
``kp_conf`` is applied to ``tailset[2]`` and ``box_conf`` is carried as data
only.  Pinned by ``test_m3_detect.py``.

Caching (PLAN §5)
-----------------
``out/<clip_id>/tracks.json`` caches the whole run, keyed by

    (clip_sha, model_sha, imgsz, conf, kp_conf, clahe_params, frame_range)

so re-review never re-runs inference.  The file holds a *map* of keys, not a
single entry: one clip legitimately gets detected more than once (the V5
ablations vary ``imgsz`` and CLAHE), and a single-entry file would silently
invalidate the previous run every time.  Both properties PLAN asks for hold —
it is written once per distinct key, and a repeat of the same key is served from
disk with zero inference.  Superset of PLAN §5; the extra is the multi-key map.

FOURTH MEASURED CORRECTION TO PLAN — the dominant-track filter deletes the clip
------------------------------------------------------------------------------
PLAN §5 puts ``track_id`` in every record and requires "``track_id`` matches the
dominant track".  That rule is **unsafe on this checkpoint**, measured on the full
18-07.mp4 (2026-10-07):

* bytetrack's ``track_high_thresh`` and ``new_track_thresh`` are both 0.25, but
  this fine-tune's box confidences run 0.12-0.30 (median 0.14) — the same
  miscalibration that forces keypoint-confidence gating above.
* It therefore starts a track exactly once in the clip: frame 0, on a **junk**
  detection whose box conf is 0.304 while its tailset keypoint conf is 0.19.
  The track survives 5 frames and is never seen again in the remaining 445.
* That 5-frame junk track is still the most common id, so it *is* the "dominant
  track" — and every one of the clip's 30 genuinely confident tailset frames
  (kp conf 0.80-0.99, box conf 0.12-0.25) is rejected for having
  ``track_id=None``.  **A clip with an obvious dog reports 0 accepted frames**,
  with no error, no warning and no low-count tell in the schema.

Fixed by gating the filter on whether tracking is actually tracking.  The dominant
id must (a) cover at least ``DetectConfig.track_coverage_min`` (default 0.5) of the
clip's detections **and** (b) have at least half its own frames carry a tailset
keypoint above ``kp_conf`` — measured 5/41 = 12% coverage and 0/5 keypoint-backed
here, against ~100% for a track that spans the dog's flight, so neither gate is a
tuned threshold.  ``tracking_available``, ``track_coverage``,
``n_dominant_frames`` and ``n_dominant_with_kp`` are carried in the result and the
cache entry, so the fallback is visible rather than silent.  Condition (b) is
load-bearing: over frames 0-8 the junk track's coverage is 62% (it clears the
coverage gate) and still rejects 3 confident frames.  Both failure modes are pinned
by ``test_m3_detect.py``.  The real fix (a tracker config whose ``new_track_thresh``
sits below this detector's box confidences) is an open item in HANDOFF.md.

Reported throughput is *measured*, not assumed: ``timings`` carries the wall-clock
ms/frame and the inference-only ms/frame, plus a projection for the whole clip.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import pose_layout, video
from .config import (
    CONF_LADDER,
    DEFAULT_CLAHE_CLIP,
    DEFAULT_CLAHE_GRID,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    DEFAULT_KP_CONF,
    DEFAULT_TRACKER,
    IMGSZ_LADDER,
    KP_CONF_LADDER,
    MIN_ACCEPTED_POINTS,
    MODEL_PATH,
    NOSE_INDEX,
    OUT_DIR,
    TAILSET_INDEX,
)

# The model of record is 2-keypoint dog pose with a single class, and its keypoint order is
# ``[nose, tailset]``.  The *shape* is still the gate — pointing the pipeline at a 24-kpt
# checkpoint silently shifts TAILSET_INDEX onto the wrong joint, which is exactly the kind of
# failure that produces plausible-looking, wrong distances — but the gate is now a lookup in
# ``pose_layout`` rather than a hard-coded 2-kpt check, so a mapped family (COCO person pose,
# 17 kpt) can run with its own, written-down mapping.  See ``pose_layout.py``.


#: 4 = the 2026-10-22 rename (``dockdogs`` -> ``dog_diving``), which moved the package, the saved
#: documents' schema ids and the report's name.  Warm ``tracks.json`` files from before it are
#: *ignored* rather than trusted, so no artifact mixes two names.
CACHE_VERSION = 4


# --- configuration ----------------------------------------------------------
@dataclass(frozen=True)
class DetectConfig:
    """Every knob that changes the *output* of detection.

    Anything in here must also appear in the cache key: two configs that differ
    only in a field the key ignores would silently share one cache entry.
    """

    model_path: Path = MODEL_PATH
    imgsz: int = DEFAULT_IMGSZ
    conf: float = DEFAULT_CONF
    kp_conf: float = DEFAULT_KP_CONF
    tracker: str = DEFAULT_TRACKER
    device: str = DEFAULT_DEVICE
    max_det: int = 1
    clahe: bool = True
    clahe_clip: float = DEFAULT_CLAHE_CLIP
    clahe_grid: tuple[int, int] = DEFAULT_CLAHE_GRID
    # Minimum share of a clip's detections that must carry a track id before the
    # dominant-track filter is trusted (see `track_stats`).  Part of the cache key:
    # it changes which frames are accepted.
    track_coverage_min: float = 0.5

    def clahe_params(self) -> dict[str, Any]:
        return (
            {"enabled": True, "clip": float(self.clahe_clip), "grid": list(self.clahe_grid)}
            if self.clahe
            else {"enabled": False}
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_path"] = str(self.model_path)
        d["clahe_grid"] = list(self.clahe_grid)
        return d


@dataclass
class DetectResult:
    """Return value of :func:`detect_clip` — records plus how they were obtained."""

    clip_id: str
    clip_path: str
    clip_sha: str
    model_sha: str
    cache_key: dict[str, Any]
    cache_key_hash: str
    cache_path: str
    frame_range: list[int] | None
    records: list[dict]
    dominant_track: int | None
    tracking_available: bool
    track_coverage: float
    n_tracked_frames: int
    n_dominant_frames: int
    n_dominant_with_kp: int
    n_frames_clip: int
    n_frames_processed: int
    n_detections: int
    n_accepted: int
    timings: dict[str, float]
    from_cache: bool
    config: dict[str, Any] = field(default_factory=dict)
    # Filled in only by :func:`detect_clip_adaptive`; ``None`` means the fixed-threshold path ran.
    ladder: dict[str, Any] | None = None
    # Which keypoint mapping produced ``records`` (``pose_layout.PoseLayout.key``), and whether
    # the tailset was a labelled keypoint or derived from another joint.  Provenance: a
    # ``coco-person-17`` run's "tailset" is a person's hips midpoint, not a dog's tail base.
    model_layout: str = ""
    tailset_derived: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def accepted_frames(self) -> list[int]:
        return [r["frame"] for r in self.records if r["accepted"]]


# --- enhancement + checkpoint ----------------------------------------------
def clahe_lab_l(
    frame: np.ndarray, clip: float = DEFAULT_CLAHE_CLIP, grid: tuple[int, int] = DEFAULT_CLAHE_GRID
) -> np.ndarray:
    """CLAHE on the L channel of LAB, exactly as ``draw_tail_trajectory.py`` does."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid[0]), int(grid[1])))
    l_enh = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_enh, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def verify_checkpoint(model: Any) -> "pose_layout.PoseLayout":
    """Resolve — or refuse — the loaded checkpoint's keypoint layout.

    Returns the :class:`dog_diving.pose_layout.PoseLayout` for the model's ``kpt_shape``
    (``dog-2kpt`` for the vendored 2-kpt checkpoint, ``coco-person-17`` for a stock COCO pose
    model).  Raises ``ValueError`` with the actual shape in the message for any family that has
    no written-down mapping — a mismatch is a configuration error, never something to warn
    about and continue past.

    The keypoint *meaning* is now carried by the layout rather than assumed by index constant:
    callers must extract with ``layout.extract(xy, conf)`` (or ``layout.tailset`` /
    ``layout.nose``) instead of indexing ``TAILSET_INDEX`` directly.
    """
    return pose_layout.layout_for_model(model)


def load_model(cfg: DetectConfig) -> Any:
    """Load + verify the checkpoint once.  Import of ultralytics is deferred so the
    module (and its cache/record helpers) stay importable without torch installed."""
    from ultralytics import YOLO  # local import: heavy, and only needed to infer

    path = Path(cfg.model_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model = YOLO(str(path))
    verify_checkpoint(model)
    return model


def reset_tracker(model: Any) -> bool:
    """Clear persisted tracker state so a run does not inherit ids from the last one.

    ``persist=True`` keeps the tracker alive across ``track`` calls by design, which
    is what we want *within* a clip and exactly what we do not want *between* runs:
    the same frame range would otherwise come back with different track ids and a
    different dominant track.  Best-effort — a missing/incompatible tracker API
    returns False and the caller records that.
    """
    pred = getattr(model, "predictor", None)
    trackers = getattr(pred, "trackers", None) if pred is not None else None
    if not trackers:
        return True
    try:
        for t in trackers:
            t.reset()
    except Exception:  # noqa: BLE001 - tracker internals are not a stable API
        return False
    return True


# --- cache ------------------------------------------------------------------
#: Which detector produced a record set.  Part of the cache key *and* written into every entry:
#: ``predict_v1`` is the trajectory fork's path (``image_set.detect_frames``, no tracker); the
#: tracker-derived entries that predate 2026-10-20 must never be served to it.
DETECTOR = "predict_v1"


def cache_key(
    clip_sha: str,
    model_sha: str,
    cfg: DetectConfig,
    frame_range: tuple[int, int] | None,
) -> dict[str, Any]:
    """PLAN §5's cache identity, plus the detector that produced it."""
    return {
        "detector": DETECTOR,
        "clip_sha": clip_sha,
        "model_sha": model_sha,
        "imgsz": int(cfg.imgsz),
        "conf": float(cfg.conf),
        "kp_conf": float(cfg.kp_conf),
        "clahe": cfg.clahe_params(),
        "track_coverage_min": float(cfg.track_coverage_min),
        "frame_range": list(frame_range) if frame_range else None,
    }


def key_hash(key: dict[str, Any]) -> str:
    blob = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def tracks_path(clip_id: str, out_dir: str | Path | None = None) -> Path:
    return (Path(out_dir) if out_dir is not None else OUT_DIR) / clip_id / "tracks.json"


def load_cache(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if doc.get("version") != CACHE_VERSION:
        return {}
    return doc


def save_cache(doc: dict, path: str | Path) -> None:
    """Atomic write — a truncated cache would be silently treated as a miss next run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


# --- acceptance -------------------------------------------------------------
def dominant_track(records: list[dict]) -> int | None:
    """The track id seen on the most frames, or ``None`` when nothing was tracked."""
    ids = [r["track_id"] for r in records if r.get("track_id") is not None]
    if not ids:
        return None
    return Counter(ids).most_common(1)[0][0]


def track_stats(
    records: list[dict], coverage_min: float = 0.5, kp_conf: float = DEFAULT_KP_CONF
) -> dict[str, Any]:
    """Decide whether the dominant-track filter can be trusted on this clip.

    PLAN §5's accept rule needs "``track_id`` matches the dominant track".  Applied
    literally on this corpus it does not filter outliers — it deletes the clip.
    Measured on 18-07.mp4, full clip (2026-10-07): bytetrack started a track at
    frame 0 on a *junk* detection (box conf 0.304 clears ``new_track_thresh=0.25``
    while its keypoint conf is 0.19), kept it alive for 5 frames, then lost it for
    the remaining 445.  That single 5-frame track is still the most common id, so it
    becomes the "dominant track" and every one of the 30 frames whose tailset
    keypoint is confident (0.80-0.99) is rejected for having ``track_id=None``.
    Net effect: a clip with an obvious dog reports **0 accepted frames**, silently.

    Two conditions must both hold before the filter is applied, both about whether
    the tracker is following *this* dog rather than merely producing ids:

    * **coverage** — the dominant id must tag at least ``coverage_min`` of the clip's
      detections.  Measured 5/41 = 12% here, against ~100% for a track spanning the
      dog's flight.
    * **keypoint backing** — at least half of the dominant track's own frames must
      carry a tailset keypoint above ``kp_conf``.  The junk track's 5 frames score
      0/5 (kp conf 0.14-0.19): it is a track of things that are not the dog.

    Either condition failing means the filter is reported as data and not applied.
    This can only ever *stop the filter from deleting frames*; with ``max_det=1``
    there is at most one detection per frame anyway, so the filter's only observable
    effect on this corpus is to discard the dog's own detections.

    Returns the raw dominant id plus the evidence for the verdict, so a reader sees
    the numbers rather than a bare boolean.
    """
    det = [r for r in records if r.get("tailset") or r.get("track_id") is not None]
    dom = dominant_track(records)
    n_tracked = sum(1 for r in records if r.get("track_id") is not None)
    coverage = (n_tracked / len(det)) if det else 0.0
    dom_frames = [r for r in records if r.get("track_id") == dom] if dom is not None else []
    n_dom = len(dom_frames)
    n_dom_kp = sum(1 for r in dom_frames if r.get("tailset") and r["tailset"][2] is not None
                   and float(r["tailset"][2]) > kp_conf)
    kp_backed = bool(n_dom) and n_dom_kp >= 0.5 * n_dom
    return {
        "dominant_track": dom,
        "track_coverage": coverage,
        "n_tracked_frames": n_tracked,
        "n_dominant_frames": n_dom,
        "n_dominant_with_kp": n_dom_kp,
        "tracking_available": dom is not None and coverage >= coverage_min and kp_backed,
    }


def apply_acceptance(records: list[dict], kp_conf: float, dominant: int | None) -> int:
    """Mark ``accepted`` in place; return the accepted count.

    Rule (PLAN §5): tailset present AND ``tailset[2] > kp_conf`` AND
    (no tracking available OR ``track_id == dominant``).  ``dominant`` must be the
    *effective* id — the caller passes ``None`` when :func:`track_stats` found the
    tracking untrustworthy, rather than rejecting every frame for a missing tracker.
    """
    n = 0
    for r in records:
        ts = r.get("tailset")
        ok = bool(ts) and ts[2] is not None and float(ts[2]) > kp_conf
        if ok and dominant is not None:
            ok = r.get("track_id") == dominant
        r["accepted"] = ok
        n += int(ok)
    return n


# --- adaptive threshold ladder (operator instruction, 2026-10-12) ------------
# "Dynamically change the threshold so that a minimum of 8 points are identified, with the
#  threshold being lowered if 8 points can't be found with the current confidence level."
#
# What "a point" means here was settled with the operator: an *accepted keypoint row* — a frame
# whose tailset clears the acceptance rule — i.e. exactly the rows :mod:`trajectory` consumes.
@dataclass
class LadderRung:
    """One ``(imgsz, conf, kp_conf)`` combination tried by :func:`detect_clip_adaptive`."""

    imgsz: int
    conf: float
    kp_conf: float
    n_detections: int
    n_accepted: int
    met_minimum: bool

    def as_dict(self) -> dict[str, Any]:
        # Deliberately no cache state: this block ships inside result.json, which is a pure function
        # of its inputs (PLAN §11), and "was this pass served from disk" is not one of them.
        return asdict(self)


@dataclass
class LadderReport:
    """What the ladder did, in the order it did it — the provenance half of the feature."""

    min_points: int
    base: dict[str, Any]
    chosen: dict[str, Any] | None
    rungs: list[dict[str, Any]]
    imgsz_tried: list[int]
    imgsz_escalated: bool
    below_min_points: bool
    n_points: int
    n_low_conf_points: int

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disabled"] = int(self.min_points) <= 0
        return d


def _rung_config(cfg: DetectConfig, *, imgsz: int, conf: float, kp_conf: float) -> DetectConfig:
    """The config for one inference pass.

    Field-by-field rather than ``replace(cfg, ...)`` for the three varied fields plus an explicit
    copy of the rest, because a knob added to ``DetectConfig`` later must be *considered* here
    rather than silently taking its default inside an ablation.
    """
    return DetectConfig(
        model_path=cfg.model_path, imgsz=int(imgsz), conf=float(conf), kp_conf=float(kp_conf),
        tracker=cfg.tracker, device=cfg.device, max_det=cfg.max_det, clahe=cfg.clahe,
        clahe_clip=cfg.clahe_clip, clahe_grid=cfg.clahe_grid,
        track_coverage_min=cfg.track_coverage_min,
    )


def _rung_view(records: list[dict], *, conf: float, kp_conf: float, coverage_min: float,
               base_conf: float, base_kp_conf: float) -> tuple[list[dict], dict[str, Any]]:
    """Apply ``conf`` as a post-hoc box pre-filter and ``kp_conf`` as the accept rule.

    Post-hoc is *exact*, not an approximation: ``model.track(conf=c)`` keeps the highest-confidence
    box of the frame (``max_det=1`` here) and drops the frame when that box is below ``c``, so
    re-reading each record's stored ``box_conf`` against ``c`` reproduces a pass at ``c`` frame for
    frame.  That is why the whole ladder costs one inference pass instead of one per rung.

    Every record is annotated with the thresholds it was judged under, and with ``low_conf`` —
    True when this point would NOT have been admitted at the run's base thresholds.
    """
    sel = [dict(r) for r in records]
    for r in sel:
        if r.get("tailset") is not None and r.get("box_conf") is not None \
                and float(r["box_conf"]) < float(conf):
            r["tailset"] = None
            r["nose"] = None
            r["box_conf"] = None
            r["track_id"] = None
        r["conf_used"] = float(conf)
        r["kp_conf_used"] = float(kp_conf)
        kp = r["tailset"][2] if r.get("tailset") else None
        r["low_conf"] = bool(
            (r.get("box_conf") is not None and float(r["box_conf"]) < float(base_conf))
            or (kp is not None and float(kp) <= float(base_kp_conf))
        )
    stats = track_stats(sel, coverage_min, float(kp_conf))
    apply_acceptance(sel, float(kp_conf), stats["dominant_track"] if stats["tracking_available"] else None)
    return sel, stats


def _rung_values(seq: Any, base: float) -> list[float]:
    """``base`` first, then every rung strictly below it, descending.  Never raises a threshold."""
    lower = {float(v) for v in seq if float(v) < float(base)}
    return [float(base), *sorted(lower, reverse=True)]


def select_rung(records: list[dict], *, min_points: int, conf_ladder: Any, kp_conf_ladder: Any,
                coverage_min: float, imgsz: int, base_conf: float, base_kp_conf: float,
                include_base_rung: bool = True,
                ) -> tuple[list[dict], LadderRung | None, list[LadderRung], bool]:
    """The most conservative rung that reaches ``min_points``; else the rung with the most points.

    Preference order is conservatism, not convenience: for each ``conf`` rung (descending) every
    ``kp_conf`` rung is tried (descending), so a strict box gate is preferred over a loosened
    keypoint gate at the same depth — the operator's stated order ("conf down through the ladder,
    then kp_conf if still short").

    Returns ``(selected_records, chosen_rung, rungs_tried, met_minimum)``.  ``selected_records``
    carries every record (with ``accepted`` set by the chosen rung), so the artifact keeps the
    frames the rung *rejected* as evidence rather than discarding them.
    """
    tried: list[LadderRung] = []
    best: tuple[LadderRung, list[dict]] | None = None
    confs = _rung_values(conf_ladder, base_conf)
    kps = _rung_values(kp_conf_ladder, base_kp_conf)
    if not include_base_rung:
        # The base-`conf` rungs were already evaluated by the pass that ran at the base `conf`, and a
        # box_conf filter can only *remove* rows — re-trying them here would pad the artifact with
        # identical rows (measured: 3 duplicated rungs before this was fixed).
        confs = [c for c in confs if c < float(base_conf)]
    for conf in confs:
        for kp_conf in kps:
            sel, _ = _rung_view(records, conf=conf, kp_conf=kp_conf, coverage_min=coverage_min,
                                base_conf=base_conf, base_kp_conf=base_kp_conf)
            n_acc = sum(1 for r in sel if r["accepted"])
            rung = LadderRung(imgsz=int(imgsz), conf=float(conf), kp_conf=float(kp_conf),
                              n_detections=sum(1 for r in sel if r["tailset"]), n_accepted=n_acc,
                              met_minimum=n_acc >= int(min_points))
            tried.append(rung)
            if best is None or n_acc > best[0].n_accepted:
                best = (rung, sel)
            if n_acc >= int(min_points):
                return sel, rung, tried, True
    if best is None:
        return [], None, tried, False
    return best[1], best[0], tried, False


def detect_clip_adaptive(
    clip: str | Path,
    cfg: DetectConfig | None = None,
    *,
    min_points: int | None = None,
    conf_ladder: Any = None,
    kp_conf_ladder: Any = None,
    imgsz_ladder: Any = None,
    frame_range: tuple[int, int] | None = None,
    cancel_event: Any = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
    out_dir: str | Path | None = None,
    use_cache: bool = True,
    model: Any | None = None,
) -> DetectResult:
    """Detect a clip, lowering the threshold until ``min_points`` accepted rows exist.

    Order of the search (all operator-settled, 2026-10-12):

    1. **The base config's own thresholds.**  If the clip already yields ``min_points`` accepted
       rows, nothing is lowered and the run is byte-identical to :func:`detect_clip` — the cost of
       this feature on a healthy clip is zero.
    2. **One pass at the floor** ``conf`` (``kp_conf`` needs no pass — see the note on ``passes``),
       then re-select from its records.  Every ``conf`` rung is tried before any ``kp_conf`` rung
       is lowered.
    3. **``imgsz`` escalation** — the final rung, one pass per larger size, only if every
       ``conf``/``kp_conf`` rung is still short.
    4. **Give up loudly**: the run proceeds with the rung that had the most points and is flagged
       ``below_min_points``.

    MEASURED CAVEAT — this does *not* make the extra points trustworthy.  On
    ``Dog Dock Diving Slow Mo 2-...-seg07`` (149 frames, 2026-10-12) the base rung gives 3 accepted
    points, ``conf=0.06`` gives 78, and the added ones include **empty sky** (frame 50, kp conf
    0.960) and a **person** at the pool edge (frame 130, kp conf 0.995), while ``imgsz=1920``
    still gives only 3.  So every point admitted below the base thresholds is marked ``low_conf``
    in the records, counted in the ``LadderReport``, and disclosed in ``result.json`` /
    ``report.txt`` (see :func:`dog_diving.report._ladder_warnings`).
    """
    cfg = cfg or DetectConfig()
    min_points = int(MIN_ACCEPTED_POINTS if min_points is None else min_points)
    conf_ladder = CONF_LADDER if conf_ladder is None else conf_ladder
    kp_conf_ladder = KP_CONF_LADDER if kp_conf_ladder is None else kp_conf_ladder
    imgsz_ladder = IMGSZ_LADDER if imgsz_ladder is None else imgsz_ladder
    base = {"imgsz": int(cfg.imgsz), "conf": float(cfg.conf), "kp_conf": float(cfg.kp_conf)}
    larger = sorted({int(v) for v in imgsz_ladder if int(v) > int(cfg.imgsz)})

    def finish(det: DetectResult, sel: list[dict], rung: LadderRung | None, tried: list[LadderRung],
               met: bool, imgss: list[int]) -> DetectResult:
        n_points = sum(1 for r in sel if r["accepted"])
        stats = (track_stats(sel, cfg.track_coverage_min, float(rung.kp_conf)) if rung else {})
        chosen = dict(rung.as_dict()) if rung else None
        if chosen is not None:
            chosen["base_conf"] = base["conf"]
            chosen["base_kp_conf"] = base["kp_conf"]
        report = LadderReport(
            min_points=int(min_points), base=base, chosen=chosen,
            rungs=[r.as_dict() for r in tried], imgsz_tried=[int(v) for v in imgss],
            imgsz_escalated=bool(imgss and imgss[-1] != base["imgsz"]),
            below_min_points=bool(rung is not None and not met),
            n_points=int(n_points),
            n_low_conf_points=int(sum(1 for r in sel if r["accepted"] and r.get("low_conf"))),
        ).as_dict()
        return replace(
            det, records=sel, n_detections=sum(1 for r in sel if r["tailset"]),
            n_accepted=int(n_points), ladder=report,
            dominant_track=stats.get("dominant_track", det.dominant_track),
            tracking_available=stats.get("tracking_available", det.tracking_available),
            track_coverage=stats.get("track_coverage", det.track_coverage),
            n_tracked_frames=stats.get("n_tracked_frames", det.n_tracked_frames),
            n_dominant_frames=stats.get("n_dominant_frames", det.n_dominant_frames),
            n_dominant_with_kp=stats.get("n_dominant_with_kp", det.n_dominant_with_kp),
        )

    # 1. the base pass — cache-friendly, and the only pass a healthy clip ever needs
    det0 = detect_clip(clip, cfg, frame_range=frame_range, cancel_event=cancel_event,
                       progress_cb=progress_cb, out_dir=out_dir, use_cache=use_cache, model=model)
    if min_points <= 0:                                    # ladder off: today's behaviour, and a
        zero = LadderReport(min_points=0, base=base, chosen=None, rungs=[],   # report that says so
                            imgsz_tried=[int(cfg.imgsz)], imgsz_escalated=False,
                            below_min_points=False, n_points=int(det0.n_accepted),
                            n_low_conf_points=0).as_dict()
        return replace(det0, ladder=zero)
    # This pass ran at the base `conf`, so its records cannot contain a frame that a *lower* conf
    # would have added — only the kp_conf rungs are evaluable here, and they are worth evaluating:
    # a clip whose boxes are confident but whose keypoints sit just under kp_conf is fixed by this
    # sweep alone, with no second inference pass at all.
    sel0, rung0, tried0, met0 = select_rung(
        det0.records, min_points=min_points, conf_ladder=(float(cfg.conf),),
        kp_conf_ladder=kp_conf_ladder, coverage_min=cfg.track_coverage_min, imgsz=cfg.imgsz,
        base_conf=cfg.conf, base_kp_conf=cfg.kp_conf)
    if met0:
        return finish(det0, sel0, rung0, tried0, True, [int(cfg.imgsz)])

    # 2./3. short at the base rung: one floor pass here, then one per larger imgsz
    floor_conf = min(_rung_values(conf_ladder, cfg.conf))
    # `kp_conf` is deliberately NOT lowered in the extra passes: it never reaches the model (only
    # `conf` is passed to `model.track`), it only sets the `accepted` flag on a row that has already
    # been stored with its own keypoint confidence.  So every kp_conf rung is applied post-hoc on
    # the stored confidences — one less inference pass, and this pass stays on the base kp_conf,
    # which is the cache entry an ordinary run already has.
    passes = [(int(cfg.imgsz), floor_conf, float(cfg.kp_conf))]
    passes += [(v, floor_conf, float(cfg.kp_conf)) for v in larger]
    tried_all = list(tried0)
    imgss: list[int] = [int(cfg.imgsz)]
    best: tuple[LadderRung | None, list[dict], DetectResult] = (rung0, sel0, det0)
    for imgsz, pconf, pkp in passes:
        pcfg = _rung_config(cfg, imgsz=imgsz, conf=pconf, kp_conf=pkp)
        # Progress: the extra passes report the *end* of their own progress against the base pass's
        # frame count, so the UI's counter stays monotonic and its final value keeps meaning "the
        # clip's frames" — a second inference pass must not restart the operator's progress bar.
        # The per-pass ETA is real.  Which passes ran is disclosed in the ladder block and the log.
        cb = None
        if progress_cb is not None and det0.n_frames_processed:
            n0 = int(det0.n_frames_processed)

            def cb(done, total, eta_s, _cb=progress_cb, _n=n0):  # noqa: ANN001
                _cb(int(_n), int(_n), float(eta_s))

        det = detect_clip(clip, pcfg, frame_range=frame_range, cancel_event=cancel_event,
                          progress_cb=cb, out_dir=out_dir, use_cache=use_cache, model=model)
        if int(imgsz) not in imgss:
            imgss.append(int(imgsz))
        sel_i, rung_i, tried_i, met_i = select_rung(
            det.records, min_points=min_points, conf_ladder=conf_ladder,
            kp_conf_ladder=kp_conf_ladder, coverage_min=cfg.track_coverage_min, imgsz=imgsz,
            base_conf=cfg.conf, base_kp_conf=cfg.kp_conf, include_base_rung=False)
        tried_all.extend(tried_i)
        if rung_i is not None and (best[0] is None or rung_i.n_accepted > best[0].n_accepted):
            best = (rung_i, sel_i, det)
        if met_i:
            return finish(det, sel_i, rung_i, tried_all, True, imgss)

    # 4. nothing reached the minimum: report the best rung and flag it
    return finish(best[2], best[1], best[0], tried_all, False, imgss)


# --- inference --------------------------------------------------------------
def _extract(res: Any, layout: "pose_layout.PoseLayout | None" = None) -> tuple[list | None, list | None, float | None, int | None]:
    """Pull (tailset, nose, box_conf, track_id) out of one ultralytics result.

    Keypoint indices come from ``layout`` (default: the 2-kpt dog model of record), so a COCO-17
    checkpoint yields its derived hips point rather than row 1 of a person's right eye.
    """
    layout = layout or pose_layout.DOG_2KPT
    kp = getattr(res, "keypoints", None)
    data = getattr(kp, "data", None)
    xy = None if data is None or len(data) == 0 else data[0].tolist()
    tailset = nose = None
    if xy is not None:
        tailset, nose = layout.extract(xy, [row[2] if len(row) > 2 else 1.0 for row in xy])
    boxes = getattr(res, "boxes", None)
    box_conf = None
    track_id = None
    if boxes is not None and len(boxes):
        box_conf = float(boxes.conf[0])
        ids = getattr(boxes, "id", None)
        if ids is not None and len(ids):
            track_id = int(ids[0])
    return tailset, nose, box_conf, track_id


class VideoFrames:
    """The four members ``image_set.detect_frames`` needs, over a video clip.

    ``detect_frames`` only ever touches ``root``, ``n_frames``, ``numbers``, ``read(i)`` and
    ``gt_px(n)``, so a clip can be handed to the trajectory project's own still-frame detector
    **unmodified**.  Frames are decoded sequentially when asked in order (the normal case) and by
    seek otherwise, so a full-clip run is still a single forward pass over the file.
    """

    def __init__(self, clip: str | Path, *, frame_range: tuple[int, int] | None = None,
                 count: int | None = None) -> None:
        self.clip = Path(clip)
        info = video.probe(self.clip, count_frames=count is not None)
        if frame_range is not None:
            start, stop = int(frame_range[0]), int(frame_range[1])
            self.numbers = list(range(start, stop))
        else:
            self.numbers = list(range(int(info.n_frames)))
        self.root = self.clip
        self.clip_frames = int(info.n_frames)      # the whole clip, not the frame_range
        self.n_frames = len(self.numbers)
        self._cap = None
        self._next = 0
        self._last_i = -1
        self._last = None

    def read(self, i: int):
        import cv2
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.clip))
        if i == self._last_i:
            return self._last
        if i != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.numbers[i]))
            self._next = i
        ok, frame = self._cap.read()
        self._next = i + 1
        if not ok:
            raise RuntimeError(f"could not decode frame {i} ({self.numbers[i]}) of {self.clip}")
        self._last_i, self._last = i, frame
        return frame

    def gt_px(self, frame_no: int) -> dict | None:
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def detect_clip(
    clip: str | Path,
    cfg: DetectConfig | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
    cancel_event: Any | None = None,
    *,
    frame_range: tuple[int, int] | None = None,
    out_dir: str | Path | None = None,
    use_cache: bool = True,
    model: Any | None = None,
) -> DetectResult:
    """Detect every frame of ``clip`` with **the trajectory project's own detector**.

    The frames go through ``image_set.detect_frames`` verbatim (PLAN §16): one ``model.predict``
    per frame, ``max_det=1``, no tracker and therefore **no dominant-track filter**.  That is the
    path the model was measured on (46/46 stills, tailset median 8.18 px) and the one that keeps a
    dog the tracker used to discard.

    Records and the ``DetectResult`` shape are unchanged, so ``trajectory.build_trajectory``,
    ``pool.pipeline.records_to_track``, ``contact`` and every artifact keep working.  Tracker
    fields are reported honestly as absent (``dominant_track=None``, ``tracking_available=False``,
    ``track_coverage=0.0``).

    ``progress_cb(done, total, eta_s)`` is called every frame; a cancelled run returns a partial
    result and does **not** write the cache.
    """
    cfg = cfg or DetectConfig()
    clip = Path(clip)
    if not clip.is_file():
        raise FileNotFoundError(clip)
    if frame_range is not None:
        start, stop = int(frame_range[0]), int(frame_range[1])
        if stop <= start:
            raise ValueError(f"frame_range must be half-open (start < stop), got {frame_range}")

    t0 = time.perf_counter()
    clip_sha = video.sha256(clip)
    model_sha = video.sha256(Path(cfg.model_path))
    key = cache_key(clip_sha, model_sha, cfg, frame_range)
    khash = key_hash(key)
    cpath = tracks_path(clip.stem, out_dir)

    if use_cache:
        doc = load_cache(cpath)
        e = (doc.get("entries") or {}).get(khash)
        if e is not None:
            return DetectResult(
                clip_id=clip.stem, clip_path=str(clip), clip_sha=clip_sha, model_sha=model_sha,
                cache_key=key, cache_key_hash=khash, cache_path=str(cpath),
                frame_range=e.get("frame_range"), records=e["records"],
                dominant_track=None, tracking_available=False, track_coverage=0.0,
                n_tracked_frames=0, n_dominant_frames=0, n_dominant_with_kp=0,
                n_frames_clip=e.get("n_frames_clip", 0),
                n_frames_processed=e.get("n_frames_processed", len(e["records"])),
                n_detections=e.get("n_detections", 0), n_accepted=e.get("n_accepted", 0),
                timings={**e.get("timings", {}), "cache_load_s": time.perf_counter() - t0},
                from_cache=True, config=cfg.as_dict(),
                model_layout=e.get("layout", ""),
                tailset_derived=bool(e.get("tailset_derived", False)),
            )

    if model is None:
        model = load_model(cfg)
    layout = verify_checkpoint(model)

    from . import image_set as _image_set          # local: keeps this module torch-free on import

    vf = VideoFrames(clip, frame_range=frame_range)
    cancelled = False
    state = {"done": 0}

    def _progress(done: int, total: int, _eta: float) -> None:
        state["done"] = int(done)
        if progress_cb is not None:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / done) * max(int(total) - int(done), 0) if done else 0.0
            progress_cb(int(done), int(total), eta)

    def _cancel() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    try:
        fd = _image_set.detect_frames(vf, cfg, model=model, progress_cb=_progress,
                                      cancel_event=_Cancelled(cancel_event))
        cancelled = _cancel()
    finally:
        vf.close()

    records = list(fd.records)
    n = len(records)
    total_s = time.perf_counter() - t0
    n_det = sum(1 for r in records if r.get("tailset"))
    n_acc = sum(1 for r in records if r.get("accepted"))
    timings = {
        "total_s": total_s,
        "ms_per_frame": (total_s / n * 1000.0) if n else float("nan"),
        "inference_ms_per_frame": float("nan"),        # predict time is not instrumented here
        "fps": (n / total_s) if (n and total_s) else float("nan"),
        "projected_clip_s": (vf.clip_frames * total_s / n) if n else float("nan"),
        "cancelled": float(cancelled),
    }

    result = DetectResult(
        clip_id=clip.stem, clip_path=str(clip), clip_sha=clip_sha, model_sha=model_sha,
        cache_key=key, cache_key_hash=khash, cache_path=str(cpath),
        frame_range=list(frame_range) if frame_range else None, records=records,
        dominant_track=None, tracking_available=False, track_coverage=0.0,
        n_tracked_frames=0, n_dominant_frames=0, n_dominant_with_kp=0,
        n_frames_clip=vf.clip_frames, n_frames_processed=n, n_detections=n_det, n_accepted=n_acc,
        timings=timings, from_cache=False, config=cfg.as_dict(),
        model_layout=layout.key, tailset_derived=bool(layout.tailset_derived),
    )

    if use_cache and not cancelled:
        doc = load_cache(cpath)
        doc.setdefault("version", CACHE_VERSION)
        doc.setdefault("clip_id", clip.stem)
        doc.setdefault("entries", {})
        doc["clip_sha"] = clip_sha
        doc["model_sha"] = model_sha
        doc["entries"][khash] = {
            # provenance INSIDE the entry: the file-level header is rewritten by every writer, so
            # an entry that only trusts it cannot say which model/detector produced it (measured:
            # the 2026-10-20 investigation could not tell from disk, and had to re-run).
            "key": key, "model_sha": model_sha, "detector": DETECTOR,
            "layout": layout.key, "tailset_derived": bool(layout.tailset_derived),
            "frame_range": result.frame_range, "records": records,
            "dominant_track": None, "tracking_available": False, "track_coverage": 0.0,
            "n_frames_clip": vf.clip_frames,
            "n_frames_processed": n, "n_detections": n_det,
            "n_accepted": n_acc, "timings": timings,
        }
        save_cache(doc, cpath)

    return result


class _Cancelled:
    """Adapter: ``image_set.detect_frames`` wants ``cancel_event.is_set()``."""

    def __init__(self, event: Any | None) -> None:
        self._event = event

    def is_set(self) -> bool:
        return bool(self._event is not None and self._event.is_set())


def detect_clip_tracked(
    clip: str | Path,
    cfg: DetectConfig | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
    cancel_event: Any | None = None,
    *,
    frame_range: tuple[int, int] | None = None,
    out_dir: str | Path | None = None,
    use_cache: bool = True,
    model: Any | None = None,
) -> DetectResult:
    """The 2026-10-12 **tracker** path, kept for the archive and for A/B.

    Superseded by :func:`detect_clip` on 2026-10-20: its ByteTrack ``dominant_track`` filter
    selected a stationary handler over the dog on the `video004_right…seg01` clip (measured: f45's
    dog detection, box 0.70, was dropped as "not the dominant track").  Do not wire this into the
    app again without re-measuring that failure.

    Detect every frame of ``clip`` (or of ``frame_range``) and cache the result.

    ``progress_cb(done, total, eta_s)`` is called every frame; ``cancel_event`` is
    any object with ``is_set()`` (a ``threading.Event``) — a cancelled run returns
    a partial result and does **not** write the cache, so a cancelled run can never
    be mistaken for a complete one.

    ``frame_range`` is the half-open ``(start, stop)`` of PLAN §5's ``--subclip``.
    It is part of the cache key: a subclip and the full clip are different runs.
    """
    cfg = cfg or DetectConfig()
    clip = Path(clip)
    if not clip.is_file():
        raise FileNotFoundError(clip)
    if frame_range is not None:
        start, stop = int(frame_range[0]), int(frame_range[1])
        if stop <= start:
            raise ValueError(f"frame_range must be half-open (start < stop), got {frame_range}")

    t0 = time.perf_counter()
    clip_sha = video.sha256(clip)
    model_sha = video.sha256(Path(cfg.model_path))
    key = cache_key(clip_sha, model_sha, cfg, frame_range)
    khash = key_hash(key)
    cpath = tracks_path(clip.stem, out_dir)

    if use_cache:
        doc = load_cache(cpath)
        entries = doc.get("entries", {})
        if khash in entries:
            e = entries[khash]
            return DetectResult(
                clip_id=clip.stem, clip_path=str(clip), clip_sha=clip_sha, model_sha=model_sha,
                cache_key=key, cache_key_hash=khash, cache_path=str(cpath),
                frame_range=e.get("frame_range"), records=e["records"],
                dominant_track=e.get("dominant_track"),
                tracking_available=e.get("tracking_available", True),
                track_coverage=e.get("track_coverage", 0.0),
                n_tracked_frames=e.get("n_tracked_frames", 0),
                n_dominant_frames=e.get("n_dominant_frames", 0),
                n_dominant_with_kp=e.get("n_dominant_with_kp", 0),
                n_frames_clip=e.get("n_frames_clip", 0),
                n_frames_processed=e.get("n_frames_processed", len(e["records"])),
                n_detections=e.get("n_detections", 0), n_accepted=e.get("n_accepted", 0),
                timings={**e.get("timings", {}), "cache_load_s": time.perf_counter() - t0},
                from_cache=True, config=cfg.as_dict(),
            )

    if model is None:
        model = load_model(cfg)
    tracker_reset = reset_tracker(model)
    layout = verify_checkpoint(model)

    info = video.probe(clip, count_frames=False)
    total = (stop - start) if frame_range else info.n_frames
    t_inf = 0.0
    records: list[dict] = []
    cancelled = False

    gen = video.iter_frames(clip, start=start if frame_range else 0,
                            stop=stop if frame_range else None)
    try:
        for idx, frame in gen:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            src = clahe_lab_l(frame, cfg.clahe_clip, cfg.clahe_grid) if cfg.clahe else frame
            t1 = time.perf_counter()
            res = model.track(
                source=src, imgsz=int(cfg.imgsz), conf=float(cfg.conf),
                max_det=int(cfg.max_det), tracker=cfg.tracker, persist=True,
                device=cfg.device, verbose=False,
            )
            t_inf += time.perf_counter() - t1
            tailset, nose, box_conf, track_id = _extract(res[0], layout)
            records.append({
                "frame": idx, "tailset": tailset, "nose": nose,
                "box_conf": box_conf, "track_id": track_id, "accepted": False,
            })
            done = len(records)
            if progress_cb is not None:
                elapsed = time.perf_counter() - t0
                eta = (elapsed / done) * max(total - done, 0) if done else 0.0
                progress_cb(done, total, eta)
    finally:
        gen.close()

    dom = dominant_track(records)
    stats = track_stats(records, cfg.track_coverage_min, cfg.kp_conf)
    # `dominant_track` reports the RAW most-common id (data); `tracking_available`
    # says whether it was trusted enough to filter with.  See track_stats().
    dom_effective = stats["dominant_track"] if stats["tracking_available"] else None
    n_acc = apply_acceptance(records, cfg.kp_conf, dom_effective)
    total_s = time.perf_counter() - t0
    n = len(records)
    timings = {
        "total_s": total_s,
        "ms_per_frame": (total_s / n * 1000.0) if n else float("nan"),
        "inference_ms_per_frame": (t_inf / n * 1000.0) if n else float("nan"),
        "fps": (n / total_s) if (n and total_s) else float("nan"),
        # measured throughput applied to the whole clip (PLAN §8 V7)
        "projected_clip_s": (info.n_frames * total_s / n) if n else float("nan"),
    }

    result = DetectResult(
        clip_id=clip.stem, clip_path=str(clip), clip_sha=clip_sha, model_sha=model_sha,
        cache_key=key, cache_key_hash=khash, cache_path=str(cpath),
        frame_range=list(frame_range) if frame_range else None, records=records,
        dominant_track=dom, tracking_available=stats["tracking_available"],
        track_coverage=stats["track_coverage"], n_tracked_frames=stats["n_tracked_frames"],
        n_dominant_frames=stats["n_dominant_frames"], n_dominant_with_kp=stats["n_dominant_with_kp"],
        n_frames_clip=info.n_frames,
        n_frames_processed=n, n_detections=sum(1 for r in records if r["tailset"]),
        n_accepted=n_acc, timings={**timings, "tracker_reset": float(tracker_reset),
                                   "cancelled": float(cancelled)},
        from_cache=False, config=cfg.as_dict(),
    )

    if use_cache and not cancelled:
        doc = load_cache(cpath)
        doc.setdefault("version", CACHE_VERSION)
        doc.setdefault("clip_id", clip.stem)
        doc.setdefault("entries", {})
        doc["clip_sha"] = clip_sha
        doc["model_sha"] = model_sha
        doc["entries"][khash] = {
            "key": key, "frame_range": result.frame_range, "records": records,
            "dominant_track": dom, "tracking_available": stats["tracking_available"],
            "track_coverage": stats["track_coverage"], "n_tracked_frames": stats["n_tracked_frames"],
            "n_dominant_frames": stats["n_dominant_frames"],
            "n_dominant_with_kp": stats["n_dominant_with_kp"],
            "n_frames_clip": info.n_frames,
            "n_frames_processed": n, "n_detections": result.n_detections,
            "n_accepted": n_acc, "timings": timings,
        }
        save_cache(doc, cpath)

    return result
