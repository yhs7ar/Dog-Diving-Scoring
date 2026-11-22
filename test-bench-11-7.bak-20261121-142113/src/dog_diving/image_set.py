"""Frame-set input: run the same tailset detection + trajectory over still images.

A CVAT export (``images/Test/frame_0000NN.png`` + ``labels/Test/frame_0000NN.txt``) has no
container and no fps, so it cannot go through ``cv2.VideoCapture``.  This module gives it the two
things ``detect`` and ``trajectory`` need:

* a frame sequence with real frame numbers (parsed from the filenames, so a gap in the export is a
  gap in the trajectory, not a silent splice), and
* records in exactly the shape ``detect.detect_clip`` produces, so ``trajectory.build_trajectory``
  consumes them unchanged.

Ground truth, when labels are present, is attached to each record under ``gt`` — the evaluation
script uses it; the trajectory does not.

What counts as a source
-----------------------
``FrameSet.open`` takes one path *or a list*: a folder, a ``.zip`` export, loose image files,
and/or ``.txt`` label files, in any mix — i.e. whatever one file dialog can hand over.  Labels
are optional and are matched per image by this order: an explicitly selected ``.txt`` with the
same stem, then a ``.txt`` beside the image, then a ``labels`` folder near the images, then the
CVAT ``images/`` <-> ``labels/`` mirror.  The ``sources`` and ``label_files`` fields record what
was selected and what matched, so the GUI can show it rather than guess.
"""
from __future__ import annotations

import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config as C  # noqa: F401  (kept: callers use dog_diving.image_set.C)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_EXTS = {".txt"}
_NUM = re.compile(r"(\d+)$")


def _as_paths(src: Any) -> list[Path]:
    """A single path or a multi-select list -> one list of distinct paths."""
    if isinstance(src, (str, Path)):
        return [Path(src)]
    out: list[Path] = []
    for s in src:
        p = Path(s)
        if p not in out:
            out.append(p)
    return out


def _num_of(f: Path) -> int:
    m = _NUM.search(f.stem)
    return int(m.group(1)) if m else -1


def _dedupe(files: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for f in files:
        k = str(f.resolve()).lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    return out


def _common_root(imgs: list[Path], dirs: list[Path]) -> Path:
    """The one directory a multi-source selection reports itself under (clip id, out/)."""
    if len(dirs) == 1 and all(dirs[0] in f.parents for f in imgs):
        return dirs[0]
    try:
        cp = Path(os.path.commonpath([str(f.parent) for f in imgs]))
    except ValueError:                       # different drives
        return imgs[0].parent
    return cp if len(cp.parts) > 1 else imgs[0].parent



def _nearby_label_dirs(frames: list[Path]) -> list[Path]:
    """``labels`` folders these images could plausibly belong to, nearest first."""
    out: list[Path] = []
    for parent in dict.fromkeys(f.parent for f in frames):
        cands = [parent / "labels", parent.parent / "labels"]
        try:                           # e.g. labels/, labels_pose/ next to the images dir
            cands.extend(d for d in sorted(parent.parent.iterdir())
                         if d.is_dir() and d.name.lower().startswith("label"))
        except OSError:
            pass
        for c in cands:
            if c not in out:
                out.append(c)
    return out


def _label_candidates(f: Path, label_dirs: list[Path]) -> list[Path]:
    """Where one image's .txt may live, in the order we trust it (after an explicit pick)."""
    cands = [f.with_suffix(".txt")]                        # same folder, same stem
    for d in label_dirs:                                   # a labels/ folder nearby
        cands.append(d / (f.stem + ".txt"))
        cands.append(d / f.parent.name / (f.stem + ".txt"))
    parts = list(f.parts)                                  # CVAT: images/ -> labels/
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].lower() == "images":
            cands.append(Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
            break
    return cands


def _first_label(f: Path, by_stem: dict[str, Path], label_dirs: list[Path]) -> Path | None:
    """The label this image resolves to, or ``None`` (labels stay optional).

    Order, as settled with the operator: an explicitly selected ``.txt`` with the same stem,
    then a ``.txt`` beside the image, then a nearby ``labels`` folder, then the CVAT
    ``images/`` <-> ``labels/`` mirror.
    """
    cands = ([by_stem[f.stem]] if f.stem in by_stem else []) + _label_candidates(f, label_dirs)
    for c in cands:
        if c.is_file():
            return c
    return None


# --------------------------------------------------------------------------- frame set
@dataclass
class FrameSet:
    root: Path                     # directory the images are reported under
    frames: list[Path]
    numbers: list[int]             # frame number per image, ascending
    labels: dict[int, dict] = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)          # exactly what was selected
    duplicates: list[Path] = field(default_factory=list)       # same frame number, dropped
    label_files: dict[int, Path] = field(default_factory=dict)  # matched .txt per frame
    _size: tuple[int, int] | None = None

    # -- construction -----------------------------------------------------------------
    @classmethod
    def open(cls, src: str | Path | list[str | Path]) -> "FrameSet":
        """Build a frame set from anything one dialog can hand over.

        ``src`` is a path **or a list of paths** — a directory (CVAT export root or its
        ``images/`` folder), a .zip export, loose image files, and/or .txt label files, in
        any mix.  Directories and zips contribute every image under them; loose images are
        used as given; .txt files are *explicit* labels, matched by filename stem before any
        directory rule is tried (see :meth:`_match_labels`).
        """
        paths = _as_paths(src)
        if not paths:
            raise ValueError("nothing selected")

        dirs: list[Path] = []
        zips: list[Path] = []
        images: list[Path] = []
        explicit: list[Path] = []
        missing: list[Path] = []
        for p in paths:
            if p.is_dir():
                dirs.append(p)
            elif p.is_file() and p.suffix.lower() == ".zip":
                zips.append(p)
            elif p.is_file() and p.suffix.lower() in IMG_EXTS:
                images.append(p)
            elif p.is_file() and p.suffix.lower() in LABEL_EXTS:
                explicit.append(p)
            else:
                missing.append(p)
        if missing:
            raise FileNotFoundError("missing: " + ", ".join(str(m) for m in missing))

        zpath = zips[0] if len(zips) == 1 else None
        for z in zips:
            dest = Path(tempfile.mkdtemp(prefix=f"dd_frames_{z.stem[:24]}_"))
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            dirs.append(dest)

        imgs = list(images)
        for d in dirs:
            imgs.extend(f for f in sorted(d.rglob("*")) if f.suffix.lower() in IMG_EXTS)
        if not imgs:
            raise ValueError("no images in " + ", ".join(str(p) for p in paths))

        imgs = _dedupe(imgs)
        imgs.sort(key=lambda f: (_num_of(f), str(f)))
        nums = [_num_of(f) for f in imgs]
        if any(n < 0 for n in nums):                      # no numeric stem -> index order
            nums = list(range(len(imgs)))

        # Frame numbers are the pipeline's identity (arc fit, label match, table rows), so a
        # frame picked twice — e.g. a .zip plus stills from it — keeps one copy and the rest are
        # reported in ``duplicates``.  When copies disagree, the one with a label wins: the
        # selection order a dialog returns says nothing about which copy is the labelled one.
        by_stem: dict[str, Path] = {}
        for lf in explicit:
            by_stem.setdefault(lf.stem, lf)
        label_dirs = _nearby_label_dirs(imgs)

        groups: dict[int, list[tuple[Path, Path | None]]] = {}
        for f, n in zip(imgs, nums):
            groups.setdefault(n, []).append((f, _first_label(f, by_stem, label_dirs)))

        kept: list[Path] = []
        kept_nums: list[int] = []
        dropped: list[Path] = []
        matched: dict[int, Path] = {}
        for n, group in groups.items():
            labelled = [g for g in group if g[1] is not None]
            f, lab = labelled[0] if labelled else group[0]
            kept.append(f)
            kept_nums.append(n)
            if lab is not None:
                matched[n] = lab
            dropped.extend(g[0] for g in group if g[0] is not f)
        imgs, nums = kept, kept_nums

        fs = cls(root=_common_root(imgs, dirs), frames=imgs, numbers=nums,
                 sources=list(paths), duplicates=dropped)
        fs.label_files = matched
        fs.labels = fs._load_labels(fs.label_files)
        return fs

    def _load_labels(self, matched: dict[int, Path]) -> dict[int, dict]:
        gt: dict[int, dict] = {}
        for n, lf in matched.items():
            try:
                text = lf.read_text()
            except OSError:
                continue
            rows = []
            for line in text.strip().splitlines():
                if not line.strip():
                    continue
                v = [float(x) for x in line.split()]
                if len(v) < 5:
                    continue
                row = {"cls": int(v[0]), "box": v[1:5]}
                kps = []
                for k in range(5, len(v) - 2, 3):
                    kps.append({"x": v[k], "y": v[k + 1], "v": int(v[k + 2])})
                row["kps"] = kps
                rows.append(row)
            if rows:
                gt[n] = rows[0]                      # single dog per frame in this export
        return gt

    # -- reading ----------------------------------------------------------------------
    @property
    def size(self) -> tuple[int, int]:
        if self._size is None:
            im = cv2.imread(str(self.frames[0]))
            if im is None:
                raise ValueError(f"cannot read {self.frames[0]}")
            self._size = (im.shape[1], im.shape[0])
        return self._size

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def read(self, i: int) -> np.ndarray:
        im = cv2.imread(str(self.frames[i]))
        if im is None:
            raise ValueError(f"cannot read {self.frames[i]}")
        return im

    def gt_px(self, frame_no: int) -> dict | None:
        """Ground truth in pixels: tailset/nose + box, or ``None``."""
        row = self.labels.get(frame_no)
        if not row:
            return None
        w, h = self.size
        bx, by, bw, bh = row["box"]
        out = {"box": [bx * w, by * h, bw * w, bh * h],
               "kps": [{"x": k["x"] * w, "y": k["y"] * h, "v": k["v"]} for k in row["kps"]]}
        if len(out["kps"]) >= 2:
            out["nose"] = [out["kps"][0]["x"], out["kps"][0]["y"]]
            out["tailset"] = [out["kps"][1]["x"], out["kps"][1]["y"]]
        return out


# --------------------------------------------------------------------------- detection
@dataclass
class FrameDetect:
    '''The slice of ``detect.DetectResult`` that ``trajectory`` and the GUI use.'''

    clip_id: str
    records: list[dict]
    n_frames_clip: int
    n_frames_processed: int
    n_detections: int
    n_accepted: int
    source: str
    config: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"clip_id": self.clip_id, "source": self.source, "config": self.config,
                "n_frames_clip": self.n_frames_clip,
                "n_frames_processed": self.n_frames_processed,
                "n_detections": self.n_detections, "n_accepted": self.n_accepted,
                "records": self.records}


def detect_frames(
    fs: FrameSet,
    cfg: Any = None,
    model: Any = None,
    progress_cb=None,
    cancel_event: Any = None,
) -> FrameDetect:
    """Predict on every still; one record per frame in ``detect_clip``'s shape.

    No tracker: a still has no temporal context, so ``box_conf``/keypoints come straight from the
    model's single best box per frame (``max_det=1``, same as the video path).
    """
    from . import detect as D  # local: keeps this module importable without torch

    cfg = cfg or D.DetectConfig()
    if model is None:
        model = D.load_model(cfg)
    layout = D.verify_checkpoint(model)

    records: list[dict] = []
    n_det = n_acc = 0
    total = fs.n_frames
    for i in range(total):
        if cancel_event is not None and cancel_event.is_set():
            break
        frame = fs.read(i)
        if cfg.clahe:
            frame = D.clahe_lab_l(frame, cfg.clahe_clip, tuple(cfg.clahe_grid))
        res = model.predict(frame, imgsz=int(cfg.imgsz), conf=float(cfg.conf),
                            max_det=1, device=cfg.device, verbose=False)[0]
        frame_no = int(fs.numbers[i])
        rec: dict[str, Any] = {"frame": frame_no, "nose": None, "tailset": None,
                               "box_conf": None, "track_id": None, "accepted": False}
        gt = fs.gt_px(frame_no)
        if gt is not None:
            rec["gt"] = gt
        kps = getattr(res, "keypoints", None)
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes):
            n_det += 1
            rec["box_conf"] = float(boxes.conf[0])
            if kps is not None and kps.xy is not None and len(kps.xy):
                xy = kps.xy[0].detach().cpu().numpy()
                conf = kps.conf[0].detach().cpu().numpy() if kps.conf is not None else np.ones(len(xy))
                nose = layout.nose(xy, conf)
                tail = layout.tailset(xy, conf)
                if tail is not None:
                    rec["nose"], rec["tailset"] = nose, tail
                    rec["accepted"] = bool(tail[2] > float(cfg.kp_conf))
                    n_acc += int(rec["accepted"])
        records.append(rec)
        if progress_cb is not None:
            progress_cb(i + 1, total, 0.0)

    return FrameDetect(clip_id=fs.root.name, records=records, n_frames_clip=total,
                       n_frames_processed=len(records), n_detections=n_det, n_accepted=n_acc,
                       source=str(fs.root), config=cfg.as_dict())
