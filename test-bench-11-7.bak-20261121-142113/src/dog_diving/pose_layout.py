"""Pose-checkpoint layouts — *which* keypoint is the tailset, per checkpoint family.

Why this module exists
----------------------
``detect.py`` used to hard-fail any checkpoint whose ``kpt_shape`` was not ``(2, 3)``.
That guard was right about the danger it was written for — pointing the pipeline at a
24-kpt (or COCO) checkpoint silently moves ``TAILSET_INDEX`` onto the wrong joint and
produces plausible-looking wrong distances — but it made "run a different pose model"
impossible, and it was really only checking a *number* where what it cared about was a
*meaning*.

A layout names that meaning explicitly: for one checkpoint family, which keypoint row is
the nose, which row is the tailset (or which two rows the tailset is derived from), and
what the model's own class names are.  Adding a checkpoint family means adding a layout
here, once, with the mapping written down — never re-pointing an index constant and hoping.

Layouts
-------
``dog-2kpt``
    The vendored/fine-tuned Dog Diving models (``models/yolo26n-pose_dogs_v5.pt`` and
    friends): 2 keypoints in the order ``[nose, tailset]``, single class ``dog``.  This is
    the model of record; its layout is the reference recipe's (``TAILSET_INDEX = 1``).

``coco-person-17``
    A stock Ultralytics COCO pose checkpoint (``yolo26s-pose.pt`` and the rest of that
    family): 17 COCO keypoints, single class ``person``.  It has **no tailset keypoint** —
    COCO-pose only ever annotated people — so the tailset is *derived* as the midpoint of
    the two hips (rows 11 and 12) and its confidence is the **minimum** of the two hip
    confidences.

    That derivation is an approximation, and it is stated rather than hidden:

    * Row 11/12 is a person's pelvis, not a dog's tail base.  On Dog Diving footage this
      checkpoint fires on the **handler** (or nothing), because COCO-pose has no dog class
      at all.  It is useful for the person-side geometry and as an A/B against the dog
      fine-tune; it is not a drop-in replacement for the tailset measurement.
    * ``min(conf_l, conf_r)`` means the existing acceptance rule (``tailset[2] >
      kp_conf``) still has to be cleared by **both** hips.  One confident hip cannot buy a
      point on its own.
    * A hip the model did not see (COCO-pose writes ``conf == 0`` for an undetected
      keypoint) makes the derived tailset ``None`` rather than a half-visible guess.

Every derived point is disclosed upward rather than per record: the layout key and a
``tailset_derived`` flag are written into the cache entry, into
``DetectResult.model_layout`` / ``.tailset_derived``, and into ``result.json``'s model
block, so an artifact can always say which mapping produced its numbers.  The record
schema itself is unchanged, because ``trajectory``, ``report`` and the cached
``tracks.json`` entries all read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: COCO-pose keypoint order (Ultralytics ``coco-pose.yaml``).
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_HIPS: tuple[int, int] = (COCO_KEYPOINTS.index("left_hip"), COCO_KEYPOINTS.index("right_hip"))
COCO_NOSE: int = COCO_KEYPOINTS.index("nose")


@dataclass(frozen=True)
class PoseLayout:
    """One checkpoint family's keypoint semantics.

    ``tailset_index`` and ``tailset_pair`` are mutually exclusive: a model either *labels*
    the tailset or the tailset is *derived* from two rows.
    """

    key: str
    kpt_shape: tuple[int, int]
    names: Mapping[int, str]
    nose_index: int = 0
    tailset_index: int | None = None
    tailset_pair: tuple[int, int] | None = None
    note: str = ""

    # --- identity ---------------------------------------------------------

    @property
    def tailset_derived(self) -> bool:
        return self.tailset_pair is not None

    @property
    def tailset_source(self) -> str:
        if self.tailset_pair is not None:
            a, b = self.tailset_pair
            return f"mid(kpt[{a}],kpt[{b}])"
        return f"kpt[{self.tailset_index}]"

    def describe(self) -> str:
        return (f"{self.key}: kpt_shape={list(self.kpt_shape)} names={dict(self.names)} "
                f"nose={self.nose_index} tailset={self.tailset_source}")

    # --- extraction -------------------------------------------------------
    @staticmethod
    def _row(xy: Sequence, conf: Sequence, idx: int) -> list[float] | None:
        """One keypoint row as ``[x, y, conf]``, or ``None`` if the row is not present."""
        if idx is None or idx < 0 or idx >= len(xy):
            return None
        return [float(xy[idx][0]), float(xy[idx][1]), float(conf[idx])]

    def nose(self, xy: Sequence, conf: Sequence) -> list[float] | None:
        return self._row(xy, conf, self.nose_index)

    def tailset(self, xy: Sequence, conf: Sequence) -> list[float] | None:
        """The tailset point for one detection, or ``None`` when it is not visible.

        For a derived layout both source keypoints must have a positive confidence; the
        reported confidence is their minimum (see the module docstring).
        """
        if self.tailset_pair is None:
            return self._row(xy, conf, int(self.tailset_index))
        a, b = self.tailset_pair
        pa, pb = self._row(xy, conf, a), self._row(xy, conf, b)
        if pa is None or pb is None:
            return None
        ca, cb = float(pa[2]), float(pb[2])
        if ca <= 0.0 or cb <= 0.0:
            return None
        return [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, min(ca, cb)]

    def extract(self, xy: Sequence, conf: Sequence) -> tuple[list[float] | None, list[float] | None]:
        """``(tailset, nose)`` — the pair every record and ``detect._extract`` speak."""
        return self.tailset(xy, conf), self.nose(xy, conf)


#: The model of record: 2 keypoints ``[nose, tailset]``, one class ``dog``.
DOG_2KPT = PoseLayout(
    key="dog-2kpt",
    kpt_shape=(2, 3),
    names={0: "dog"},
    nose_index=0,
    tailset_index=1,
    note="vendored Dog Diving fine-tune; tailset is a labelled keypoint (the reference recipe's "
         "TAILSET_INDEX = 1)",
)

#: Stock COCO pose (``yolo26s-pose.pt`` et al.): 17 keypoints, one class ``person``.
COCO_PERSON_17 = PoseLayout(
    key="coco-person-17",
    kpt_shape=(17, 3),
    names={0: "person"},
    nose_index=COCO_NOSE,
    tailset_index=None,
    tailset_pair=COCO_HIPS,
    note="stock COCO person pose — no dog class and no tailset keypoint; the tailset is DERIVED "
         "as the hips midpoint with min(hip confidence), so it tracks a person's pelvis",
)

LAYOUTS: tuple[PoseLayout, ...] = (DOG_2KPT, COCO_PERSON_17)

#: kpt_shape -> layout.  Shape alone decides; a name mismatch is reported, not obeyed, because
#: a fine-tune keeps the same index semantics with different class labels.
_BY_SHAPE: dict[tuple[int, int], PoseLayout] = {tuple(l.kpt_shape): l for l in LAYOUTS}




def resolve(kpt_shape: Iterable[Any], names: Mapping[Any, Any] | None = None) -> PoseLayout:
    """Pick the layout for a checkpoint's ``kpt_shape`` (its ``names`` are carried, not judged).

    Raises ``ValueError`` with the actual shape in the message: an unmapped checkpoint is a
    configuration error, never something to warn about and continue past — running it would
    apply this pipeline's index constants to keypoints that mean something else.
    """
    shape = tuple(int(v) for v in kpt_shape or ())
    layout = _BY_SHAPE.get(shape[:2])
    if layout is None:
        raise ValueError(
            f"unsupported checkpoint: kpt_shape={list(shape)} — known layouts are "
            f"{', '.join(f'{l.key} {list(l.kpt_shape)}' for l in LAYOUTS)}. "
            f"Add a PoseLayout in dog_diving/pose_layout.py for a new keypoint family rather than "
            f"re-pointing a keypoint index."
        )
    out = layout
    if names:
        out = PoseLayout(
            key=layout.key, kpt_shape=layout.kpt_shape,
            names={int(k): str(v) for k, v in dict(names).items()},
            nose_index=layout.nose_index, tailset_index=layout.tailset_index,
            tailset_pair=layout.tailset_pair, note=layout.note,
        )
    return out


def layout_for_model(model: Any) -> PoseLayout:
    """Resolve the layout of an already-loaded Ultralytics model (its inner ``.model``)."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError("not a loaded ultralytics model: no .model attribute")
    kpt_shape = tuple(int(v) for v in (getattr(inner, "kpt_shape", ()) or ()))
    names = dict(getattr(inner, "names", {}) or {})
    return resolve(kpt_shape, names)
