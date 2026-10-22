"""M27 gate: pose layouts — which checkpoint families may run, and what their keypoints mean.

The gate that used to guard this (`detect.verify_checkpoint`) hard-failed anything that was not
``kpt_shape == (2, 3)``.  That protected the tailset index and blocked every other pose model, so
it now resolves a *named mapping* (``dog_diving/pose_layout.py``) instead: ``dog-2kpt`` keeps the
reference recipe's indices (nose 0, tailset 1, class ``dog``), ``coco-person-17`` runs a stock COCO
pose checkpoint with nose 0 and a **derived** tailset (hips midpoint, min hip confidence), and any
other shape is still refused with the actual shape in the message.

What this asserts
  1. ``dog-2kpt`` extraction is unchanged — same indices, same values as the old constants.  This
     is the no-regression half: everything measured in ``models/PROVENANCE.md`` must still hold.
  2. ``coco-person-17`` derives the tailset correctly, and **one confident hip cannot buy a point**:
     an undetected hip (conf 0) gives no tailset at all, and a marginal hip caps the derived
     confidence at the *minimum*, so ``apply_acceptance`` still rejects it.
  3. ``detect._extract`` and ``detect.verify_checkpoint`` honour the layout (fake result objects;
     no torch needed).
  4. An unmapped ``kpt_shape`` is refused, loudly, with the shape in the message.
  5. The installed ``models/yolo26s-pose.pt`` really is COCO person pose and resolves to
     ``coco-person-17`` (gated on the file existing; loads ultralytics once).
  6. The installed ``models/yolo26s-pose_dogs_clahe.pt`` — a **yolo26s** backbone fine-tuned on the
     same 2-keypoint dog labels — resolves to ``dog-2kpt``, not to the COCO layout.  Two checkpoints
     with ``yolo26s`` in the name but different keypoint semantics is exactly the case a filename
     could get wrong: the layout is decided by ``kpt_shape``, never by the file name.

Run:  <venv-python> -m pytest src/tests/test_m27_pose_layout.py -s
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import config as C  # noqa: E402
from dog_diving import detect as D  # noqa: E402
from dog_diving import pose_layout as PL  # noqa: E402

#: The checkpoint this gate was written against (``yolo26s-pose.pt``, stock COCO person pose).
COCO_CHECKPOINT = PROJECT_ROOT / "models" / "yolo26s-pose.pt"
COCO_CHECKPOINT_SHA = "a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646"

#: The s-scale **dog** fine-tune (yolo26s-pose backbone, the 2-kpt labels from
#: ``dog-pose-2kpt-clahe``).  Same backbone name as the COCO file above, different keypoints.
SS_DOG_CHECKPOINT = PROJECT_ROOT / "models" / "yolo26s-pose_dogs_clahe.pt"
SS_DOG_CHECKPOINT_SHA = "45252fc7ce305c336e9b05f05d3eb2cb6b169cdb9f5f2f98a236d13cb6a7f0c7"


class _Boxes:
    """Minimal stand-in for an ultralytics ``Boxes`` (needs ``len``, ``conf``, ``id``)."""

    def __init__(self, conf, ids) -> None:
        self.conf = np.array(conf, dtype=np.float32)
        self.id = np.array(ids, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)


def _coco_arrays(*, nose=(100.0, 50.0, 0.9), left_hip=(200.0, 300.0, 0.8),
                 right_hip=(300.0, 340.0, 0.6)):
    """A 17-keypoint detection as the model emits it: ``xy`` is (17, 2), ``conf`` is (17,)."""
    xy = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    for idx, row in ((PL.COCO_NOSE, nose), (PL.COCO_HIPS[0], left_hip), (PL.COCO_HIPS[1], right_hip)):
        xy[idx] = [row[0], row[1]]
        conf[idx] = row[2]
    return xy, conf


# --- 1. the model of record is unchanged -------------------------------------
def test_dog_layout_keeps_the_reference_recipe() -> None:
    lay = PL.DOG_2KPT
    assert lay.kpt_shape == (2, 3), lay.kpt_shape
    assert dict(lay.names) == {0: "dog"}, lay.names
    # the constants the rest of the tree (and models/PROVENANCE.md) were measured with
    assert lay.nose_index == C.NOSE_INDEX, (lay.nose_index, C.NOSE_INDEX)
    assert lay.tailset_index == C.TAILSET_INDEX, (lay.tailset_index, C.TAILSET_INDEX)
    assert lay.tailset_derived is False

    xy = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    conf = np.array([0.9, 0.8], dtype=np.float32)
    tailset, nose = lay.extract(xy, conf)
    assert tailset == pytest.approx([30.0, 40.0, 0.8]), tailset
    assert nose == pytest.approx([10.0, 20.0, 0.9]), nose
    print(f"dog-2kpt      : tailset={tailset} nose={nose} (unchanged)")


# --- 2. COCO person pose: derived tailset, conservative confidence ------------
def test_coco17_derives_tailset_from_hips() -> None:
    lay = PL.resolve((17, 3), {0: "person"})
    assert lay.key == "coco-person-17", lay.key
    assert dict(lay.names) == {0: "person"}, lay.names
    assert lay.tailset_derived is True
    assert lay.tailset_source == "mid(kpt[11],kpt[12])", lay.tailset_source

    xy, conf = _coco_arrays()
    tailset, nose = lay.extract(xy, conf)
    assert nose == pytest.approx([100.0, 50.0, 0.9]), nose
    # midpoint of the two hips, confidence = the LOWER of the two hips (0.6, not 0.8)
    assert tailset == pytest.approx([250.0, 320.0, 0.6]), tailset
    print(f"coco-person-17: tailset(derived)={tailset} nose={nose}")


def test_one_visible_hip_cannot_buy_a_point() -> None:
    lay = PL.COCO_PERSON_17

    xy, conf = _coco_arrays(right_hip=(300.0, 340.0, 0.0))       # COCO writes conf 0 = not seen
    tailset, _ = lay.extract(xy, conf)
    assert tailset is None, f"half-visible hips produced {tailset}"

    # marginal hip: the derived point exists but its confidence is the minimum, so the existing
    # acceptance rule rejects it instead of admitting a high-confidence torso point
    xy, conf = _coco_arrays(left_hip=(200.0, 300.0, 0.95), right_hip=(300.0, 340.0, 0.05))
    tailset, _ = lay.extract(xy, conf)
    assert tailset is not None and tailset[2] == pytest.approx(0.05), tailset
    records = [{"frame": 0, "tailset": tailset, "nose": None, "track_id": None,
                "accepted": False}]
    assert D.apply_acceptance(records, kp_conf=0.30, dominant=None) == 0, records
    print(f"min-hip rule  : hidden hip -> None; hip@0.05 -> conf {tailset[2]:.2f} (rejected)")


# --- 3. the wiring: verify_checkpoint + _extract honour the layout -----------
def test_verify_checkpoint_accepts_both_families_and_refuses_others() -> None:
    dog = SimpleNamespace(model=SimpleNamespace(kpt_shape=(2, 3), names={0: "dog"}))
    assert D.verify_checkpoint(dog).key == "dog-2kpt"

    coco = SimpleNamespace(model=SimpleNamespace(kpt_shape=(17, 3), names={0: "person"}))
    assert D.verify_checkpoint(coco).key == "coco-person-17"

    wide = SimpleNamespace(model=SimpleNamespace(kpt_shape=(24, 3), names={0: "dog"}))
    with pytest.raises(ValueError) as exc:
        D.verify_checkpoint(wide)
    msg = str(exc.value)
    assert "24" in msg and "coco-person-17" in msg and "dog-2kpt" in msg, msg

    with pytest.raises(ValueError):
        D.verify_checkpoint(SimpleNamespace())                 # not a loaded model at all
    print(f"verify        : dog-2kpt / coco-person-17 accepted; 24-kpt refused -> {msg[:60]}…")


def test_extract_uses_the_layout_not_the_index_constants() -> None:
    xy, conf = _coco_arrays()
    res = SimpleNamespace(
        keypoints=SimpleNamespace(data=np.stack([np.column_stack([xy, conf])])),  # (1,17,3)
        boxes=_Boxes([0.77], [4]),
    )

    # Without a layout the legacy default (dog-2kpt) would read row 1 — a person's left eye.
    tailset, nose, box_conf, track_id = D._extract(res, PL.COCO_PERSON_17)
    assert tailset is not None and tailset[:2] == pytest.approx([250.0, 320.0]), tailset
    assert nose[:2] == pytest.approx([100.0, 50.0]), nose
    assert box_conf == pytest.approx(0.77) and track_id == 4

    legacy_tailset, _, _, _ = D._extract(res)                  # dog default, unchanged
    assert legacy_tailset == [0.0, 0.0, 0.0], legacy_tailset   # row 1 is an unseen keypoint here
    print(f"_extract      : layout={PL.COCO_PERSON_17.key} tailset={tailset} "
          f"(dog default would give {legacy_tailset})")


# --- 4./5. the installed COCO checkpoint -------------------------------------
def test_coco_checkpoint_installed_and_mapped() -> None:
    if not COCO_CHECKPOINT.is_file():
        pytest.skip(f"not installed: {COCO_CHECKPOINT}")
    sha = hashlib.sha256(COCO_CHECKPOINT.read_bytes()).hexdigest()
    assert sha == COCO_CHECKPOINT_SHA, f"{COCO_CHECKPOINT.name} sha256 {sha} != pinned"

    from ultralytics import YOLO

    model = YOLO(str(COCO_CHECKPOINT))
    lay = D.verify_checkpoint(model)
    assert lay.key == "coco-person-17", lay.key
    assert list(model.model.kpt_shape) == [17, 3], model.model.kpt_shape
    print(f"checkpoint    : {COCO_CHECKPOINT.name} -> layout={lay.key} "
          f"names={dict(lay.names)} tailset={lay.tailset_source}")

    cfg = D.DetectConfig(model_path=COCO_CHECKPOINT)
    assert cfg.model_path == COCO_CHECKPOINT
    print(f"DetectConfig  : accepts the COCO checkpoint as model_path "
          f"(cache key already carries model_sha {sha[:12]}…)")


# --- 6. the installed s-scale dog fine-tune ----------------------------------
def test_sscale_dog_checkpoint_is_the_dog_layout() -> None:
    """A yolo26s backbone does not make it a different *layout* — its labels still do."""
    if not SS_DOG_CHECKPOINT.is_file():
        pytest.skip(f"not installed: {SS_DOG_CHECKPOINT}")
    sha = hashlib.sha256(SS_DOG_CHECKPOINT.read_bytes()).hexdigest()
    assert sha == SS_DOG_CHECKPOINT_SHA, f"{SS_DOG_CHECKPOINT.name} sha256 {sha} != pinned"

    from ultralytics import YOLO

    model = YOLO(str(SS_DOG_CHECKPOINT))
    lay = D.verify_checkpoint(model)
    assert lay.key == "dog-2kpt", f"{SS_DOG_CHECKPOINT.name} resolved to {lay.key}"
    assert lay.tailset_derived is False, lay.tailset_derived
    assert lay.tailset_index == C.TAILSET_INDEX, lay.tailset_index
    assert list(model.model.kpt_shape) == [2, 3], model.model.kpt_shape
    assert dict(model.model.names) == {0: "dog"}, dict(model.model.names)

    # the pair of yolo26s-named checkpoints must not be interchangeable
    if COCO_CHECKPOINT.is_file():
        coco = D.verify_checkpoint(YOLO(str(COCO_CHECKPOINT)))
        assert coco.key != lay.key, (coco.key, lay.key)
        assert coco.tailset_source != lay.tailset_source
    print(f"checkpoint    : {SS_DOG_CHECKPOINT.name} -> layout={lay.key} "
          f"names={dict(lay.names)} tailset={lay.tailset_source} (labelled, not derived)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
