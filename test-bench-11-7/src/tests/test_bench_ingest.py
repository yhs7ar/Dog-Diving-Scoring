"""M2 — ingest + manifest/meta.

Covers the Flash-scoped M2 deliverable:
  * manifest schema round-trip + bad-schema refusal,
  * clip-id sanitization + extension dispatch,
  * a real MP4 ingest whose ``meta`` block equals the parent ``video.probe`` and
    whose ``source.sha256`` equals ``video.sha256`` (the same identity the parent's
    ``result.json`` records),
  * re-ingest preserving placed analysis (plane/ground-truth) rather than wiping it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# src/tests/test_*.py -> parents[1] == src, parents[2] == test-bench, parents[3] == repo
SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from dog_diving import video as V  # noqa: E402
from test_bench import ingest as ING  # noqa: E402
from test_bench import manifest as M  # noqa: E402
from test_bench import meta as MT  # noqa: E402

CLIP_CANDIDATES = (
    REPO_ROOT / "Our_dog_diving_videos" / "test-clip-video004_right.mp4",
    REPO_ROOT / "end-to-end-app" / "out" /
    "video004_right-00.00.36.812-00.00.38.328-seg01" / "annotated.mp4",
)


def _first_clip() -> Path | None:
    return next((c for c in CLIP_CANDIDATES if c.is_file()), None)


def test_sanitize_clip_id():
    assert M.sanitize_clip_id("a b/c@d") == "a_b_c_d"
    assert M.sanitize_clip_id("plain-id_1.mp4") == "plain-id_1.mp4"
    with pytest.raises(ValueError):
        M.sanitize_clip_id("@@@")


def test_clip_kind_dispatch():
    assert MT.clip_kind("x.mp4") == "mp4"
    assert MT.clip_kind("x.MOV") == "mp4"
    assert MT.clip_kind("x.svo2") == "svo2"
    with pytest.raises(ValueError):
        MT.clip_kind("x.txt")


def test_svo2_probe_without_sdk_is_a_clean_error():
    # On an SDK-less desktop this must raise RuntimeError, not ImportError/traceback.
    with pytest.raises(RuntimeError):
        MT.probe_clip("fake.svo2")


def test_manifest_round_trip(tmp_path: Path):
    man = M.new_manifest("c1", source={"sha256": "abc", "kind": "mp4"}, meta={"width": 960})
    d = tmp_path / "c1"
    d.mkdir()
    p = M.write_manifest(man, d)
    assert p.is_file()
    back = M.read_manifest(d)
    assert back == man
    assert back["schema"] == M.SCHEMA


def test_write_manifest_refuses_bad_schema(tmp_path: Path):
    with pytest.raises(ValueError):
        M.write_manifest({"schema": "wrong/0"}, tmp_path)
    (tmp_path / "manifest.json").write_text('{"schema": "wrong/0"}', encoding="utf-8")
    with pytest.raises(ValueError):
        M.read_manifest(tmp_path)


def test_ingest_mp4_matches_parent_probe(tmp_path: Path):
    clip = _first_clip()
    if clip is None:
        pytest.skip("no local clip available")
    root = tmp_path / "analyzed-clips"
    man = ING.ingest_clip(clip, root)
    cid = man["clip_id"]
    d = root / cid

    # clip is NOT copied — the manifest maps to the original path
    assert not (d / f"clip{clip.suffix.lower()}").exists()
    assert (d / "manifest.json").is_file()
    assert (d / "meta.json").is_file()
    assert man["source"]["original_path"] == str(clip)
    assert "clip_file" not in man["source"]
    assert M.clip_video_path(man, d) == clip

    # meta block equals the parent's own probe
    src = V.probe(clip)
    m = man["meta"]
    assert m["width"] == src.width and m["height"] == src.height
    assert abs(m["fps"] - src.fps) < 1e-6
    assert m["n_frames"] == src.n_frames
    assert m["codec"] == src.codec
    assert m["vfr"] == src.vfr
    assert m["declared_vs_decoded"][0] == src.n_frames_declared

    # source identity matches the parent's sha256
    assert man["source"]["sha256"] == V.sha256(clip)
    assert man["source"]["kind"] == "mp4"

    # fresh (pre-analysis) state
    assert man["plane"]["present"] is False
    assert man["detections"] == {}
    assert man["results"] == []

    # persisted == returned
    assert M.read_manifest(d) == man


def test_reingest_preserves_analysis(tmp_path: Path):
    clip = _first_clip()
    if clip is None:
        pytest.skip("no local clip available")
    root = tmp_path / "analyzed-clips"
    man = ING.ingest_clip(clip, root)
    d = root / man["clip_id"]
    # simulate a prepared clip
    man["plane"] = {"path": "plane.json", "setup_id": "s1", "present": True}
    man["ground_truth"] = {"path": "ground_truth.json", "judge_present": True, "gt_in": 98.0}
    M.write_manifest(man, d)

    man2 = ING.ingest_clip(clip, root)
    assert man2["plane"]["setup_id"] == "s1"
    assert man2["ground_truth"]["gt_in"] == 98.0
    # but the clip + metadata were refreshed
    assert man2["source"]["sha256"] == V.sha256(clip)


def test_clip_video_path_prefers_original_and_falls_back(tmp_path: Path):
    cdir = tmp_path / "c1"
    cdir.mkdir()
    orig = tmp_path / "orig.mp4"
    orig.write_bytes(b"v")
    man = M.new_manifest("c1", source={"original_path": str(orig),
                                       "sha256": "0" * 64, "kind": "mp4"}, meta={})
    # original resolves -> returned, even with a stale clip_file present
    assert M.clip_video_path(man, cdir) == orig
    # original missing -> falls back to the in-repo copy
    orig.unlink()
    copy = cdir / "clip.mp4"
    copy.write_bytes(b"v")
    man["source"]["clip_file"] = "clip.mp4"
    assert M.clip_video_path(man, cdir) == copy
    # neither -> None
    copy.unlink()
    assert M.clip_video_path(man, cdir) is None


def test_migrate_drop_copied_clips(tmp_path: Path):
    root = tmp_path / "analyzed-clips"
    for cid, orig_exists in (("a", True), ("b", False)):
        cdir = root / cid
        cdir.mkdir(parents=True)
        orig = tmp_path / f"{cid}.mp4"
        if orig_exists:
            orig.write_bytes(b"v")
        M.write_manifest(M.new_manifest(
            cid,
            source={"original_path": str(orig), "sha256": "0" * 64, "kind": "mp4",
                    "clip_file": "clip.mp4"},
            meta={}), cdir)
        (cdir / "clip.mp4").write_bytes(b"v")

    report = M.migrate_drop_copied_clips(root)

    # a: original exists -> copy removed + clip_file dropped
    assert not (root / "a" / "clip.mp4").exists()
    assert "clip_file" not in M.read_manifest(root / "a")["source"]
    # b: original missing -> copy kept + clip_file retained for the fallback
    assert (root / "b" / "clip.mp4").exists()
    assert "clip_file" in M.read_manifest(root / "b")["source"]
    assert len(report) == 2

