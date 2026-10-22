"""Headless-capable still-frame calibration tool (M2 deliverable, PLAN §5).

Loads one frame, walks the operator through the PLAN §0 D1/D5 click sequence —
dock edge (2 pts) -> waterline (2 pts) -> foot marks 0, 1, 15, 16 [30, 31] — and
writes the PLAN §5 calibration JSON.

Every click is a **seed**, then refined sub-pixel by ``calibration.refine_*``
(PLAN §5 "keeps calibration reproducible without demanding pixel-perfect
clicking").  The click sequence and snapping are pure logic in
``CalibrationSession``; the cv2 window is a thin shell over it, so the same code
path is drivable from a script or a test via ``--seed``.

Usage
-----
Interactive::

    python src/calibrate_tool.py --clip data/clips/18-07.mp4 --frame-index 120 \\
        --setup-id poolA --out data/calib/poolA.json

Headless (scripted / tests) — ``--seed`` is ``x,y;x,y;...`` in the same order the
prompts ask for::

    python src/calibrate_tool.py --frame f.png --setup-id t \\
        --seed "410,470;415,680;0,700;1920,710;520,700;530,700;900,700;920,700" \\
        --skip-optional --out /tmp/t.json --print-gates
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dog_diving import calibration as C  # noqa: E402
from dog_diving.config import MODEL_PATH  # noqa: E402

# PLAN §0 D1: 0-1 ft and 15-16 ft are required, 30-31 ft is skippable if out of frame.
REQUIRED_MARKS: list[tuple[int, int]] = [(0, 0), (1, 0), (15, 0), (16, 0)]
OPTIONAL_MARKS: list[tuple[int, int]] = [(30, 0), (31, 0)]


def mark_label(ft: int, inch: int) -> str:
    return f"{ft}'{inch:02d}\""


@dataclass
class CalibrationSession:
    """Ordered click collection. No cv2 calls — drivable headlessly."""

    frame: np.ndarray
    setup_id: str
    model_sha: str = ""
    notes: str = ""
    snap: bool = True
    skip_optional: bool = False
    dock_pts: list[tuple[float, float]] = field(default_factory=list)
    water_pts: list[tuple[float, float]] = field(default_factory=list)
    marks: list[dict] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    # --- prompt state machine ------------------------------------------------
    @property
    def _mark_prompts(self) -> list[tuple[int, int]]:
        return REQUIRED_MARKS + ([] if self.skip_optional else OPTIONAL_MARKS)

    @property
    def done(self) -> bool:
        return len(self.marks) >= len(self._mark_prompts)

    @property
    def prompt(self) -> str:
        if len(self.dock_pts) < 2:
            return f"click DOCK EDGE point {len(self.dock_pts) + 1}/2"
        if len(self.water_pts) < 2:
            return f"click WATERLINE point {len(self.water_pts) + 1}/2"
        if self.done:
            return "complete — press 's' to save, 'g' to toggle the 1-ft grid, 'z' undo"
        ft, inch = self._mark_prompts[len(self.marks)]
        return f"click foot mark {mark_label(ft, inch)}  ({len(self.marks) + 1}/{len(self._mark_prompts)})"

    # --- input ---------------------------------------------------------------
    def click(self, x: float, y: float) -> str:
        if len(self.dock_pts) < 2:
            self.dock_pts.append((float(x), float(y)))
        elif len(self.water_pts) < 2:
            self.water_pts.append((float(x), float(y)))
        elif not self.done:
            ft, inch = self._mark_prompts[len(self.marks)]
            self.marks.append({"label": mark_label(ft, inch), "ft": ft, "inch": inch,
                               "x_px": float(x), "y_px": float(y)})
        else:
            return "already complete"
        self.history.append("click")
        return self.prompt

    def undo(self) -> str:
        if self.marks:
            self.marks.pop()
        elif self.water_pts:
            self.water_pts.pop()
        elif self.dock_pts:
            self.dock_pts.pop()
        if self.history:
            self.history.pop()
        return self.prompt

    def skip(self) -> str:
        self.skip_optional = True
        if len(self.dock_pts) == 2 and len(self.water_pts) == 2 and self.done:
            self.history.append("skip")
        return self.prompt

    # --- assembly ------------------------------------------------------------
    def _refined_lines(self) -> tuple[list, list]:
        dock, water = list(self.dock_pts), list(self.water_pts)
        if self.snap and len(dock) == 2:
            dock = list(C.refine_dock_edge(self.frame, dock[0], dock[1]))
        if self.snap and len(water) == 2:
            water = list(C.refine_waterline(self.frame, water[0], water[1]))
        return dock, water

    def build(self) -> dict:
        if not self.done:
            raise ValueError(f"calibration incomplete — {self.prompt}")
        dock, water = self._refined_lines()
        marks, shifts = (C.snap_marks(self.frame, self.marks) if self.snap
                         else (list(self.marks), [0.0] * len(self.marks)))
        h, w = self.frame.shape[:2]
        calib = C.build(self.setup_id, (w, h), dock, water, marks,
                        model_sha=self.model_sha, notes=self.notes)

        # Surface, never hide, how well snapping actually worked. A click landing
        # outside the safe radius can latch onto the NEIGHBOURING tick instead of
        # doing nothing, which is a silent full-foot error — so an out-of-radius
        # shift and a failed snap are both reported here.
        radius = C.safe_mark_radius(self.marks, 12) if self.snap else 0
        unsnapped = [m["label"] for m in marks if not m.get("snapped", True)]
        out_of_radius = [m["label"] for m, s in zip(marks, shifts) if self.snap and abs(s) >= radius]
        warnings: list[str] = []
        if radius:
            warnings.append(f"snap radius {radius} px (bounded by tick spacing)")
        if unsnapped:
            warnings.append(f"no tick found for {unsnapped} — kept the raw click")
        if out_of_radius:
            warnings.append(
                f"click exceeded the snap radius on {out_of_radius} — the mark may have "
                "latched onto a neighbouring tick"
            )

        calib["click_shift_px"] = [round(float(s), 3) for s in shifts]
        calib["max_click_shift_px"] = round(max((abs(s) for s in shifts), default=0.0), 3)
        calib["snap_radius_px"] = radius
        calib["unsnapped_marks"] = unsnapped
        calib["snap_warnings"] = warnings
        return calib

    def gates(self) -> dict:
        return C.validate_gates(self.build())


# --- overlay -----------------------------------------------------------------
def draw_grid(frame: np.ndarray, calib: dict, *, upto_ft: int = 31,
              color=(0, 200, 255)) -> np.ndarray:
    """Project the 1-ft grid onto the frame — PLAN §4.3 gate G4's visual check."""
    out = frame.copy()
    water = calib["waterline"]
    y_top, y_bot = int(min(water[0][1], water[1][1])), int(max(water[0][1], water[1][1]))
    for pt in C.grid_points(upto_ft + 1):
        inches = pt["nominal_in"]
        x = C.inverse_ft(inches, calib)
        if not np.isfinite(x) or not (0 <= x < out.shape[1]):
            continue
        major = pt["ft"] % 5 == 0
        cv2.line(out, (int(x), y_top), (int(x), y_bot),
                 color if major else (0, 120, 160), 2 if major else 1)
        if major:
            cv2.putText(out, str(pt["ft"]), (int(x) + 3, y_top - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_overlay(frame: np.ndarray, session: CalibrationSession, *, show_grid: bool = False
                 ) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    dock, water = session._refined_lines()
    if len(dock) == 2:
        p1, p2 = dock
        cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
        for i, p in enumerate(session.dock_pts):
            _cand(out, p, "xx" if i else "oo", (0, 255, 0))
    if len(water) == 2:
        p1, p2 = water
        cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)
        for p in session.water_pts:
            _cand(out, p, "xx", (255, 255, 0))
    for m in session.marks:
        cv2.circle(out, (int(m["x_px"]), int(m["y_px"])), 5, (0, 0, 255), -1)
        cv2.putText(out, m["label"], (int(m["x_px"]) + 6, int(m["y_px"]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if show_grid and session.done:
        try:
            out = draw_grid(out, session.build())
        except Exception as exc:  # noqa: BLE001 - overlay must never crash the loop
            cv2.putText(out, f"grid unavailable: {exc}", (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(out, session.prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    if session.done:
        try:
            gates = session.gates()
            txt = "  ".join(f"{k}:{'ok' if v['pass'] else 'FAIL'}" for k, v in gates.items())
            cv2.putText(out, txt, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if C.gates_pass(gates) else (0, 0, 255), 2)
        except Exception:  # noqa: BLE001
            pass
    return out


def _cand(img, p, tag, colour) -> None:
    x, y = int(p[0]), int(p[1])
    cv2.drawMarker(img, (x, y), colour, cv2.MARKER_CROSS, 14, 2)
    cv2.putText(img, tag, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


# --- frame acquisition ------------------------------------------------------
def load_frame(path: Path, frame_index: int | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
        from dog_diving import video
        idx = 0 if frame_index is None else frame_index
        for i, frame in video.iter_frames(path, start=idx, stop=idx + 1):
            return frame
        raise RuntimeError(f"could not read frame {idx} of {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read image {path}")
    return img


# --- interactive shell ------------------------------------------------------
def run_gui(frame: np.ndarray, session: CalibrationSession, out_path: Path,
            window: str = "calibrate — click sequence (z undo, k skip 30-31, "
                          "g grid, s save, q quit)") -> int:
    state = {"grid": False, "saved": False}
    scale = 1.0
    h, w = frame.shape[:2]
    max_w, max_h = 1500, 850
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)

    def render():
        vis = draw_overlay(frame, session, show_grid=state["grid"])
        if scale != 1.0:
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
        cv2.imshow(window, vis)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            session.click(x / scale, y / scale)
            render()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    render()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("q"):
            break
        if key == ord("z"):
            session.undo()
            render()
        elif key == ord("k"):
            session.skip()
            render()
        elif key == ord("g"):
            state["grid"] = not state["grid"]
            render()
        elif key == ord("r"):
            session.dock_pts.clear()
            session.water_pts.clear()
            session.marks.clear()
            session.history.clear()
            render()
        elif key == ord("s"):
            if session.done:
                calib = session.build()
                C.save(calib, out_path)
                state["saved"] = True
                print(f"saved {out_path}")
                print(json.dumps(C.validate_gates(calib), indent=2, default=str))
                break
            print(f"cannot save yet: {session.prompt}")
    cv2.destroyAllWindows()
    return 0 if state["saved"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", type=Path, help="still image")
    src.add_argument("--clip", type=Path, help="video file (use --frame-index)")
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--setup-id", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model-sha", default=None,
                    help="defaults to the sha256 of models/best.pt (provenance)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--seed", default=None,
                    help="headless: 'x,y;x,y;...' in prompt order (no window)")
    ap.add_argument("--skip-optional", action="store_true")
    ap.add_argument("--no-snap", action="store_true", help="record raw clicks (for A/B)")
    ap.add_argument("--print-gates", action="store_true")
    args = ap.parse_args(argv)

    path = args.frame or args.clip
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 2
    frame = load_frame(path, args.frame_index if args.clip else None)
    out_path = args.out or C.setup_path(args.setup_id)

    model_sha = args.model_sha
    if model_sha is None:
        from dog_diving import video
        model_sha = video.sha256(MODEL_PATH) if Path(MODEL_PATH).is_file() else ""

    session = CalibrationSession(
        frame=frame, setup_id=args.setup_id, model_sha=model_sha,
        notes=args.notes, snap=not args.no_snap, skip_optional=args.skip_optional,
    )

    if args.seed:
        for pair in args.seed.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            x, y = (float(v) for v in pair.split(","))
            session.click(x, y)
        if not session.done:
            print(f"seed did not complete the sequence — {session.prompt}", file=sys.stderr)
            return 2
        calib = session.build()
        C.save(calib, out_path)
        print(f"frame      : {path.name} {frame.shape[1]}x{frame.shape[0]}")
        print(f"snap       : {not args.no_snap}")
        print(f"click shift: max {calib['max_click_shift_px']} px "
              f"(per-mark {calib['click_shift_px']})")
        print(f"saved      : {out_path}")
        if args.print_gates:
            gates = C.validate_gates(calib)
            for k, v in gates.items():
                print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}"
                      f"{' (warn)' if v['warn'] else ''}  {v['detail']}")
            print(f"  overall: {'PASS' if C.gates_pass(gates) else 'FAIL'}")
        return 0

    return run_gui(frame, session, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
