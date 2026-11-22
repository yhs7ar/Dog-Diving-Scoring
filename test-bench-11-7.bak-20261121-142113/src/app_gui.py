"""Test-bench GUI — six panes over the modules in ``test_bench`` (M7).

Ingest / Analyze / Ground truth / Run suite / Results / Compare.  Every pane
reads and writes the same ``analyzed-clips/`` schema the CLI uses, so the two
interfaces are interchangeable (ROADMAP §7, §8).

Launched by ``run_app.cmd`` / ``run_app.bash`` (which resolve the interpreter);
direct: ``python src/app_gui.py`` from the repo root.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_bench import ground_truth as gt_mod       # noqa: E402
from test_bench import ingest as ingest_mod         # noqa: E402
from test_bench import manifest as manifest_mod     # noqa: E402
from test_bench import suite as suite_mod           # noqa: E402
from test_bench import analyze as analyze_mod       # noqa: E402
from test_bench import accuracy as accuracy_mod     # noqa: E402
from test_bench.frame_view import ZOOM_STEPS, FrameView  # noqa: E402
from dog_diving import trajectory as trajectory_mod  # noqa: E402
from dog_diving.config import MODEL_PATH  # noqa: E402
from calibrate_tab import CalibrateTab  # noqa: E402

DEFAULT_ROOT = SRC.parent / "analyzed-clips"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_frame(clip_dir: Path, man: dict, frame_index: int):
    """The BGR frame at ``frame_index`` for a clip, or None."""
    clip = manifest_mod.clip_video_path(man, clip_dir)
    if clip is None:
        return None
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


class _FramePane(ttk.Frame):
    """A clip selector + frame scrubber + zoomable FrameView, shared by GT/Compare."""
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Label(bar, text="Clip:").pack(side="left")
        self.clip_var = tk.StringVar()
        self.clip_box = ttk.Combobox(bar, textvariable=self.clip_var, width=42, state="readonly")
        self.clip_box.pack(side="left", padx=4)
        self.clip_box.bind("<<ComboboxSelected>>", lambda e: self._on_clip())
        self.frame_lbl = ttk.Label(bar, text="")
        self.frame_lbl.pack(side="right")

        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(self, from_=0, to=1, variable=self.frame_var,
                                     command=lambda _v: self._on_frame())
        self.frame_scale.pack(fill="x", padx=6)

        zoom = ttk.Frame(self)
        zoom.pack(fill="x", padx=6)
        ttk.Button(zoom, text="Fit", command=lambda: self.view.fit()).pack(side="left")
        for z in ZOOM_STEPS:
            ttk.Button(zoom, text=f"{z:g}x", width=4,
                       command=lambda zz=z: self.view.set_zoom(zz)).pack(side="left", padx=1)
        self.cursor_var = tk.StringVar(value="x=      0.00  y=      0.00 px")
        ttk.Label(zoom, textvariable=self.cursor_var).pack(side="right")

        self.canvas = tk.Canvas(self, background="black")
        self.canvas.pack(fill="both", expand=True, padx=6, pady=4)
        self.view = FrameView(self.canvas, max_w=900, max_h=560)
        self.view.cursor_var = self.cursor_var
        self.view.bind_zoom()

        self.man = None
        self._n_frames = 0

    def refresh_clips(self):
        self.clip_box["values"] = self.app.list_clips()
        self.select_default_clip()

    def select_default_clip(self):
        """Load the first clip when the pane has none selected (startup or refresh).

        Panes that override ``refresh()`` must call this: without it the combobox
        stays empty and a processed clip's video/points/GT never render until the
        operator picks one by hand.
        """
        cid = self.clip_var.get()
        ids = self.app.list_clips()
        if ids and cid not in ids:
            self.clip_var.set(ids[0])
            self._on_clip()
        elif ids and (self.man is None or self.man.get("clip_id") != cid):
            self._on_clip()

    def _on_clip(self):
        cid = self.clip_var.get()
        if not cid:
            return
        self.man = manifest_mod.read_manifest(self.app.clip_dir(cid))
        self._n_frames = int(self.man.get("meta", {}).get("n_frames") or 0)
        self.frame_scale.configure(to=max(1, self._n_frames - 1))
        self.frame_var.set(0)
        self._on_frame()

    def _on_frame(self):
        if not self.man:
            return
        idx = int(self.frame_var.get())
        self.frame_lbl.config(text=f"frame {idx} / {self._n_frames - 1}")
        frame = _load_frame(self.app.clip_dir(self.clip_var.get()), self.man, idx)
        if frame is not None:
            self.view.show(frame, reset_view=(not self.view._shown))


class IngestPane(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._pending_paths: list[str] = []   # clips staged by the (multi-select) Browse dialog
        row = ttk.Frame(self)
        row.pack(fill="x", padx=8, pady=8)
        self.path_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.path_var, width=55).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self._browse).pack(side="left", padx=4)
        ttk.Button(row, text="Select", command=self._ingest).pack(side="left")
        idrow = ttk.Frame(self)
        idrow.pack(fill="x", padx=8)
        ttk.Label(idrow, text="Clip id (optional):").pack(side="left")
        self.id_var = tk.StringVar()
        ttk.Entry(idrow, textvariable=self.id_var, width=40).pack(side="left", padx=4)
        self.out = tk.Text(self, height=8, state="disabled")
        self.out.pack(fill="both", expand=True, padx=8, pady=8)

    def _log(self, msg):
        self.out.configure(state="normal")
        self.out.insert("end", msg + "\n")
        self.out.configure(state="disabled")

    def _browse(self):
        paths = filedialog.askopenfilenames(title="Clips",
            filetypes=[("video", "*.mp4 *.mov *.m4v *.avi *.svo2"), ("all", "*.*")])
        if paths:
            self._pending_paths = list(paths)
            self.path_var.set("; ".join(str(p) for p in paths))

    def _ingest(self):
        sources = list(self._pending_paths)
        if not sources:
            single = self.path_var.get().strip()
            sources = [single] if single else []
        if not sources:
            return
        # A single clip may take the optional id; a multi-select derives ids from file names.
        clip_id = None if len(sources) > 1 else (self.id_var.get().strip() or None)
        try:
            for src in sources:
                man = ingest_mod.ingest_clip(src, self.app.root_dir, clip_id=clip_id)
                m = man["meta"]
                self._log(f"selected {man['clip_id']}  ({m.get('width')}x{m.get('height')} "
                          f"@ {m.get('fps')} fps, {m.get('n_frames')} frames, {m.get('codec')})")
            self._pending_paths = []
            self.app.on_clips_changed()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("select clips", str(e))


class AnalyzePane(_FramePane):
    """Analyze + Review + Ground truth in one pane.

    The top is the parent's clip selector + frame scrubber + zoomable canvas; the
    Analyze controls (model / method / plane) sit below it, and the Review controls
    (Select Ground Truth / judge / save GT / live readout) sit above the canvas.
    """

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.pending = None          # (x_px, y_px) the operator clicked as GT
        self._select_gt_mode = False
        self._deselect_mode = True
        self.excluded: set[int] = set()
        self._dirty = False         # deselection changed but not yet saved
        self._traj = None            # loaded Trajectory for the current clip
        self._plane_doc = None
        self._analyze_results = []   # [(method_name, meas)] from the last Analyze

        # --- review controls (top) ---
        rtop = ttk.Frame(self)
        rtop.pack(fill="x", padx=6, pady=2)
        self.deselect_btn = ttk.Button(rtop, text="Deselect points: ON", command=self._toggle_deselect_mode)
        self.deselect_btn.pack(side="left", padx=2)
        self.select_gt_btn = ttk.Button(rtop, text="Select Ground Truth", command=self._toggle_gt_mode)
        self.select_gt_btn.pack(side="left", padx=2)
        ttk.Button(rtop, text="Include all", command=self._include_all).pack(side="left", padx=2)
        ttk.Button(rtop, text="Save Detected Points", command=self._save_detections).pack(side="left", padx=2)
        self.exclude_lbl = ttk.Label(rtop, text="")
        self.exclude_lbl.pack(side="left", padx=6)
        self.readout_var = tk.StringVar(value="score: —")
        ttk.Label(rtop, textvariable=self.readout_var).pack(side="right")

        # --- analyze controls (bottom) ---
        self.ctl = ttk.Frame(self)
        self.ctl.pack(fill="x", padx=8, pady=4)
        ttk.Label(self.ctl, text="Model:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(self.ctl, textvariable=self.model_var, width=30)
        self.model_box.pack(side="left", padx=4)

        mrow = ttk.Frame(self)
        mrow.pack(fill="x", padx=8, pady=2)
        ttk.Label(mrow, text="Method:").pack(side="left")
        self.method_vars = {
            "3D Arc": tk.BooleanVar(value=False),
            "End-to-End Plane": tk.BooleanVar(value=True),
        }
        for name, var in self.method_vars.items():
            ttk.Checkbutton(mrow, text=name, variable=var).pack(side="left", padx=5)
        ttk.Button(mrow, text="Analyze", command=self._analyze).pack(side="right")

        prow = ttk.Frame(self)
        prow.pack(fill="x", padx=8, pady=2)
        ttk.Label(prow, text="Plane:").pack(side="left")
        self.plane_var = tk.StringVar()
        self.plane_box = ttk.Combobox(prow, textvariable=self.plane_var, width=34, state="readonly")
        self.plane_box.pack(side="left", padx=4)
        ttk.Button(prow, text="Use plane", command=self._use_plane).pack(side="left", padx=2)
        self.plane_lbl = ttk.Label(prow, text="")
        self.plane_lbl.pack(side="left", padx=6)

        gtw = ttk.Frame(self)
        gtw.pack(fill="x", padx=8, pady=2)
        ttk.Button(gtw, text="Save Ground Truth", command=self._save_gt).pack(side="left", padx=2)
        ttk.Label(gtw, text="Judge:").pack(side="left", padx=(8, 0))
        self.judge_ft_var = tk.StringVar()
        ttk.Entry(gtw, textvariable=self.judge_ft_var, width=4).pack(side="left", padx=2)
        ttk.Label(gtw, text="ft").pack(side="left")
        self.judge_in_var = tk.StringVar()
        ttk.Entry(gtw, textvariable=self.judge_in_var, width=4).pack(side="left", padx=2)
        ttk.Label(gtw, text="in").pack(side="left")
        ttk.Button(gtw, text="Save Judge Score", command=self._save_judge).pack(side="left", padx=2)
        self.gt_lbl = ttk.Label(gtw, text="GT: none")
        self.gt_lbl.pack(side="right")

        self.status = ttk.Label(self, text="", foreground="#0369a1")
        self.status.pack(fill="x", padx=8, pady=2)
        self.out = tk.Text(self, height=6, state="disabled")
        self.out.pack(fill="x", padx=8, pady=2)

        self.view.on_click(self._on_click)
        self.view.on_marquee(self._on_marquee)   # armed by default (deselect mode ON)

    # --- clip / model / plane refresh -------------------------------------
    def refresh(self):
        self.clip_box["values"] = self.app.list_clips()
        models = sorted(p.name for p in (self.app.root_dir.parent / "models").glob("*.pt"))
        self.model_box["values"] = models
        if models and not self.model_var.get():
            # Default to the end-to-end app's model of record, not the alphabetical first.
            self.model_var.set(MODEL_PATH.name if MODEL_PATH.name in models else models[0])
        self.plane_box["values"] = manifest_mod.list_planes()
        # Overriding _FramePane.refresh_clips() dropped the auto-select, so an
        # already-processed clip never rendered its points/GT until picked by hand.
        self.select_default_clip()

    def _log(self, msg):
        self.out.configure(state="normal"); self.out.insert("end", msg + "\n"); self.out.configure(state="disabled")

    def _use_plane(self):
        cid = self.clip_var.get()
        setup_id = self.plane_var.get()
        if not cid or not setup_id:
            return
        cdir = self.app.clip_dir(cid)
        man = manifest_mod.read_manifest(cdir)
        man["plane"] = {"setup_id": setup_id, "present": True}
        manifest_mod.write_manifest(man, cdir)
        self.plane_lbl.config(text=f"plane: {setup_id}")
        self.app.on_clips_changed()
        self._reload_trajectory()

    def _on_clip(self):
        super()._on_clip()
        self._reload_trajectory()

    # --- review state loading ---------------------------------------------
    def _reload_trajectory(self):
        cid = self.clip_var.get()
        self.excluded = set()
        self._dirty = False
        self.pending = None
        self._traj = None
        self._plane_doc = None
        self._analyze_results = []
        self.gt_lbl.config(text="GT: none")
        self.exclude_lbl.config(text="")
        if not cid or not self.man:
            return
        cdir = self.app.clip_dir(cid)
        self.excluded = set(manifest_mod.read_excluded_frames(cdir))

        # Plane: resolve it and say which one was used.  A manifest's setup_id can
        # name a plane that was re-saved under a timestamped id; the resolver
        # falls back by camera tag, and that must never be silent (it is geometry).
        plane = manifest_mod.plane_resolution(self.man)
        self._plane_doc = plane["doc"]
        if plane["doc"] is None:
            self.plane_lbl.config(text=f"plane: none — {plane['error'] or 'not assigned'}")
        elif plane["source"] == "camera-auto":
            self.plane_lbl.config(text=f"plane: {plane['setup_id']} (auto-matched)")
        else:
            self.plane_lbl.config(text=f"plane: {plane['setup_id']}")

        # Ground truth: only a placed *pixel* can be drawn.  A record with null
        # x_px/y_px must leave `pending` None — (None, None) is truthy and used
        # to send a marker at null coordinates into the overlay list.
        gtp = cdir / "ground_truth.json"
        if gtp.is_file():
            try:
                gt = json.loads(gtp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                gt = {}
            x, y = gt.get("x_px"), gt.get("y_px")
            gt_in = gt.get("gt_in")
            judge = gt.get("judge_in")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                self.pending = (float(x), float(y))
                if gt.get("frame") is not None:
                    self.frame_var.set(int(gt["frame"]))
            txt = f"GT: {gt_in:.2f} in" if isinstance(gt_in, (int, float)) else "GT: not placed"
            if isinstance(judge, (int, float)):
                txt += f"  judge: {judge:.2f} in"
            self.gt_lbl.config(text=txt)

        tpath = cdir / "trajectory.json"
        if tpath.is_file():
            try:
                self._traj = _Trajectory(json.loads(tpath.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                self._traj = None
            if self._traj is not None and self._traj.points:
                self._n_frames = int(self.man.get("meta", {}).get("n_frames") or 0)
                self.frame_scale.configure(to=max(1, self._n_frames - 1))
                self.frame_var.set(int(self._traj.last_frame()))
        # Always redraw: the saved points and the GT pixel must appear without the
        # operator scrubbing a frame first.
        self._on_frame()
        self._sync_exclude_lbl()

    # --- frame / overlay ----------------------------------------------------
    def _on_frame(self):
        if not self.man:
            return
        super()._on_frame()
        self._redraw_review()

    def _redraw_review(self):
        overlays = []
        traj = self._traj
        if traj is not None:
            kept = [p for p in traj.points if int(p["frame"]) not in self.excluded]
            if len(kept) >= 2:
                overlays.append(("curve", [(p["x"], p["y"]) for p in kept], "yellow"))
            for p in traj.inliers:
                if int(p["frame"]) not in self.excluded:
                    overlays.append(("marker", (p["x"], p["y"], 1.0), "magenta"))
            for f in sorted(self.excluded):
                p = traj.point_at(int(f))
                if p is not None:
                    overlays.append(("marker", (p["x"], p["y"], 1.2), "gray"))
            if kept:
                overlays.append(("marker", (kept[-1]["x"], kept[-1]["y"], 1.5), "red"))
        if self.pending:
            overlays.append(("marker", (self.pending[0], self.pending[1], 2.0), "lime"))
        self.view.set_overlays(overlays)
        self._update_readout()

    def _update_readout(self):
        if self._traj is None or self._plane_doc is None:
            self.readout_var.set("score: —")
            return
        from dog_diving import measure as measure_mod
        parts = []
        for name, var in self.method_vars.items():
            if not var.get():
                continue
            try:
                if name == "3D Arc":
                    meas = measure_mod.trajectory_plane_distance(self._traj, self._plane_doc,
                                                                 excluded=self.excluded)
                else:
                    meas = measure_mod.plane_only_distance(self._traj, self._plane_doc,
                                                           excluded=self.excluded)
            except Exception:  # noqa: BLE001 — the 3-D method is known-broken
                meas = None
            v = meas.get("total_inches") if meas else None
            parts.append(f"{name}: {v:.2f} in" if v is not None else f"{name}: null")
        self.readout_var.set("score: " + "   ".join(parts))

    # --- click: deselect a point or set ground truth -----------------------
    def _on_click(self, ix, iy):
        if self._select_gt_mode:
            self.pending = (float(ix), float(iy))
            self._redraw_review()
            return
        if not self._deselect_mode:
            return
        if self._traj is None:
            return
        p, d = self._nearest_point(ix, iy)
        if p is None or d > 12.0:
            return
        f = int(p["frame"])
        if f in self.excluded:
            self.excluded.discard(f)
        else:
            self.excluded.add(f)
        self._dirty = True
        self._redraw_review()

    def _on_marquee(self, x0, y0, x1, y1):
        """Box-drag deselection: every detection inside the drawn rectangle is
        marked faulty.  The rectangle itself is transient (never persisted) — the
        point is to sweep out false detections that sit on top of one another."""
        if not self._deselect_mode or self._traj is None:
            return
        newly = 0
        for p in self._traj.points:
            if x0 <= p["x"] <= x1 and y0 <= p["y"] <= y1:
                self.excluded.add(int(p["frame"]))
                newly += 1
        if newly:
            self._dirty = True
            self._redraw_review()

    def _nearest_point(self, ix, iy):
        best, best_d = None, float("inf")
        for p in self._traj.points:
            d = (p["x"] - ix) ** 2 + (p["y"] - iy) ** 2
            if d < best_d:
                best, best_d = p, d
        return best, best_d ** 0.5

    def _save_detections(self):
        """Commit the draft deselection to the saved detections and report it.

        Deselecting (marquee / click) only marks a *draft*; this bakes the
        deselected frames into ``detections.json`` (``accepted=False``), rebuilds
        the trajectory, and tells the operator exactly how many tailset detections
        are now saved."""
        cid = self.clip_var.get()
        if not cid:
            return
        cdir = self.app.clip_dir(cid)
        if not (cdir / "detections.json").is_file():
            messagebox.showinfo("detections", "no detections to save (run Analyze first)")
            return
        try:
            traj = analyze_mod.deselect_detections(cdir, cid, self.excluded)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("detections", str(e))
            return
        self.man = manifest_mod.read_manifest(cdir)
        self._traj = _Trajectory(manifest_mod.read_json(cdir / "trajectory.json"))
        self._dirty = False
        kept, dropped = traj.n_points, len(self.excluded)
        if dropped:
            frames = ", ".join(str(f) for f in sorted(self.excluded))
            messagebox.showinfo("detections saved",
                f"Saved {kept} accepted tailset detection(s).\n"
                f"Deselected {dropped} (frames {frames}).")
        else:
            messagebox.showinfo("detections saved",
                f"Saved {kept} accepted tailset detection(s) (none deselected).")
        self._redraw_review()
        self._sync_exclude_lbl()

    def _sync_exclude_lbl(self):
        n = len(self.excluded)
        unsaved = " (unsaved)" if getattr(self, "_dirty", False) else ""
        self.exclude_lbl.config(
            text=f"deselected: {n}{unsaved}" if n else
            "drag a box to deselect faulty detections" + unsaved)

    def _sync_marquee(self):
        # marquee is armed only in deselect mode (GT placement uses click only)
        self.view.on_marquee(self._on_marquee if self._deselect_mode else None)

    def _toggle_deselect_mode(self):
        self._deselect_mode = not self._deselect_mode
        if self._deselect_mode:
            self._select_gt_mode = False
            self.select_gt_btn.config(text="Select Ground Truth")
        self.deselect_btn.config(text="Deselect points: ON" if self._deselect_mode
                                 else "Deselect points")
        self._sync_marquee()

    def _toggle_gt_mode(self):
        self._select_gt_mode = not self._select_gt_mode
        if self._select_gt_mode:
            self._deselect_mode = False
            self.deselect_btn.config(text="Deselect points")
        self.select_gt_btn.config(text="Select Ground Truth: ON" if self._select_gt_mode
                                  else "Select Ground Truth")
        self._sync_marquee()

    def _include_all(self):
        self.excluded = set()
        self._dirty = True
        self._redraw_review()

    def _save_gt(self):
        cid = self.clip_var.get()
        if not cid or not self.pending:
            messagebox.showinfo("ground truth", "click 'Select Ground Truth' and place a point first")
            return
        cdir = self.app.clip_dir(cid)
        plane = manifest_mod.clip_plane(manifest_mod.read_manifest(cdir))
        if plane is None:
            messagebox.showerror("ground truth", "assign a plane first (Use plane)")
            return
        # preserve any judge score already saved on this clip
        judge = None
        gtp = cdir / "ground_truth.json"
        if gtp.is_file():
            try:
                judge = json.loads(gtp.read_text(encoding="utf-8")).get("judge_in")
            except Exception:  # noqa: BLE001
                judge = None
        try:
            gt = gt_mod.set_ground_truth(plane, frame=int(self.frame_var.get()),
                                         x_px=self.pending[0], y_px=self.pending[1],
                                         judge_in=judge, updated=_now())
        except ValueError as e:
            messagebox.showerror("ground truth", str(e))
            return
        gt_mod.write_ground_truth(gt, cdir)
        man = manifest_mod.read_manifest(cdir)
        man["ground_truth"] = {"path": "ground_truth.json", "judge_present": gt["judge_present"],
                               "gt_in": gt["gt_in"]}
        manifest_mod.write_manifest(man, cdir)
        self.gt_lbl.config(text=f"GT: {gt['gt_in']:.2f} in"
                                + (f"  judge: {judge:.2f} in" if judge is not None else ""))
        self.app.on_clips_changed()

    def _save_judge(self):
        cid = self.clip_var.get()
        if not cid:
            return
        cdir = self.app.clip_dir(cid)
        gtp = cdir / "ground_truth.json"
        if not gtp.is_file():
            messagebox.showinfo("judge", "save the Ground Truth point first")
            return
        ft_txt = self.judge_ft_var.get().strip()
        in_txt = self.judge_in_var.get().strip()
        if not ft_txt and not in_txt:
            messagebox.showinfo("judge", "enter a judge distance (feet and/or inches)")
            return
        try:
            feet = float(ft_txt) if ft_txt else 0.0
            inches = float(in_txt) if in_txt else 0.0
        except ValueError:
            messagebox.showerror("judge", "judge distance must be numbers (feet and inches)")
            return
        if feet < 0 or inches < 0:
            messagebox.showerror("judge", "judge distance cannot be negative")
            return
        judge = feet * 12.0 + inches
        gt = json.loads(gtp.read_text(encoding="utf-8"))
        gt = gt_mod.set_judge(gt, judge, updated=_now())
        gt_mod.write_ground_truth(gt, cdir)
        man = manifest_mod.read_manifest(cdir)
        man["ground_truth"] = {"path": "ground_truth.json", "judge_present": gt["judge_present"],
                               "gt_in": gt.get("gt_in")}
        manifest_mod.write_manifest(man, cdir)
        gt_in = gt.get("gt_in")
        self.gt_lbl.config(text=f"GT: {gt_in:.2f} in  judge: {judge:.2f} in" if gt_in is not None
                            else f"judge: {judge:.2f} in")
        self.app.on_clips_changed()

    # --- analyze ------------------------------------------------------------
    def _analyze(self):
        cid = self.clip_var.get()
        if not cid or getattr(self, "_busy", False):
            return
        cdir = self.app.clip_dir(cid)
        man = manifest_mod.read_manifest(cdir)
        if not man.get("plane", {}).get("present"):
            messagebox.showerror("analyze", "Please select and 'Use plane' first.")
            return

        from dog_diving import detect as detect_mod
        from dog_diving import measure as measure_mod
        if man.get("source", {}).get("kind") == "svo2":
            messagebox.showerror("analyze", "SVO2 analyze needs the ZED X SDK (not wired)")
            return
        enabled = {name for name, var in self.method_vars.items() if var.get()}
        if not enabled:
            messagebox.showwarning("analyze", "Please select at least one method.")
            return
        model = self.app.root_dir.parent / "models" / self.model_var.get()
        cfg = detect_mod.DetectConfig(model_path=model)
        clip = manifest_mod.clip_video_path(man, cdir)
        if clip is None:
            src = man.get("source", {})
            messagebox.showerror(
                "analyze", "clip file not found — looked for "
                f"{src.get('filename')!r} in {manifest_mod.SOURCE_CLIPS_DIR} "
                f"(ingest path: {src.get('original_path')!r})")
            return
        self._busy = True
        self.status.config(text="analyzing…")

        def _after(cb):
            try:
                self.app.root.after(0, cb)
            except tk.TclError:
                pass

        def prog(done, total, eta):
            _after(lambda d=done, t=total, e=eta:
                   self.status.config(text=f"detecting {d}/{t}  (eta {e:.0f}s)"))

        def worker():
            try:
                det = detect_mod.detect_clip(clip, cfg, progress_cb=prog)
                manifest_mod.write_json(cdir / "detections.json", det.records)
                w, h = int(man["meta"]["width"]), int(man["meta"]["height"])
                traj = trajectory_mod.build_trajectory(det.records, clip_id=cid, frame_size=(w, h))
                manifest_mod.write_json(cdir / "trajectory.json", traj.as_dict())
                man["detections"] = {"path": "detections.json", "model_sha": det.model_sha,
                                     "n_detections": det.n_detections, "n_accepted": det.n_accepted}
                man["trajectory"] = {"path": "trajectory.json", "n_points": traj.n_points}
                plane = manifest_mod.clip_plane(man)
                results = []
                if plane is not None:
                    man["plane"]["present"] = True
                    for name in sorted(enabled):
                        if name == "3D Arc":
                            meas = measure_mod.trajectory_plane_distance(traj, plane)
                        elif name == "End-to-End Plane":
                            meas = measure_mod.plane_only_distance(traj, plane)
                        else:
                            meas = None
                        results.append((name, meas))
                manifest_mod.write_manifest(man, cdir)
                _after(lambda: self._analyze_done(cid, det, results))
            except Exception as e:  # noqa: BLE001
                _after(lambda err=str(e): self._analyze_fail(err))

        threading.Thread(target=worker, daemon=True).start()

    def _analyze_done(self, cid, det, results):
        self._busy = False
        self.status.config(text="done")
        if not results:
            self._log(f"{cid}: {det.n_detections} detections, {det.n_accepted} accepted "
                      f"(no plane — distance null)")
        else:
            for name, meas in results:
                if meas and meas.get("total_inches") is not None:
                    self._log(f"{cid} [{name}]: {meas['total_inches']:.2f} in  "
                              f"({meas['display']}) @ frame {meas['frame']}")
                else:
                    self._log(f"{cid} [{name}]: distance null")
        self._analyze_results = results
        self.app.on_clips_changed()
        self._reload_trajectory()

    def _analyze_fail(self, err):
        self._busy = False
        self.status.config(text="failed")
        messagebox.showerror("analyze", err)


class _Trajectory:
    """Lightweight wrapper around a trajectory.json dict for the review pane."""

    def __init__(self, d):
        self._d = d
        self.points = d.get("points", [])
        self.inliers = d.get("inliers", [])
        self.coeffs = d.get("coeffs")
        self.airborne_frame_range = d.get("airborne_frame_range")

    def point_at(self, frame):
        for p in self.points:
            if int(p["frame"]) == int(frame):
                return p
        return None

    def last_frame(self):
        return max((int(p["frame"]) for p in self.points), default=0)



class CheckGroup(ttk.Frame):
    """A collapsible checkbox filter — the usual expanding filter-GUI menu.

    ``options`` is a list of ``(label, value, enabled)``.  ``selected()`` returns
    the checked values; ``on_change`` fires after any toggle.  Default-checked
    values are passed via ``default_checked``.
    """

    def __init__(self, parent, title, options, on_change=None, default_checked=None):
        super().__init__(parent)
        self.title = title
        self.on_change = on_change
        self._btn = ttk.Label(self, text=f"▾ {title}", cursor="hand2")
        self._btn.pack(anchor="w", fill="x")
        self._btn.bind("<Button-1>", lambda e: self._toggle())
        self.body = ttk.Frame(self)
        self.body.pack(fill="x", padx=(14, 0))
        self._items: list[tuple] = []   # (value, BooleanVar)
        checked = set(default_checked or ())
        for label, value, enabled in options:
            var = tk.BooleanVar(value=(value in checked))
            cb = ttk.Checkbutton(self.body, text=str(label), variable=var,
                                 command=self._changed)
            if not enabled:
                cb.state(["disabled"])
                var.set(False)
            cb.pack(anchor="w")
            self._items.append((value, var))

    def _changed(self):
        if self.on_change is not None:
            self.on_change()

    def _toggle(self):
        if self.body.winfo_ismapped():
            self.body.pack_forget()
            self._btn.config(text=f"▸ {self.title}")
        else:
            self.body.pack(fill="x", padx=(14, 0))
            self._btn.config(text=f"▾ {self.title}")

    def selected(self) -> list:
        return [value for value, var in self._items if var.get()]


class RunSuitePane(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        left = ttk.Frame(self); left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        ttk.Label(left, text="Prepared clips (4+ detected pts, GT placed, plane resolved):").pack(anchor="w")
        self.clip_list = tk.Listbox(left, selectmode="extended", exportselection=False)
        self.clip_list.pack(fill="both", expand=True)
        self.summary = ttk.Label(left, text="", wraplength=520, justify="left")
        self.summary.pack(fill="x", pady=(4, 0))
        right = ttk.Frame(self); right.pack(side="left", fill="y", padx=8, pady=8)

        from test_bench.scoring import contract, registry
        from test_bench.scoring.arc_2d import SELECTIONS
        implemented = set(registry.list_methods())
        methods = [(f"{m} (not implemented)" if m not in implemented else m, m, m in implemented)
                   for m in contract.METHODS]
        self.method_group = CheckGroup(right, "methods", methods,
                                       default_checked=list(implemented))
        n_opts = [(str(v), v, True) for v in (4, 6, 8, 12, 20, "all")]
        self.n_group = CheckGroup(right, "n_points", n_opts,
                                  default_checked=(4, 6, 8, 12, 20, "all"))
        sel_opts = [(s, s, True) for s in SELECTIONS]
        self.sel_group = CheckGroup(right, "selections", sel_opts,
                                    default_checked=("last_n", "all", "ransac_inliers"))
        self.method_group.pack(fill="x", pady=2)
        self.n_group.pack(fill="x", pady=2)
        self.sel_group.pack(fill="x", pady=2)

        ttk.Button(right, text="Run suite", command=self._run).pack(fill="x", pady=8)
        self.status = ttk.Label(right, text="", wraplength=200)
        self.status.pack()

    def refresh(self):
        if not self._visible():
            # Not on screen: defer to on_shown() so launching the app stays
            # responsive (walking 116 clips' artifacts costs ~2 s cold).
            return
        self.clip_list.delete(0, "end")
        # Readiness comes from the clip's artifact files, not the manifest caches
        # (manifest["detections"]["n_accepted"] is written only by the Analyze
        # stage, so a clip processed anywhere else arrived with {} and vanished).
        skipped = []
        for cid in self.app.list_clips():
            try:
                st = manifest_mod.clip_state(self.app.clip_dir(cid))
            except Exception as e:  # noqa: BLE001 — one broken clip must not blank the pane
                skipped.append((cid, [f"unreadable: {e}"]))
                continue
            if st["ready"]:
                self.clip_list.insert("end", cid)
            else:
                skipped.append((cid, st["blockers"]))
        n_ready = self.clip_list.size()
        if n_ready:
            msg = f"{n_ready} clip(s) ready"
        elif skipped:
            cid, why = skipped[0]
            msg = (f"0 of {len(skipped)} clip(s) ready — e.g. {cid}: "
                   + "; ".join(why))
        else:
            msg = "no clips selected yet"
        self.summary.config(text=msg)

    def _visible(self):
        try:
            return self.app.nb.nametowidget(self.app.nb.select()) is self
        except (KeyError, tk.TclError):
            return True

    def on_shown(self):
        """Refresh now that this tab is on screen (see refresh())."""
        self.refresh()

    def _run(self):
        sel_idx = self.clip_list.curselection()
        if not sel_idx:
            messagebox.showinfo("run suite", "select at least one prepared clip")
            return
        ids = [self.clip_list.get(i) for i in sel_idx]
        methods = self.method_group.selected()
        n_vals = self.n_group.selected()
        sel_vals = self.sel_group.selected()
        if not methods:
            messagebox.showwarning("run suite", "select at least one method")
            return
        if not n_vals or not sel_vals:
            messagebox.showwarning("run suite", "select at least one n_points value and one selection")
            return
        configs = suite_mod.grid_configs(n_vals, sel_vals, methods=methods)
        clips = []
        prov: dict = {}
        for cid in ids:
            cdir = self.app.clip_dir(cid)
            man = manifest_mod.read_manifest(cdir)
            recs = json.loads((cdir / "detections.json").read_text(encoding="utf-8"))
            points = trajectory_mod.assemble(recs)
            st = manifest_mod.clip_state(cdir, man)
            plane = st["plane_doc"]
            gt = dict(st["gt"] or {})
            # gt_in is always derived from the GT pixel through the plane, never a
            # stored number, so a record that carries a pixel but no distance (or
            # was saved under an older plane) scores instead of raising.
            if plane is not None and gt.get("gt_in") is None and st["gt_placed"]:
                try:
                    gt = gt_mod.set_ground_truth(
                        plane, frame=int(gt.get("frame") or 0),
                        x_px=float(gt["x_px"]), y_px=float(gt["y_px"]),
                        judge_in=gt.get("judge_in"), updated=gt.get("updated"))
                except ValueError as e:
                    messagebox.showerror("run suite", f"{cid}: {e}")
                    return
            clips.append({"clip_id": cid, "points": points, "plane_doc": plane, "gt": gt,
                          "frame_size": [man["meta"]["width"], man["meta"]["height"]]})
        try:
            res = suite_mod.run_suite(clips, configs)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("run suite", str(e))
            return
        run_id = _now().replace(":", "-").replace("+", "Z")
        out = self.app.root_dir / "results" / f"{run_id}.json"
        manifest_mod.write_json(out, {"run_id": run_id, "config": {"clips": ids, "methods": configs},
                                      "provenance": prov, **res})
        self.status.config(text=f"wrote {run_id}.json\n{len(res['rows'])} clip(s) x {len(configs)} config(s)")
        self.app.last_run = run_id
        self.app.on_clips_changed()


class ResultsPane(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.avg_lbl = ttk.Label(self, text="")
        self.avg_lbl.pack(fill="x", padx=8, pady=4)
        cols = ("clip", "method", "n_points", "selection", "pixel_px", "score_in",
                "judge_in", "accuracy_pct", "status")
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        headings = {"pixel_px": "pixel err (px)", "score_in": "score err (in)",
                    "judge_in": "judge err (in)", "accuracy_pct": "accuracy %"}
        for c, w in zip(cols, (200, 90, 60, 90, 90, 90, 90, 90, 120)):
            self.tree.heading(c, text=headings.get(c, c))
            self.tree.column(c, width=w, anchor="center", stretch=True)
        self.tree.column("clip", anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        if not getattr(self.app, "last_run", None):
            return
        doc = self._latest_run_doc()
        if not doc:
            return
        recorded = doc.get("provenance", {})
        for row in doc.get("rows", []):
            cid = row["clip_id"]
            stale = self._clip_stale(cid, recorded.get(cid))
            for r in row.get("runs", []):
                status = r["score"].get("status") or ""
                if stale:
                    status = (status + " (stale)").strip()
                self.tree.insert("", "end", values=(
                    row["clip_id"], r["method"], r.get("n_points"), r.get("selection"),
                    _fmt(r.get("pixel_error_px")), _fmt(r.get("score_error_in")),
                    _fmt(r.get("judge_error_in")), _fmt_pct(r.get("accuracy_pct")),
                    status))
        av = doc.get("averages", {}).get("overall", {})
        self.avg_lbl.config(text="averages — pixel: %s px   score: %s in   judge: %s in   accuracy: %s" % (
            _fmt(av.get("pixel_error_px")), _fmt(av.get("score_error_in")),
            _fmt(av.get("judge_error_in")), _fmt_pct(av.get("accuracy_pct"))))

    def _clip_stale(self, cid, recorded):
        cdir = self.app.clip_dir(cid)
        try:
            man = manifest_mod.read_manifest(cdir)
            plane = manifest_mod.clip_plane(man)
            gt_p = cdir / "ground_truth.json"
            gt = json.loads(gt_p.read_text(encoding="utf-8")) if gt_p.is_file() else None
        except Exception:  # noqa: BLE001
            return True
        return suite_mod.is_stale(recorded, man, plane, gt)

    def _latest_run_doc(self):
        rdir = self.app.root_dir / "results"
        if not rdir.is_dir():
            return None
        files = sorted(rdir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return json.loads(files[-1].read_text(encoding="utf-8")) if files else None

    def _on_select(self):
        sel = self.tree.selection()
        if not sel:
            return
        clip_id = self.tree.item(sel[0], "values")[0]
        self.app.open_compare(clip_id)


class ComparePane(_FramePane):
    """Per-method visual review: GT, each run's selected detections, fitted curve,
    entry point, and the GT→entry error vector — all with the dynamic line style."""

    #: Stable per-run palette (one colour per checked run, shown in the legend).
    PALETTE = ("#00e5ff", "#ff5cff", "#3cf03c", "#ffa500", "#ffe14d",
               "#ff6b6b", "#b388ff", "#69f0ae", "#ff80ab", "#80d8ff")
    #: The judge reference line (drawn at X = judge distance) and its legend swatch.
    JUDGE_COLOUR = "#ffca28"
    #: The accuracy cone — a shaded ±X fan around the fitted trajectory.  Red
    #: when the compared clip's own error falls outside the suite's error range.
    CONE_COLOUR = "#5b7d99"
    CONE_OUT_COLOUR = "#b00020"

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.runs = []          # [(run_dict, colour)] from the active run
        self.gt = None
        self.plane_doc = None
        self.points = []        # accepted detections (post-deselection)
        self._run_doc = None    # newest results doc containing this clip
        self.filter_frame = ttk.Frame(self)
        self.filter_frame.pack(fill="x", padx=8, pady=4)
        self.legend = ttk.Frame(self)
        self.legend.pack(fill="x", padx=8, pady=2)
        self.method_group = None
        self.n_group = None
        self.sel_group = None

        op = ttk.Frame(self)
        op.pack(fill="x", padx=8, pady=2)
        ttk.Label(op, text="element opacity:").pack(side="left")
        self.element_opacity = tk.DoubleVar(value=100.0)
        tk.Scale(op, from_=0, to=100, orient="horizontal", variable=self.element_opacity,
                 showvalue=False, command=self._on_opacity).pack(side="left", padx=(4, 0))
        ttk.Label(op, text="grid opacity:").pack(side="left", padx=(16, 0))
        self.grid_opacity = tk.DoubleVar(value=100.0)
        tk.Scale(op, from_=0, to=100, orient="horizontal", variable=self.grid_opacity,
                 showvalue=False, command=self._on_opacity).pack(side="left", padx=(4, 0))
        self.show_det = tk.BooleanVar(value=True)
        self.show_vec = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        self.show_cone = tk.BooleanVar(value=True)
        for text, var in (("detections", self.show_det), ("error vectors", self.show_vec),
                          ("grid", self.show_grid)):
            ttk.Checkbutton(op, text=text, variable=var,
                            command=self._redraw).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(op, text="accuracy cone", variable=self.show_cone,
                        command=self._on_cone_toggle).pack(side="left", padx=(12, 0))

    # -- loading ------------------------------------------------------------
    def load_clip(self, clip_id):
        self.clip_var.set(clip_id)
        self._on_clip()

    def _on_clip(self):
        super()._on_clip()
        self._load_data()
        self._rebuild_filters()

    def _load_data(self):
        cid = self.clip_var.get()
        self.gt = None
        self.plane_doc = None
        self.points = []
        self._run_doc = None
        if not cid or not self.man:
            return
        cdir = self.app.clip_dir(cid)
        gt_path = cdir / "ground_truth.json"
        self.gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else None
        self.plane_doc = manifest_mod.clip_plane(self.man)
        self._run_doc = self._active_run_doc()
        det_path = cdir / "detections.json"
        if det_path.is_file():
            recs = json.loads(det_path.read_text(encoding="utf-8"))
            self.points = trajectory_mod.assemble(recs)

    # -- overlay model ------------------------------------------------------
    def _rebuild_filters(self):
        for w in self.filter_frame.winfo_children():
            w.destroy()
        cid = self.clip_var.get()
        self.runs = []
        if cid:
            runs = self._active_runs_for(cid)
            for i, r in enumerate(runs):
                self.runs.append((r, self.PALETTE[i % len(self.PALETTE)]))
        methods = sorted({r["method"] for r, _ in self.runs})
        n_pts = sorted({r.get("n_points") for r, _ in self.runs},
                       key=lambda v: (v == "all", v if v != "all" else 0))
        sels = sorted({r.get("selection") for r, _ in self.runs})
        self.method_group = CheckGroup(self.filter_frame, "methods",
                                       [(m, m, True) for m in methods],
                                       on_change=self._on_filter, default_checked=methods)
        self.n_group = CheckGroup(self.filter_frame, "n_points",
                                  [(str(v), v, True) for v in n_pts],
                                  on_change=self._on_filter, default_checked=n_pts)
        self.sel_group = CheckGroup(self.filter_frame, "selections",
                                    [(s, s, True) for s in sels],
                                    on_change=self._on_filter, default_checked=sels)
        self.method_group.pack(side="left", anchor="n", padx=4)
        self.n_group.pack(side="left", anchor="n", padx=4)
        self.sel_group.pack(side="left", anchor="n", padx=4)
        self._rebuild_legend()
        self._redraw()

    def _on_filter(self):
        self._rebuild_legend()
        self._redraw()

    def _on_cone_toggle(self):
        self._rebuild_legend()
        self._redraw()

    def _run_visible(self, r):
        if self.method_group is not None and r["method"] not in self.method_group.selected():
            return False
        if self.n_group is not None and r.get("n_points") not in self.n_group.selected():
            return False
        if self.sel_group is not None and r.get("selection") not in self.sel_group.selected():
            return False
        return True

    def _legend_row(self, colour, text):
        row = ttk.Frame(self.legend)
        row.pack(side="left", padx=8)
        swatch = tk.Canvas(row, width=12, height=12, highlightthickness=0)
        swatch.create_rectangle(1, 1, 11, 11, fill=colour, outline="")
        swatch.pack(side="left")
        ttk.Label(row, text=text).pack(side="left")

    def _rebuild_legend(self):
        for w in self.legend.winfo_children():
            w.destroy()
        if self.gt and self.gt.get("judge_present") and self.gt.get("judge_in") is not None:
            self._legend_row(self.JUDGE_COLOUR, f"judge {float(self.gt['judge_in']):.1f} in")
        visible = [rc for rc in self.runs if self._run_visible(rc[0])]
        if not visible:
            if not self.legend.winfo_children():
                ttk.Label(self.legend, text="no runs for this clip (run a suite first)").pack(side="left")
            return
        if self.show_cone.get():
            geo = self._cone_geometry(visible[0][0])
            if geo is not None:
                col = self.CONE_OUT_COLOUR if geo["outlier"] else self.CONE_COLOUR
                tag = "OUTLIER " if geo["outlier"] else ""
                self._legend_row(col, f"{tag}accuracy cone ±{geo['X_in']:.1f}in "
                                      f"(n={geo['n_other']} others)")
        for r, colour in visible:
            px = _fmt(r.get("pixel_error_px"))
            inch = _fmt(r.get("score_error_in"))
            self._legend_row(colour, f"{r['method']} n={r.get('n_points')} {r.get('selection')}  "
                                     f"Δ{px}px/{inch}in")

    def _active_run_doc(self):
        """The newest results document that contains the current clip (or None)."""
        cid = self.clip_var.get()
        rdir = self.app.root_dir / "results"
        if not cid or not rdir.is_dir():
            return None
        files = sorted(rdir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for p in reversed(files):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if any(row.get("clip_id") == cid for row in doc.get("rows", [])):
                return doc
        return None

    def _active_runs_for(self, cid):
        """The runs for ``cid`` in the newest results file (the active bench run)."""
        doc = self._active_run_doc()
        if not doc:
            return []
        for row in doc.get("rows", []):
            if row.get("clip_id") == cid:
                return row["runs"]
        return []

    def _selected_px_for(self, r):
        fit = (r["score"].get("fit") or {})
        sel = fit.get("selected_px")
        if sel:
            return sel
        if not self.points:
            return []
        try:
            from test_bench.scoring.arc_2d import _select
            pts = _select(self.points, r.get("selection", "all"), r.get("n_points", "all"))
            return [[float(p["x"]), float(p["y"])] for p in pts]
        except Exception:  # noqa: BLE001
            return []

    def _grid_lines(self):
        doc = self.plane_doc
        if not doc or not (doc.get("plane") or {}):
            return []
        try:
            from dog_diving.pool import calib as _calib
            from dog_diving.pool import plane as _plane
            block = doc["plane"]
            length = float(block.get("length_ft") or _plane.DEFAULT_LENGTH_FT)
            width = float(block.get("width_ft") or _plane.DEFAULT_WIDTH_FT)
            lens = _calib.lens_from_document(doc)
            spec = _plane.grid_spec(doc["homography"], length_ft=length, width_ft=width, lens=lens)
        except Exception:  # noqa: BLE001
            return []
        return [(tuple(ln["a"]), tuple(ln["b"]), bool(ln.get("major")))
                for ln in spec.get("lines", [])]

    def _judge_line(self):
        """The judge reference as an image-space line at ``X = judge distance``.

        Returns ``(a, b)`` image-px endpoints (across the pool width), or ``None``
        when there is no judge, no plane, or the projection folds.
        """
        gt = self.gt
        if not gt or not gt.get("judge_present") or gt.get("judge_in") is None:
            return None
        doc = self.plane_doc
        if not doc or not (doc.get("plane") or {}):
            return None
        try:
            import numpy as _np
            from dog_diving.pool import calib as _calib
            from dog_diving.pool import homography as _hom
            from dog_diving.pool import lens as _lens
            from dog_diving.pool import plane as _plane
            block = doc["plane"]
            width = float(block.get("width_ft") or _plane.DEFAULT_WIDTH_FT)
            judge_ft = float(gt["judge_in"]) / 12.0
            H = _np.asarray(doc["homography"], dtype=float).reshape(3, 3)
            pts = _hom.apply_homography(H, [(judge_ft, 0.0), (judge_ft, width)])
            try:
                lens = _calib.lens_from_document(doc)
            except Exception:  # noqa: BLE001 — a plane without a lens is still drawable
                lens = None
            if lens is not None:
                pts = _lens.distort_points(pts, lens)
            a = (float(pts[0][0]), float(pts[0][1]))
            b = (float(pts[1][0]), float(pts[1][1]))
            if not (_np.all(_np.isfinite(a)) and _np.all(_np.isfinite(b))):
                return None
            return a, b
        except Exception:  # noqa: BLE001
            return None

    def _cone_geometry(self, run):
        """Accuracy-cone geometry around ``run``'s trajectory, or None.

        The cone is a fan from the launch point to the landing line spanning
        ±X inches, where X is the robust error bound (Tukey fence on absolute
        score error) of the *other* clips at this run's config.  Returns
        ``{"polygon": [(x,y)...], "ticks": [(x,y,is_outlier)...], "X_in": X,
        "outlier": bool, "this_signed": float|None, "n_other": int}``.
        """
        sc = run.get("score") or {}
        ep = sc.get("entry_px")
        fit = sc.get("fit") or {}
        curve = fit.get("curve")
        if not ep or not curve or len(curve) < 2:
            return None
        in_per_px = accuracy_mod.local_in_per_px_at_entry(self.plane_doc, ep)
        if in_per_px is None:
            in_per_px = accuracy_mod.nominal_in_per_px(self.plane_doc)
        if not in_per_px or in_per_px <= 0:
            return None
        cl = fit.get("centerline_line_px")
        if not cl or len(cl) < 2:
            return None
        dirv = np.asarray(cl[1], dtype=float) - np.asarray(cl[0], dtype=float)
        norm = float(np.linalg.norm(dirv))
        if norm <= 0:
            return None
        dirv = dirv / norm
        # The centerline's endpoint order is arbitrary; orient it in the +X
        # (down-pool) direction via the homography so overshoot/short map to
        # the physically correct sides.
        H = (self.plane_doc or {}).get("homography")
        if H is not None:
            try:
                Hm = np.asarray(H, dtype=float).reshape(3, 3)
                Xc = float(sc.get("distance_in") or 0.0) / 12.0
                Yc = float(fit.get("centerline_yc_ft") or 0.0)
                p0 = Hm @ np.array([Xc, Yc, 1.0], dtype=float)
                p1 = Hm @ np.array([Xc + 1.0, Yc, 1.0], dtype=float)
                if abs(float(p0[2])) > 1e-12 and abs(float(p1[2])) > 1e-12:
                    plus_x = (p1[:2] / p1[2]) - (p0[:2] / p0[2])
                    if float(np.dot(plus_x, dirv)) < 0:
                        dirv = -dirv
            except Exception:  # noqa: BLE001
                pass

        cfg = {"method": run.get("method"), "n_points": run.get("n_points"),
               "selection": run.get("selection")}
        cid = self.clip_var.get()
        signed = []
        for row in (self._run_doc or {}).get("rows") or []:
            if row.get("clip_id") == cid:
                continue
            gt_in = (row.get("gt") or {}).get("gt_in")
            cell = accuracy_mod.find_run(row.get("runs"), cfg)
            d_in = ((cell or {}).get("score") or {}).get("distance_in")
            if gt_in is not None and d_in is not None:
                signed.append(float(d_in) - float(gt_in))
        if len(signed) < 2:
            return None
        abs_err = np.abs(np.asarray(signed, dtype=float))
        if len(signed) >= 4:
            q1, q3 = np.percentile(abs_err, [25.0, 75.0])
            X = float(q3 + 1.5 * (q3 - q1))
        else:
            X = float(abs_err.max())
        if X <= 0:
            return None
        dx = X / in_per_px

        pts = [np.asarray(p, dtype=float) for p in curve]
        pts.append(np.asarray(ep, dtype=float))
        n = len(pts)
        t = np.linspace(0.0, 1.0, n)
        long_pts = [tuple(pts[i] + (t[i] * dx) * dirv) for i in range(n)]
        short_pts = [tuple(pts[i] - (t[i] * dx) * dirv) for i in range(n)]
        polygon = long_pts + list(reversed(short_pts))

        ep_arr = np.asarray(ep, dtype=float)
        ticks = []
        for s in signed:
            pos = ep_arr + (s / in_per_px) * dirv
            ticks.append((float(pos[0]), float(pos[1]), abs(s) > X))

        this_signed = None
        if sc.get("distance_in") is not None and (self.gt or {}).get("gt_in") is not None:
            this_signed = float(sc["distance_in"]) - float(self.gt["gt_in"])
        outlier = this_signed is not None and abs(this_signed) > X

        return {"polygon": polygon, "ticks": ticks, "X_in": X,
                "outlier": outlier, "this_signed": this_signed, "n_other": len(signed)}

    def _cone_overlays(self):
        """Overlays for the accuracy cone (empty when off / no data)."""
        if not self.show_cone.get() or not self.plane_doc:
            return []
        visible = [r for r, _ in self.runs if self._run_visible(r)]
        if not visible:
            return []
        geo = self._cone_geometry(visible[0])
        if geo is None:
            return []
        colour = self.CONE_OUT_COLOUR if geo["outlier"] else self.CONE_COLOUR
        overlays = [("polygon", geo["polygon"], colour, 0.30)]
        for x, y, is_out in geo["ticks"]:
            overlays.append(("dot", (x, y, 0.8),
                             self.CONE_OUT_COLOUR if is_out else self.CONE_COLOUR))
        if geo["outlier"] and geo["this_signed"] is not None:
            ep = (visible[0].get("score") or {}).get("entry_px")
            s = geo["this_signed"]
            overlays.append(("label", (ep[0], ep[1],
                                       f"OUTLIER {'+' if s >= 0 else ''}{s:.1f}in "
                                       f"(cone ±{geo['X_in']:.1f}in)"),
                             self.CONE_OUT_COLOUR))
        return overlays

    def _build_overlays(self):
        overlays = list(self._cone_overlays())
        if self.gt and self.gt.get("x_px") is not None:
            overlays.append(("marker", (self.gt["x_px"], self.gt["y_px"], 2.2), "red"))
        judge = self._judge_line()
        if judge is not None:
            a, b = judge
            overlays.append(("segment", (a[0], a[1], b[0], b[1]), self.JUDGE_COLOUR))
            overlays.append(("label", (a[0], a[1], f"judge {float(self.gt['judge_in']):.1f}in"),
                             self.JUDGE_COLOUR))
        if self.show_grid.get():
            gl = self._grid_lines()
            if gl:
                overlays.append(("grid", gl, "#3a6ea5"))
        for r, colour in self.runs:
            if not self._run_visible(r):
                continue
            sc = r["score"]
            if self.show_det.get():
                for px in self._selected_px_for(r):
                    overlays.append(("dot", (px[0], px[1], 1.0), colour))
            fit = sc.get("fit") or {}
            curve = fit.get("curve")
            if curve and len(curve) >= 2 and fit.get("curve_space", "px") == "px":
                overlays.append(("curve", curve, colour))
            ep = sc.get("entry_px")
            if ep:
                overlays.append(("marker", (ep[0], ep[1], 1.5), colour))
                if self.show_vec.get() and self.gt and self.gt.get("x_px") is not None:
                    overlays.append(("segment", (self.gt["x_px"], self.gt["y_px"], ep[0], ep[1]), colour))
                    px = r.get("pixel_error_px")
                    if px is not None:
                        overlays.append(("label", (ep[0], ep[1], f"Δ{px:.1f}px"), colour))
        return overlays

    def _on_opacity(self, _v=None):
        self.view.element_opacity = self.element_opacity.get() / 100.0
        self.view.grid_opacity = self.grid_opacity.get() / 100.0
        self.view.set_overlays(self._build_overlays())

    def _redraw(self):
        self.view.element_opacity = self.element_opacity.get() / 100.0
        self.view.grid_opacity = self.grid_opacity.get() / 100.0
        self.view.set_overlays(self._build_overlays())

    def _on_frame(self):
        super()._on_frame()
        self._redraw()


class CalibratePane(ttk.Frame):
    """The parent's Calibrate tab, plus a clip picker (it calibrates one clip at a time)."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=4)
        ttk.Label(bar, text="Clip:").pack(side="left")
        self.clip_var = tk.StringVar()
        self.clip_box = ttk.Combobox(bar, textvariable=self.clip_var, width=44, state="readonly")
        self.clip_box.pack(side="left", padx=4)
        ttk.Button(bar, text="Load for calibration", command=self._load).pack(side="left", padx=4)
        self.cal = CalibrateTab(self, on_plane_saved=self._on_plane_saved)
        self.cal.pack(fill="both", expand=True)

    def refresh(self):
        self.clip_box["values"] = self.app.list_clips()

    def _load(self):
        cid = self.clip_var.get()
        if not cid:
            return
        cdir = self.app.clip_dir(cid)
        man = manifest_mod.read_manifest(cdir)
        clip = manifest_mod.clip_video_path(man, cdir)
        if clip is None:
            messagebox.showerror("calibrate", "clip file not found")
            return
        self.cal.load_clip(clip)

    def _on_plane_saved(self, setup_id, path):
        # A new plane landed in the saved-planes folder: refresh the Analyze dropdown.
        self.app.on_clips_changed()


def _fmt(v):
    return "" if v is None else f"{v:.2f}"


def _fmt_pct(v):
    return "" if v is None else f"{v:.1f}%"


class AccuracyPane(ttk.Frame):
    """Read-only, proposal-facing accuracy metrics over the active run's results.

    Selects a run JSON + one method config, then tabulates each selected clip's
    nominal / local in-per-pixel, score error (in and %), judge error (in and %),
    and the ±2 / ±4 / ±6 in pass marks, with a mean/median summary footer and a
    stale banner when the run's recorded provenance no longer matches the clip's
    current inputs.  Never writes results.
    """

    COLS = ("clip", "nominal_px", "local_px", "score_in", "score_pct",
            "judge_in", "judge_pct", "w2", "w4", "w6")

    SUM_COLS = ("config", "n", "mean_in", "median_in", "mean_pct", "median_pct",
                "w2", "w4", "w6", "judge_in", "judge_pct")
    SUM_HEADINGS = {"config": "config", "n": "n", "mean_in": "mean (in)",
                    "median_in": "median (in)", "mean_pct": "mean (%)",
                    "median_pct": "median (%)", "w2": "±2 in", "w4": "±4 in",
                    "w6": "±6 in", "judge_in": "judge err (in)",
                    "judge_pct": "judge err (%)"}
    SUM_WIDTHS = {"config": 180, "n": 44, "mean_in": 60, "median_in": 60,
                  "mean_pct": 60, "median_pct": 60, "w2": 48, "w4": 48, "w6": 48,
                  "judge_in": 66, "judge_pct": 66}

    # plot colours (kept distinct from the tree so the two views read clearly)
    BAR_COLOUR = "#5b7d99"
    SEL_COLOUR = "#0077c8"
    MED_COLOUR = "#00a651"
    MEAN_COLOUR = "#ff6d00"
    OUT_COLOUR = "#b00020"
    CAM0_COLOUR = "#0077c8"      # camera0 plane in the landing-location plot
    CAM1_COLOUR = "#e07b00"      # camera1 plane

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.doc = None
        self._configs: list = []          # doc["config"]["methods"] entries
        self._run_paths: dict = {}        # label -> results/*.json path

        # larger, legible fonts for the summary ("not a thin bar at the bottom")
        self._font_fam = tkfont.nametofont("TkDefaultFont").actual("family")
        _style = ttk.Style()
        _style.configure("AccuracyHeader.TLabel", font=(self._font_fam, 12, "bold"))
        _style.configure("AccuracySum.Treeview", font=(self._font_fam, 11), rowheight=26)
        _style.configure("AccuracySum.Treeview.Heading", font=(self._font_fam, 11, "bold"))

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=4)
        ttk.Label(bar, text="Run:").pack(side="left")
        self.run_var = tk.StringVar()
        self.run_box = ttk.Combobox(bar, textvariable=self.run_var, width=30, state="readonly")
        self.run_box.pack(side="left", padx=4)
        self.run_box.bind("<<ComboboxSelected>>", lambda e: self._load_run())
        ttk.Label(bar, text="Method config:").pack(side="left", padx=(12, 0))
        self.cfg_var = tk.StringVar()
        self.cfg_box = ttk.Combobox(bar, textvariable=self.cfg_var, width=38, state="readonly")
        self.cfg_box.pack(side="left", padx=4)
        self.cfg_box.bind("<<ComboboxSelected>>", lambda e: self._render())
        ttk.Label(bar, text="Group by:").pack(side="left", padx=(12, 0))
        self.group_var = tk.StringVar(value="none")
        self.group_box = ttk.Combobox(bar, textvariable=self.group_var, width=9,
                                      state="readonly", values=("none", "camera", "dog"))
        self.group_box.pack(side="left", padx=4)
        self.group_box.bind("<<ComboboxSelected>>", lambda e: self._render())

        # average-error summary — one row per method config in the active run
        sframe = ttk.Frame(self)
        sframe.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(sframe, text="Average error by config (active suite):",
                  style="AccuracyHeader.TLabel", anchor="w").pack(fill="x")

        sum_hold = ttk.Frame(self)
        sum_hold.pack(fill="x", padx=8, pady=(2, 6))
        self.sum_tree = ttk.Treeview(sum_hold, columns=self.SUM_COLS, show="headings",
                                     style="AccuracySum.Treeview")
        for c in self.SUM_COLS:
            self.sum_tree.heading(c, text=self.SUM_HEADINGS.get(c, c))
            self.sum_tree.column(c, width=self.SUM_WIDTHS.get(c, 60), anchor="center",
                                 stretch=True)
        self.sum_tree.column("config", anchor="w")
        self.sum_tree.pack(side="left", fill="both", expand=True)
        _sb = ttk.Scrollbar(sum_hold, orient="vertical", command=self.sum_tree.yview)
        self.sum_tree.configure(yscrollcommand=_sb.set)
        _sb.pack(side="right", fill="y")

        # four plots in a 2x2 grid: error-by-config/group + landing-location on
        # top; per-clip error + judge comparison below (selected config)
        self._plots = ttk.Frame(self)
        self._plots.pack(fill="x", padx=8, pady=(0, 4))
        top = ttk.Frame(self._plots)
        top.pack(fill="x")
        bot = ttk.Frame(self._plots)
        bot.pack(fill="x")
        self.fig_cfg = Figure(figsize=(3.6, 2.2), dpi=100, tight_layout=True)
        self.ax_cfg = self.fig_cfg.add_subplot(111)
        self.canvas_cfg = FigureCanvasTkAgg(self.fig_cfg, master=top)
        self.canvas_cfg.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.fig_land = Figure(figsize=(3.6, 2.2), dpi=100, tight_layout=True)
        self.ax_land = self.fig_land.add_subplot(111)
        self.canvas_land = FigureCanvasTkAgg(self.fig_land, master=top)
        self.canvas_land.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.fig_sel = Figure(figsize=(3.6, 2.2), dpi=100, tight_layout=True)
        self.ax_sel = self.fig_sel.add_subplot(111)
        self.canvas_sel = FigureCanvasTkAgg(self.fig_sel, master=bot)
        self.canvas_sel.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.fig_judge = Figure(figsize=(3.6, 2.2), dpi=100, tight_layout=True)
        self.ax_judge = self.fig_judge.add_subplot(111)
        self.canvas_judge = FigureCanvasTkAgg(self.fig_judge, master=bot)
        self.canvas_judge.get_tk_widget().pack(side="left", fill="both", expand=True)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Clips (empty = all):").pack(anchor="w")
        self.clip_list = tk.Listbox(left, selectmode="extended", exportselection=False,
                                    width=28, height=10)
        self.clip_list.pack(fill="y", expand=True, padx=(0, 6))
        self.clip_list.bind("<<ListboxSelect>>", lambda e: self._on_clip_select())

        self.tree = ttk.Treeview(body, columns=self.COLS, show="headings")
        headings = {"nominal_px": "nominal in/px", "local_px": "local in/px @entry",
                    "score_in": "score err (in)", "score_pct": "score err (%)",
                    "judge_in": "judge err (in)", "judge_pct": "judge err (%)",
                    "w2": "±2 in", "w4": "±4 in", "w6": "±6 in"}
        widths = {"clip": 220, "nominal_px": 90, "local_px": 100, "score_in": 90,
                  "score_pct": 90, "judge_in": 90, "judge_pct": 90,
                  "w2": 55, "w4": 55, "w6": 55}
        for c in self.COLS:
            self.tree.heading(c, text=headings.get(c, c))
            self.tree.column(c, width=widths.get(c, 100), anchor="center")
        self.tree.column("clip", anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)

        self.stale_lbl = ttk.Label(self, text="", foreground="#b00020", anchor="w")
        self.res_lbl = ttk.Label(self, text="", foreground="#b05000", anchor="w")

    # -- refresh ------------------------------------------------------------
    def refresh(self):
        rdir = self.app.root_dir / "results"
        files = sorted(rdir.glob("*.json"), key=lambda p: p.stat().st_mtime) if rdir.is_dir() else []
        self._run_paths = {}
        for p in files:
            label = p.stem.split(".", 1)[0] if "." in p.stem else p.stem
            self._run_paths[label] = p
        labels = list(self._run_paths)
        self.run_box["values"] = labels
        self.run_var.set(labels[-1] if labels else "")   # newest (mtime-sorted)
        self._load_run()

    def _load_run(self):
        self.doc = None
        self._configs = []
        path = self._run_paths.get(self.run_var.get())
        if path is None or not path.is_file():
            self._clear()
            return
        try:
            self.doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self.doc = None
            self._clear()
            return
        self._configs = list((self.doc.get("config") or {}).get("methods") or [])
        cfg_labels = [f"{c.get('method')} · n={c.get('n_points')} · {c.get('selection')}"
                      for c in self._configs]
        self.cfg_box["values"] = cfg_labels
        self.cfg_var.set(cfg_labels[0] if cfg_labels else "")
        self._populate_clips()
        self._render()

    def _populate_clips(self):
        self._populating = True
        try:
            self.clip_list.delete(0, "end")
            if not self.doc:
                return
            gt_indices = []
            for i, row in enumerate(self.doc.get("rows") or []):
                self.clip_list.insert("end", row.get("clip_id"))
                if (row.get("gt") or {}).get("gt_in") is not None:
                    gt_indices.append(i)
            for i in gt_indices:
                self.clip_list.selection_set(i)   # default: all clips with GT
        finally:
            self._populating = False

    def _on_clip_select(self):
        if getattr(self, "_populating", False):
            return
        self._render()

    def _clear(self):
        self.tree.delete(*self.tree.get_children())
        self.sum_tree.delete(*self.sum_tree.get_children())
        self.clip_list.delete(0, "end")
        self.cfg_box["values"] = []
        self.cfg_var.set("")
        self.stale_lbl.pack_forget()
        self.res_lbl.pack_forget()
        for ax, cv in ((self.ax_cfg, self.canvas_cfg), (self.ax_land, self.canvas_land),
                       (self.ax_sel, self.canvas_sel), (self.ax_judge, self.canvas_judge)):
            ax.cla()
            cv.draw_idle()

    def _render(self):
        self.tree.delete(*self.tree.get_children())
        self.sum_tree.delete(*self.sum_tree.get_children())
        self.stale_lbl.pack_forget()
        self.res_lbl.pack_forget()
        if not self.doc or not self._configs:
            return
        idx = self.cfg_box.current()
        method_cfg = self._configs[idx if idx >= 0 else 0]

        sel = self.clip_list.curselection()
        clip_ids = [self.clip_list.get(i) for i in sel] if sel else None

        recorded = self.doc.get("provenance") or {}
        rows = []
        any_stale = False
        res_mismatch = []
        for row in self.doc.get("rows") or []:
            r = dict(row)
            cid = row.get("clip_id")
            man = plane = gt_disk = None
            try:
                cdir = self.app.clip_dir(cid)
                man = manifest_mod.read_manifest(cdir)
                plane = manifest_mod.clip_plane(man)
                gt_p = cdir / "ground_truth.json"
                gt_disk = json.loads(gt_p.read_text(encoding="utf-8")) if gt_p.is_file() else None
            except Exception:  # noqa: BLE001
                man = plane = gt_disk = None
            r["plane_doc"] = plane
            r["_labels"] = (man or {}).get("labels") or {}
            r["dog"] = (r["_labels"].get("dog")
                        or ((row.get("gt") or {}).get("reference") or {}).get("dog"))
            if suite_mod.is_stale(recorded.get(cid), man, plane, gt_disk):
                any_stale = True
            psize = (plane or {}).get("image_size")
            meta = (man or {}).get("meta") or {}
            mw, mh = meta.get("width"), meta.get("height")
            if (isinstance(psize, (list, tuple)) and len(psize) == 2
                    and mw is not None and mh is not None
                    and (int(psize[0]) != int(mw) or int(psize[1]) != int(mh))):
                res_mismatch.append(cid)
            rows.append(r)

        res = accuracy_mod.aggregate(rows, method_cfg, clip_ids=clip_ids)

        for clip in res["clips"]:
            self.tree.insert("", "end", values=(
                clip["clip_id"],
                _fmt(clip.get("in_per_px_nominal")),
                _fmt(clip.get("in_per_px_local")),
                _fmt(clip.get("score_error_in")),
                _fmt_pct(clip.get("score_error_pct")),
                _fmt(clip.get("judge_error_in")),
                _fmt_pct(clip.get("judge_error_pct")),
                self._pass(clip, "within_2"),
                self._pass(clip, "within_4"),
                self._pass(clip, "within_6"),
            ))

        self._render_summary(rows, clip_ids)
        self._render_plots(rows, clip_ids, method_cfg, res)

        if any_stale:
            self.stale_lbl.config(text="stale: recorded provenance no longer matches current inputs")
            self.stale_lbl.pack(fill="x", padx=8, pady=2)

        if res_mismatch:
            self.res_lbl.config(text="in/px assumes native-resolution plane: " + ", ".join(res_mismatch))
            self.res_lbl.pack(fill="x", padx=8, pady=2)

    def _render_summary(self, rows, clip_ids):
        """Average-error table: one row per method config in the active run."""
        self.sum_tree.delete(*self.sum_tree.get_children())
        if not self.doc or not self._configs:
            return
        cur = self.cfg_box.current()
        cur = cur if cur >= 0 else 0
        for i, cfg in enumerate(self._configs):
            res = accuracy_mod.aggregate(rows, cfg, clip_ids=clip_ids)
            s = res["summary"]
            bins = s.get("bins") or {}
            self.sum_tree.insert("", "end", values=(
                f"{cfg.get('method')} · n={cfg.get('n_points')} · {cfg.get('selection')}",
                s.get("n_clips", 0),
                _fmt(s.get("mean_in")),
                _fmt(s.get("median_in")),
                _fmt_pct(s.get("mean_pct")),
                _fmt_pct(s.get("median_pct")),
                _fmt_pct(bins.get(2, {}).get("pct")),
                _fmt_pct(bins.get(4, {}).get("pct")),
                _fmt_pct(bins.get(6, {}).get("pct")),
                _fmt(s.get("judge_mean_in")),
                _fmt_pct(s.get("judge_mean_pct")),
            ), tags=("current",) if i == cur else ())
        self.sum_tree.tag_configure("current", font=(self._font_fam, 11, "bold"))
        self.sum_tree.configure(height=max(3, min(len(self._configs), 12)))

    # -- plots ----------------------------------------------------------------

    def _cfg_short(self, cfg):
        """Compact x-axis label; n_points only when method/selection are fixed."""
        keys = {(c.get("method"), c.get("selection")) for c in self._configs}
        n = cfg.get("n_points")
        if len(keys) == 1:
            return f"n={n}"
        m = str(cfg.get("method") or "")
        if m.startswith("arc_2d_"):
            m = m[len("arc_2d_"):]
        elif m == "arc_2d":
            m = "a2d"
        return f"{m}/{cfg.get('selection')}/n{n}"

    @staticmethod
    def _short_clip(cid):
        """``<HHMMSS>_cameraN_...`` -> ``<HHMMSS>/cN`` (compact tick label)."""
        parts = str(cid).split("_")
        if len(parts) >= 2 and parts[1].startswith("camera"):
            return f"{parts[0]}/{parts[1].replace('camera', 'c')}"
        return str(cid)

    @staticmethod
    def _camera_key(cid, labels=None):
        """``camera0`` / ``camera1`` for a clip, from labels or its id."""
        cam = (labels or {}).get("camera") or str(cid)
        for part in str(cam).split("_"):
            if part.startswith("camera"):
                return part
        return "unknown"

    def _render_plots(self, rows, clip_ids, method_cfg, sel_res):
        cur = self.cfg_box.current()
        cur = cur if cur >= 0 else 0
        mode = self.group_var.get()
        meta = {r["clip_id"]: r for r in rows}

        # Plot 1 — error distribution: box + jittered per-clip strip.  By
        # default it is grouped by config (find the best n); the "group by"
        # selector regroups the SELECTED config's errors by camera or dog.
        ax = self.ax_cfg
        ax.cla()
        groups: list = []          # (label, [score_error_in, ...])
        sel_pos = None
        if mode == "none":
            for i, cfg in enumerate(self._configs):
                res = accuracy_mod.aggregate(rows, cfg, clip_ids=clip_ids)
                errs = [c["score_error_in"] for c in res["clips"]
                        if c["score_error_in"] is not None]
                if errs:
                    if i == cur:
                        sel_pos = len(groups)
                    groups.append((self._cfg_short(cfg), errs))
        else:
            keyed: dict = {}
            for c in (sel_res.get("clips") or []):
                if c.get("score_error_in") is None:
                    continue
                r = meta.get(c["clip_id"]) or {}
                key = (self._camera_key(c["clip_id"], r.get("_labels")) if mode == "camera"
                       else (r.get("dog") or "unknown"))
                keyed.setdefault(key, []).append(c["score_error_in"])
            for key in sorted(keyed):
                groups.append((key, keyed[key]))
        if groups:
            labels = [g[0] for g in groups]
            pos = np.arange(len(groups))
            bp = ax.boxplot(
                [g[1] for g in groups], positions=pos, widths=0.55,
                patch_artist=True, showfliers=False, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor=self.MEAN_COLOUR,
                               markeredgecolor=self.MEAN_COLOUR, markersize=6),
                medianprops=dict(color="black", lw=1.5),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#dbe7f3")
                patch.set_edgecolor(self.BAR_COLOUR)
            rng = np.random.default_rng(0)   # deterministic jitter
            for xi, errs in zip(pos, (g[1] for g in groups)):
                errs = np.asarray(errs, dtype=float)
                q1, q3 = np.percentile(errs, [25.0, 75.0])
                hi = q3 + 1.5 * (q3 - q1)
                out = errs > hi
                jit = rng.uniform(-0.18, 0.18, errs.size)
                ax.scatter(xi + jit[~out], errs[~out], s=16, color=self.BAR_COLOUR,
                           alpha=0.6, zorder=3)
                ax.scatter(xi + jit[out], errs[out], s=26, color=self.OUT_COLOUR,
                           zorder=4)
            ax.set_xticks(pos)
            ax.set_xticklabels(labels)
            if sel_pos is not None:
                ax.get_xticklabels()[sel_pos].set_fontweight("bold")
                ax.get_xticklabels()[sel_pos].set_color(self.SEL_COLOUR)
            ax.set_ylabel("score error (in)")
            if mode == "none":
                ax.set_title("Error by config — box + per-clip points", fontsize=10)
            else:
                ax.set_title(f"Error by {mode} ({self._cfg_short(method_cfg)})", fontsize=10)
            ax.legend(handles=[
                Line2D([0], [0], color="black", lw=1.5, label="median"),
                Line2D([0], [0], marker="D", color="none",
                       markerfacecolor=self.MEAN_COLOUR,
                       markeredgecolor=self.MEAN_COLOUR, markersize=6,
                       label="mean"),
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=self.OUT_COLOUR, markersize=7,
                       label="outlier (>Q3+1.5·IQR)"),
            ], fontsize=8, loc="best", frameon=False)
            ax.set_xlim(-0.7, len(groups) - 0.3)
            ax.set_ylim(bottom=0)
        else:
            ax.set_title("Error — no GT", fontsize=10)
            ax.axis("off")
        self.canvas_cfg.draw_idle()

        # Plot 2 — the selected config: one point per clip, ranked by error,
        # with statistical outliers named so they can be removed from the suite.
        ax = self.ax_sel
        ax.cla()
        clips = [c for c in (sel_res.get("clips") or [])
                 if c.get("score_error_in") is not None]
        if clips:
            clips = sorted(clips, key=lambda c: c["score_error_in"], reverse=True)
            errs = np.array([c["score_error_in"] for c in clips], dtype=float)
            q1, q3 = np.percentile(errs, [25.0, 75.0])
            hi = q3 + 1.5 * (q3 - q1)
            out = errs > hi
            x = np.arange(len(clips))
            ax.scatter(x[~out], errs[~out], s=22, color=self.BAR_COLOUR,
                       alpha=0.7, zorder=3)
            ax.scatter(x[out], errs[out], s=40, color=self.OUT_COLOUR, zorder=4)
            for i in np.where(out)[0]:
                ax.annotate(self._short_clip(clips[i]["clip_id"]), (x[i], errs[i]),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            rotation=90, fontsize=6, color=self.OUT_COLOUR)
            summary = sel_res.get("summary") or {}
            mean_v, med_v = summary.get("mean_in"), summary.get("median_in")
            if mean_v is not None:
                ax.axhline(mean_v, color=self.MEAN_COLOUR, lw=1.0, ls="--",
                           label=f"mean {mean_v:.1f}")
            if med_v is not None:
                ax.axhline(med_v, color=self.MED_COLOUR, lw=1.0, ls=":",
                           label=f"median {med_v:.1f}")
            ax.set_xticks(x)
            ax.set_xticklabels([self._short_clip(c["clip_id"]) for c in clips],
                               rotation=90, fontsize=6)
            ax.set_ylabel("score error (in)")
            ax.set_title(f"{self._cfg_short(method_cfg)} — per-clip error", fontsize=10)
            ax.legend(fontsize=8, loc="best", frameon=False)
            ax.set_xlim(-0.6, len(clips) - 0.4)
            ax.set_ylim(bottom=0)
        else:
            ax.set_title("Selected config — no scored clips", fontsize=10)
            ax.axis("off")
        self.canvas_sel.draw_idle()

        # Plot 3 — automated score error vs human judge error (inches) for the
        # selected config.  Points above the diagonal are clips where the
        # pipeline is less accurate than the judge (the removal candidates).
        ax = self.ax_judge
        ax.cla()
        pts = [(c["score_error_in"], c["judge_error_in"], c["clip_id"])
               for c in (sel_res.get("clips") or [])
               if c.get("score_error_in") is not None
               and c.get("judge_error_in") is not None]
        if pts:
            score = np.array([p[0] for p in pts], dtype=float)
            judge = np.array([p[1] for p in pts], dtype=float)
            q1, q3 = np.percentile(score, [25.0, 75.0])
            hi = q3 + 1.5 * (q3 - q1)
            out = score > hi
            ax.scatter(judge[~out], score[~out], s=22, color=self.BAR_COLOUR,
                       alpha=0.7, zorder=3)
            ax.scatter(judge[out], score[out], s=40, color=self.OUT_COLOUR, zorder=4)
            for i in np.where(out)[0]:
                ax.annotate(self._short_clip(pts[i][2]), (judge[i], score[i]),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=6, color=self.OUT_COLOUR)
            ax.axline((0.0, 0.0), slope=1.0, color="#666666", lw=1.0, ls="--",
                      label="score = judge")
            ax.set_xlabel("judge error (in)")
            ax.set_ylabel("score error (in)")
            ax.set_title(f"{self._cfg_short(method_cfg)} — score vs judge", fontsize=10)
            ax.legend(fontsize=8, loc="best", frameon=False)
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)
        else:
            ax.set_title("Selected config — no judge data", fontsize=10)
            ax.axis("off")
        self.canvas_judge.draw_idle()

        # Plot 4 — score error vs GT landing location, coloured by plane
        # (camera0 vs camera1).  Shows whether error grows with jump length and
        # whether one camera plane is systematically biased.
        ax = self.ax_land
        ax.cla()
        cam_colours = {"camera0": self.CAM0_COLOUR, "camera1": self.CAM1_COLOUR}
        pts = []
        for c in (sel_res.get("clips") or []):
            if c.get("score_error_in") is None:
                continue
            r = meta.get(c["clip_id"]) or {}
            gt_in = (r.get("gt") or {}).get("gt_in")
            if gt_in is None or float(gt_in) <= 0:
                continue
            cam = self._camera_key(c["clip_id"], r.get("_labels"))
            pts.append((cam, float(gt_in), float(c["score_error_in"])))
        if pts:
            for cam in ("camera0", "camera1"):
                xs = [p[1] for p in pts if p[0] == cam]
                ys = [p[2] for p in pts if p[0] == cam]
                if xs:
                    ax.scatter(xs, ys, s=24, color=cam_colours.get(cam, "#888888"),
                               alpha=0.8, label=cam)
            ax.set_xlabel("GT landing location (in)")
            ax.set_ylabel("score error (in)")
            ax.set_title(f"{self._cfg_short(method_cfg)} — error vs landing", fontsize=10)
            ax.legend(fontsize=8, loc="best", frameon=False)
            ax.set_xlim(0, None)
            ax.set_ylim(bottom=0)
        else:
            ax.set_title("Error vs landing — no GT", fontsize=10)
            ax.axis("off")
        self.canvas_land.draw_idle()

    @staticmethod
    def _pass(clip, key):
        if clip.get("score_error_in") is None:
            return "—"
        return "✓" if clip.get(key) else "✗"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root_dir = Path(DEFAULT_ROOT)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "results").mkdir(exist_ok=True)
        self.last_run = None
        root.title("Dock-Diving Test Bench")
        root.geometry("1180x780")
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._refresh_visible_pane())
        self.ingest = IngestPane(self.nb, self)
        self.calibrate = CalibratePane(self.nb, self)
        self.analyze = AnalyzePane(self.nb, self)
        self.run = RunSuitePane(self.nb, self)
        self.results = ResultsPane(self.nb, self)
        self.compare = ComparePane(self.nb, self)
        self.accuracy = AccuracyPane(self.nb, self)
        for name, pane in (("Select clips", self.ingest), ("Calibrate", self.calibrate),
                           ("Analyze", self.analyze),
                           ("Run suite", self.run),
                           ("Results", self.results), ("Compare", self.compare),
                           ("Accuracy", self.accuracy)):
            self.nb.add(pane, text=name)
        self.on_clips_changed()

    def clip_dir(self, clip_id):
        return manifest_mod.clip_dir(self.root_dir, clip_id)

    def list_clips(self):
        return sorted(p.name for p in self.root_dir.iterdir()
                      if p.is_dir() and (p / "manifest.json").is_file())

    def on_clips_changed(self):
        self.calibrate.refresh()
        self.analyze.refresh()
        self.compare.refresh_clips()
        self.run.refresh()
        self.results.refresh()
        self.accuracy.refresh()

    def _refresh_visible_pane(self):
        """Give the newly shown pane a chance to load its (deferred) data."""
        try:
            pane = self.nb.nametowidget(self.nb.select())
        except (KeyError, tk.TclError):
            return
        fn = getattr(pane, "on_shown", None)
        if callable(fn):
            fn()

    def open_compare(self, clip_id):
        self.compare.load_clip(clip_id)
        self.nb.select(self.compare)


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
