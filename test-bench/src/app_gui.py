"""Test-bench GUI — six panes over the modules in ``test_bench`` (M7).

Ingest / Analyze / Ground truth / Run suite / Results / Compare.  Every pane
reads and writes the same ``analyzed-clips/`` schema the CLI uses, so the two
interfaces are interchangeable (ROADMAP §7, §8).

Launched by ``run_app.cmd`` / ``run_app.bash`` (which resolve the interpreter);
direct: ``python src/app_gui.py`` from the repo root.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_bench import ground_truth as gt_mod       # noqa: E402
from test_bench import ingest as ingest_mod         # noqa: E402
from test_bench import manifest as manifest_mod     # noqa: E402
from test_bench import plane_draw as plane_draw_mod # noqa: E402
from test_bench import suite as suite_mod           # noqa: E402
from test_bench.frame_view import ZOOM_STEPS, FrameView  # noqa: E402
from dog_diving import trajectory as trajectory_mod  # noqa: E402

DEFAULT_ROOT = SRC.parent / "analyzed-clips"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_frame(clip_dir: Path, man: dict, frame_index: int):
    """The BGR frame at ``frame_index`` for a clip, or None."""
    clip_file = man.get("source", {}).get("clip_file")
    if not clip_file:
        kind = man.get("source", {}).get("kind", "mp4")
        clip_file = f"clip.{'svo2' if kind == 'svo2' else 'mp4'}"
    cap = cv2.VideoCapture(str(clip_dir / clip_file))
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
        ids = self.app.list_clips()
        self.clip_box["values"] = ids
        if ids and self.clip_var.get() not in ids:
            self.clip_var.set(ids[0])
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
        row = ttk.Frame(self)
        row.pack(fill="x", padx=8, pady=8)
        self.path_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.path_var, width=55).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self._browse).pack(side="left", padx=4)
        ttk.Button(row, text="Ingest", command=self._ingest).pack(side="left")
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
        p = filedialog.askopenfilename(title="Clip",
            filetypes=[("video", "*.mp4 *.mov *.m4v *.avi *.svo2"), ("all", "*.*")])
        if p:
            self.path_var.set(p)

    def _ingest(self):
        src = self.path_var.get().strip()
        if not src:
            return
        try:
            man = ingest_mod.ingest_clip(src, self.app.root_dir,
                                         clip_id=self.id_var.get().strip() or None)
            m = man["meta"]
            self._log(f"ingested {man['clip_id']}  ({m.get('width')}x{m.get('height')} "
                      f"@ {m.get('fps')} fps, {m.get('n_frames')} frames, {m.get('codec')})")
            self.app.on_clips_changed()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("ingest", str(e))


class AnalyzePane(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        row = ttk.Frame(self)
        row.pack(fill="x", padx=8, pady=8)
        ttk.Label(row, text="Clip:").pack(side="left")
        self.clip_var = tk.StringVar()
        self.clip_box = ttk.Combobox(row, textvariable=self.clip_var, width=40, state="readonly")
        self.clip_box.pack(side="left", padx=4)

        row2 = ttk.Frame(self)
        row2.pack(fill="x", padx=8)
        ttk.Label(row2, text="Model:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(row2, textvariable=self.model_var, width=40)
        self.model_box.pack(side="left", padx=4)
        ttk.Button(row2, text="Assign plane…", command=self._assign_plane).pack(side="left", padx=8)
        ttk.Button(row2, text="Re-draw plane…", command=self._redraw_plane).pack(side="left")
        self.plane_lbl = ttk.Label(row2, text="plane: none")
        self.plane_lbl.pack(side="left")
        ttk.Button(row2, text="Analyze", command=self._analyze).pack(side="right")

        self.status = ttk.Label(self, text="", foreground="#0369a1")
        self.status.pack(fill="x", padx=8, pady=4)
        self.out = tk.Text(self, height=10, state="disabled")
        self.out.pack(fill="both", expand=True, padx=8, pady=8)

    def refresh(self):
        self.clip_box["values"] = self.app.list_clips()
        models = sorted(p.name for p in (self.app.root_dir.parent / "models").glob("*.pt"))
        self.model_box["values"] = models
        if models and not self.model_var.get():
            self.model_var.set(models[0])

    def _log(self, msg):
        self.out.configure(state="normal"); self.out.insert("end", msg + "\n"); self.out.configure(state="disabled")

    def _assign_plane(self):
        p = filedialog.askopenfilename(title="plane document",
                                       filetypes=[("plane", "*.json"), ("all", "*.*")])
        if not p or not self.clip_var.get():
            return
        cdir = self.app.clip_dir(self.clip_var.get())
        manifest_mod.write_json(cdir / "plane.json", json.loads(Path(p).read_text(encoding="utf-8")))
        self.plane_lbl.config(text=f"plane: {Path(p).name}")
        self.app.on_clips_changed()

    def _redraw_plane(self):
        cid = self.clip_var.get()
        if not cid:
            return
        PlaneDrawDialog(self.app, cid)

    def _analyze(self):
        cid = self.clip_var.get()
        if not cid or getattr(self, "_busy", False):
            return
        from dog_diving import detect as detect_mod
        from dog_diving import measure as measure_mod
        cdir = self.app.clip_dir(cid)
        man = manifest_mod.read_manifest(cdir)
        if man.get("source", {}).get("kind") == "svo2":
            messagebox.showerror("analyze", "SVO2 analyze needs the ZED X SDK (not wired)")
            return
        model = self.app.root_dir.parent / "models" / self.model_var.get()
        cfg = detect_mod.DetectConfig(model_path=model)
        clip = cdir / man["source"]["clip_file"]
        self._busy = True
        self.status.config(text="analyzing…")

        def _after(cb):
            try:
                self.app.root.after(0, cb)
            except tk.TclError:
                pass  # window gone; the daemon thread exits quietly

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
                meas = None
                if (cdir / "plane.json").is_file():
                    plane = json.loads((cdir / "plane.json").read_text(encoding="utf-8"))
                    meas = measure_mod.plane_distance(traj, plane)
                    man["plane"]["present"] = True
                manifest_mod.write_manifest(man, cdir)
                _after(lambda: self._analyze_done(cid, det, meas))
            except Exception as e:  # noqa: BLE001
                _after(lambda err=str(e): self._analyze_fail(err))

        threading.Thread(target=worker, daemon=True).start()

    def _analyze_done(self, cid, det, meas):
        self._busy = False
        self.status.config(text="done")
        if meas and meas.get("total_inches") is not None:
            self._log(f"{cid}: {meas['total_inches']:.2f} in  ({meas['display']}) @ frame {meas['frame']}")
        else:
            self._log(f"{cid}: {det.n_detections} detections, {det.n_accepted} accepted "
                      f"(no plane — distance null)")
        self.app.on_clips_changed()

    def _analyze_fail(self, err):
        self._busy = False
        self.status.config(text="failed")
        messagebox.showerror("analyze", err)


class GroundTruthPane(_FramePane):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.pending = None   # (x_px, y_px)
        self.view.on_click(self._on_click)
        ctl = ttk.Frame(self)
        ctl.pack(fill="x", padx=8, pady=4)
        ttk.Label(ctl, text="Judge distance (in):").pack(side="left")
        self.judge_var = tk.StringVar()
        ttk.Entry(ctl, textvariable=self.judge_var, width=10).pack(side="left", padx=4)
        ttk.Button(ctl, text="Save GT + judge", command=self._save).pack(side="left", padx=8)
        self.gt_lbl = ttk.Label(ctl, text="GT: none")
        self.gt_lbl.pack(side="right")

    def _on_click(self, ix, iy):
        self.pending = (ix, iy)
        self._redraw()

    def _redraw(self):
        overlays = []
        if self.pending:
            overlays.append(("marker", (self.pending[0], self.pending[1], 6.0), "yellow"))
        self.view.set_overlays(overlays)

    def _save(self):
        cid = self.clip_var.get()
        if not cid or not self.pending:
            return
        cdir = self.app.clip_dir(cid)
        plane_path = cdir / "plane.json"
        if not plane_path.is_file():
            messagebox.showerror("ground truth", "assign a plane first (Analyze pane)")
            return
        plane = json.loads(plane_path.read_text(encoding="utf-8"))
        judge = None
        if self.judge_var.get().strip():
            try:
                judge = float(self.judge_var.get().strip())
            except ValueError:
                messagebox.showerror("judge", "judge distance must be a number (inches)")
                return
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
        self.gt_lbl.config(text=f"GT: {gt['gt_in']:.2f} in" + (f"  judge: {judge:.2f} in" if judge is not None else ""))
        self.app.on_clips_changed()


class RunSuitePane(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        left = ttk.Frame(self); left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        ttk.Label(left, text="Prepared clips (detected + GT):").pack(anchor="w")
        self.clip_list = tk.Listbox(left, selectmode="extended", exportselection=False)
        self.clip_list.pack(fill="both", expand=True)
        right = ttk.Frame(self); right.pack(side="left", fill="y", padx=8, pady=8)
        ttk.Label(right, text="n_points:").pack(anchor="w")
        self.n_var = tk.StringVar(value="4,6,8,12,20,all")
        ttk.Entry(right, textvariable=self.n_var, width=16).pack()
        ttk.Label(right, text="selections:").pack(anchor="w")
        self.sel_var = tk.StringVar(value="last_n,all,ransac_inliers")
        ttk.Entry(right, textvariable=self.sel_var, width=16).pack()
        ttk.Button(right, text="Run suite", command=self._run).pack(fill="x", pady=8)
        self.status = ttk.Label(right, text="", wraplength=180)
        self.status.pack()

    def refresh(self):
        self.clip_list.delete(0, "end")
        for cid in self.app.list_clips():
            cdir = self.app.clip_dir(cid)
            if (cdir / "detections.json").is_file() and (cdir / "ground_truth.json").is_file():
                self.clip_list.insert("end", cid)

    def _run(self):
        sel_idx = self.clip_list.curselection()
        if not sel_idx:
            messagebox.showinfo("run suite", "select at least one prepared clip")
            return
        ids = [self.clip_list.get(i) for i in sel_idx]
        n_vals = [v.strip() for v in self.n_var.get().split(",") if v.strip()]
        sel_vals = [v.strip() for v in self.sel_var.get().split(",") if v.strip()]
        n_vals = [int(v) if v != "all" else "all" for v in n_vals]
        configs = suite_mod.grid_configs(n_vals, sel_vals)
        clips = []
        prov: dict = {}
        for cid in ids:
            cdir = self.app.clip_dir(cid)
            man = manifest_mod.read_manifest(cdir)
            recs = json.loads((cdir / "detections.json").read_text(encoding="utf-8"))
            points = trajectory_mod.assemble(recs)
            plane = json.loads((cdir / "plane.json").read_text(encoding="utf-8"))
            gt = json.loads((cdir / "ground_truth.json").read_text(encoding="utf-8"))
            clips.append({"clip_id": cid, "points": points, "plane_doc": plane, "gt": gt,
                          "frame_size": [man["meta"]["width"], man["meta"]["height"]]})
            prov[cid] = suite_mod.clip_provenance(man, plane, gt)
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
        cols = ("clip", "method", "n_points", "selection", "pixel_px", "score_in", "judge_in", "status")
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        for c, w in zip(cols, (260, 70, 60, 90, 70, 70, 70, 90)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
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
                    _fmt(r.get("judge_error_in")), status))
        av = doc.get("averages", {}).get("overall", {})
        self.avg_lbl.config(text="averages — pixel: %s px   score: %s in   judge: %s in" % (
            _fmt(av.get("pixel_error_px")), _fmt(av.get("score_error_in")), _fmt(av.get("judge_error_in"))))

    def _clip_stale(self, cid, recorded):
        cdir = self.app.clip_dir(cid)
        try:
            man = manifest_mod.read_manifest(cdir)
            plane_p = cdir / "plane.json"
            gt_p = cdir / "ground_truth.json"
            plane = json.loads(plane_p.read_text(encoding="utf-8")) if plane_p.is_file() else None
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
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.run_vars = []     # (tk.BooleanVar, run_dict)
        self.gt = None
        self.ctl = ttk.Frame(self)
        self.ctl.pack(fill="x", padx=8, pady=4)

    def load_clip(self, clip_id):
        self.clip_var.set(clip_id)
        self._on_clip()
        self._rebuild_overlay_toggles()

    def _rebuild_overlay_toggles(self):
        for w in self.ctl.winfo_children():
            w.destroy()
        cid = self.clip_var.get()
        if not cid:
            return
        gt_path = self.app.clip_dir(cid) / "ground_truth.json"
        self.gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else None
        self.run_vars = []
        for doc in self._run_docs_for(cid):
            for r in doc["runs"]:
                v = tk.BooleanVar(value=True)
                cb = ttk.Checkbutton(self.ctl, text=f"{r['method']} n={r.get('n_points')} {r.get('selection')}",
                                     variable=v, command=self._redraw)
                cb.pack(side="left", padx=4)
                self.run_vars.append((v, r))
        self._redraw()

    def _run_docs_for(self, cid):
        rdir = self.app.root_dir / "results"
        if not rdir.is_dir():
            return []
        out = []
        for p in sorted(rdir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            doc = json.loads(p.read_text(encoding="utf-8"))
            for row in doc.get("rows", []):
                if row["clip_id"] == cid:
                    out.append({"runs": row["runs"], "gt": row.get("gt", {})})
        return out

    def _redraw(self):
        overlays = []
        if self.gt and self.gt.get("x_px") is not None:
            overlays.append(("marker", (self.gt["x_px"], self.gt["y_px"], 7.0), "red"))
        for i, (v, r) in enumerate(self.run_vars):
            if not v.get():
                continue
            colour = ("cyan", "magenta", "lime", "orange", "yellow")[i % 5]
            sc = r["score"]
            if sc.get("entry_px"):
                overlays.append(("marker", (sc["entry_px"][0], sc["entry_px"][1], 5.0), colour))
            curve = (sc.get("fit") or {}).get("curve")
            if curve and len(curve) >= 2:
                overlays.append(("curve", curve, colour))
        self.view.set_overlays(overlays)

    def _on_frame(self):
        super()._on_frame()
        self._redraw()


class PlaneDrawDialog(tk.Toplevel):
    """Click the four pool corners (00, 10, 11, 01) to re-fit the plane homography."""

    _ORDER = ("00", "10", "11", "01")
    _COLOURS = ("#ff5555", "#ffaa00", "#55ff55", "#55aaff")

    def __init__(self, app, clip_id):
        super().__init__(app.root)
        self.app = app
        self.clip_id = clip_id
        self.title(f"Re-draw plane — {clip_id}")
        self.corners: dict[str, list[float]] = {}
        self.idx = 0

        cdir = app.clip_dir(clip_id)
        man = manifest_mod.read_manifest(cdir)
        self.plane_p = cdir / "plane.json"
        if not self.plane_p.is_file():
            messagebox.showerror("re-draw", "no plane.json — assign a plane first", parent=self)
            self.destroy()
            return
        self.existing = json.loads(self.plane_p.read_text(encoding="utf-8"))
        frame = _load_frame(cdir, man, 0)
        if frame is None:
            messagebox.showerror("re-draw", "could not read a frame from the clip", parent=self)
            self.destroy()
            return

        self.inst = ttk.Label(self, text=f"click corner {self._ORDER[0]}")
        self.inst.pack(fill="x", padx=8, pady=4)
        self.canvas = tk.Canvas(self, background="black")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self.view = FrameView(self.canvas, max_w=900, max_h=560)
        self.view.bind_zoom()
        self.view.on_click(self._on_click)
        self.view.show(frame)

        ov = []
        for k, col in zip(self._ORDER, self._COLOURS):
            c = (self.existing.get("plane", {}).get("corners") or {}).get(k)
            if c:
                ov.append(("marker", (c[0], c[1], 5.0), col))
        self.view.set_overlays(ov)

    def _on_click(self, ix, iy):
        if self.idx >= len(self._ORDER):
            return
        k = self._ORDER[self.idx]
        self.corners[k] = [ix, iy]
        self.idx += 1
        self.view.set_overlays([
            ("marker", (v[0], v[1], 6.0), self._COLOURS[self._ORDER.index(kk)])
            for kk, v in self.corners.items()])
        if self.idx < len(self._ORDER):
            self.inst.config(text=f"click corner {self._ORDER[self.idx]}")
        else:
            self._finish()

    def _finish(self):
        try:
            redrawn = plane_draw_mod.redraw_plane(self.existing, self.corners)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("re-draw", str(e), parent=self)
            return
        manifest_mod.write_json(self.plane_p, redrawn)
        cdir = self.app.clip_dir(self.clip_id)
        gt_p = cdir / "ground_truth.json"
        if gt_p.is_file():
            try:
                gt = json.loads(gt_p.read_text(encoding="utf-8"))
                gt2 = gt_mod.set_ground_truth(redrawn, frame=int(gt["frame"]),
                                               x_px=float(gt["x_px"]), y_px=float(gt["y_px"]),
                                               judge_in=gt.get("judge_in"))
                gt_mod.write_ground_truth(gt2, cdir)
            except Exception:  # noqa: BLE001
                pass  # GT stays; it will be flagged stale on load
        self.app.on_clips_changed()
        self.destroy()


def _fmt(v):
    return "" if v is None else f"{v:.2f}"


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
        self.ingest = IngestPane(self.nb, self)
        self.analyze = AnalyzePane(self.nb, self)
        self.gt = GroundTruthPane(self.nb, self)
        self.run = RunSuitePane(self.nb, self)
        self.results = ResultsPane(self.nb, self)
        self.compare = ComparePane(self.nb, self)
        for name, pane in (("Ingest", self.ingest), ("Analyze", self.analyze),
                           ("Ground truth", self.gt), ("Run suite", self.run),
                           ("Results", self.results), ("Compare", self.compare)):
            self.nb.add(pane, text=name)
        self.on_clips_changed()

    def clip_dir(self, clip_id):
        return manifest_mod.clip_dir(self.root_dir, clip_id)

    def list_clips(self):
        return sorted(p.name for p in self.root_dir.iterdir()
                      if p.is_dir() and (p / "manifest.json").is_file())

    def on_clips_changed(self):
        self.analyze.refresh()
        self.gt.refresh_clips()
        self.compare.refresh_clips()
        self.run.refresh()
        self.results.refresh()

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
