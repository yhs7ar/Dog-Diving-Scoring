def _setup_capture_tab(self) -> None:
    import tkinter as tk
    from tkinter import ttk
    
    self.camera_selector = tk.StringVar(value="both")
    controls = ttk.Frame(self.tab_capture)
    controls.pack(side="top", fill="x")
    ttk.Radiobutton(controls, text="Camera 1", variable=self.camera_selector, value="cam1").pack(side="left")
    ttk.Radiobutton(controls, text="Camera 2", variable=self.camera_selector, value="cam2").pack(side="left")
    ttk.Radiobutton(controls, text="Both", variable=self.camera_selector, value="both").pack(side="left")
    
    self.video_label = ttk.Label(self.tab_capture)
    self.video_label.pack(expand=True, fill="both")
    
    self._update_video_stream()

def _update_video_stream(self) -> None:
    if self.tab_capture.winfo_ismapped() and self.zed_recorder:
        from PIL import Image, ImageTk
        import cv2
        import numpy as np
        
        frames = {}
        if self.camera_selector.get() in ["cam1", "both"]:
            f1 = self.zed_recorder._synthetic_frame(self.zed_recorder._frame_seq, 1200, 720, cam=1)
            frames["cam1"] = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        if self.camera_selector.get() in ["cam2", "both"]:
            f2 = self.zed_recorder._synthetic_frame(self.zed_recorder._frame_seq, 1200, 720, cam=2)
            frames["cam2"] = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        
        if frames:
            # Stack them or just show one based on selector
            combined = np.hstack(list(frames.values()))
            img = ImageTk.PhotoImage(image=Image.fromarray(combined))
            self.video_label.img = img
            self.video_label.configure(image=img)
    self.root.after(30, self._update_video_stream)
