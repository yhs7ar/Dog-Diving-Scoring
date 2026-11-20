#!/usr/bin/env python3
"""
Generate a synthetic stationary-camera "dock + pool" video with a known
ground-truth waterline, for testing detect_waterline.py without real footage.

Scene:
  * Static textured dock/wall above a gently curved, slanted waterline.
  * Water below with moving ripples + light flicker (temporal noise).
  * Sensor noise everywhere (so static pixels are not perfectly constant).
  * A flickering sun-glare spot on the dock and a "foliage" patch in the
    top-right corner -- high-variance regions that are NOT water.
  * Optional slow lighting drift over the whole clip (--drift).
  * After the calm segment, a "dog" (dark blob) jumps across and splashes.

Writes <out>.mp4 and <out>_truth.json (waterline y for every x, calm range).
"""

import argparse
import json

import cv2
import numpy as np

W, H, FPS = 640, 360, 30


def truth_y(x: np.ndarray) -> np.ndarray:
    return 170 + 0.08 * (x - W / 2) + 6 * np.sin(x / 60.0)


def make(out: str, calm: int = 150, active: int = 90, drift: float = 0.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    xs = np.arange(W)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wl = truth_y(xs)
    water = yy >= wl[None, :]

    # Static dock: planks + fixed random texture.
    dock = 120 + 25 * ((xx // 40) % 2) + cv2.GaussianBlur(
        rng.normal(0, 20, (H, W)).astype(np.float32), (0, 0), 2)
    dock[np.abs(yy - wl[None, :] + 8) < 3] = 70  # dark dock edge just above the water
    base_water = 90 + 30 * (yy / H)

    n = calm + active
    vw = cv2.VideoWriter(out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for t in range(n):
        ph = t / FPS
        ripples = (6 * np.sin(0.09 * xx + 0.21 * yy - 2.1 * ph)
                   + 5 * np.sin(-0.05 * xx + 0.33 * yy - 3.3 * ph)
                   + 4 * np.sin(0.15 * xx - 0.11 * yy + 1.7 * ph))
        frame = np.where(water, base_water + ripples, dock)

        # Sun glare on the dock: bright, flickering, saturating spot.
        g = np.exp(-((xx - 110) ** 2 + (yy - 60) ** 2) / (2 * 12 ** 2))
        frame = frame + g * (150 + 80 * np.sin(7 * ph + rng.normal(0, 0.5)))
        # Foliage: flickering texture in the top-right corner.
        fol = (xx > 540) & (yy < 70)
        frame[fol] += 15 * rng.normal(0, 1, fol.sum())

        # Slow lighting drift (e.g. clouds), applied to the whole frame.
        frame = frame + drift * t / max(n - 1, 1)

        # The dog: enters after the calm segment, runs off the dock and splashes.
        if t >= calm:
            k = t - calm
            cx, cy = 80 + 6 * k, 110 + min(k, 30) * 3.0
            frame[((xx - cx) / 35) ** 2 + ((yy - cy) / 18) ** 2 < 1] = 30
            if k > 20:
                sp = ((xx - cx) ** 2 + (yy - wl[np.clip(int(cx), 0, W - 1)]) ** 2) < (k - 20) ** 2 * 4
                frame[sp] = 230 - 2 * (k - 20)

        frame = frame + rng.normal(0, 2.0, (H, W))  # sensor noise
        f8 = np.clip(frame, 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(f8, cv2.COLOR_GRAY2BGR))
    vw.release()

    with open(out + "_truth.json", "w") as f:
        json.dump({"waterline_y": wl.tolist(), "calm_frames": [0, calm], "fps": FPS,
                   "drift": drift, "water_side": "below"}, f)
    print(f"wrote {out}.mp4 ({n} frames, calm=[0,{calm}), drift={drift})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--calm", type=int, default=150)
    ap.add_argument("--active", type=int, default=90)
    ap.add_argument("--drift", type=float, default=0.0, help="total gray-level drift over the clip")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.out, a.calm, a.active, a.drift, a.seed)
