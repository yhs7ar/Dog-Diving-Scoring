# Waterline detection — lite version

`detect_waterline_lite.py` finds the waterline in a short clip from a
**stationary camera** and fits a straight line to it. It is a slimmed-down
version of `detect_waterline.py`: it keeps only what's needed to get a
reliable line, with a handful of options and four output files.

## How it works

1. **Watch each pixel over time.** On the wall or dock, brightness barely
   changes. On the water, it keeps flickering from ripples and reflections.
2. **Make a noise map.** Each pixel is scored by how much it flickered. Slow
   lighting changes, like a cloud passing, are removed first so they don't
   count as flicker. Water comes out bright; static structure comes out dark.
3. **Split into water / not water.** A cut-off is picked automatically (Otsu's
   method), or you can set it yourself. Small specks are removed, small gaps
   are filled, and only the biggest water area is kept.
4. **Find the edge.** In each column, the first water pixel from the top is
   the waterline. The result is lightly smoothed.
5. **Fit a straight line.** A line is fitted, points far from it are thrown
   out, and the line is refitted. This stops things like people moving behind
   the wall from pulling the line off.

## Requirements and assumptions

- **Stationary camera.** If it moves or gets bumped, rerun the script.
- **Calm clip.** Nothing should move in front of the waterline: no dog,
  splash or people at the water's edge. About **1 second** of footage has
  been enough in testing.
- **Water below the line.** The camera looks across the water at a wall or
  dock, so the water is in the lower part of the image.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# By frame number: the first 30 frames
python detect_waterline_lite.py --video videos/testvideo001.mp4 --end-frame 30 \
    --roi 0,200,1920,800 --output-dir out_lite/

# By time: 12.5 s to 13.5 s
python detect_waterline_lite.py --video videos/testvideo001.mp4 --start-time 12.5 --end-time 13.5 \
    --roi 0,200,1920,800 --output-dir out_lite/
```

If `python` isn't recognised on Windows, use `py` instead.

## Options

| Option | Default | What it does |
|---|---|---|
| `--video` | required | The input video. |
| `--start-frame` | 0 | First frame to use. |
| `--end-frame` | end of video | Stop *before* this frame. |
| `--start-time` | – | Start in seconds; use instead of `--start-frame`. |
| `--end-time` | – | End in seconds; use instead of `--end-frame`. |
| `--roi` | whole frame | Only search inside this box, `x1,y1,x2,y2` in pixels (top-left and bottom-right corners). |
| `--threshold` | automatic | Cut-off (0–255) on the noise map for "this is water". Leave it out to pick it automatically. |
| `--output-dir` | `out_lite` | Where results are saved. |

You can mix frames and seconds (e.g. `--start-time 5 --end-frame 200`). You
can't give both forms for the same end; the script stops with an error.

### Choosing the ROI
Draw the box around the waterline and the water below it. Leave out anything
else that flickers: sky, trees, flags, spectators. Boundaries on the edge of
the box are ignored, so the box can cut through the water without creating a
false waterline.

### Choosing a threshold
The automatic cut-off usually works. The value used is printed on every run.
If it fails, open `noise_map.png` and `mask.png`, then rerun with `--threshold`
near the printed value:
- **Raise it** if parts of the wall are being counted as water.
- **Lower it** if parts of the water are being missed.

## Outputs

| File | Contents |
|---|---|
| `overlay.png` | The middle frame of the clip. Green = waterline points used for the line; red = points thrown out; magenta = fitted line; cyan = ROI. |
| `noise_map.png` | Bright = flickered a lot (water); dark = static. |
| `mask.png` | White = pixels classed as water, after cleanup. |
| `waterline.json` | All results, described below. |

### `waterline.json`

| Field | Meaning |
|---|---|
| `video`, `roi`, `fps` | The inputs used. |
| `frame_range`, `time_range_s` | The clip that was analysed, as `[start, end)` in frames and in seconds. |
| `threshold` | The cut-off used on the 0–255 noise map. |
| `line.slope`, `line.intercept` | The fitted line: `y = slope * x + intercept`, in pixels (x right, y down). |
| `line.angle_deg` | Tilt of the line. 0 = level; positive = slopes down to the right. |
| `line.endpoints` | Where the line starts and ends, over the range of points used. |
| `line.rms_residual_px` | Average distance of the used points from the line. Small = the edge is close to straight. |
| `line.inlier_fraction` | Share of points used for the line (the rest were thrown out). |
| `points_xy` | Every detected waterline point, `[x, y]` in pixels. |
| `inlier` | For each point, `true` if it was used for the line. |

To get the waterline height at any column `x`, or a point `N` pixels above it:

```python
import json
line = json.load(open("out_lite/waterline.json"))["line"]
y = line["slope"] * x + line["intercept"]   # on the waterline
y_above = y - N                             # N pixels above it (y grows downward)
```

## Reading the console output

```
Frames 0-30 (0.00-1.00 s at 30.00 fps), threshold 89/255
Waterline: 1808 points, 97% used for the line
Line: y = -0.01517 * x + 634.23  (angle -0.87 deg, RMS 7.28 px)
```

- **Percentage used.** If it's below 70%, the script prints a warning. Look
  at `overlay.png` (red = thrown out), then tighten `--roi` or pick a calmer
  clip.
- **RMS.** A few pixels is normal, because ripples make the edge slightly
  wavy. A large value means the edge isn't straight, or something else was
  detected as water.

## Tuning constants

The less commonly changed settings are constants at the top of the script
rather than command-line options:

| Constant | Default | What it does |
|---|---|---|
| `PRE_BLUR` | 3 | Blurs each frame slightly to smooth out camera sensor speckle. |
| `POST_BLUR` | 5 | Blurs the noise map so the water area is smooth. |
| `OPEN_SIZE` | 5 | Removes bright specks smaller than this (px). |
| `CLOSE_SIZE` | 15 | Fills dark gaps in the water smaller than this (px). |
| `SMOOTH` | 9 | Smooths the waterline over this many neighbouring columns. |
| `OUTLIER_PX` | 3.0 | Points closer than this to the line are never thrown out. |

## What the full version has that this one doesn't

Use `detect_waterline.py` (see `README.md`) for these:
- Other noise map methods and a side-by-side comparison
- Water above, left of or right of the line
- Curved shorelines (contour tracing)
- Polygon ROI
- Downscaling and frame skipping for speed
- Stability check across the clip
- Overlay video
- Glare and lighting-drift warnings

## Tests

```bash
python -m pytest tests/test_detect_waterline_lite.py -q
```

The tests cover drift removal, outlier rejection, ignoring the ROI edge, and
an end-to-end run on a synthetic video with a known waterline. That run also
checks that selecting the clip by seconds or by frames gives the same result.
