# Waterline detection from temporal noise

Finds the waterline in video from a **stationary camera**. It measures how much
each pixel's brightness changes over time. Rigid structure (dock, pool wall,
deck) stays nearly constant. Water keeps flickering from ripples and
reflections, even when it looks calm. Thresholding that "temporal noise map"
separates water from structure, and the boundary between them is the waterline.

The script is meant to run on a **calm segment**, before any dog jumps in. A
dog or splash adds a lot of motion, which corrupts the statistics. The output
is a *baseline* waterline position, not a tracker that works through a splash.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Pick the calm segment (frames 0-299 here) and a box around the dock/water edge
python detect_waterline.py --video dock.mp4 --start-frame 0 --end-frame 300 \
    --roi 0,200,1920,800 --output-dir out/

# 2. Open out/overlay.png and out/noise_map_comparison.png and check the result
# 3. If glare or reflections are picked up, tighten --roi or set a manual threshold
python detect_waterline.py --video dock.mp4 --start-frame 0 --end-frame 300 \
    --roi 0,200,1920,800 --threshold-mode manual --threshold-value 140 --output-dir out/
```

Try it without real footage. This generates a synthetic dock/pool video with a
known waterline, sun glare, foliage flicker, lighting drift and a "dog" after
frame 150:

```bash
python tests/make_synthetic_video.py --out synthetic --drift 30
python detect_waterline.py --video synthetic.mp4 --end-frame 150 --roi 0,100,640,300 --clip --clip-extra 90 --output-dir out_syn/
```

Run the tests with `python -m pytest tests -q`.

## Plain-language guide

This section explains each idea without the math. The detailed tables in
[Parameters](#parameters) give the exact defaults.

### The basic idea

Pick one pixel and watch its brightness over a few seconds of video:

* On the **dock**, the brightness hardly changes. The wood isn't moving.
* On the **water**, it keeps wobbling up and down as ripples and reflections
  pass through.

A **noise map** is an image where each pixel is coloured by *how much it
wobbled*. Water comes out bright and the dock comes out dark. The waterline is
where bright meets dark.

### The four noise map methods (`--method`)

These are four ways of scoring "how much did this pixel wobble?"

* **`variance`**: how far, on average, the pixel strays from its own average
  brightness. It is the simplest score. Its weakness is that if the whole
  scene slowly gets brighter (the sun comes out), every pixel "strays from its
  average", so the dock starts to look like water too.
* **`std`** (standard deviation): the same thing as `variance`, just
  square-rooted so the numbers are in normal brightness units. It finds the
  same waterline; it is there because some people find it easier to read.
* **`detrended`** (the default): like `variance`, but it first ignores any
  slow, steady brightening or darkening, then measures the wobble that's
  left. Think of drawing a straight trend line through each pixel's
  brightness over time and only counting the bumps around that line. This
  keeps the dock dark even if the sun comes out.
* **`diff`** (frame differencing): compares each frame to the one just before
  it and counts how much it changed. Slow lighting changes barely register
  between neighbouring frames, so it also ignores them. But slow, gentle
  ripples barely register either, so calm water can look almost as still as
  the dock. `--diff-lag` compares frames further apart (e.g. 4 frames
  instead of 1), which helps it notice slower ripples.

**Which one to use:** start with `detrended`. Every run also saves all four
maps side by side (`noise_map_comparison.png`), so you can see which one shows
the water most clearly on your video.

### Getting the right part of the video

* **`--start-frame` / `--end-frame`** (or **`--start-time` / `--end-time`**):
  choose the calm stretch *before* the dog jumps. Anything that moves in this
  stretch (a dog, a person, a splash) will be mistaken for water.
* **`--step`**: skip frames to go faster. `2` means "use every other frame".
* **`--scale`**: shrink the video before processing to go faster. `0.5` means
  half width and half height, about 4× faster. The waterline coordinates are
  still reported at full size.
* **`--color`**: load the video in colour. This only makes the output
  pictures colour; the detection itself always uses brightness.

### Cleaning up the noise map

* **`--pre-blur`**: blurs each frame slightly before measuring. Every camera
  has a little random speckle on every pixel, even on the dock. A slight blur
  smooths that speckle out but keeps ripples, which are bigger than one pixel.
* **`--post-blur`**: blurs the finished noise map slightly, so the water shows
  up as one smooth bright area instead of a patchy one.

### Deciding what counts as water

* **`--roi`** (region of interest): a box or shape telling the script "only
  look here". Use it to leave out things that also flicker but aren't the
  pool, such as sun glare on the dock, trees or the sky. A box is
  `x1,y1,x2,y2` (top-left and bottom-right corners in pixels).
* **`--threshold-mode`**: how to decide "bright enough to be water".
  * `otsu` picks the cut-off automatically by looking for the natural gap
    between dark (pool wall/dock) and bright (water) pixels.
  * `manual` lets you choose the cut-off yourself with `--threshold-value`.
* **`--threshold-value`**: your cut-off, from 0 (darkest) to 255 (brightest)
  on the noise map. Raise it if dock areas are being called water; lower it
  if parts of the water are being missed. Each run prints the automatic
  value, so start from that.
* **`--open-size`**: removes small bright specks that aren't water, like a
  single twinkling highlight on the dock. Bigger value = removes bigger specks.
* **`--close-size`**: fills small dark gaps inside the water, where the
  surface happened to be still for a moment. Bigger value = fills bigger gaps.

After this cleanup, only the single biggest bright area is kept as the water.

### Turning the water area into a line

* **`--line-method`**: how to trace the edge.
  * `scan` goes across the image one column at a time and finds where the
    water starts. It is best when the camera looks across the pool at the dock.
  * `contour` traces around the outline of the water area. Use it when the
    edge curves around, e.g. a pool corner seen from above.
* **`--water-side`**: tells the script which side of the line the water is on
  (`below`, `above`, `left`, `right`). `auto` works it out, but you can set it
  if it guesses wrong.
* **`--smooth`**: evens out small zig-zags in the line. Bigger = smoother.
  It still follows real bends; it does *not* make the line straight.
* **`--fit line`**: draws the single straight line that best matches the
  detected edge, and reports its equation and angle. Use it when the real
  edge is straight (e.g. a dock or pool wall). A few bad points, like a
  glare blob, barely pull on the fit. The script also reports how far the
  detected points are from the straight line. If the distance is large, the
  edge isn't really straight, or a wide-angle lens is bending it.

### Checking the result

* **`--stability-window`**: the script repeats the detection on several
  shorter slices of the calm stretch. If the method is working, the waterline
  should land in almost the same place every time. This sets how many frames
  each slice has. Slices need to be long enough to see a full ripple go past;
  too short and the line jumps around for no real reason. `0` skips the check.
* **`--stability-stride`**: how far apart the slices start. The default
  overlaps them by half.
* **`--clip`**: saves a video with the detected waterline drawn on it, so you
  can watch it and check it sits on the real edge.
* **`--clip-extra`**: keeps the clip going for this many frames after the calm
  stretch, so you can see the line while the dog is in the water.

## Parameters

### Frame range
Use these to pick out the calm segment.

| Flag | Default | Meaning |
|---|---|---|
| `--video` | required | Input video. |
| `--start-frame` / `--end-frame` | whole video | Frame range `[start, end)`. Takes precedence over times. |
| `--start-time` / `--end-time` | – | Same range, in seconds. |
| `--step` | 1 | Use every Nth frame. |
| `--scale` | 1.0 | Resize factor for processing, e.g. `0.5` for 4K footage. Output coordinates are always in **original** pixels. |
| `--color` | off | Load colour frames. Noise is still measured on intensity; colour is only used for the overlays. |

### Noise map
| Flag | Default | Meaning |
|---|---|---|
| `--method` | `detrended` | Statistic used for segmentation: `variance`, `std`, `detrended` or `diff` (see below). All four maps are always saved so you can compare them. |
| `--diff-lag` | 1 | For `diff`: mean \|f[t] − f[t−k]\|. Use 3–5 if the ripples are slow. |
| `--pre-blur` | 3 | Gaussian blur applied to each frame first, to suppress per-pixel sensor and compression noise. `0` turns it off. | 
| `--post-blur` | 5 | Gaussian blur applied to the noise map before thresholding. |

### Segmentation
| Flag | Default | Meaning |
|---|---|---|
| `--roi` | whole frame | `x1,y1,x2,y2` box, or `x1,y1;x2,y2;x3,y3;...` polygon. Only this region is thresholded and searched. Boundaries that coincide with the ROI edge are ignored, because they are crop artifacts, not waterline. |
| `--threshold-mode` | `otsu` | `otsu` picks the threshold automatically from ROI pixels. `manual` uses `--threshold-value`. |
| `--threshold-value` | – | Threshold on the 0–255 normalized noise map. The Otsu value is printed on every run, so start near it. |
| `--open-size` / `--close-size` | 5 / 15 | Morphological kernels. Opening removes specks on structure; closing fills holes in the water. The largest blob is kept as the water. |

### Waterline
| Flag | Default | Meaning |
|---|---|---|
| `--line-method` | `scan` | `scan` returns one point per column (or per row). It is best when the camera looks across the water at the dock and the line is single-valued. `contour` uses `cv2.findContours` on the water blob, for curved or closed shorelines. |
| `--water-side` | `auto` | Which side of the line the water is on: `below`, `above`, `left`, `right`, or `auto` (inferred). Set it explicitly if `auto` guesses wrong. |
| `--smooth` | 9 | Median filter window for `scan` output. |
| `--fit` | `none` | `line` also fits one straight line to the detected points with `cv2.fitLine` and a Huber loss, which is robust to outliers. Results go in `waterline.json` → `line_fit`: equation, angle, endpoints and residuals. The line is drawn in magenta on `overlay.png` and the clip. The detected points themselves are still saved unchanged. |

### Validation
| Flag | Default | Meaning |
|---|---|---|
| `--stability-window` | max(2 s, segment/3) | Length of the sliding sub-windows used to measure waterline jitter. `0` skips the check. |
| `--stability-stride` | window/2 | Step between sub-windows. |
| `--clip` | off | Write `overlay_clip.mp4`: the baseline line in green, and the current sub-window's line in yellow. |
| `--clip-extra` | 0 | Also render N frames after the segment, to see the baseline once the dog is in. |

## Outputs (`--output-dir`)

| File | Contents |
|---|---|
| `waterline.json` | `points_xy` in original pixels, plus all parameters, the threshold used, method comparison, stability stats, drift and glare diagnostics, and warnings. |
| `waterline.npy` | `(K, 2)` float32 array of x, y. |
| `noise_map_<method>.png` / `_color.png` | Normalized noise maps (bright = more temporal change). |
| `noise_map_comparison.png` | All four methods side by side, labelled with a separability score. |
| `mask_raw.png` / `mask_clean.png` | Thresholded mask before and after cleanup. Check `mask_raw.png` for glare blobs. |
| `overlay.png` | Waterline (green), ROI (cyan) and, with `--fit line`, the fitted straight line (magenta) on the middle frame of the segment. |
| `stability_lines.png` | The waterline from every sub-window, overlaid. They should sit on top of each other. |
| `overlay_clip.mp4` | Only with `--clip`. |

## Reading the console output

* **Method separability (Fisher ratio).** How cleanly each map splits into
  two classes inside the ROI; higher is better. A high score with a tiny
  `water_fraction` means the split isolated glare rather than the water. The
  script flags this.
* **Lighting drift.** Brightness change on static pixels across the
  segment. Above about 3 gray levels, plain `variance` becomes unreliable.
* **Stability.** Standard deviation of the waterline position across
  sub-windows, in original pixels. On a calm segment the median should be
  about 0.5–2 px. Larger values mean the segment isn't calm, the sub-window
  is shorter than the ripple period, or the threshold is unstable.
* **Line fit** (with `--fit line`). Shows the line equation in pixels, its
  angle, and the residuals: how far the detected points sit from the
  straight line.
  * An RMS residual of about 1–2 px means a straight line describes the edge
    well.
  * If the residuals are much larger at the `ends` than the `centre`, a
    wide-angle lens is probably bending a straight edge. Use a narrower
    `--roi`, or correct the lens distortion first.
  * The "Line-fit stability" line shows how much the fitted line tilts
    (degrees) and shifts (px) between sub-windows.

## Choosing a method (what testing showed)

Tested on the synthetic video against the known ground-truth line (640×360,
150 calm frames, ROI excluding glare). Error = mean absolute error in pixels.

| Method | No drift | +30 gray-level drift |
|---|---|---|
| `variance` | 0.44 px | 1.56 px (separability 42 → 12) |
| `detrended` (default) | 0.44 px | **0.46 px** (separability unaffected) |
| `diff`, lag 1 | 9.1 px | 15.5 px |
| `diff`, lag 4 | 4.5 px | 4.1 px |

* **Drift inflates raw variance.** A linear brightness change of total size D
  adds about D²/12 to *every* pixel's variance, so static structure starts to
  look like water.
* **`detrended` removes that.** It subtracts each pixel's best-fit line over
  time before measuring variance. That removes linear drift exactly and still
  captures slow ripples. It does not remove non-linear lighting changes, such
  as a cloud passing over and leaving. For those, shorten the calm window or
  try `diff`.
* **Frame differencing is drift-robust, but blind to slow ripples.** Plain
  consecutive-frame `diff` only sees *fast* change. With slow, smooth ripples,
  the water signal was buried under sensor noise and mp4 block artifacts, and
  the line was biased several pixels up into the dock. A lag of 3–5 frames
  helps a lot. Real water with fast glints may favour `diff` more than the
  synthetic video did, so compare the maps on your own footage.

## Known limitations / edge cases

* **Sun glare and specular reflections.** These are high-variance regions
  that aren't water. They can win Otsu outright: on the synthetic video
  without an ROI, `diff` segmented only the glare and foliage. Mitigations:
  * Set `--roi` around the dock/water edge.
  * Set a manual threshold.
  * Rely on the largest-blob rule (the cleanup keeps only the biggest blob).
  * Act on the warnings. The script lists every separate high-noise region
    and flags blobs that saturate at times as glare suspects. Sun glints *on*
    the water can also trigger that flag, so check `mask_raw.png`.
* **Motion breaks the calm assumption.** A dog, splash, person or boat in
  the segment will be detected as "water". Only use frames from before the
  jump. The stability check is the alarm for this.
* **The stability window must cover a full ripple period.** A single frame
  has no temporal variance, so stability is measured over sliding
  sub-windows. If a sub-window is shorter than the swell or ripple period,
  the water mask goes patchy and the line jitters. On the synthetic video,
  30-frame windows gave about 2.5–5 px of jitter and 90-frame windows about
  0.5 px. Give the baseline itself several seconds of calm footage.
* **Compression artifacts.** mp4 macroblocks cause blocky noise near
  high-motion areas. `--pre-blur` and `--post-blur` reduce it.
* **Camera shake.** Any camera movement makes edges look "noisy". The camera
  must be truly fixed; otherwise stabilise the video first.
* **Viewing angle.** `scan` assumes one boundary crossing per column (or
  row). If the shoreline curves back on itself, use `--line-method contour`.
