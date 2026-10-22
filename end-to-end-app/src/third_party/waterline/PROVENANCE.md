# Provenance — vendored Waterline-Detection

Upstream: <https://github.com/quintinlevy/Waterline-Detection>
Author: quintinlevy
Commit vendored: `e0c412607ea92d557da127b1ac72f791f23efb96` (`main`, 2026-10-12)
Fetched: 2026-10-20, `git clone --depth 1`
Vendored by: dog-diving-agent-work (same team; see the licence note below)

## Licence

**Upstream ships no LICENSE file.** Under default copyright that means "all rights reserved", even
between teammates. The project owner (harri) decided on 2026-10-20 to vendor it anyway for this
personal/team project and not to block on it. Before this tree is published or distributed outside
the team, the author should be asked to add a licence (MIT or Apache-2.0) — or this directory must
be removed from the distribution.

## What is here

Every file upstream ships, copied byte-for-byte and **not edited**:

| file | role |
|---|---|
| `detect_waterline.py` | the pipeline: noise maps → mask → waterline → line fit + diagnostics |
| `detect_waterline_lite.py` | upstream's reduced variant (not called by this project) |
| `estimate_water_plane.py` | water plane `n·X = d` from a waterline + ZED depth + IMU (needs the ZED recording; not reachable from `.mp4`) |
| `zed_real.py`, `zed_stub.py` | ZED camera / stub used by `estimate_water_plane.py` |
| `requirements.txt` | `opencv-python>=4.8`, `numpy>=1.24`, `pytest>=7` — all already in the app's venv |
| `README.md`, `README_lite.md` | upstream documentation |
| `tests/` | upstream's own tests, runnable from this directory (`python -m pytest tests -q`) |

`SHA256SUMS` pins each file. `src/dog_diving/waterline.py` is the only adapter; it imports this
package and does **not** modify it. If a future need demands a change to upstream's code, add a new
function in the adapter instead — a byte change here invalidates the hashes and the provenance
record, and the app records this hash in every calibration document it seeds.

## Why it is vendored rather than imported from GitHub

The app must run offline from `G:` and must be able to answer "which algorithm placed this plane?"
years later. A pinned, hashed copy in-tree does that; a `pip install git+…` does not.

## How the adapter uses it

`src/dog_diving/waterline.py` reuses, unmodified: `compute_noise_maps`, `fit_normalization`,
`normalize_noise_map`, `threshold_noise_map`, `clean_mask`, `separability`, `component_report`,
`lighting_drift`, `auto_stability_window`, `fit_straight_line`, and `load_frames`. The three-edge
extension (top+bottom per column, row scans for the dock contact, run splitting, classification)
lives entirely in the adapter, because upstream returns one boundary per scan and has no notion of
"which edge is the dock".
