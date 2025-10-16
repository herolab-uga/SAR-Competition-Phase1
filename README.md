# SAR-Competition-Phase1
IEEE SMCS SAR Competition Phase 1 Submission

# Project README — Proposed Solution (ROSbot controller)

This repository provides a Webots ROSbot controller implementing a safety-centric multi-robot navigation framework. Key features include a control-barrier-function (CBF) based motion controller, robust stuck detection with repulsive recovery behaviors, prioritized victim detection and mission handling (color-based victim identification), consensus-driven rendezvous and navigation, persistent map and GRAPH_SUMMARY merging, and inter-robot/supervisor messaging. The primary entry point is `proposed_solution.py`.

---

## Quick summary

- **Purpose:** multi-robot search & victim-detection controller for Webots-style robots. Prioritizes red-victim detection (tight HSV swatch), supports greedy pixel-centering pursuit when depth/world coords are unavailable, and uses a timed consensus cycle for rendezvous when no victim mission is active. Safety features include short-range stops, CBF mid-field safety using LIDAR, repulsive recovery, and emergency backoff.
- **Key files:**
  - `proposed_solution.py` — main controller implementation
  - `README.md` — this file
  - `requirements.txt` — third-party Python packages required for local development


## Strategy summary

Here's the summary of our strategy.

- **Exploration approach:** We use a simple *inverted rendezvous* strategy. During the first 30 seconds each robot follows a rendezvous schedule; if rendezvous is reached or the timeout expires, robots switch to *inverted rendezvous* where they apply velocities to move away from other robots to improve coverage.

- **Parallel pipelines:** Motion planning and victim detection run in parallel. The planner switches between *exploration* and *victim-seeking* modes; obstacle avoidance is applied whenever the robot moves toward any goal or direction.

- **Victim detection & response:** Victim detection relies on color mapping and additional attributes to identify a victim. When a victim is detected, the robot immediately switches to victim-seeking mode, approaches the victim to a reporting distance, notifies the supervisor, then returns to exploration mode.

- **Mode switching:** In motion planning the robot dynamically switches between exploration and victim-seeking modes. Victim-seeking has higher priority: once a victim is detected the robot interrupts exploration to pursue and report.

---

## Prerequisites

1. **Webots**: This controller uses the `controller` Python module supplied by Webots. Install and run the controller inside Webots. The Webots runtime provides the `controller` package; you do **not** install that via pip.
2. **Python**: Recommended/testing Python version: **3.10** (works with 3.9–3.11 in most environments). Check your version with:

```bash
python --version
```

3. Optional but recommended: `opencv-python` for robust red-victim detection and debug image output. If OpenCV is not installed, the controller will still run but red-victim detection will be disabled.

---

## Setup (venv + pip)

From the project root (where `proposed_solution.py` lives):

**Create a virtual environment** (recommended):

```bash
# cross-platform (POSIX)
python -m venv .venv
source .venv/bin/activate

# windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# windows (cmd)
python -m venv .venv
.venv\Scripts\activate.bat
```

(You can also use the short path used in some examples: `.venv/Scripts/activate` on Windows when invoked from a compatible shell.)

**Install dependencies**:

```bash
pip install -r requirements.txt
```

If you prefer not to install OpenCV (to avoid large wheels), you can remove `opencv-python` from `requirements.txt`; detection will just be disabled.

---

## Running

Run the controller inside Webots as a robot controller. If you want to run the Python file standalone for static checking or to inspect logic (outside of Webots), you can still run it, but many hardware-specific calls will be `None` or fail without the Webots runtime.

To start the script (outside of Webots, for linting/testing):

```bash
python proposed_solution.py
```

**Note:** When launching as a Webots controller, Webots passes environment variables and injects the `controller` API; run the controller via the Webots robot configuration or the `webots --stdout --stderr` command-line options that spawn controllers.

---

## Files produced at runtime

- `example_camera_outputs/` — debug image outputs (RGB frames, depth frames, annotated detection images)
- `example_camera_outputs/map.json` — persisted map nodes
- `example_camera_outputs/map_history.jsonl` — append-only event history
- `example_camera_outputs/victims.json` — persisted victim sightings

---

## Notes and tips

- The code attempts to operate gracefully when sensors or optional libraries are missing. If OpenCV is unavailable, red-victim detection is skipped (a warning is printed). If GPS, LIDAR, or IMU aren't present in the Webots robot definition you use, the controller will fall back to safer behaviors where possible.
- If you plan to run multiple robot instances on the same machine, create separate controller copies or unique working directories to avoid file collisions in the `example_camera_outputs` folder.
- To reproduce the environment in CI, pin exact versions in `requirements.txt` (this file contains minimal version constraints; feel free to pin to exact versions if you need deterministic builds).

---

## `requirements.txt` (contents)

The `requirements.txt` file included with this project contains the Python packages used during development. For reproducibility it is intentionally conservative (no strict pins):

```
numpy>=1.24
Pillow>=9.0
opencv-python>=4.7
```

**Important:** The Webots `controller` package is supplied by Webots and is **not** listed in `requirements.txt`.

---

## License

(Include a LICENSE file if you wish — not provided here.)

---

## Contact / Author

If you want me to also add a `LICENSE` file, CI configuration, or pin exact package versions, tell me which OS/CI you target and I will add them.
