# IEEE SMCS SAR Competition — Phase 1 Submission

This repository contains our **Webots controller** for the SAR competition.
It includes everything needed to **recreate the Python environment** and run the
controller as `controllers/proposed_solution/proposed_solution.py`.

> **Device names expected by the controller**
> - Laser: `laser`
> - GPS: `gps`
> - Compass: `imu compass`
> - Inertial Unit: `imu inertial_unit`
> - RGB Camera: `camera rgb` 
> - Depth Camera: `camera depth`
> - Wheels (diff drive): `fl_wheel_joint`, `fr_wheel_joint`, `rl_wheel_joint`, `rr_wheel_joint`

The controller also **detects red victims**, **reports** sightings/confirmations
to the supervisor/squad, and **reroutes** around obstacles using the laser.

---

## 📦 Repo Layout

```
controllers/
└── proposed_solution/
    ├── proposed_solution.py         # MAIN ENTRY picked up by Webots
    ├── requirements.txt             # If using venv/pip
    ├── environment.yml              # If using conda (alternative to requirements.txt)
    ├── readme.md                    # This file
    └── (any other helper modules your controller imports)
.gitignore
```

> **Important:** Do not include compiled binaries or extraneous large files.

---

## 🐍 Python Version

We tested with **Python 3.10**. Use the same for reproducibility (or adjust below).

---

## ✅ venv + pip (Windows-friendly)

From the repo root (where this README lives):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```


## ▶️ Running in Webots

1. Place this repository so that **Webots** sees it as:
   `controllers/proposed_solution/` (Webots clones into this location for grading).
2. Ensure your `proposed_solution.py` is the **exact** file name at that path.
3. Start the simulation; Webots will automatically run the controller.

> If devices differ in your world, update the device names in `proposed_solution.py` accordingly.

---

## 🧪 Quick Local Check (optional)

After activating your environment:

```bash
python -c "import cv2, numpy; print('OK: OpenCV', cv2.__version__)"
python -c "import numpy as np; print('OK: NumPy', np.__version__)"
```

OpenCV is optional, but needed for robust victim detection from RGB frames.

---

