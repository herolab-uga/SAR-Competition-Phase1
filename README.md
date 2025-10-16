# SAR-Competition-Phase1
IEEE SMCS SAR Competition Phase 1 Submission

# Multi‑Robot Victim-First Controller (Webots)

## What this repo contains
- `proposed_solution.py` — main Webots Python controller implementing:
  - Victim‑first behavior (preemptive red‑victim pursuit and confirmation)
  - Consensus position cycling (forward/reverse cycle)
  - Short‑range safety, CBF-style mid‑field safety, repulsive recovery
  - Map merging (GRAPH_SUMMARY), mission reporting, and supervisor handling
- `README.md` — this file
- `requirements.txt` — Python packages required for running the controller (see notes)

## Python version
This project targets **Python 3.10** (3.10.x). Use Python 3.10 to maximize compatibility with Webots controller bindings.

## Quick setup (Windows)
```powershell
# from the repo root
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# then run in Webots as a controller (see Webots notes below)
```

## Quick setup (Linux / macOS)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# then run in Webots as a controller (see Webots notes below)
```

## Running with Webots
The `controller` module used by `proposed_solution.py` is provided by the Webots runtime. Two common ways to run:
1. **Inside Webots:** Put `proposed_solution.py` in your Webots project `controllers/<your_controller>/` folder and select that controller from the robot's controller field in the Webots scene. Use Webots' built‑in Python (or point Webots to your venv Python).
2. **Command line (advanced):** Ensure Webots' Python bindings are available to your environment (typical on your system when Webots is installed). If you get `ModuleNotFoundError: No module named 'controller'` -> run the script from inside Webots or configure PYTHONPATH to include Webots' `lib` directory.

## Notes on optional features
- **OpenCV (`opencv-python`)** is optional but recommended for robust red victim detection and producing debug images. If OpenCV is not installed, detection falls back to disabled mode and the controller still runs.
- Depth‑to‑world approximation requires a depth camera device in your Webots robot; otherwise the controller uses greedy pixel pursuit for victims.

## Troubleshooting
- `ModuleNotFoundError: No module named 'controller'` — The `controller` package comes from Webots. Run the controller from Webots or configure your Python to use Webots' controller bindings.
- If images are not saved or OpenCV fails, ensure `opencv-python` and `Pillow` are installed in the active venv.
- On Windows the venv activation command is `.\.venv\Scripts\activate` (PowerShell) or `.\.venv\Scripts\Activate.ps1`.

## Recommended additional steps
- Create a `.gitignore` that excludes `.venv/` and `example_camera_outputs/`.
- Pin package versions after you confirm everything works: `pip freeze > requirements.txt` (optional).

## License
Choose a license for your repo (e.g. MIT) or keep for internal use.

