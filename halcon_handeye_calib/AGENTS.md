# Repository Guidelines

## Project Structure & Module Organization
- `capture_img.py` captures calibration images from an Intel RealSense D405 and writes them to `images/`.
- `halcon_files/` contains HDevelop scripts (`.hdev`) and the calibration target descriptor (`my_caltab_40.descr`).
- `images/` stores captured RGB images named `img_XX.png`.
- `poses/` stores robot/tool poses aligned with each image, named `pose_XX.txt`.
- `result/` holds calibration outputs such as `pose_final_campar.dat` and diagnostics logs.

## Build, Test, and Development Commands
- `python capture_img.py` runs the RealSense capture loop; press `s` to save images, `q` to quit.
- Open `halcon_files/hand_eye_movingcam_calibration_halcon_demo.hdev` or `halcon_files/my_calibration.hdev` in HDevelop and run the script to compute calibration results.
- There is no build system or test runner in this repo; execution is via Python and HDevelop.

## Coding Style & Naming Conventions
- Python follows PEP 8 style: 4-space indentation, descriptive variable names, and short functions where possible.
- File naming uses numbered pairs: `images/img_XX.png` matches `poses/pose_XX.txt`.
- Keep calibration artifacts in `result/` and avoid mixing raw inputs and outputs.

## Testing Guidelines
- No automated tests are currently configured.
- When modifying calibration logic, verify by re-running the HDevelop script and checking `result/pose_calibration_results.txt` and `result/pose_pose_diagnosis_log.txt`.

## Commit & Pull Request Guidelines
- Git history uses short, plain messages (e.g., "update"). Prefer concise, imperative summaries like `add pose parser`.
- PRs should include a brief description, mention the data or scripts touched (e.g., `images/`, `poses/`), and attach before/after calibration logs if results change.

## Configuration & Data Notes
- Hardware dependencies: Intel RealSense D405, OpenCV, and HALCON/HDevelop.
- Keep `images/` and `poses/` in sync; mismatched counts will break calibration scripts.
