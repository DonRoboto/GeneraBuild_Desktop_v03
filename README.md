# GeneraBuild Studio Desktop v0.3.2

**Geometry-Constrained Urban Scenario Visualization — Research Software Prototype**

A native **PySide6 / Qt desktop application** for calibrating four photographs of an urban site, editing a shared 3D building volume, preparing 8–10-floor densification scenarios, and generating geometry-conditioned photographic visualizations. This release uses a **photo-first, two-pane layout**: a large calibrated-photo workspace alongside a resizable controls panel. It retains the English interface and reversible four-camera synchronization.

**Status:** CPU geometry/calibration operations and regression tests were exercised with synthetic photographs. A visual Qt session and CUDA inference must still be validated on the target Ubuntu workstation. Camera poses and physical dimensions are approximations, not a geodetic or architectural survey.

## Installation on Ubuntu

```bash
unzip GeneraBuild_Desktop_v03_2_PhotoFirst_Complete.zip
cd GeneraBuild_Desktop_v03_2_PhotoFirst
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m generabuild_desktop
```

Recommended: Python 3.11 or 3.12. If Qt fails to open on Linux, inspect the platform plugin and `xcb` system dependencies. CPU mode does not require CUDA.

When upgrading an existing v0.2.2 installation, prefer the **patch ZIP** instead of replacing the entire directory. The patch does not include `workspace/` and therefore does not overwrite existing projects or runs.

## Main workflow

### 1. Create a project or open a calibrated site

- Select **File → New project (4 photographs)** to create a site from scratch. Photos 1–2 are aerial, photos 3–4 are street-level perspectives; the original bytes are copied unchanged to `workspace/sites/<id>/inputs/`.
- Alternatively, use **Open project folder** or **Import Colab ZIP** for an existing FR2 calibration.
- Saving a draft does not create fake camera poses; the application requires successful PnP calibration before a scenario can be generated.

### 2. Calibration

1. In **Calibration**, choose one of the four views. Mark only genuinely identifiable physical vertices, using consistent IDs `0..7` across the project.
2. Enter approximate original width `W`, depth `D`, and height `H₀`. Set the **horizontal FOV separately for every camera**. The app preserves whatever FOV you calibrated: 110° was effective for one particular Guanajuato 8 street image, but is not assumed for other photographs.
3. Select **Calibrate selected view** or **Calibrate all four cameras**. Initial poses are estimated using OpenCV PnP.
4. Inspect both the original reference prism and the proposed building prism. Add **permitted-edit** and **protected-context polygons**; optionally record **observed structural edges** and vanishing-point guides.
5. Confirm the visual review and click **Save calibration and enable generation**. This creates `calibration_multiview.json`, masks, Canny images, and previews.

Do **not** invent rear vertices that cannot be seen. Fitting RMSE alone does not establish geometric accuracy. A common local coordinate system is used across all four photographs.

### 3. Camera Performance — four-camera synchronization (new)

Open **Camera Performance** after saving the four-camera PnP calibration. You will see a table containing each view's number of anchors, horizontal FOV, and current reprojection RMSE.

1. Click **Synchronize cameras · Propose refinement**. An optimization runs in a worker thread without overwriting the saved calibration.
2. The solver adjusts **four camera poses and the common original building width and depth** using restricted bundle adjustment. It **holds original height and all four FOVs fixed**. If you recorded *real* structural edges, you can use them as additional angular constraints. Automatically drawn VP guide lines are never treated as independent observations.
3. Inspect the new pooled RMSE, each camera's before/after RMSE, width/depth changes, and warnings.
4. Toggle **Preview synchronized geometry in all four photographs**. View the original and proposed cameras with the global 2×2 / single-view selector. Zoom and check actual building boundaries and protected neighbors.
5. Click **Discard proposal** to keep the original calibration, or check **I reviewed all four reprojections and the protected regions**, then click **Accept and save synchronization**. A confirmation dialog is required.

**Acceptance** keeps `calibration_backup_*.json`, saves an audit report in `diagnostics/camera_sync/`, rebuilds geometric masks and previews, and leaves original photographs and historical run directories intact. If 2.5D neighbor volumes exist, review their placements after changes to the building footprint.

**Important limitation:** joint refinement requires observable depth evidence (genuine front and rear corners in at least one aerial image). With only front-face planar points, optimization is rejected rather than fabricating depth. A lower fitting RMSE is not independent geometric validation. See [`docs/CAMERA_SYNC.md`](docs/CAMERA_SYNC.md).

### 4. Scenario generation

In **Scenario**, select 8, 9, or 10 floors, one of the existing generation modes, and the photos to render.

- **Prepare scenario · CPU** generates masks, Canny conditioning, and previews without diffusion. This is useful for validation on any laptop.
- **Generate · Stable Diffusion + ControlNet (GPU)** uses the FR2 inpainting pipeline and optionally runs façade and roof finishing. CUDA and compatible pretrained model weights are required; GPU inference was not tested in this build environment.
- Geometry and camera calibration are not re-estimated when changing floor count.

For optional CUDA support, install the correct PyTorch/CUDA build for your system, then use `requirements-gpu.txt`. Model downloads and their licensing terms are managed independently of this source code.

### 5. Optional 2.5D neighborhood context

In **2.5D Neighbors**, define approximate ground footprints and heights. Neighbors are projected onto all four photographs; no separate 3D viewer is required. Their JSON is separate from camera calibration. **Use neighbors as generation occluders** is experimental and off by default; it can introduce errors if footprints/heights are inaccurate.

### 6. Result Viewer

Open **Results** to select a saved run and switch between **Original**, **Generated**, **Compare**, **Base (before enhancements)**, and **Geometry preview (CPU)**.

Use the top **View layout** switch for a **2×2 grid** or a **single enlarged photo**. The comparison slider controls a vertical reveal of original/generated pixels. Export a 2×2 montage or the run artifacts. Missing generated views are explicitly labeled rather than fabricated.

## Photo-first workspace (v0.3.2)

The separate axonometric 3D canvas was removed **from the interface only**. The four-photo canvas now occupies the left workspace, with Calibration, Scenario, 2.5D Neighbors, Camera Performance, and Results in a resizable panel on the right. Drag the vertical splitter to give more space to photos or controls. The **Four views (2×2)** / **One enlarged view** selector, point annotation, zoom, pan, projected building outlines, manual vanishing lines, synchronization preview, and before/after Result Viewer all remain available. Original project data and 3D mathematics are unchanged.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Ctrl+G` | Four-view grid (2×2) |
| `Ctrl+L` | One enlarged photograph |
| `Ctrl+1` to `Ctrl+4` | Select a photo |
| Mouse wheel | Zoom in/out |
| Drag outside annotation mode | Pan photographs |

## Reproducibility and tests

```bash
python -m pytest -q
python -m generabuild.cli --site workspace/sites/synthetic_demo --floors 9 --mode ampliacion --preview-only
```

The CLI still uses internal mode identifiers `ampliacion` / `completo` for backward compatibility with calibration, run manifests and notebooks. These **machine-readable identifiers are intentionally unchanged**; interface labels are English.

- `calibration_multiview.json`: saved view poses, matrices, correspondences, polygons, and camera FOVs.
- `project_draft.json`: incomplete annotations before camera calibration.
- `urban_context_2p5d.json`: optional approximate neighbor volumes.
- `runs/<id>/`: immutable per-scenario artifacts and evaluation reports.
- `diagnostics/camera_sync/`: audits for accepted joint calibrations.


See [`METHOD.md`](METHOD.md), [`docs/CAMERA_SYNC.md`](docs/CAMERA_SYNC.md), and [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

## Floor-to-floor height for Guanajuato 8 (v0.3.1)

Use **Scenario → Assumed height per floor = 3.00 m**. The default nine-floor scenario is **27.00 m**; eight floors = 24 m and ten floors = 30 m. A live label shows the exact scenario height. If a legacy project saved a different floor-height assumption (for example 5 m), the Scenario tab warns you, but your original camera calibration is not modified. Details: [`docs/FLOOR_HEIGHT.md`](docs/FLOOR_HEIGHT.md).
