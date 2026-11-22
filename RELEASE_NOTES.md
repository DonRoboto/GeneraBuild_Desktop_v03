# Release notes — GeneraBuild Studio Desktop v0.3.0

Based on **Desktop v0.2.2 Result Viewer**, retaining FR2 generation and the optional 2.5D neighborhood editor.

## Added

- English-language Qt interface: menus, new-project wizard, calibration, scenarios, neighbors, Camera Performance, Results, status and dialog messages.
- **Camera Performance** panel with four per-view reprojection metrics, fixed horizontal FOV display, and restricted four-camera optimization using SciPy/OpenCV.
- Threaded optimization with two-stage acceptance: inspect projection overlays, review metrics, then explicitly accept or discard.
- Safe persisted `camera_sync` report and calibration backup on acceptance.
- Checks rejecting proposals when the saved calibration changed or unsaved annotation edits exist.
- Automated regression tests for no-write proposal, explicit approval, FOV and height preservation, backup creation, masks and stale calibrations.

## Retained

- Four-view 2×2 and single-photo enlargement, camera corner calibration, vanishing-line guides, protected polygons, Result Viewer comparison/history, CPU preview, optional neighboring building volumes, optional façade/roof enhancement and FR2 GPU generation pipeline.

## Not implemented

- Automatic calibration from unknown photo collections without manual correspondences.
- An exact 3D reconstruction of the surrounding urban fabric.
- Joint camera synchronization from front-only coplanar points; such cases are rejected.
- GPU inference validation or complete offscreen Qt interaction testing in the packaging environment.

## Compatibility

`calibration_multiview.json`, `project_draft.json`, run manifests, and internal CLI modes `ampliacion` / `completo` remain backward compatible with v0.2.2. Do not overwrite `workspace/` when upgrading an installed project.

## v0.3.1 — explicit floor-to-floor height (November 2026)

- Added a **Scenario → Assumed height per floor** control, defaulting to **3.00 m**.
- Added a live total-height display (8 floors = 24 m, 9 = 27 m, 10 = 30 m for the 3 m assumption) and a one-click 3 m reset.
- Made the CPU preview, 3D view (when present in v0.3.1), projections, and per-run generation configuration use the explicit scenario assumption instead of silently inheriting a possibly stale value from site metadata.
- Saved original reference geometry, cameras, FOVs, masks, project JSON and historical runs remain untouched. Each new run records the chosen scenario height in its configuration.
- Tests cover the legacy case where a project JSON contains `floor_height_m=5`.

See [`docs/FLOOR_HEIGHT.md`](docs/FLOOR_HEIGHT.md).

## v0.3.2 — Photo-first desktop workspace (November 2026)

- Removed the nonessential axonometric 3D canvas and its show/hide checkbox from the Qt interface.
- Photo grid and control inspector are now the only two panes in a draggable horizontal splitter; default allocation prioritizes photos.
- Retained 2×2 and single-view modes, image zoom/pan, overlays, PnP calibration, four-camera synchronization, optional 2.5D neighbors, GPU generator, and Result Viewer.
- No changes to calibrated geometry, stored FOVs, the 3 m/floor assumption, run data, generation masks or inference.
- The new complete package omits the unused scene3d.py; an existing copy left in place by the patch is harmless.

See [`docs/PHOTO_WORKSPACE.md`](docs/PHOTO_WORKSPACE.md).
