# Camera Performance — Four-camera synchronization

## Why this feature exists

The initial calibrated cameras are fitted separately to a common reference prism. If width or depth is uncertain, a camera fitted to front-face-only correspondences may project the rear edges inaccurately. **Restricted joint bundle adjustment** uses the four camera observations together to refine the shared width/depth and all poses while keeping the original height and each view's FOV fixed.

## Controls

- **Geometry prior fraction (default 0.08)**: strength of the prior on original width/depth; smaller values impose stronger constraints against change.
- **Maximum W/D change (default 0.15)**: restricts each shared dimension to ±15% of its original assumption.
- **Use manually observed structural lines**: uses measured architectural edges, not model-generated guides, as soft angular constraints.
- **Synchronize cameras · Propose refinement**: runs optimization without editing the source JSON.
- **Preview synchronized geometry**: toggles between current and proposed projections over the same four photographs.
- **Accept** requires the explicit visual-review checkbox and a confirmation dialog; **Discard** returns to the saved calibration.

## Required evidence

All four views must have a valid saved PnP pose and at least four physical-corner annotations. There must be genuine front and rear corner evidence in one or more views; otherwise depth is unidentifiable and the proposal is rejected.

The app does not use automatically drawn vanishing-point guides as if they were independent measurements. If the FOV or annotations have changed but were not saved, save the calibration first.

## What is and is not synchronized

**Updated after acceptance:** camera pose `rvec`, `tvec`, `P`, shared `W`, shared `D`, masks, Canny conditioning and previews.

**Not updated:** original building reference height `H0`, each view's `FOV`, source photos, historical runs, or manually stored 2.5D neighbor coordinates. The neighbor model must be visually revalidated when the ground-plane geometry changes.

## Fitting metrics

The table shows the number of observed anchors, FOV, current RMSE and joint RMSE for each camera. The detailed report includes pooled RMSE, measured depth evidence, optimizer convergence, dimension changes and warnings. These are **training-fit diagnostics** only; independently marked structural edges and extra image observations should be used for external validation.

## Recovering prior calibration

Accepted camera synchronization creates a timestamped `calibration_backup_*.json` alongside the active calibration and an audit file in `diagnostics/camera_sync/`. To restore an older calibration safely, close the application, back up the project, then restore a previous calibration JSON and **regenerate masks/previews** with the original camera parameters. Do not restore only the JSON and assume its old masks are still consistent.

A proposal that fails to converge or worsens fitting RMSE beyond the acceptance guard cannot be accepted through this version's UI.
