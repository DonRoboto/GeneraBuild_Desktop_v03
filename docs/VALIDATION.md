# Validation protocol

1. Validate the original prism against independently visible corners and structural lines; note that fitting RMSE is not independent ground truth.
2. Inspect all four images with original FOVs fixed. Record errors and warnings per camera.
3. If synchronizing cameras, compare current and proposed RMSE, candidate W/D values and actual building boundaries before accepting. A lower fitting error does not prove a physically correct pose.
4. For an 8/9/10-floor parameter sweep, keep the building footprint, calibration, model configuration and seed unchanged; change only target floor count.
5. Verify that generated pixels are unchanged outside approved masks. Check the actual mask footprint to ensure no neighbor was wrongly permitted for editing.
6. Run automated CPU tests with `python -m pytest -q`. Test the full Qt UI and optional GPU generation in the deployment environment.

The 2.5D context is approximate. Occlusion masking based on neighbor volumes is experimental and disabled by default.
