# Photo-first UI (Desktop v0.3.2)

The app has two resizable horizontal areas:

- **Left: Photography workspace (default ~70%)** — four photographs in a 2×2 grid or one enlarged photo. All annotations, calibrated building outlines, original/proposed height overlays, vanishing-point guides, observed lines and optional 2.5D building outlines are still drawn on the actual photographs.
- **Right: Controls (default ~30%)** — Calibration, Scenario, 2.5D Neighbors, Camera Performance and Results. Drag the splitter handle to adjust the ratio.

The axonometric 3D illustration is no longer shown or instantiated. GeneraBuild **still uses a shared 3D coordinate system and camera matrices**; only its redundant display widget was removed. The photo viewer displays the same pixel-perfect original images and continues to use native image coordinates for annotations.

**Shortcuts:** Ctrl+G = 2×2 grid, Ctrl+L = enlarge selected photo, Ctrl+1–4 = select view. Double-click an image when not annotating to toggle between 2×2 and single view.

**Compatibility:** The change is UI-only. The calibration JSON, sites in `workspace/`, 2.5D neighbor JSON, masks, existing run folders, GPU pipeline and 3 m/floor scenario default have not changed.

**Upgrade:** Unzip the v0.3.2 patch into the root of your installed project. It deliberately omits `workspace/`. On older installations `generabuild_desktop/scene3d.py` may remain on disk as an unused file. You may remove it after verifying the new interface.
