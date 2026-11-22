# GeneraBuild: method overview

## Scope

GeneraBuild is a geometry-constrained multiview visualization system for hypothetical urban densification. A single local 3D building model is projected into four photographs. The geometric parameters determine where the structure appears; pretrained diffusion models synthesize visual detail within an explicitly defined edit mask. This is a research-software engineering pipeline, **not** a method for certifying true building geometry or physical camera intrinsics.

## Data and shared geometry

Four photographs of the same site are treated as independent calibrated observations. The reference building is parameterized by width `W`, depth `D`, and original height `H0`, with eight ordered 3D corners. Target height is `N × floor_height`, where `N` is the requested number of floors. Footprint is held fixed while preparing a scenario.

## Initial camera calibration

For each image `i`, OpenCV estimates its pose using manually identified correspondences between observed pixel corners and the fixed 3D reference. Projection follows `λ p_ij = K_i [R_i | t_i] V_j`. The effective horizontal FOV is a per-view parameter. The initial camera poses are estimated separately, and the corresponding reprojection RMSEs are recorded. False back-face annotations, incorrect FOVs, and coplanarity can produce misleading poses despite small fitting RMSE.

## Optional joint camera synchronization

The restricted joint objective is:

`min_(R_i,t_i,W,D) Σ_i Σ_j ρ(||π(K_i,R_i,t_i,V_j(W,D,H0)) − p_ij||²) + R(W,D)`

where `R` regularizes changes in shared reference width/depth. The **reference height `H0` and all K_i (and therefore horizontal FOVs) are kept fixed**. Independently observed image-edge orientations can supplement the objective. A sufficient amount of genuine depth information is required; the method rejects front-only planar annotations.

The solver produces a **candidate** calibration. Its fitting RMSE, width/depth changes, warnings and projected building edges are available for human review. It is never committed without explicit confirmation. An accepted candidate overwrites the active calibration only after making a JSON backup and regenerating masks and previews; existing photos and historical run directories are preserved.

## Spatial conditioning and extension volume

The proposed building volume is projected into every camera. In the *extension* mode, the spatial edit region is derived from the volume between the reference roof and the new target roof, **not** by subtracting the full 2D silhouette of the reference building. This is essential for aerial photos, where new floors may occlude the old roof. User-approved edit polygons and manually protected regions further restrict modifications.

For the diffusion process, an architectural edge map combines projected geometry with the original photograph's Canny edges. Inference uses Stable Diffusion 1.5 Inpainting and ControlNet Canny on a localized crop to preserve effective resolution, with fixed seed and saved generation parameters.

## Optional architectural enhancement

Post-processing is not required for base generation. A deterministic, geometry-aware module can improve window panes, frames and wall colors; another optional module uses a shared projected roof texture. These operations must respect the authorized masks and be evaluated independently of the base diffusion stage.

## Evaluation

The system computes fitting reprojection RMSE, exact-pixel context-preservation checks outside the mask, and mask consistency diagnostics. Fitting RMSE is **not** an independent estimate of true metric camera accuracy. Further independent testing would compare generated building silhouettes with a separately estimated segmentation or structural boundaries, and measure parametric consistency for 8-, 9- and 10-floor scenarios using fixed cameras and a fixed seed.

## Limitations and research ethics

The geometry, original floor heights, and camera FOVs are approximate unless supported by survey and camera metadata. The four views do not reconstruct the surrounding neighborhood photogrammetrically. A 2.5D neighbor context is optional and can improve or worsen occlusion handling depending on how accurately buildings are placed. This prototype cannot prove regulatory or architectural feasibility and must not be represented as an exact reconstruction of a building or its neighbors.
