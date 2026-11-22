# Architecture

```text
Qt Desktop UI (PySide6)
 ├─ Calibration Studio (four photographs, correspondences, masks)
 ├─ Camera Performance (independent PnP, optional restricted joint BA)
 ├─ Scenario Builder (height-only change; 2.5D context optional)
 └─ Result Viewer (history, before/after, 2×2 montage)
                  │
         Python geometric core
         OpenCV + NumPy + SciPy
          ├─ camera poses / reference prism
          ├─ masks and Canny conditioning
          ├─ optional context occlusion
          └─ FR2 scenario preparation
                  │
      Optional GPU generation engine
         PyTorch + Diffusers
         Stable Diffusion 1.5 / ControlNet
                  │
      Optional architectural enhancement
         façade finishing + rooftop atlas
                  │
        Audit, evaluation and export
```

Photographs are kept unchanged. The authoritative calibration is `calibration_multiview.json`; the original height and per-view FOVs remain fixed during camera synchronization. Generated runs are stored separately and do not overwrite the original inputs.
