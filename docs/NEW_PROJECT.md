# New-project workflow

1. Click **File → New project (4 photographs)** and choose four original images: views 1–2 aerial, views 3–4 street-level.
2. Set reference dimensions `W,D,H0` and the assumed floor height. These are geometric assumptions, not field-survey measurements.
3. In **Calibration**, mark consistent physical vertex IDs across all visible photographs. Avoid assigning non-visible rear corners.
4. Set the horizontal FOV for each image. Use OpenCV PnP to calibrate each view. Verify the original reference prism, roof corners, and ground corners.
5. Draw an edit-permitted polygon and protected context polygons as needed. Optional observed structural lines can help later joint refinement.
6. Save a draft to return later, or review all four views and **Save calibration and enable generation**.
7. In **Camera Performance**, optionally propose and visually inspect a synchronized refinement.
8. In **Scenario**, choose 8–10 floors and generate a CPU geometric preview or a GPU photorealistic output.

The first click in a different photo activates that view but does not automatically assign a vertex, preventing accidental index mismatches.
