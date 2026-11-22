# Floor-height assumption in GeneraBuild Studio v0.3.1

For the Guanajuato 8 research scenario, **3.00 metres per floor** is the modelling assumption. It is **not** a surveyed floor-to-floor height.

| Existing | 8-floor scenario | 9-floor scenario | 10-floor scenario |
| --- | --- | --- | --- |
| 3 floors / 9 m | 24 m | 27 m | 30 m |

The **Scenario** tab now contains **Assumed height per floor** (initial value **3.00 m**) and displays the calculated total height in metres. It no longer silently inherits `floor_height_m` from older site calibration JSON files.

To change this assumption for a different site, edit the scenario-specific control. The override is stored in each run's `scenario_config.json`; the saved `calibration_multiview.json` and all camera matrices remain unchanged.

If the calibration JSON says 5 m/floor, the Scenario tab explicitly reports the discrepancy. This does **not** mean the reference geometry itself is correct: verify that `reference_box_m.H` still represents the *original three-floor building height* (approximately 9 m for Guanajuato 8). If the original was calibrated at 15 m by mistake, re-estimating or revisiting the calibration is necessary. Do not simply edit `reference_box_m.H` after calibrating, because camera pose estimation used that original geometry.

To check a project on disk:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('workspace/sites/guanajuato8/calibration_multiview.json')
m = json.loads(p.read_text())
print('Saved floor height:', m.get('floor_height_m'), 'm/floor')
print('Original height H0:', m['reference_box_m']['H'], 'm')
PY
```

The scenario-specific override is applied consistently in the projected photo overlays, CPU preview, generation masks, and the GPU generation configuration. This patch does not change the existing calibration and is backward compatible with saved projects and runs.
