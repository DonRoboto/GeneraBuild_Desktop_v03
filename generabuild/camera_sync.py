"""Safe, opt-in joint camera synchronization for GeneraBuild Desktop.

This module intentionally has no GUI dependencies. Joint refinement estimates four
camera poses and a shared W/D footprint while keeping the original height and all
horizontal FOVs fixed. It only proposes changes; acceptance requires a user review
and leaves a timestamped backup of the previous calibration JSON.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .calibration import save_calibration
from .geometry import sha256_file
from .joint_calibration import refine_joint_cameras
from .storage import image_paths, load_metadata


def build_camera_sync_proposal(
    site: Path,
    *,
    max_nfev: int = 120,
    geometry_prior_fraction: float = 0.08,
    allowed_dimension_change: float = 0.15,
    use_observed_lines: bool = True,
) -> dict[str, Any]:
    """Read a saved FR2 calibration and return a non-mutating joint BA proposal.

    At least one image must have *observed* front and rear building corners.
    Generated vanishing-point guides are never fed to optimization as measurements.
    """
    folder = Path(site).resolve()
    metadata = load_metadata(folder)
    paths = image_paths(folder, metadata)
    for path, view in zip(paths, metadata['views']):
        if view.get('image_sha256') and sha256_file(path) != view['image_sha256']:
            raise ValueError(f'Input photo changed since calibration: {path.name}')
    source = folder / 'calibration_multiview.json'
    initial_sha = sha256_file(source)
    dims = metadata['reference_box_m']
    views = metadata['views']
    cameras = [{
        'ok': view.get('P') is not None,
        'P': view['P'], 'K': view['K'], 'rvec': view['rvec'], 'tvec': view['tvec'],
    } for view in views]
    anchors = [v.get('anchors_px', {}) for v in views]
    fovs = [float(v['fov_horizontal_deg']) for v in views]
    sizes = [tuple(map(int, v['image_size_wh'])) for v in views]
    lines = ([v.get('observed_axis_segments_px', []) for v in views]
             if use_observed_lines else None)
    proposal = refine_joint_cameras(
        cameras, anchors, fovs, sizes,
        (dims['W'], dims['D'], dims['H']),
        max_nfev=max_nfev,
        geometry_prior_fraction=geometry_prior_fraction,
        allowed_dimension_change=allowed_dimension_change,
        observed_lines_by_view=lines,
    )
    return {
        'schema': 'GeneraBuild_JointCameraProposal_v1',
        'source_sha256': initial_sha,
        'source_site': str(folder),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'proposal': proposal,
    }


def accept_camera_sync_proposal(site: Path, candidate: dict, *,
                                reviewed: bool = False) -> dict:
    """Persist accepted proposal, regenerate masks/previews, and keep a backup.

    Refuses a stale candidate, failed solver, or a substantially worse fit. The
    previous FR2 calibration is backed up by ``save_calibration`` automatically.
    Existing run folders and input photos remain unchanged.
    """
    if not reviewed:
        raise ValueError('Inspect all four reprojections before accepting synchronization.')
    folder = Path(site).resolve()
    if candidate.get('schema') != 'GeneraBuild_JointCameraProposal_v1':
        raise ValueError('Unsupported joint calibration proposal format.')
    if str(folder) != candidate.get('source_site'):
        raise ValueError('This proposal belongs to a different project.')
    current_path = folder / 'calibration_multiview.json'
    if sha256_file(current_path) != candidate.get('source_sha256'):
        raise ValueError('The calibration changed after the proposal. Re-run synchronization.')
    metadata = load_metadata(folder)
    joint = candidate['proposal']
    report = joint['report']
    if not report.get('solver_success'):
        raise ValueError('Joint calibration did not converge; the current cameras were preserved.')
    before, after = (float(report[key]) for key in (
        'initial_pooled_rmse_px', 'refined_pooled_rmse_px'))
    if not (0 <= after <= before * 1.05 + 0.1):
        raise ValueError('Joint calibration increased reprojection error; proposal rejected.')
    original = metadata['reference_box_m']
    dims = joint['reference_dimensions_m']
    if abs(float(dims['H']) - float(original['H'])) > 1e-8:
        raise ValueError('Reference height must not change during camera synchronization.')
    if len(joint.get('cameras', [])) != 4:
        raise ValueError('Joint calibration requires exactly four camera poses.')
    fovs = [float(v['fov_horizontal_deg']) for v in metadata['views']]
    if any(abs(a-b)>1e-8 for a,b in zip(fovs, report['fovs_locked'])):
        raise ValueError('Field-of-view values changed; synchronization refused.')
    payload = {
        'ref_W': float(dims['W']), 'ref_D': float(dims['D']),
        'ref_H': float(dims['H']),
        'num_floors': int(metadata['num_floors']),
        'floor_height_m': float(metadata['floor_height_m']),
        'views_anchors': [v['anchors_px'] for v in metadata['views']],
        'fov_deg_by_view': fovs,
        'protected_polygons_px': [v.get('protected_polygons_px', []) for v in metadata['views']],
        'allowed_polygons_px': [v.get('allowed_polygon_px') for v in metadata['views']],
        'guide_lines': [v.get('manual_vp_guides_px', []) for v in metadata['views']],
        'observed_lines': [v.get('observed_axis_segments_px', []) for v in metadata['views']],
        'visual_inspection_confirmed': True,
    }
    existing_backups = set(folder.glob('calibration_backup_*.json'))
    updated = save_calibration(payload, joint['cameras'], image_paths(folder, metadata), folder)
    new_backups = list(set(folder.glob('calibration_backup_*.json')) - existing_backups)
    diagnostics = folder / 'diagnostics' / 'camera_sync'
    diagnostics.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    report_path = diagnostics / f'camera_sync_{stamp}.json'
    audit = {
        'event': 'accepted_joint_camera_calibration',
        'accepted_utc': datetime.now(timezone.utc).isoformat(),
        'original_calibration_sha256': candidate['source_sha256'],
        'new_calibration_sha256': sha256_file(current_path),
        'backup_files': [p.name for p in new_backups],
        'camera_report': report,
        'notes': [
            'Existing photographs and historical runs were not overwritten.',
            'Saved 2.5D neighbor coordinates must be reviewed after geometry changes.',
            'PnP fitting error is not independent metric ground truth.',
        ],
    }
    report_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf8')
    return {'metadata': updated, 'audit_path': str(report_path),
            'backup_paths': [str(p) for p in new_backups]}
