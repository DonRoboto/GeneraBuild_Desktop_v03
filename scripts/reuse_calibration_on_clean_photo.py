#!/usr/bin/env python3
"""Rebind one calibrated camera to a cleaner but geometrically identical photograph.

Creates a NEW GeneraBuild site. It never overwrites the original site or calibration.
Runs on CPU. Calibrated pose and reference geometry are unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Support invocation from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generabuild.geometry import (architectural_canny, make_diagnostic_preview,
                                 mask_for_view, sha256_file, validate_project)
from generabuild.storage import load_metadata


def check_alignment(old_path: Path, new_path: Path, *, max_drift_px: float = 3.0) -> dict:
    """Robust image-to-image registration check using natural image features.

    Only a near-identity transformation is accepted. This is not a camera
    calibration or proof of identical viewpoint, but catches most crop/shift errors.
    """
    old = cv2.imread(str(old_path), cv2.IMREAD_GRAYSCALE)
    new = cv2.imread(str(new_path), cv2.IMREAD_GRAYSCALE)
    if old is None or new is None:
        raise ValueError('Could not decode one of the photographs')
    if old.shape != new.shape:
        raise ValueError(f'Image sizes differ: {old.shape[::-1]} versus {new.shape[::-1]}')
    h, w = old.shape
    if np.array_equal(old, new):
        return {"passed": True, "method": "identical_pixels", "median_drift_px": 0.0,
                "matches": None, "inliers": None}
    detector = cv2.ORB_create(nfeatures=4500, fastThreshold=8)
    a, da = detector.detectAndCompute(old, None)
    b, db = detector.detectAndCompute(new, None)
    if da is None or db is None or len(a) < 12 or len(b) < 12:
        return {"passed": False, "reason": "Not enough stable photographic features"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    matches = [m for group in pairs if len(group) == 2 for m, n in [group] if m.distance < .72 * n.distance]
    if len(matches) < 15:
        return {"passed": False, "reason": "Insufficient ORB correspondences", "matches": len(matches)}
    src = np.float32([a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inlier_mask is None:
        return {"passed": False, "reason": "No robust image homography"}
    inliers = int(inlier_mask.sum())
    if inliers < 12:
        return {"passed": False, "reason": "Too few geometrically consistent matches", "inliers": inliers}
    test = np.float32([[x * w, y * h]
                       for x in (.15, .35, .5, .65, .85)
                       for y in (.15, .35, .5, .65, .85)]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(test, H)
    drifts = np.linalg.norm((transformed - test).reshape(-1, 2), axis=1)
    median = float(np.median(drifts))
    maximum = float(np.max(drifts))
    passed = bool(np.isfinite(maximum) and median <= max_drift_px and maximum <= max_drift_px * 2)
    return {"passed": passed, "method": "ORB_RANSAC_Homography_vs_identity",
            "median_drift_px": round(median, 3), "max_grid_drift_px": round(maximum, 3),
            "matches": len(matches), "inliers": inliers, "threshold_px": max_drift_px}


def transfer(site: Path, image: Path, view_number: int, dest: Path,
             *, allow_unverified: bool = False) -> dict:
    site, image, dest = (p.expanduser().resolve() for p in (site, image, dest))
    if not 1 <= view_number <= 4:
        raise ValueError('View number must be 1, 2, 3 or 4')
    if not site.is_dir() or not image.is_file():
        raise FileNotFoundError('Site directory or new image does not exist')
    if dest.exists():
        raise FileExistsError(f'Destination already exists: {dest}')
    meta = load_metadata(site)
    v = meta['views'][view_number-1]
    previous_rel = Path(v['image_relative'])
    if previous_rel.is_absolute() or '..' in previous_rel.parts or previous_rel.parts[0] != 'inputs':
        raise ValueError('Unsafe existing image path')
    old_path = site / previous_rel
    if not old_path.exists():
        raise FileNotFoundError(old_path)
    if v.get('image_sha256') and sha256_file(old_path) != v['image_sha256']:
        raise ValueError('Original image no longer matches its calibration checksum')
    with Image.open(old_path) as im_old, Image.open(image) as im_new:
        if im_old.size != im_new.size:
            raise ValueError(f'Different pixel dimensions: {im_old.size} vs {im_new.size}')
        image_size = im_new.size
    alignment = check_alignment(old_path, image)
    if not alignment.get('passed') and not allow_unverified:
        raise RuntimeError('Cannot confirm unchanged camera framing: '
                           + json.dumps(alignment, ensure_ascii=False)
                           + '. Inspect the two images; use --allow-unverified only when certain.')

    for dirname in ('inputs', 'masks', 'canny', 'previews'):
        origin = site / dirname
        if not origin.exists():
            raise FileNotFoundError(f'Missing project folder: {origin}')
    try:
        dest.mkdir(parents=True, exist_ok=False)
        for dirname in ('inputs', 'masks', 'canny', 'previews'):
            shutil.copytree(site / dirname, dest / dirname)
        old_view_path = dest / previous_rel
        old_view_path.unlink()
        new_suffix = image.suffix.lower()
        if new_suffix not in ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'):
            raise ValueError('Unsupported image extension')
        new_relative = f'inputs/Vista_{view_number}{new_suffix}'
        new_view_path = dest / new_relative
        if new_view_path.exists():
            raise ValueError('New image path conflicts with another file')
        shutil.copy2(image, new_view_path)
        previous_hash = v.get('image_sha256')
        v.update({'image_relative': new_relative,
                  'image_original_path': str(new_view_path.resolve()),
                  'image_sha256': sha256_file(new_view_path),
                  'image_size_wh': list(image_size),
                  'name': image.name,
                  'calibration_transferred_from_image_sha256': previous_hash,
                  'calibration_transfer_requires_visual_review': True})
        # Make all absolute path hints point to the new independent site.
        for other in meta['views']:
            local_input = dest / other['image_relative']
            other['image_original_path'] = str(local_input.resolve())
        from copy import deepcopy
        original_cameras = [deepcopy({k: row[k] for k in ('K', 'rvec', 'tvec', 'P')})
                            for row in meta['views']]
        with Image.open(new_view_path) as im:
            clean = im.convert('RGB')
        P = np.asarray(v['P'], dtype=np.float64)
        ref = np.asarray(meta['reference_vertices_3d_m'], dtype=np.float64)
        proposed = np.asarray(meta['vertices_3d_m'], dtype=np.float64)
        mask, stats = mask_for_view(P, proposed, clean.size,
                                    v.get('protected_polygons_px', []),
                                    v.get('allowed_polygon_px'))
        mask_path = dest / v['mask_relative']
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
        canny = architectural_canny(clean, mask, P, proposed,
                                     v['rvec'], v['tvec'], int(meta['num_floors']))
        canny_path = dest / v['canny_relative']
        canny_path.parent.mkdir(parents=True, exist_ok=True)
        canny.save(canny_path)
        preview = make_diagnostic_preview(clean, mask, P, ref, proposed,
                                           v.get('anchors_px', {}),
                                           v.get('protected_polygons_px', []),
                                           v.get('allowed_polygon_px'))
        preview_path = dest / v['preview_relative']
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(preview_path)
        v['mask_stats'] = stats
        meta['calibration_transfer'] = {
            'utc': datetime.now(timezone.utc).isoformat(),
            'source_project_name': site.name,
            'replaced_view': view_number,
            'previous_image_sha256': previous_hash,
            'new_image_sha256': v['image_sha256'],
            'camera_parameters_reused_unchanged': True,
            'alignment_check': alignment,
            'unverified_override': bool(not alignment.get('passed')),
            'note': 'Old PnP residuals refer to source photograph; review new landmarks and protected polygons.'
        }
        (dest / 'calibration_multiview.json').write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        for original, row in zip(original_cameras, meta['views']):
            if any(original[k] != row[k] for k in original):
                raise RuntimeError('Camera matrices or poses unexpectedly changed')
        checks = validate_project(dest, meta)
        if not all(row['mask_reproducible_exact'] for row in checks):
            raise RuntimeError('Mask verification failed')
        (dest / 'photo_transfer_audit.json').write_text(
            json.dumps({**meta['calibration_transfer'],
                        'mask_reproducibility': checks}, indent=2, ensure_ascii=False),
            encoding='utf-8')
        return {'project': str(dest), 'alignment': alignment, 'mask_checks': checks}
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True, help='Existing calibrated site folder')
    parser.add_argument('--view', type=int, required=True, choices=[1,2,3,4])
    parser.add_argument('--image', type=Path, required=True, help='Cleaner image with identical framing')
    parser.add_argument('--dest', type=Path, required=True, help='New site folder; must not exist')
    parser.add_argument('--allow-unverified', action='store_true',
                        help='Override the feature-alignment test after manual verification')
    args = parser.parse_args()
    result = transfer(args.site, args.image, args.view, args.dest,
                      allow_unverified=args.allow_unverified)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print('Review the projection and protected polygons before generating new scenarios.')


if __name__ == '__main__':
    main()
