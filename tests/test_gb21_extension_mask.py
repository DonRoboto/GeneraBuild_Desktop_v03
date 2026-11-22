"""pytest -q test_gb21_extension_mask.py"""
import numpy as np
import cv2
from PIL import Image
from gb21_extension_mask import (extension_work_mask,extension_mask_diagnostics,
    polygon_silhouette, extension_volume)
from test_gb21_photo_finish import setup_project


def _project_old(meta,idx,shape):
    return polygon_silhouette(meta['views'][idx-1]['P'],
             meta['reference_vertices_3d_m'],shape)


def test_aerial_extension_recovers_original_silhouette_overlap(tmp_path):
    meta=setup_project(tmp_path)
    for idx in [1,2]:
        saved=np.array(Image.open(tmp_path/f'mask{idx}.png'))
        mask=extension_work_mask(saved,meta,idx,True)
        old=_project_old(meta,idx,saved.shape)
        recovered=(mask>0)&(old>0)
        assert np.count_nonzero(mask)>0
        assert np.count_nonzero(recovered)>0, f'Vista {idx}: el solape es esencial'
        assert extension_mask_diagnostics(saved,meta,idx)[
            'pixels_extension_hidden_by_legacy_mask']==np.count_nonzero(recovered)
        assert np.array_equal(mask[saved==0],np.zeros_like(mask[saved==0]))


def test_street_extension_preserves_base_and_context(tmp_path):
    meta=setup_project(tmp_path)
    for idx in [3,4]:
        saved=np.array(Image.open(tmp_path/f'mask{idx}.png'))
        mask=extension_work_mask(saved,meta,idx,True)
        old=_project_old(meta,idx,saved.shape)
        assert np.count_nonzero(mask)>0
        assert np.all(mask[saved==0]==0)
        assert np.count_nonzero((old>0)&(mask==0))>0
        assert meta['views'][idx-1]['fov_horizontal_deg']==([65,110][idx-3])


def test_full_mode_unchanged(tmp_path):
    meta=setup_project(tmp_path)
    for idx in range(1,5):
        saved=np.array(Image.open(tmp_path/f'mask{idx}.png'))
        assert np.array_equal(extension_work_mask(saved,meta,idx,False),saved)


def test_protected_occlusion_applies_in_aerial(tmp_path):
    meta=setup_project(tmp_path)
    saved=np.array(Image.open(tmp_path/'mask1.png'))
    mask=extension_work_mask(saved,meta,1,True)
    ys,xs=np.where(mask>0)
    assert len(xs)>10
    cy,cx=int(np.median(ys)),int(np.median(xs))
    saved[max(0,cy-8):cy+8,max(0,cx-9):cx+9]=0
    corrected=extension_work_mask(saved,meta,1,True)
    assert not corrected[max(0,cy-8):cy+8,max(0,cx-9):cx+9].any()


def test_height_parameter_and_footprint_validation(tmp_path):
    meta=setup_project(tmp_path)
    V=extension_volume(meta)
    assert V.shape==(8,3)
    assert np.allclose(V[:4,2],9)
    assert np.allclose(V[4:,2],27)
    original=meta['vertices_3d_m']
    meta['vertices_3d_m']=[row[:] for row in original]
    meta['vertices_3d_m'][4][0] += 3
    saved=np.array(Image.open(tmp_path/'mask1.png'))
    import pytest
    with pytest.raises(ValueError, match='ancho/profundidad'):
        extension_work_mask(saved,meta,1,True)


def test_zero_height_change_empty(tmp_path):
    meta=setup_project(tmp_path)
    meta['vertices_3d_m']=[row[:] for row in meta['reference_vertices_3d_m']]
    saved=np.array(Image.open(tmp_path/'mask1.png'))
    assert not np.any(extension_work_mask(saved,meta,1,True))


def test_aerial_base_preservation_is_not_pixelwise_entire_hull(tmp_path):
    meta=setup_project(tmp_path)
    saved=np.array(Image.open(tmp_path/'mask2.png'))
    work=extension_work_mask(saved,meta,2,True)
    old=_project_old(meta,2,saved.shape)
    orig=np.array(Image.open(tmp_path/'view2.png').convert('RGB'))
    out=orig.copy(); out[work>0]=[230,213,198]
    # La base permanece idéntica donde todavía es visible.
    retained=(old>0)&(work==0)
    assert np.array_equal(out[retained],orig[retained])
    assert np.count_nonzero((out!=orig).any(2)&(old>0))>0
    assert np.array_equal(out[work==0],orig[work==0])
