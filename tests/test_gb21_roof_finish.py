"""Pruebas CPU de v2.1-FR con el mismo proyecto sintético de v2.1-F.

pytest -q test_gb21_roof_finish.py
"""
import json
from pathlib import Path

import numpy as np
import cv2
import pytest
from PIL import Image

from gb21_roof_finish import (RoofConfig, make_roof_atlas,
                              finish_roof_one_view, finish_existing_roofs)
from test_gb21_photo_finish import setup_project


def test_roof_atlas_same_seed_and_no_grid():
    a=make_roof_atlas(10,20,RoofConfig(seed=2026,pixels_per_m=22))
    b=make_roof_atlas(10,20,RoofConfig(seed=2026,pixels_per_m=22))
    assert a.shape==(440,220,3)
    assert np.array_equal(a,b)
    assert not np.array_equal(a,make_roof_atlas(10,20,RoofConfig(seed=2,pixels_per_m=22)))
    assert np.mean(a)>140


def test_roof_mask_changed_only_aerial_and_outside_unchanged(tmp_path):
    meta=setup_project(tmp_path)
    cfg=RoofConfig(pixels_per_m=25,roof_equipment=0)
    atlas=make_roof_atlas(10,20,cfg)
    for idx in [1,2]:
        source=np.array(Image.open(tmp_path/'generated_realistic'/f'Vista_{idx}_9pisos_completo_s1234.png'))
        mask=np.array(Image.open(tmp_path/f'mask{idx}.png'))
        output,roof,report=finish_roof_one_view(Image.fromarray(source),mask,meta,idx,atlas,cfg)
        out=np.asarray(output)
        assert report['status']=='ok'
        assert report['roof_edited_px']>0
        assert not np.array_equal(out[roof>0],source[roof>0])
        assert np.array_equal(out[roof==0],source[roof==0])
        assert np.array_equal(out[mask==0],source[mask==0])
    # Street views are kept pixel-identical even with the same roof atlas.
    source=np.array(Image.open(tmp_path/'generated_realistic'/'Vista_4_9pisos_completo_s1234.png'))
    mask=np.array(Image.open(tmp_path/'mask4.png'))
    image, roof, rep=finish_roof_one_view(Image.fromarray(source),mask,meta,4,atlas,cfg)
    assert np.array_equal(np.asarray(image),source)
    assert not roof.any()


def test_preserve_original_building(tmp_path):
    meta=setup_project(tmp_path)
    cfg=RoofConfig(pixels_per_m=21,preserve_existing=True)
    atlas=make_roof_atlas(10,20,cfg)
    source=np.array(Image.open(tmp_path/'generated_realistic'/'Vista_1_9pisos_ampliacion_s1234.png'))
    mask=np.array(Image.open(tmp_path/'mask1.png'))
    result,roof,stat=finish_roof_one_view(Image.fromarray(source),mask,meta,1,atlas,cfg,True)
    uv=np.asarray(meta['views'][0]['P'])@np.c_[np.asarray(meta['reference_vertices_3d_m']),np.ones(8)].T
    uv=(uv[:2]/uv[2]).T
    old=np.zeros(mask.shape,np.uint8)
    cv2.fillConvexPoly(old,cv2.convexHull(np.rint(uv).astype(np.int32)),255)
    from gb21_extension_mask import extension_work_mask
    permitted=extension_work_mask(mask,meta,1,True)
    assert np.count_nonzero(roof)>0
    assert np.array_equal(np.asarray(result)[(old>0)&(permitted==0)],
                          source[(old>0)&(permitted==0)])
    assert np.array_equal(np.asarray(result)[roof==0],source[roof==0])


def test_run_roof_project_no_json_mutation_and_png_outputs(tmp_path):
    meta=setup_project(tmp_path)
    js=tmp_path/'calibration_multiview.json'
    baseline=js.read_bytes()
    sources={i:tmp_path/'generated_realistic'/f'Vista_{i}_9pisos_completo_s1234.png' for i in [1,2]}
    rec,atlas=finish_existing_roofs(tmp_path,views=(1,2),
                config=RoofConfig(pixels_per_m=20),source_paths=sources,preview=False)
    assert js.read_bytes()==baseline
    assert len(rec)==2
    assert rec[0]['status']=='ok' and rec[1]['status']=='ok'
    for item in rec:
        assert Path(item['output']).exists()
        assert Path(item['roof_mask']).exists()
    old=np.array(Image.open(sources[1]))
    new=np.array(Image.open(rec[0]['output']))
    assert not np.array_equal(old,new)
    out2,atlas2=finish_existing_roofs(tmp_path,views=(1,2),
                config=RoofConfig(pixels_per_m=20),source_paths=sources,preview=False)
    assert np.array_equal(atlas,atlas2)
    assert np.array_equal(np.asarray(Image.open(rec[0]['output'])),new)
    assert meta['views'][3]['fov_horizontal_deg']==110


def test_disallow_invalid_and_deformed_roof(tmp_path):
    with pytest.raises(ValueError):
        RoofConfig(pixels_per_m=5).validate()
    meta=setup_project(tmp_path)
    meta['vertices_3d_m'][6][2]+=2
    (tmp_path/'calibration_multiview.json').write_text(json.dumps(meta),encoding='utf-8')
    with pytest.raises(ValueError,match='horizontal'):
        finish_existing_roofs(tmp_path,preview=False)
