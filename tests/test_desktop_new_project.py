"""Regresión: creación desde cero de un sitio y calibración FR2 completa."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from generabuild.project_setup import new_draft, load_draft, save_draft
from generabuild.storage import image_paths, load_metadata
from generabuild.calibration import calibrate_all, save_calibration
from generabuild.scenario import config_from_metadata, prepare_run


DEMO = Path(__file__).resolve().parents[1] / 'workspace' / 'sites' / 'synthetic_demo'


def setup_site(tmp_path):
    reference=load_metadata(DEMO)
    base=reference['reference_box_m']
    source_images=image_paths(DEMO,reference)
    target=new_draft('test_new_site',source_images,workspace=tmp_path,
                     ref_w=base['W'],ref_d=base['D'],ref_h=base['H'],
                     fovs=[v['fov_horizontal_deg'] for v in reference['views']])
    return target,reference


def test_new_project_preserves_four_images_and_draft(tmp_path):
    target,reference=setup_site(tmp_path)
    draft=load_draft(target)
    assert draft['format']=='GeneraBuild_Desktop_Draft_v1'
    assert len(draft['views'])==4
    assert not (target/'calibration_multiview.json').exists()
    assert [str(p.name) for p in image_paths(target)] == [f'Vista_{i}.png' for i in range(1,5)]
    assert [v['fov_horizontal_deg'] for v in draft['views']] == [65,68,65,110]
    for original,copied in zip(image_paths(DEMO,reference),image_paths(target)):
        assert original.read_bytes()==copied.read_bytes()


def test_draft_edits_persist_and_cannot_overwrite_images(tmp_path):
    target,_=setup_site(tmp_path)
    saved=save_draft(target,{'reference_box_m':{'W':10,'D':20,'H':9},
                             'views':[{'anchors_px':{'0':{'x':123,'y':44}}},
                                      {},{},{}]})
    assert load_draft(target)['views'][0]['anchors_px']['0']['x']==123
    assert all(p.exists() for p in image_paths(target))
    with pytest.raises(FileExistsError):
        new_draft('test_new_site',image_paths(target),workspace=tmp_path)


def test_new_project_rejects_invalid_photo_without_creating_site(tmp_path):
    original=Image.new('RGB',(400,300),'white')
    paths=[]
    for i in range(3):
        p=tmp_path/f'photo_{i}.png';original.save(p);paths.append(p)
    bad=tmp_path/'photo_4.png';bad.write_bytes(b'not an image');paths.append(bad)
    with pytest.raises(ValueError):
        new_draft('broken',paths,workspace=tmp_path)
    assert not (tmp_path/'broken').exists()


def test_complete_project_workflow_with_real_core_and_four_cameras(tmp_path):
    target,reference=setup_site(tmp_path)
    base=reference['reference_box_m']
    fovs=[v['fov_horizontal_deg'] for v in reference['views']]
    annotations=[v['anchors_px'] for v in reference['views']]
    draft_data={'reference_box_m':base,'floor_height_m':3,'num_floors':9,
                'views':[{'anchors_px':a} for a in annotations]}
    save_draft(target,draft_data)
    payload={
        'ref_W':base['W'],'ref_D':base['D'],'ref_H':base['H'],
        'floor_height_m':3.0,'num_floors':9,
        'views_anchors':annotations,'fov_deg_by_view':fovs,
        'allowed_polygons_px':[v['allowed_polygon_px'] for v in reference['views']],
        'protected_polygons_px':[v['protected_polygons_px'] for v in reference['views']],
        'guide_lines':[[] for _ in range(4)],'observed_lines':[[] for _ in range(4)],
        'visual_inspection_confirmed':True,
    }
    cams=calibrate_all(payload,image_paths(target))
    assert all(c['ok'] for c in cams)
    assert all(c['rmse_px']<0.01 for c in cams)
    meta=save_calibration(payload,cams,image_paths(target),target)
    assert (target/'calibration_multiview.json').exists()
    assert len(meta['views'])==4
    assert np.allclose(np.asarray(meta['views'][3]['P']),np.asarray(reference['views'][3]['P']),atol=1e-4)
    config=config_from_metadata(meta,floors=9,mode='ampliacion')
    run=prepare_run(target,config)
    assert (run/'scenario_config.json').is_file()
    for view in meta['views']:
        assert (run/view['mask_relative']).is_file()
        assert (run/view['canny_relative']).is_file()
        assert (run/view['preview_relative']).is_file()
    assert (target/'calibration_multiview.json').read_bytes() # original intact
