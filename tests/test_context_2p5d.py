import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from generabuild.context_2p5d import (
    image_to_ground, world_to_image, new_neighbor, save_context,
    load_context, validate_footprint, NeighborBuilding,
)
from generabuild.context_occlusion import neighbor_occlusion_mask,apply_context_occlusions_to_run
from generabuild.storage import load_metadata
from generabuild.scenario import config_from_metadata, prepare_run

DEMO = Path(__file__).resolve().parents[1]/'workspace/sites/synthetic_demo'


def test_backprojection_four_realistic_camera_matrices():
    meta=load_metadata(DEMO)
    base=np.array([[2,3,0],[8,4,0],[7,13,0],[1,13,0]],float)
    for view in meta['views']:
        P=np.asarray(view['P'],float)
        uv=world_to_image(P,base)
        actual=image_to_ground(P,uv)
        assert np.allclose(actual,base[:,:2],atol=1e-8)


def test_serialization_is_independent_of_calibration(tmp_path):
    calib=tmp_path/'calibration_multiview.json'
    calib.write_text('{"a":1}',encoding='utf-8')
    initial=hashlib.sha256(calib.read_bytes()).hexdigest()
    b=new_neighbor('Edificio alto',[[11,-5],[22,-5],[22,8],[11,8]],16,source_view=2)
    save_context(tmp_path,[b],calibration_sha256=initial)
    loaded=load_context(tmp_path)
    assert len(loaded)==1 and loaded[0].id==b.id
    assert loaded[0].height_m==16
    assert hashlib.sha256(calib.read_bytes()).hexdigest()==initial


def test_reject_self_crossing_and_concave_polygons():
    with pytest.raises(ValueError):validate_footprint([[0,0],[4,4],[0,4],[4,0]])
    with pytest.raises(ValueError):validate_footprint([[0,0],[4,0],[2,1],[4,4],[0,4]])


def test_neighbor_mesh_has_closed_faces():
    b=new_neighbor('N', [[0,0],[5,0],[5,7],[0,7]],12)
    assert b.vertices().shape==(8,3)
    assert len(b.triangles())==12


def test_depth_mask_detects_occluder_in_front():
    meta=load_metadata(DEMO)
    P=np.array(meta['views'][0]['P'],float)
    target=np.array(meta['vertices_3d_m'],float)
    b=new_neighbor('Occluder',[[17,-17],[23,-17],[23,-10],[17,-10]],34)
    blocked=neighbor_occlusion_mask(P,target,[b],(820,590))
    assert np.count_nonzero(blocked)>0
    assert blocked.dtype==np.uint8
    assert np.all(np.isin(np.unique(blocked),[0,255]))


def test_disabled_neighborhood_does_not_affect_images():
    meta=load_metadata(DEMO)
    b=new_neighbor('Invisible',[[17,-17],[23,-17],[23,-10],[17,-10]],34)
    b.enabled=False
    blocked=neighbor_occlusion_mask(meta['views'][0]['P'],meta['vertices_3d_m'],[b],(820,590))
    assert np.count_nonzero(blocked)==0


def test_optional_occlusion_clips_only_run_and_not_site(tmp_path):
    import shutil
    site=tmp_path/'site'
    shutil.copytree(DEMO,site)
    calibration=site/'calibration_multiview.json'
    before=calibration.read_bytes()
    b=new_neighbor('Occluder',[[17,-17],[23,-17],[23,-10],[17,-10]],34)
    save_context(site,[b],calibration_sha256=hashlib.sha256(before).hexdigest())
    meta=load_metadata(site)
    config=config_from_metadata(meta,floors=9,mode='ampliacion',photo_finish=False,roof_finish=False)
    run=prepare_run(site,config,run_id='test_demo_context')
    image0=np.asarray(Image.open(run/meta['views'][0]['mask_relative']).convert('L'))
    report=apply_context_occlusions_to_run(run,site)
    image1=np.asarray(Image.open(run/meta['views'][0]['mask_relative']).convert('L'))
    assert report['applied']
    assert np.all(image1<=image0)
    assert calibration.read_bytes()==before
    assert (run/'context_occlusion_report.json').exists()


def test_reject_calibration_mismatch(tmp_path):
    import shutil
    site=tmp_path/'site';shutil.copytree(DEMO,site)
    meta=load_metadata(site)
    b=new_neighbor('Occluder',[[17,-17],[23,-17],[23,-10],[17,-10]],34)
    save_context(site,[b],calibration_sha256='bad_hash')
    run=prepare_run(site,config_from_metadata(meta),run_id='test_mismatch')
    with pytest.raises(ValueError,match='calibración'):
        apply_context_occlusions_to_run(run,site)
