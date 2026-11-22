"""Pruebas de regresión: OpenCV/Hough y BA con geometría compartida."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from scripts.create_demo_site import look_at
from generabuild.geometry import box_vertices, project_camera, calibrate_view, intrinsic_matrix
from generabuild.vanishing import (
    detect_segments, camera_vanishing_points, assign_segments,
    draw_vanishing_overlay, estimate_vanishing_candidates,
)
from generabuild.joint_calibration import refine_joint_cameras


def test_hough_returns_original_coordinates_after_downsample():
    image=np.zeros((1100,1800,3),np.uint8)
    cv2.line(image,(100,160),(1590,160),(255,255,255),6)
    cv2.line(image,(280,260),(1700,920),(255,255,255),5)
    segments=detect_segments(Image.fromarray(image),max_side=900,canny_low=50)
    assert len(segments)>=2
    assert any(max(s[0],s[2])>1450 for s in segments)  # escala original
    assert all(0<=s[0]<=1800 and 0<=s[2]<=1800 for s in segments)


def test_projective_vanishing_points_and_line_assignment():
    W,H=820,590
    c=look_at(W,H,65,(25,-30,24),(5,8,11))
    c['ok']=True
    vps=camera_vanishing_points(c,(W,H))
    assert len(vps)==3
    assert {v['axis'] for v in vps}=={0,1,2}
    xyz=np.array([[0,0,0],[5,0,0],[10,0,0]],float)
    uv=project_camera(c['P'],xyz)
    segment=[float(uv[0,0]),float(uv[0,1]),float(uv[2,0]),float(uv[2,1])]
    lines,stats=assign_segments([segment],vps,tolerance_deg=2)
    assert len(lines)==1 and lines[0]['axis']==0
    overlay=draw_vanishing_overlay(Image.new('RGB',(W,H),(70,80,90)),{
        'vanishing_points':vps,'segments':lines,'empirical_vps':[]})
    assert overlay.size==(W,H)


def test_ransac_vp_candidates_independent_of_camera():
    rng=np.random.default_rng(42)
    vp=np.array([285.,165.]);segs=[]
    for angle in np.linspace(0.1,2.8,17):
        v=np.array([np.cos(angle),np.sin(angle)])
        a=vp+35*v+rng.normal(0,.1,2)
        b=vp+235*v+rng.normal(0,.1,2)
        segs.append([a[0],a[1],b[0],b[1]])
    found=estimate_vanishing_candidates(segs,(850,600),n_candidates=1,trials=450,
                                        threshold_px=3.0,min_inliers=10)
    assert found and found[0]['support']>=14
    assert np.linalg.norm(np.asarray(found[0]['xy'])-vp)<3.0


def _synthetic_case():
    W,H=960,720
    fovs=[65,68,65,110]
    poses=[((33,-37,39),(5,9,12)),((-27,-29,36),(5,9,12)),
           ((14,-28,7),(5,9,12)),((18,-24,6),(5,9,12))]
    cams=[look_at(W,H,fov,*pose) for fov,pose in zip(fovs,poses)]
    original=box_vertices(10,20,9)
    rng=np.random.default_rng(2026)
    anchors=[];base=[]
    for i,c in enumerate(cams):
        uv=project_camera(c['P'],original)
        annotations={str(j):{'x':float(p[0]+rng.normal(0,.65)),
                             'y':float(p[1]+rng.normal(0,.65))}
                     for j,p in enumerate(uv) if 0<p[0]<W and 0<p[1]<H}
        anchors.append(annotations)
        r=calibrate_view(annotations,W,H,10.6,18.6,9,fovs[i])
        assert r['ok'],r
        base.append(r)
    return base,anchors,fovs,[(W,H)]*4


def test_joint_ba_recovers_common_geometry_and_keeps_fov():
    cams,anchors,fovs,sizes=_synthetic_case()
    res=refine_joint_cameras(cams,anchors,fovs,sizes,(10.6,18.6,9),max_nfev=90)
    report=res['report']
    assert report['solver_success']
    assert report['refined_pooled_rmse_px']<report['initial_pooled_rmse_px']*.40
    assert abs(res['reference_dimensions_m']['W']-10.)<.25
    assert abs(res['reference_dimensions_m']['D']-20.)<.3
    assert report['fovs_locked']==fovs
    for i,c in enumerate(res['cameras']):
        assert np.allclose(c['K'],intrinsic_matrix(*sizes[i],fovs[i]))
        assert c['ok'] and np.isfinite(c['rmse_px'])


def test_joint_ba_refuses_unobservable_depth():
    cams,anchors,fovs,sizes=_synthetic_case()
    front={j:pt for j,pt in anchors[0].items() if int(j) in (0,1,4,5)}
    with pytest.raises(ValueError,match='depth'):
        refine_joint_cameras(cams,[front]*4,fovs,sizes,(10.6,18.6,9))


def test_joint_proposal_can_be_saved_with_backup_and_rebuilt_masks(tmp_path,monkeypatch):
    """Simula aceptación explícita y verifica formato FR2 y backup previo."""
    from scripts import create_demo_site
    from generabuild.storage import load_metadata, image_paths
    from generabuild.calibration import save_calibration
    from generabuild.geometry import validate_project
    monkeypatch.setattr(create_demo_site,'site_path',lambda name:tmp_path/name)
    folder=create_demo_site.create_demo('demo',size=(680,480))
    original=(folder/'calibration_multiview.json').read_bytes()
    meta=load_metadata(folder)
    cams=[{'P':v['P'],'K':v['K'],'rvec':v['rvec'],'tvec':v['tvec'],'ok':True} for v in meta['views']]
    anchors=[v['anchors_px'] for v in meta['views']]
    fovs=[v['fov_horizontal_deg'] for v in meta['views']]
    sizes=[tuple(v['image_size_wh']) for v in meta['views']]
    dim=meta['reference_box_m']
    candidate=refine_joint_cameras(cams,anchors,fovs,sizes,(dim['W'],dim['D'],dim['H']))
    d=candidate['reference_dimensions_m']
    payload={
        'ref_W':d['W'],'ref_D':d['D'],'ref_H':d['H'],
        'num_floors':9,'floor_height_m':3.,'views_anchors':anchors,
        'fov_deg_by_view':fovs,
        'protected_polygons_px':[v['protected_polygons_px'] for v in meta['views']],
        'allowed_polygons_px':[v['allowed_polygon_px'] for v in meta['views']],
        'visual_inspection_confirmed':True,
    }
    updated=save_calibration(payload,candidate['cameras'],image_paths(folder,meta),folder)
    assert len(updated['views'])==4
    assert updated['views'][3]['fov_horizontal_deg']==110
    backups=list(folder.glob('calibration_backup_*.json'))
    assert backups and backups[0].read_bytes()==original
    assert all(row['mask_reproducible_exact'] for row in validate_project(folder,updated))
