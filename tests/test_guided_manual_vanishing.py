"""v0.5: guía trazada por VP, aristas independientes y persistencia sin tocar calibración."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.create_demo_site import look_at, create_demo
from generabuild.geometry import box_vertices,project_camera
from generabuild.manual_lines import (
    line_through_point_and_vp, normalize_line_annotations,
    fit_manual_vanishing, manual_line_report,manual_axis_error_deg,
    predicted_line_angle_residual_deg,save_manual_annotations,
)
from generabuild.vanishing import camera_vanishing_points,draw_manual_annotations_overlay
from generabuild.joint_calibration import refine_joint_cameras
from tests.test_geometric_core_multiview import _synthetic_case


def test_guide_goes_through_chosen_pixel_and_vp_outside_screen():
    point={'x':330.,'y':240.}
    vp={'xy':[1450.,-620.], 'at_infinity':False}
    result=line_through_point_and_vp(point,vp,(880,600))
    assert result and len(result)==4
    p=np.array([result[0],result[1]])
    q=np.array([result[2],result[3]])
    v=np.asarray(vp['xy'])
    assert abs(np.linalg.det(np.stack((q-p,v-p))))<1e-6
    assert 0<=p[0]<880 and 0<=q[0]<880
    assert 0<=p[1]<600 and 0<=q[1]<600


def test_parallel_guide_uses_vp_at_infinity():
    point={'x':210.,'y':180.}
    vp={'at_infinity':True,'xy':None,'direction':[1.,0.]}
    result=line_through_point_and_vp(point,vp,(600,400))
    assert result and np.allclose([result[1],result[3]],[180,180])
    assert np.allclose([result[0],result[2]],[0,599])


def test_manual_segments_infer_vp_independently():
    truevp=np.array([580.,135.])
    segments=[]
    for angle in (.23,.91,1.5,2.25):
        direction=np.array([np.cos(angle),np.sin(angle)])
        a=truevp+45*direction
        b=truevp+240*direction
        segments.append({'axis':0,'p1':{'x':float(a[0]),'y':float(a[1])},
                         'p2':{'x':float(b[0]),'y':float(b[1])}})
    # a.y beyond height? use 950x900
    fit=fit_manual_vanishing(segments,(950,900),0)
    assert fit['status']=='ok',fit
    assert np.linalg.norm(np.array(fit['xy'])-truevp)<.1
    report=manual_line_report(segments,(950,900),[{'axis':0,'xy':truevp.tolist(),'at_infinity':False}])
    assert report['axes'][0]['observed_segments']==4
    assert report['axes'][0]['mean_angle_to_camera_deg'] < .001
    assert report['axes'][1]['manual_vp']['status']=='insufficient'


def test_manual_validation_and_visual_overlay():
    guide=[{'axis':0,'source':'opencv_0','point':{'x':90,'y':40}}]
    seg=[{'axis':1,'p1':{'x':30,'y':25},'p2':{'x':220,'y':145}}]
    g,o=normalize_line_annotations(guide,seg,(400,250))
    assert g[0]['source']=='opencv_0' and len(o)==1
    overlay=draw_manual_annotations_overlay(Image.new('RGB',(400,250),(100,100,100)),g,o,
                                             [{'axis':0,'xy':[800.,-350.],'at_infinity':False}],
                                             [{'xy':[170.,10.]}])
    assert overlay.size==(400,250)
    assert np.count_nonzero(np.array(overlay)!=100)>200


def test_save_only_manual_annotation_preserves_camera_geometry(tmp_path,monkeypatch):
    from scripts import create_demo_site
    monkeypatch.setattr(create_demo_site,'site_path',lambda name:tmp_path/name)
    folder=create_demo_site.create_demo('testmanual',size=(680,480))
    path=folder/'calibration_multiview.json'
    before=json.loads(path.read_text())
    all_guids=[[],[],[],[]]
    all_lines=[[],[],[],[]]
    all_guids[3]=[{'axis':2,'source':'camera','point':{'x':300.,'y':100.}}]
    all_lines[0]=[{'axis':0,'p1':{'x':10.,'y':10.},'p2':{'x':200.,'y':20.}}]
    saved=save_manual_annotations(folder,all_guids,all_lines)
    assert saved['views'][3]['manual_vp_guides_px'][0]['axis']==2
    assert saved['views'][0]['observed_axis_segments_px'][0]['axis']==0
    for a,b in zip(before['views'],saved['views']):
        for key in ('K','P','rvec','tvec','fov_horizontal_deg','image_sha256','mask_relative'):
            assert a[key]==b[key]
    assert (folder/saved['views'][0]['mask_relative']).is_file()
    assert list(folder.glob('calibration_backup_manual_*.json'))


def test_joint_ba_can_include_real_lines_without_using_guides():
    initial,anchors,fovs,sizes=_synthetic_case()
    model=box_vertices(10,20,9)
    source=look_at(*sizes[0], fovs[0],(33,-37,39),(5,9,12))
    uv=project_camera(source['P'],model)
    lines=[[],[],[],[]]
    for axis,a,b in [(0,0,1),(0,4,5),(1,0,3),(1,1,2),(2,0,4),(2,1,5)]:
        p1,p2=uv[a],uv[b]
        if np.all(p1>=0) and np.all(p2>=0) and np.all(p1<np.array(sizes[0])) and np.all(p2<np.array(sizes[0])):
            lines[0].append({'axis':axis,
                'p1':{'x':float(p1[0]),'y':float(p1[1])},
                'p2':{'x':float(p2[0]),'y':float(p2[1])}})
    assert lines[0], 'Se necesita al menos una arista visible en el caso sintético'
    result=refine_joint_cameras(initial,anchors,fovs,sizes,(10.6,18.6,9),
        observed_lines_by_view=lines,line_weight_px_per_degree=1.5,max_nfev=90)
    report=result['report']
    assert report['solver_success']
    assert report['observed_manual_segments']==len(lines[0])
    assert np.isfinite(report['initial_manual_mean_angular_error_deg'])
    assert report['refined_manual_mean_angular_error_deg'] < report['initial_manual_mean_angular_error_deg'] + .5
    assert report['fovs_locked']==fovs
