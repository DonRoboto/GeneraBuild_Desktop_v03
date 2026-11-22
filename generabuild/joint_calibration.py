"""Refinamiento multivista con geometría compartida (bundle adjustment restringido).

IMPORTANTE: La geometría 3D NO puede mantenerse completamente fija si se desea
acoplar matemáticamente las cuatro cámaras usando solo correspondencias vértice↔píxel.
Este módulo optimiza simultáneamente W y D (comunes), y las cuatro poses; H0
permanece fija para establecer la escala. Los FOV por vista permanecen FIJOS.
Los resultados son candidatos, no se guardan automáticamente.
"""
from __future__ import annotations

import math
from typing import Sequence
import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import box_vertices, intrinsic_matrix, project_camera
from .manual_lines import normalize_line_annotations, predicted_line_angle_residual_deg


def _cam_data(cam: dict, w: int, h: int, fov: float):
    if not cam or cam.get('ok') is False:
        raise ValueError('All four initial cameras must have valid PnP poses')
    if not all(k in cam for k in ('rvec', 'tvec')):
        raise ValueError('Camera rvec/tvec missing for joint refinement')
    K = intrinsic_matrix(w,h,fov)
    return (K, np.r_[np.asarray(cam['rvec'],float).reshape(3),
                     np.asarray(cam['tvec'],float).reshape(3)])


def _observations(anchors: dict, w: int, h: int):
    ids = sorted(int(i) for i in anchors)
    if len(ids) < 4 or len(set(ids)) != len(ids) or any(i<0 or i>7 for i in ids):
        raise ValueError('Each view needs at least four valid physical corners')
    pts=np.array([[float(anchors[str(i)]['x']),float(anchors[str(i)]['y'])]
                  for i in ids],float)
    if not np.all(np.isfinite(pts)) or np.any(pts[:,0]<0) or np.any(pts[:,0]>=w) or \
       np.any(pts[:,1]<0) or np.any(pts[:,1]>=h):
        raise ValueError('Photo annotations are outside image bounds or non-finite')
    return np.array(ids,dtype=int),pts


def refine_joint_cameras(initial_cameras: list[dict],
                         anchors_by_view: list[dict],
                         fovs: list[float],
                         image_sizes: list[tuple[int,int]],
                         ref_dimensions: tuple[float,float,float],
                         *,geometry_prior_fraction: float=.08,
                         allowed_dimension_change: float=.15,
                         max_nfev: int=160,
                         observed_lines_by_view: list[list[dict]] | None=None,
                         line_weight_px_per_degree: float=1.5) -> dict:
    """BA conjunto 4 cámaras + 2 parámetros compartidos W,D, H0 fija.

    La región posterior necesita observaciones reales en al menos una vista;
    de lo contrario la profundidad no es identificable y se rechaza el ajuste.
    La solución incluye poses y dimensiones *propuestas*; no modifica archivos.
    """
    if len(initial_cameras)!=4 or len(anchors_by_view)!=4 or len(fovs)!=4 or len(image_sizes)!=4:
        raise ValueError('Joint synchronization requires exactly four cameras and annotation sets')
    W0,D0,H0=map(float,ref_dimensions)
    if min(W0,D0,H0)<=0 or not np.all(np.isfinite([W0,D0,H0])):
        raise ValueError('Invalid reference dimensions')
    if not (0.01<=geometry_prior_fraction<=0.50 and 0.01<=allowed_dimension_change<=.5):
        raise ValueError('Regularization parameters are out of range')
    observations=[]
    intrinsics=[]
    pose0=[]
    manual_observations=[]
    if observed_lines_by_view is not None and len(observed_lines_by_view)!=4:
        raise ValueError('Observed structural lines must cover four views')
    if not np.isfinite(line_weight_px_per_degree) or not 0 <= line_weight_px_per_degree <= 10:
        raise ValueError('Angular constraint weight is out of range')
    depth_constrained_views=0
    for i in range(4):
        w,h=map(int,image_sizes[i])
        ids,pts=_observations(anchors_by_view[i],w,h)
        if any(j in (0,1,4,5) for j in ids) and any(j in (2,3,6,7) for j in ids):
            depth_constrained_views+=1
        observations.append((ids,pts))
        K,pose=_cam_data(initial_cameras[i],w,h,float(fovs[i]))
        intrinsics.append(K)
        pose0.append(pose)
        if observed_lines_by_view is not None:
            _, observed_lines = normalize_line_annotations([], observed_lines_by_view[i], (w,h))
        else:
            observed_lines = []
        manual_observations.append(observed_lines)
    if depth_constrained_views==0:
        raise ValueError(
            'Shared depth D is unobservable: mark physical front and '
            'rear corners in at least one aerial view. Do not invent hidden points.'
        )

    x0=np.r_[np.asarray(pose0).reshape(-1),W0,D0]
    lb=np.r_[np.full(24,-np.inf),W0*(1-allowed_dimension_change),D0*(1-allowed_dimension_change)]
    ub=np.r_[np.full(24,np.inf),W0*(1+allowed_dimension_change),D0*(1+allowed_dimension_change)]
    sigma_W=max(.0001,W0*geometry_prior_fraction)
    sigma_D=max(.0001,D0*geometry_prior_fraction)
    n_obs=sum(len(o[0]) for o in observations)
    weights=[1./max(1.,math.hypot(*image_sizes[i])/1100.) for i in range(4)]

    def residual(x,with_priors=True):
        W,D=float(x[-2]),float(x[-1])
        box=box_vertices(W,D,H0)
        parts=[]
        for i,((ids,obs),K) in enumerate(zip(observations,intrinsics)):
            rvec=x[6*i:6*i+3].reshape(3,1)
            tvec=x[6*i+3:6*i+6].reshape(3,1)
            xyz=box[ids]
            projected,_=cv2.projectPoints(xyz,rvec,tvec,K,None)
            parts.append((projected.reshape(-1,2)-obs).ravel()*weights[i])
            # Evita voltear las cámaras durante la optimización.
            R,_=cv2.Rodrigues(rvec)
            depths=(R@box.T+tvec).T[:,2]
            parts.append(np.maximum(0.,.05-depths)*100.)
            # Únicamente aristas observadas (dos extremos reales), NO guías por VP.
            if manual_observations[i] and line_weight_px_per_degree>0:
                parts.append(np.asarray([
                    predicted_line_angle_residual_deg(seg,K,R) * line_weight_px_per_degree
                    for seg in manual_observations[i]
                ]))
        if with_priors:
            # Dos restricciones *compartidas*, expresadas en píxeles equivalentes.
            parts.append(np.array([2.*(W-W0)/sigma_W,2.*(D-D0)/sigma_D]))
        return np.concatenate(parts)

    opt=least_squares(residual,x0,method='trf',loss='soft_l1',f_scale=3.,
                      bounds=(lb,ub),max_nfev=int(max_nfev),x_scale='jac')
    if not np.all(np.isfinite(opt.x)):
        raise RuntimeError('Joint optimization produced non-finite parameters')
    W,D=float(opt.x[-2]),float(opt.x[-1])
    box=box_vertices(W,D,H0)
    refined=[]
    initial_errors=[];final_errors=[]
    initial_line_errors=[];final_line_errors=[]
    for i,((ids,obs),K) in enumerate(zip(observations,intrinsics)):
        rvec=opt.x[i*6:i*6+3].reshape(3,1)
        tvec=opt.x[i*6+3:i*6+6].reshape(3,1)
        R,_=cv2.Rodrigues(rvec)
        depths=(R@box.T+tvec).T[:,2]
        if min(depths)<=1e-6:
            raise RuntimeError(f'Camera {i+1} ended with points behind it')
        P=K@np.column_stack([R,tvec])
        if manual_observations[i]:
            final_line_errors.extend(abs(predicted_line_angle_residual_deg(seg,K,R))
                                     for seg in manual_observations[i])
        uv=project_camera(P,box[ids])
        err=np.linalg.norm(uv-obs,axis=1)
        final_errors.extend(err.tolist())
        cam0=initial_cameras[i]
        oldK,oldx=_cam_data(cam0,*image_sizes[i],fovs[i])
        Rold,_=cv2.Rodrigues(oldx[:3].reshape(3,1))
        if manual_observations[i]:
            initial_line_errors.extend(abs(predicted_line_angle_residual_deg(seg,K,Rold))
                                       for seg in manual_observations[i])
        Pold=oldK@np.c_[Rold,oldx[3:].reshape(3,1)]
        olduv=project_camera(Pold,box_vertices(W0,D0,H0)[ids])
        olderr=np.linalg.norm(olduv-obs,axis=1)
        initial_errors.extend(olderr.tolist())
        width,height=image_sizes[i]
        rmse=float(np.sqrt(np.mean(err**2)))
        centered=box[ids]-box[ids].mean(axis=0)
        planar=bool(np.linalg.matrix_rank(centered,tol=1e-7)<=2)
        refined.append({
            'ok':True,'K':K.tolist(),'rvec':rvec.ravel().tolist(),'tvec':tvec.ravel().tolist(),
            'P':P.tolist(),'camera_center_m':(-R.T@tvec).ravel().tolist(),
            'rmse_px':rmse,'rmse_diagonal_pct':100.*rmse/math.hypot(width,height),
            'per_vertex_error_px':{str(j):float(e) for j,e in zip(ids,err)},
            'n_anchors':len(ids),'planar':planar,
            'warnings':(['Coplanar points only: verify depth using aerial views'] if planar else []),
            'joint_refined':True,
        })
    e0=np.asarray(initial_errors)
    e1=np.asarray(final_errors)
    report={
        'method':'joint_geometry_and_poses_fixed_FOV',
        'description':'Restricted BA: four poses + shared width/depth. H0 height and all FOVs fixed.',
        'solver_success':bool(opt.success),
        'solver_message':str(opt.message),
        'nfev':int(opt.nfev),
        'initial_pooled_rmse_px':float(np.sqrt(np.mean(e0**2))),
        'refined_pooled_rmse_px':float(np.sqrt(np.mean(e1**2))),
        'reference_dimensions_m':{'W':W0,'D':D0,'H':H0},
        'suggested_dimensions_m':{'W':W,'D':D,'H':H0},
        'width_change_pct':100*(W/W0-1),
        'depth_change_pct':100*(D/D0-1),
        'total_observations':int(n_obs),
        'observed_manual_segments':len(final_line_errors),
        'manual_line_weight_px_per_degree':float(line_weight_px_per_degree),
        'initial_manual_mean_angular_error_deg':float(np.mean(initial_line_errors)) if initial_line_errors else None,
        'refined_manual_mean_angular_error_deg':float(np.mean(final_line_errors)) if final_line_errors else None,
        'n_views_with_depth_evidence':depth_constrained_views,
        'fovs_locked':list(map(float,fovs)),
        'warnings':[
            'Joint fitting does not establish absolute metric accuracy or replace survey data.',
            'RMSE uses fitted points; validate using corners or edges excluded from fitting.',
            'An incorrect prism footprint can bias all fitted camera poses.',
            'Only REAL observed structural edges provide angular constraints; VP guides are not independent evidence.'
        ],
    }
    if not opt.success:
        report['warnings'].append('Optimizer did not report convergence; do NOT accept automatically')
    if max(abs(report['width_change_pct']),abs(report['depth_change_pct']))>10:
        report['warnings'].append('Footprint change >10%: verify measured geometry and aerial images')
    if report['refined_pooled_rmse_px']>report['initial_pooled_rmse_px']*1.08+.5:
        report['warnings'].append('RMSE increased: the joint result may be worse than the original poses')
    return {'cameras':refined,'reference_dimensions_m':{'W':W,'D':D,'H':H0},'report':report}
