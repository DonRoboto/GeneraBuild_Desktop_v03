"""Crea cuatro vistas sintéticas (NO representan Guanajuato 8) y su calibración.
Permite probar Calibration Studio, edición geométrica y CLI sin fotos del usuario.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from generabuild.geometry import box_vertices, intrinsic_matrix,project_camera
from generabuild.calibration import calibrate_all,save_calibration
from generabuild.storage import site_path


def look_at(w,h,fov,center,target):
    C=np.asarray(center,float);T=np.asarray(target,float)
    fwd=T-C;fwd/=np.linalg.norm(fwd)
    right=np.cross(fwd,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(fwd,right)
    R=np.stack([right,down,fwd]);t=-R@C
    K=intrinsic_matrix(w,h,fov)
    rvec,_=cv2.Rodrigues(R)
    return dict(P=K@np.c_[R,t],rvec=rvec.ravel().tolist(),tvec=t.tolist(),
                K=K.tolist(),camera_center_m=C.tolist())


def create_demo(name='synthetic_demo',size=(820,590)):
    folder=site_path(name)
    if (folder/'calibration_multiview.json').exists():
        print('Demo ya existe:',folder);return folder
    folder.mkdir(parents=True,exist_ok=True)
    paths=[];W,H=size
    cameras=[look_at(W,H,65,(33,-37,39),(5,9,12)),
             look_at(W,H,68,(-27,-29,36),(5,9,13)),
             look_at(W,H,65,(14,-28,7),(5,8,11)),
             look_at(W,H,110,(18,-24,6),(5,8,11))]
    old=box_vertices(10,20,9)
    for i,c in enumerate(cameras,1):
        image=np.zeros((H,W,3),np.uint8)
        image[:H//2,:]=[171,194,211];image[H//2:,:]=[171,163,151]
        cv2.rectangle(image,(0,H//2),(W//3,H-1),(130,126,112),-1)
        cv2.rectangle(image,(W*3//4,H//3),(W-1,H-1),(147,154,158),-1)
        uv=np.rint(project_camera(c['P'],old)).astype(np.int32)
        # Prisma abstracto de referencia, completamente sintético.
        for face in [[0,1,5,4],[1,2,6,5],[4,5,6,7]]:
            cv2.fillPoly(image,[uv[face]],(202,206,207))
            cv2.polylines(image,[uv[face]],True,(53,69,79),2)
        for t in (1/3,2/3):
            a=old[0]*(1-t)+old[4]*t;b=old[1]*(1-t)+old[5]*t
            xx=np.rint(project_camera(c['P'],np.array([a,b]))).astype(int)
            cv2.line(image,tuple(xx[0]),tuple(xx[1]),(70,83,89),2)
        cv2.putText(image,'SYNTHETIC DEMO - NOT GUANAJUATO 8',(22,36),
                    cv2.FONT_HERSHEY_SIMPLEX,.57,(55,60,65),2)
        path=folder/'inputs'/f'Vista_{i}.png';path.parent.mkdir(exist_ok=True,parents=True)
        Image.fromarray(image).save(path)
        paths.append(path)
    anchors=[]
    for cam in cameras:
        uv=project_camera(cam['P'],old)
        anchors.append({str(j):{'x':float(uv[j,0]),'y':float(uv[j,1])}
                       for j in range(8) if 0<=uv[j,0]<W and 0<=uv[j,1]<H})
    payload={
        'ref_W':10.,'ref_D':20.,'ref_H':9.,'num_floors':9,'floor_height_m':3.,
        'views_anchors':anchors,'fov_deg_by_view':[65,68,65,110],
        'protected_polygons_px':[[],[],[],[]],
        'allowed_polygons_px':[None]*4,'visual_inspection_confirmed':True,
    }
    results=calibrate_all(payload,paths)
    if not all(x['ok'] for x in results):
        raise RuntimeError('Error calibrando demo: '+str(results))
    save_calibration(payload,results,paths,folder)
    print('Demo creada:',folder)
    return folder

if __name__=='__main__':create_demo()
