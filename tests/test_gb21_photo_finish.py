"""CPU: pytest -q test_gb21_photo_finish.py (sin GPU ni Google Colab)."""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from gb21_photo_finish import PhotoFinishConfig, create_photo_finish, finish_previous_results
from gb21_photo_finish import _matching_work_mask as working_mask_for_realism


def _camera(w,h,fov,c,target):
    C=np.array(c,dtype=float); target=np.array(target,dtype=float)
    forward=(target-C);forward/=np.linalg.norm(forward)
    right=np.cross(forward,np.array([0.,0.,1.]));right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    R=np.stack([right,down,forward],axis=0)
    t=-R@C
    f=w/(2*np.tan(np.deg2rad(fov)/2))
    P=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])@np.column_stack([R,t])
    r,_=cv2.Rodrigues(R)
    return {'rvec':r.ravel().tolist(),'tvec':t.tolist(),
            'P':P.tolist(),'fov_horizontal_deg':fov}


def setup_project(root,shape=(640,960)):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    H,W=shape
    width,depth,h0,h1=10.,20.,9.,27.
    v0=np.array([[0,0,0],[width,0,0],[width,depth,0],[0,depth,0],
                 [0,0,h0],[width,0,h0],[width,depth,h0],[0,depth,h0]],float)
    v1=v0.copy();v1[4:,2]=h1
    cameras=[_camera(W,H,65,[36,-43,35],[5,8,14]),
             _camera(W,H,68,[-37,-31,32],[5,8,14]),
             _camera(W,H,65,[17,-30,6],[5,7,14]),
             _camera(W,H,110,[18,-26,6],[5,7,14])]
    meta={'num_floors':9,'reference_vertices_3d_m':v0.tolist(),
          'vertices_3d_m':v1.tolist(),'views':[]}
    folder=root/'generated_realistic';folder.mkdir(exist_ok=True)
    for i,c in enumerate(cameras,1):
        orig=np.zeros((H,W,3),dtype=np.uint8)
        orig[:H//2]=[171,195,210]
        orig[H//2:]=[174,171,161]
        # Divisiones que simulan el contexto del edificio.
        cv2.rectangle(orig,(0,H//2),(W//4,H),(137,129,113),-1)
        cv2.rectangle(orig,(3*W//4,H//2),(W,H),(155,140,129),-1)
        uv=np.array(c['P'])@np.c_[v1,np.ones(len(v1))].T
        uv=(uv[:2]/uv[2]).T
        mask=np.zeros((H,W),np.uint8)
        cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype(np.int32)),255)
        Image.fromarray(orig).save(root/f'view{i}.png')
        Image.fromarray(mask).save(root/f'mask{i}.png')
        meta['views'].append({**c,'image_relative':f'view{i}.png',
                              'mask_relative':f'mask{i}.png'})
    (root/'calibration_multiview.json').write_text(json.dumps(meta),encoding='utf-8')
    # Simulamos un edificio inpainted gris, sin cristales; sin tocar el fondo.
    for i,c in enumerate(cameras,1):
        orig=np.array(Image.open(root/f'view{i}.png').convert('RGB'))
        mask=np.array(Image.open(root/f'mask{i}.png').convert('L'))
        for mode,protect in (('completo',False),('ampliacion',True)):
            work=working_mask_for_realism(mask,meta,i,protect)>127
            painted=orig.copy()
            texture=np.linspace(0,18,H)[:,None,None]
            gray=np.array([143,143,137],dtype=np.float32)[None,None,:] +texture
            painted[work]=np.broadcast_to(gray,(H,W,3)).astype(np.uint8)[work]
            Image.fromarray(painted).save(folder/f'Vista_{i}_9pisos_{mode}_s1234.png')
    return meta


def test_finish_existing_all_views(tmp_path):
    meta=setup_project(tmp_path)
    original_bytes=(tmp_path/'calibration_multiview.json').read_bytes()
    cfg=PhotoFinishConfig(material_seed=44,render_px_per_m=28,details_opacity=0.20)
    for i in range(1,5):
        src=tmp_path/'generated_realistic'/f'Vista_{i}_9pisos_completo_s1234.png'
        out,report=create_photo_finish(tmp_path,i,src,cfg,show_preview=False)
        assert out.exists()
        before=np.asarray(Image.open(src).convert('RGB'))
        after=np.asarray(Image.open(out).convert('RGB'))
        original=np.asarray(Image.open(tmp_path/f'view{i}.png').convert('RGB'))
        allowed=np.asarray(Image.open(tmp_path/f'mask{i}.png').convert('L'))>127
        assert np.array_equal(after[~allowed],original[~allowed])
        assert np.count_nonzero(after[allowed]!=before[allowed])>100
        assert report['glass_pixels_in_edit']>5
        assert report['frame_pixels_in_edit']>5
        assert report['context_pixels_changed']==0
        assert report['fov_horizontal_deg']==meta['views'][i-1]['fov_horizontal_deg']
    assert original_bytes==(tmp_path/'calibration_multiview.json').read_bytes()


def test_preserve_old_3_floors_and_determinism(tmp_path):
    meta=setup_project(tmp_path)
    src=tmp_path/'generated_realistic'/'Vista_4_9pisos_ampliacion_s1234.png'
    cfg=PhotoFinishConfig(material_seed=12,render_px_per_m=22)
    out,report=create_photo_finish(tmp_path,4,src,cfg,show_preview=False)
    a=np.asarray(Image.open(out).convert('RGB')).copy()
    original=np.asarray(Image.open(tmp_path/'view4.png').convert('RGB'))
    saved=np.asarray(Image.open(tmp_path/'mask4.png').convert('L'))
    allowed=working_mask_for_realism(saved,meta,4,True)>127
    assert np.array_equal(a[~allowed],original[~allowed])
    assert report['preserve_existing_3_floors'] is True
    create_photo_finish(tmp_path,4,src,cfg,show_preview=False)
    b=np.asarray(Image.open(out).convert('RGB'))
    assert np.array_equal(a,b)


def test_auto_mode_and_existing_images(tmp_path):
    setup_project(tmp_path)
    outputs=finish_previous_results(tmp_path,views=(3,4),
                      config=PhotoFinishConfig(render_px_per_m=22),show_preview=False)
    assert len(outputs)==2
    assert all(a.exists() for a,_ in outputs)
    assert all(audit['context_pixels_changed']==0 for _,audit in outputs)


def test_protected_occlusion_is_untouched(tmp_path):
    meta=setup_project(tmp_path)
    # Proteger una franja dentro de la silueta, como un vehículo delante.
    p=tmp_path/'mask4.png'
    masked=np.array(Image.open(p).convert('L'))
    y,x=np.unravel_index(np.argmax(masked),masked.shape)
    yy,xx=np.where(masked>0)
    cy,cx=int(np.median(yy)),int(np.median(xx))
    masked[max(0,cy-20):cy+20,max(0,cx-35):cx+35]=0
    Image.fromarray(masked).save(p)
    # Actualizar render en modo completo para que el archivo de entrada también
    # respete la nueva protección, como haría v2.1 con las máscaras correctas.
    src=tmp_path/'generated_realistic'/'Vista_4_9pisos_completo_s1234.png'
    source=np.array(Image.open(src).convert('RGB'))
    original=np.array(Image.open(tmp_path/'view4.png').convert('RGB'))
    source[masked==0]=original[masked==0]
    Image.fromarray(source).save(src)
    out,_=create_photo_finish(tmp_path,4,src,PhotoFinishConfig(render_px_per_m=22),show_preview=False)
    im=np.array(Image.open(out).convert('RGB'))
    assert np.array_equal(im[masked==0],original[masked==0])


def test_party_wall_can_have_zero_windows(tmp_path):
    setup_project(tmp_path)
    cfg=PhotoFinishConfig(side_windows=0,render_px_per_m=21)
    p=tmp_path/'generated_realistic'/'Vista_3_9pisos_completo_s1234.png'
    out,report=create_photo_finish(tmp_path,3,p,cfg,show_preview=False)
    assert out.is_file() and report['context_pixels_changed']==0
