"""Oclusiones geométricas opcionales mediante depth buffers de prismas convexos.

Se aplica únicamente sobre la carpeta de una EJECUCIÓN, nunca sobre el
calibration_multiview.json ni las máscaras originales del sitio.
La opción está DESACTIVADA por defecto: requiere verificación visual.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .context_2p5d import NeighborBuilding, load_context
from .geometry import architectural_canny, make_diagnostic_preview, sha256_file


def _triangle_depth(buf:np.ndarray, P:np.ndarray, xyz:np.ndarray) -> None:
    """Profundidad de triángulo con interpolación perspectiva correcta de 1/Z.

    La tercera fila de K[R|t] coincide con la profundidad en la cámara (K[2,:]=[0,0,1]).
    """
    xyz=np.asarray(xyz,dtype=np.float64).reshape(3,3)
    homogeneous=(np.asarray(P,float) @ np.column_stack([xyz,np.ones(3)]).T).T
    z=homogeneous[:,2]
    if not np.all(np.isfinite(homogeneous)) or np.any(z<=1e-5):return
    uv=homogeneous[:,:2]/z[:,None]
    if np.max(np.abs(uv))>1e6:return
    H,W=buf.shape
    xmin=max(0,int(np.floor(uv[:,0].min())))
    xmax=min(W-1,int(np.ceil(uv[:,0].max())))
    ymin=max(0,int(np.floor(uv[:,1].min())))
    ymax=min(H-1,int(np.ceil(uv[:,1].max())))
    if xmin>xmax or ymin>ymax:return
    (ax,ay),(bx,by),(cx,cy)=uv
    den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
    if abs(den)<1e-10:return
    gy,gx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
    # Sample at pixel center (small tolerance for boundary rasterization).
    fx=gx+0.5; fy=gy+0.5
    l0=((by-cy)*(fx-cx)+(cx-bx)*(fy-cy))/den
    l1=((cy-ay)*(fx-cx)+(ax-cx)*(fy-cy))/den
    l2=1.0-l0-l1
    inside=(l0>=-1e-5)&(l1>=-1e-5)&(l2>=-1e-5)
    if not inside.any():return
    invz=l0/z[0]+l1/z[1]+l2/z[2]
    valid=inside&(invz>1e-9)
    if not valid.any():return
    roi=buf[ymin:ymax+1,xmin:xmax+1]
    roi[valid]=np.minimum(roi[valid],(1/invz[valid]).astype(np.float32))


def _target_triangles(vertices:np.ndarray) ->list[np.ndarray]:
    v=np.asarray(vertices,dtype=float)
    if v.shape!=(8,3):raise ValueError('El edificio objetivo requiere 8 vértices')
    faces=((0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7),(0,3,2,1))
    return [v[list(ids)] [list(ix)] for ids in faces for ix in ((0,1,2),(0,2,3))]


def rasterize_depth(P, triangles:list[np.ndarray], size_wh:tuple[int,int])->np.ndarray:
    w,h=map(int,size_wh)
    if not (16<=w<=10000 and 16<=h<=10000):raise ValueError('Imagen inválida')
    depth=np.full((h,w),np.inf,dtype=np.float32)
    for tri in triangles:_triangle_depth(depth,np.asarray(P,dtype=float),tri)
    return depth


def neighbor_occlusion_mask(P, target_vertices, neighbors:list[NeighborBuilding],
                            size_wh:tuple[int,int], tolerance_m:float=.04):
    """Máscara binaria de vecinos frente a superficies visibles del edificio objetivo.

    No oculta árboles/vehículos, ni modela objetos que no estén en `neighbors`.
    """
    P=np.asarray(P,dtype=float)
    target=rasterize_depth(P,_target_triangles(np.asarray(target_vertices,float)),size_wh)
    active=[b for b in neighbors if b.enabled and b.occluder]
    if not active:return np.zeros(target.shape,dtype=np.uint8)
    other=rasterize_depth(P,[tri for b in active for tri in b.triangles()],size_wh)
    blocked=(np.isfinite(target)&np.isfinite(other)&(other<target-float(tolerance_m)))
    return (blocked.astype(np.uint8)*255)


def apply_context_occlusions_to_run(run_dir:Path, site_dir:Path,
                                    *, require_calibration_match:bool=True)->dict:
    """Recorta las máscaras y regenera el Canny de la ejecución de forma reversible.

    Exige el mismo hash de calibración que se utilizó al dibujar las huellas,
    si ese dato se conservó. No cambia jamás `site_dir`.
    """
    run_dir=Path(run_dir);site_dir=Path(site_dir)
    context_path=site_dir/'urban_context_2p5d.json'
    if not context_path.is_file():
        return {'applied':False,'reason':'No existe urban_context_2p5d.json','views':[]}
    raw=json.loads(context_path.read_text(encoding='utf-8'))
    canonical=site_dir/'calibration_multiview.json'
    if require_calibration_match and raw.get('calibration_sha256'):
        if raw['calibration_sha256']!=sha256_file(canonical):
            raise ValueError('La calibración cambió desde que se anotaron los vecinos: revalida las huellas')
    neighbors=load_context(site_dir)
    if not any(b.enabled and b.occluder for b in neighbors):
        return {'applied':False,'reason':'No hay vecinos habilitados para oclusión','views':[]}
    from .storage import load_metadata
    from .geometry import box_vertices
    meta=load_metadata(run_dir)
    vertices=np.asarray(meta['vertices_3d_m'],dtype=float)
    refs=np.asarray(meta['reference_vertices_3d_m'],dtype=float)
    report={'applied':True,'source_context_file':str(context_path), 'views':[]}
    for idx,view in enumerate(meta['views'],1):
        with Image.open(run_dir/view['image_relative']) as im:
            original=im.convert('RGB')
        maskfile=run_dir/view['mask_relative']
        with Image.open(maskfile) as im: mask=np.asarray(im.convert('L'))
        cover=neighbor_occlusion_mask(view['P'],vertices,neighbors,original.size)
        clipped=np.where(cover>127,0,mask).astype(np.uint8)
        removed=int(np.count_nonzero((mask>127)&(cover>127)))
        Image.fromarray(clipped).save(maskfile)
        # Guardar exclusivamente resultados diagnósticos dentro de la EJECUCIÓN.
        occlusion_dir=run_dir/'context_diagnostics';occlusion_dir.mkdir(exist_ok=True)
        Image.fromarray(cover).save(occlusion_dir/f'occlusion_view_{idx}.png')
        control=architectural_canny(original,clipped,np.asarray(view['P'],float),vertices,
                    np.asarray(view['rvec'],float),np.asarray(view['tvec'],float),
                    int(meta['num_floors']))
        control.save(run_dir/view['canny_relative'])
        preview=make_diagnostic_preview(original,clipped,np.asarray(view['P'],float),
                    refs,vertices,view.get('anchors_px',{}),
                    view.get('protected_polygons_px',[]),view.get('allowed_polygon_px'))
        preview.save(run_dir/view['preview_relative'])
        view.setdefault('mask_stats',{})['editable_px']=int(np.count_nonzero(clipped))
        report['views'].append({'view':idx,'occluded_editable_px':removed,
                                'remaining_editable_px':int(np.count_nonzero(clipped))})
    (run_dir/'calibration_multiview.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copy2(context_path,run_dir/'urban_context_2p5d.json')
    (run_dir/'context_occlusion_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report
