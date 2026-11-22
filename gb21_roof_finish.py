"""GeneraBuild v2.1-FR: acabado de azotea de una torre ya generada.

Reutiliza SIN CAMBIOS los vertices 3D y las matrices de camara de v2.1.
Reemplaza exclusivamente la cara superior visible (vertices 4,5,6,7) dentro
la mascara guardada, sin tocar fachadas ni inmuebles contiguos. El techo es
un UNICO atlas en coordenadas metricas, reutilizado por todas las vistas.

No requiere GPU. La cubierta es una aproximacion grafica; no simula
un pretil con volumen, sombras proyectadas reales ni oclusiones 3D complejas.

Requiere los modulos gb21_photo_finish.py y gb21_shared_facades.py de v2.1-F.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from hashlib import sha256
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from gb21_photo_finish import _matching_work_mask, _mode_from_name
from gb21_shared_facades import _project, _camera_center


@dataclass(frozen=True)
class RoofConfig:
    """Cubierta plana de azotea, dimensiones en metros reales del modelo."""
    style: str = 'concreto_impermeabilizado'
    seed: int = 2026
    pixels_per_m: int = 36
    field_rgb: tuple[int, int, int] = (184, 180, 170)
    parapet_rgb: tuple[int, int, int] = (211, 206, 194)
    parapet_width_m: float = 0.22
    inner_shadow_m: float = 0.28
    joints: bool = True
    roof_equipment: int = 1  # 0 = cubierta sin objetos; 1 = pequeno bloque tecnico
    opacity: float = 1.0  # 1 elimina la reticula anterior por completo
    feather_px: float = 0.65
    aerial_blur_px: float = 0.45
    preserve_existing: bool | None = None  # infiere modo por nombre si None
    reference_buffer_px: int = 3
    skip_street_views: bool = True

    def validate(self):
        if self.style not in ('concreto_impermeabilizado', 'impermeabilizante_rojo'):
            raise ValueError('style debe ser concreto_impermeabilizado o impermeabilizante_rojo')
        if not (12 <= self.pixels_per_m <= 110):
            raise ValueError('pixels_per_m fuera de [12,110]')
        if not (0.05 <= self.parapet_width_m <= 0.5):
            raise ValueError('parapet_width_m fuera de rango')
        if not (0 <= self.inner_shadow_m <= 0.7):
            raise ValueError('inner_shadow_m fuera de rango')
        if self.roof_equipment not in (0, 1):
            raise ValueError('roof_equipment debe ser 0 o 1')
        if not (0 <= self.opacity <= 1):
            raise ValueError('opacity debe estar en [0,1]')
        if not (0 <= self.feather_px <= 3 and 0 <= self.aerial_blur_px <= 3):
            raise ValueError('suavizados fuera de rango')
        for col in (self.field_rgb, self.parapet_rgb):
            if len(col)!=3 or any(int(v)<0 or int(v)>255 for v in col):
                raise ValueError('Se requiere RGB entero en [0,255]')
        return self


def _atlas_noise(shape, rng):
    """Ruido no periodico de tres frecuencias; sin patron en mosaico."""
    h,w=shape
    a = rng.normal(0, 1, (max(3,h//75), max(3,w//75))).astype(np.float32)
    b = rng.normal(0, 1, (max(3,h//18), max(3,w//18))).astype(np.float32)
    c = rng.normal(0, 1, (h,w)).astype(np.float32)
    a=cv2.resize(a,(w,h),interpolation=cv2.INTER_CUBIC)
    b=cv2.resize(b,(w,h),interpolation=cv2.INTER_LINEAR)
    return 3.1*a + 1.2*b + 0.70*c


def make_roof_atlas(W_m: float, D_m: float, config: RoofConfig=RoofConfig()) -> np.ndarray:
    """Techo maestro RGB en orientacion UV: arriba=atras, abajo=frente."""
    cfg=config.validate()
    if not (np.isfinite(W_m) and np.isfinite(D_m) and 3<W_m<250 and 3<D_m<250):
        raise ValueError('Ancho/profundidad de cubierta invalidos')
    ppm=cfg.pixels_per_m
    w=max(64,int(round(W_m*ppm)))
    h=max(64,int(round(D_m*ppm)))
    if w*h>12_000_000:
        raise ValueError('Atlas demasiado grande, reduce pixels_per_m')
    rng=np.random.default_rng(int(cfg.seed))
    base=(cfg.field_rgb if cfg.style=='concreto_impermeabilizado'
          else (170, 103, 87))
    noise=_atlas_noise((h,w),rng)
    shade=np.linspace(1.1,-1.7,h,dtype=np.float32)[:,None]
    rgb=np.clip(np.array(base,dtype=np.float32)[None,None,:]
                +noise[...,None] + shade[...,None],0,255).astype(np.uint8)

    # Fisuras/juntas de impermeabilizante: trazos debiles NO una reticula.
    if cfg.joints:
        if D_m >= 9:
            yy=int(round(h*0.52))
            cv2.line(rgb,(int(w*.10),yy),(int(w*.91),yy),
                     tuple(int(max(0,v-7)) for v in base),1,cv2.LINE_AA)
        if W_m >= 9:
            xx=int(round(w*.53))
            cv2.line(rgb,(xx,int(h*.10)),(xx,int(h*.88)),
                     tuple(int(max(0,v-5)) for v in base),1,cv2.LINE_AA)

    # Pretil bajo visual, dentro del mismo cuadrilatero: NO aumenta el H.
    e=max(2,int(round(cfg.parapet_width_m*ppm)))
    e=min(e,max(1,min(w,h)//12))
    halo=max(1,int(round(cfg.inner_shadow_m*ppm)))
    # Banda oscura interior: sombras sutiles al lado interno del pretil.
    inner_rgb=tuple(int(max(0,v-21)) for v in base)
    cv2.rectangle(rgb,(e+halo//2,e+halo//2),
                  (w-1-e-halo//2,h-1-e-halo//2),inner_rgb,max(1,halo))
    cv2.rectangle(rgb,(0,0),(w-1,h-1),cfg.parapet_rgb,2*e)
    # Iluminacion de aristas opuestas para relieve de cubierta.
    cv2.line(rgb,(e,e),(w-e-1,e),
             tuple(min(255,int(v+10)) for v in cfg.parapet_rgb),max(1,e//3),cv2.LINE_AA)
    cv2.line(rgb,(w-e-1,e),(w-e-1,h-e-1),
             tuple(max(0,int(v-17)) for v in cfg.parapet_rgb),max(1,e//2),cv2.LINE_AA)

    # Elemento tecnico pequeno y discreto, no poblado de cajas inventadas.
    if cfg.roof_equipment:
        ww=max(5,int(round(.8*ppm)))
        hh=max(5,int(round(1.05*ppm)))
        x=int(w*.65)
        y=int(h*.36)
        if x+ww+e*2<w and y+hh+e*2<h and x>2*e and y>2*e:
            shadow=(90,86,78)
            cv2.rectangle(rgb,(x+3,y+4),(x+ww+4,y+hh+4),shadow,-1)
            cv2.rectangle(rgb,(x,y),(x+ww,y+hh),(152,154,150),-1)
            cv2.rectangle(rgb,(x+2,y+2),(x+ww-2,y+hh-2),(173,176,171),-1)
            cv2.line(rgb,(x,y),(x+ww,y),(206,207,197),2,cv2.LINE_AA)

    # Desague discreto dentro de cubierta; sin cables o instalaciones exageradas.
    cx=max(e+2,int(w*.13))
    cy=max(e+2,int(h*.15))
    r=max(2,int(round(.075*ppm)))
    if cx+r<w-e and cy+r<h-e:
        cv2.circle(rgb,(cx,cy),r,(77,78,75),-1,cv2.LINE_AA)
    return rgb


def _h_from_meta(meta):
    V=np.asarray(meta['vertices_3d_m'],dtype=np.float64)
    if V.shape!=(8,3) or not np.all(np.isfinite(V)):
        raise ValueError('Se requieren exactamente 8 vertices 3D validos')
    W=float(np.linalg.norm(V[5]-V[4]))
    D=float(np.linalg.norm(V[7]-V[4]))
    if np.max(np.abs(V[4:,2]-V[4,2]))>1e-4:
        raise ValueError('La azotea debe ser horizontal')
    if np.linalg.norm(V[6]-V[5]-V[7]+V[4])>1e-4:
        raise ValueError('La cubierta no es un paralelogramo; revisa la calibracion')
    return V, W, D


def _roi_and_homography(uv, shape, atlas_shape):
    H,W=shape[:2]
    min_xy=np.floor(np.min(uv,axis=0)-2).astype(int)
    max_xy=np.ceil(np.max(uv,axis=0)+2).astype(int)
    x0=max(0,int(min_xy[0]));y0=max(0,int(min_xy[1]))
    x1=min(W,int(max_xy[0]+1));y1=min(H,int(max_xy[1]+1))
    if x1<=x0 or y1<=y0:
        return None
    tex_h,tex_w=atlas_shape[:2]
    # V4 = frontal izquierda; V5 = frontal derecha; V6 = fondo derecha; V7 = fondo izquierda
    src=np.float32([[0,tex_h-1],[tex_w-1,tex_h-1],[tex_w-1,0],[0,0]])
    dst=(np.asarray(uv,dtype=np.float32)-np.float32([x0,y0]))
    transform=cv2.getPerspectiveTransform(src,dst)
    if not np.all(np.isfinite(transform)):
        raise ValueError('Homografia de cubierta no finita')
    return (x0,y0,x1,y1),transform,dst


def finish_roof_one_view(photo: Image.Image, saved_mask: np.ndarray,
                         metadata: dict, view_index: int,
                         atlas: np.ndarray, config: RoofConfig,
                         preserve_existing: bool=False):
    """Sustituye SOLO pixels del techo, recortados por la mascara aprobada.

    (PIL original/salida, máscara final del techo uint8, reporte dict)
    Se restauran pixels originales fuera de la mascara editable.
    """
    cfg=config.validate()
    if view_index not in range(1,len(metadata['views'])+1):
        raise ValueError('Vista fuera de rango')
    input_rgb=np.asarray(photo.convert('RGB'),dtype=np.uint8).copy()
    h,w=input_rgb.shape[:2]
    mask=np.asarray(saved_mask,dtype=np.uint8)
    if mask.shape!=(h,w):
        raise ValueError('La mascara y la foto difieren en resolucion')
    V,_,_=_h_from_meta(metadata)
    view=metadata['views'][view_index-1]
    if cfg.skip_street_views and view_index>2:
        return Image.fromarray(input_rgb),np.zeros((h,w),np.uint8),{
            'view':view_index,'status':'vista_de_calle_omitida',
            'roof_changed_pixels':0,'roof_edited_px':0}
    # V4,V5,V6,V7: normal +Z. Nunca pintar cubierta desde abajo.
    C=_camera_center(view)
    if C[2] <= float(V[4,2])+0.05:
        return Image.fromarray(input_rgb),np.zeros((h,w),np.uint8),{
            'view':view_index,'status':'camara_bajo_azotea',
            'roof_changed_pixels':0,'roof_edited_px':0}
    uv=_project(np.asarray(view['P'],dtype=np.float64),V[[4,5,6,7]])
    geometry=_roi_and_homography(uv,(h,w),atlas.shape)
    if geometry is None:
        return Image.fromarray(input_rgb),np.zeros((h,w),np.uint8),{
            'view':view_index,'status':'cubierta_fuera_de_encuadre',
            'roof_changed_pixels':0,'roof_edited_px':0}
    (x0,y0,x1,y1),Hwarp,dst=geometry
    rh,rw=y1-y0,x1-x0
    projected=cv2.warpPerspective(atlas,Hwarp,(rw,rh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    if cfg.aerial_blur_px>0 and view_index<=2:
        projected=cv2.GaussianBlur(projected,(0,0),cfg.aerial_blur_px)
    roofpoly=np.zeros((rh,rw),np.uint8)
    cv2.fillConvexPoly(roofpoly,np.rint(dst).astype(np.int32),255)
    allowed=_matching_work_mask(mask,metadata,view_index,
                                preserve_existing,
                                transition_px=cfg.reference_buffer_px)
    # NO bloquear la silueta completa del edificio original: en cámaras
    # aéreas la azotea nueva está delante de la azotea original.
    # _matching_work_mask ya restringe a la ampliación H0..H1.
    effective=(roofpoly>127)&(allowed[y0:y1,x0:x1]>127)
    roof_mask=np.zeros((h,w),np.uint8)
    roof_mask[y0:y1,x0:x1]=np.where(effective,255,0).astype(np.uint8)
    if not np.any(effective):
        return Image.fromarray(input_rgb),roof_mask,{
            'view':view_index,'status':'cubierta_fuera_de_mascara',
            'roof_changed_pixels':0,'roof_edited_px':0,
            'roof_projected_corners_px':uv.round(3).tolist()}

    # Suavizar solo dentro del poligono de techo sin contaminar contextos.
    if cfg.feather_px>0:
        d=cv2.distanceTransform((roofpoly>0).astype(np.uint8),
                                cv2.DIST_L2,3)
        alpha=np.clip(d/max(cfg.feather_px,1e-3),0,1)*cfg.opacity
    else:
        alpha=np.full((rh,rw),cfg.opacity,dtype=np.float32)
    alpha *= effective.astype(np.float32)
    output=input_rgb.copy()
    region=output[y0:y1,x0:x1]
    candidate=(region.astype(np.float32)*(1-alpha[...,None])+
               projected.astype(np.float32)*alpha[...,None]).clip(0,255).astype(np.uint8)
    region[effective]=candidate[effective]
    # Como salidas de render entran ya compuestas, fuera del techo preservamos
    # la foto procesada, sin tocar ventanas ni fachadas.
    if not np.array_equal(output[roof_mask==0],input_rgb[roof_mask==0]):
        raise AssertionError('Se modificaron pixels fuera del techo')
    if preserve_existing:
        # Los píxeles originales SOLO se protegen donde el volumen nuevo
        # no los oculta. La comparación exterior a roof_mask lo garantiza.
        from gb21_extension_mask import extension_mask_diagnostics
        recovered=extension_mask_diagnostics(mask,metadata,view_index)[
            'pixels_extension_hidden_by_legacy_mask']
    else:
        recovered=0
    stats={
        'view':int(view_index),
        'status':'ok',
        'fov_horizontal_deg':view.get('fov_horizontal_deg'),
        'roof_projected_corners_px':uv.round(3).tolist(),
        'roof_edited_px':int(np.count_nonzero(effective)),
        'extension_overlap_with_original_px':int(recovered),
        'roof_changed_pixels':int(np.count_nonzero(np.any(output!=input_rgb,axis=2))),
        'changed_pixels_outside_roof_mask':0,
    }
    return Image.fromarray(output),roof_mask,stats


def _newest_source(project_dir:Path,view_index:int)->Path:
    """El acabado ya existente tiene prioridad; nunca sobrescribe la fuente."""
    folders=('generated_photofinish','generated_realistic')
    for folder in folders:
        candidates=[]
        for p in (project_dir/folder).glob(f'Vista_{view_index}_*.png'):
            if not ('_completo_' in p.stem or '_ampliacion_' in p.stem):
                continue
            if any(p.stem.endswith(s) for s in
                   ('_mask','_canny','_materiales','_verificacion','_diagnostico')):
                continue
            if folder=='generated_photofinish' and not p.stem.endswith('_fotofinish'):
                continue
            if folder=='generated_realistic' and p.stem.endswith('_fotofinish'):
                continue
            candidates.append(p)
        if candidates:
            return max(candidates,key=lambda p:p.stat().st_mtime)
    raise FileNotFoundError(f'No hay resultados v2.1/F de la Vista {view_index}')


def finish_existing_roofs(project_dir: str|Path,
                          views: Sequence[int]=(1,2),
                          config: RoofConfig=RoofConfig(),
                          source_paths: dict[int,str|Path]|None=None,
                          preview: bool=True):
    """Corrige las azoteas de resultados anteriores. Guardado SIN sobreescritura.

    Retorna (rutas_y_estadisticas, atlas). Requiere calibracion JSON y
    fotografias generadas presentes en el directorio del proyecto.
    """
    project_dir=Path(project_dir)
    meta_path=project_dir/'calibration_multiview.json'
    if not meta_path.is_file():
        raise FileNotFoundError('Falta calibration_multiview.json en '+str(project_dir))
    original_meta_bytes=meta_path.read_bytes()
    meta=json.loads(original_meta_bytes)
    V,W,D=_h_from_meta(meta)
    cfg=config.validate()
    atlas=make_roof_atlas(W,D,cfg)
    folder=project_dir/'generated_roof_finish'
    folder.mkdir(exist_ok=True,parents=True)
    atlas_path=folder/f'roof_atlas_seed{cfg.seed}.png'
    Image.fromarray(atlas).save(atlas_path)
    records=[]
    for idx in views:
        idx=int(idx)
        v=meta['views'][idx-1]
        src=Path(source_paths[idx]) if source_paths and idx in source_paths else _newest_source(project_dir,idx)
        if not src.is_absolute() and not src.is_file():
            src=project_dir/src
        if not src.is_file():
            raise FileNotFoundError(str(src))
        preserve=cfg.preserve_existing
        if preserve is None:
            preserve=_mode_from_name(src)
        with Image.open(src) as im:
            before=im.convert('RGB')
        with Image.open(project_dir/v['mask_relative']) as im:
            allowed=np.asarray(im.convert('L'))
        if allowed.shape != (before.height,before.width):
            raise ValueError(f'Vista {idx}: foto y mascara incompatibles')
        after,roof_mask,stats=finish_roof_one_view(before,allowed,meta,idx,atlas,cfg,bool(preserve))
        outfile=folder/f'{src.stem}_rooffinish.png'
        after.save(outfile)
        maskfile=folder/f'{src.stem}_roof_mask.png'
        Image.fromarray(roof_mask).save(maskfile)
        rec={'view':idx,'source':str(src),'output':str(outfile),
             'roof_mask':str(maskfile),'preserve_existing':bool(preserve),**stats}
        records.append(rec)
        print(f"Vista {idx} | azotea: {stats['status']} | píxeles tratados: {stats['roof_edited_px']}")
        if preview:
            # Comparacion a resolucion original reducida SOLO para visualizar.
            a=before.copy(); b=after.copy()
            a.thumbnail((780,540)); b.thumbnail((780,540))
            canv=Image.new('RGB',(a.width+b.width,max(a.height,b.height)+31),'#e9e9e9')
            canv.paste(a,(0,31)); canv.paste(b,(a.width,31))
            d=ImageDraw.Draw(canv)
            d.text((8,8),f'Vista {idx}: antes',fill=(30,30,30))
            d.text((a.width+8,8),'Cubierta plana mejorada',fill=(30,30,30))
            try:
                from IPython.display import display
                display(canv)
            except ImportError:
                canv.save(folder/f'Vista_{idx}_comparacion.png')
    report={'version':'GeneraBuild_v2.1-FR','config':asdict(cfg),
            'roof_atlas':str(atlas_path),'roof_atlas_sha256':sha256(atlas.tobytes()).hexdigest(),
            'calibration_sha256':sha256(original_meta_bytes).hexdigest(),'views':records,
            'note':'Azotea compartida por homografia, cambios limitados al poligono y mascara.'}
    (folder/'roof_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if meta_path.read_bytes()!=original_meta_bytes:
        raise AssertionError('La calibracion fue alterada')
    return records, atlas
