"""GeneraBuild v2.1-F: acabado fotográfico sobre las salidas originales v2.1.

No reemplaza el edificio generado por un render 3D plano. Utiliza la propia imagen
producida con inpainting y proyecta solamente VIDRIO, MARCOS y detalles sutiles
procedentes de un atlas de fachada común en coordenadas 3D.

Requisitos CPU: numpy, opencv, Pillow y gb21_shared_facades.py.
Nunca recalibra P_i ni altera la huella, altura o FOV de ninguna vista.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from hashlib import sha256
from typing import Optional, Sequence
import json
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from gb21_shared_facades import (
    FacadeConfig, make_atlases, render_one_view, SEMANTICS,
)


@dataclass(frozen=True)
class PhotoFinishConfig:
    """La geometría/las ventanas dependen de un solo archivo de calibración."""
    front_windows: int = 3
    side_windows: int = 4
    material_seed: int = 2026
    stucco_rgb: tuple[int,int,int] = (208, 204, 196)
    glass_rgb: tuple[int,int,int] = (63, 94, 110)
    frame_rgb: tuple[int,int,int] = (38, 43, 48)
    glass_opacity: float = 0.98
    frame_opacity: float = 0.98
    details_opacity: float = 0.30
    wall_tint: float = 0.22
    reflection_brightness: float = 0.86
    feather_sigma: float = 0.55
    aerial_reduction: float = 0.80
    render_px_per_m: int = 48

    def validate(self):
        if not (1 <= self.front_windows <= 8 and 0 <= self.side_windows <= 12):
            raise ValueError('Ventanas: número fuera de rango')
        for name in ('glass_opacity','frame_opacity','details_opacity','wall_tint','aerial_reduction'):
            if not (0 <= getattr(self,name) <= 1):
                raise ValueError(f'{name} debe estar en [0,1]')
        if self.feather_sigma<0 or self.feather_sigma>3:
            raise ValueError('feather_sigma debe estar en [0,3]')
        for col in (self.stucco_rgb,self.glass_rgb,self.frame_rgb):
            if len(col)!=3 or any(not(0<=int(a)<=255) for a in col):
                raise ValueError('Color RGB inválido')
        return self

    def facade_config(self):
        self.validate()
        return FacadeConfig(
            front_windows=self.front_windows,
            side_windows=self.side_windows,
            material_seed=self.material_seed,
            render_px_per_m=self.render_px_per_m,
            stucco_rgb=self.stucco_rgb,
            glass_rgb=self.glass_rgb,
            frame_rgb=self.frame_rgb,
            window_width_fraction=0.60,
            glass_light=self.reflection_brightness,
            floor_band_cm=9,
            balconies=False,
            aerial_softness_px=0.40,
            seam_feather_px=0.0, # Acabado/píxeles se componen aparte
            preserve_original_buffer_px=2,
        )


def _alpha_for_label(label:np.ndarray, value:int, opacity:float,
                     allowed:np.ndarray, sigma:float)->np.ndarray:
    """Suaviza el contorno de cada material; nunca cambia el exterior permitido."""
    x=(label==value).astype(np.float32)
    if sigma>0:
        x=cv2.GaussianBlur(x,(0,0),float(sigma))
    return np.clip(x*float(opacity),0,1) * allowed.astype(np.float32)


def _mix_rgb(dst:np.ndarray, src:np.ndarray, alpha:np.ndarray)->np.ndarray:
    if not np.any(alpha>0):
        return dst
    # Procesamiento vectorial RGB, no modulación directa de pixeles sin máscara.
    return np.clip(dst.astype(np.float32)*(1-alpha[...,None])+
                   src.astype(np.float32)*alpha[...,None],0,255).astype(np.uint8)


def _warm_finish(original: np.ndarray, wall_mask:np.ndarray,
                 rgb_target:tuple[int,int,int], strength:float)->np.ndarray:
    """Cambia levemente el tono de los muros conservando sombras/textura SD.

    Mantiene el canal L de CIELab casi intacto; no pega una pared sintética encima.
    """
    if strength <= 0 or not np.any(wall_mask):
        return original
    lab=cv2.cvtColor(original,cv2.COLOR_RGB2LAB).astype(np.float32)
    base=np.zeros((1,1,3),dtype=np.uint8)
    base[0,0]=tuple(int(x) for x in rgb_target)
    target=cv2.cvtColor(base,cv2.COLOR_RGB2LAB).astype(np.float32)[0,0]
    # Luminosidad solo 6 puntos a fuerza 1, para no borrar sombras.
    lab[...,0]=np.where(wall_mask,lab[...,0]+6.0*strength,lab[...,0])
    lab[...,1]=np.where(wall_mask,(1-strength)*lab[...,1]+strength*target[1],lab[...,1])
    lab[...,2]=np.where(wall_mask,(1-strength)*lab[...,2]+strength*target[2],lab[...,2])
    converted=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2RGB)
    out=original.copy()
    out[wall_mask]=converted[wall_mask]
    return out


def _mode_from_name(image_path:Path)->bool:
    """'True' solo si la imagen fue generada en modo ampliación original."""
    name=image_path.stem.lower()
    if '_ampliacion_' in name:
        return True
    if '_completo_' in name:
        return False
    raise ValueError('No se reconoce el modo de la imagen. Indica preserve_existing explícitamente.')


def _last_generated(project_dir:Path,view_index:int)->Path:
    folder=project_dir/'generated_realistic'
    options=[]
    for p in folder.glob(f'Vista_{view_index}_*pisos_*.png'):
        if not ('_ampliacion_' in p.stem or '_completo_' in p.stem):
            continue
        if p.stem.endswith(('_mask','_canny','_fotofinish')):
            continue
        options.append(p)
    if not options:
        raise FileNotFoundError(f'No hay imágenes generadas de Vista {view_index} en {folder}')
    return max(options,key=lambda p:p.stat().st_mtime)


def _matching_work_mask(saved_mask: np.ndarray, meta: dict, view_index: int,
                        preserve_existing: bool, transition_px: int=0) -> np.ndarray:
    """Geometría de la ampliación (H0..H1), no sustracción 2D del original.

    Se conserva la firma de v2.1-F para compatibilidad. El parámetro
    transition_px ya no erosiona el edificio existente: una erosion 2D
    destruye la visibilidad en las tomas aéreas.
    """
    from gb21_extension_mask import extension_work_mask
    return extension_work_mask(saved_mask,meta,view_index,preserve_existing)


def create_photo_finish(
    project_dir: str|Path,
    view_index: int,
    generated_path: str|Path|None=None,
    config: PhotoFinishConfig|None=None,
    preserve_existing:bool|None=None,
    show_preview:bool=True,
    save_debug:bool=True,
) -> tuple[Path,dict]:
    """Mejora la fachada existente sin GPU y sin recalibrar cámaras.

    Precondición: las fotografías, las máscaras y calibration_multiview.json
    del proyecto deben coincidir con la salida generada v2.1.

    - Todo píxel fuera de la máscara editable es restaurado del original.
    - Los huecos de vidrio y sus carpinterías comparten atlas 3D en las 4 vistas.
    - Conserva los 3 pisos originales si se eligió modo 'ampliación'.
    - No dibuja un muro 3D nuevo; se conserva el aspecto fotográfico de SD1.5.
    """
    cfg=(config or PhotoFinishConfig()).validate()
    project_dir=Path(project_dir)
    json_path=project_dir/'calibration_multiview.json'
    if not json_path.exists():
        raise FileNotFoundError(f'Falta la calibración, sin la cual no pueden proyectarse ventanas: {json_path}')
    meta=json.loads(json_path.read_text(encoding='utf-8'))
    if not (1<=int(view_index)<=len(meta['views'])):
        raise ValueError('Vista inválida')
    view=meta['views'][view_index-1]
    src=Path(generated_path) if generated_path is not None else _last_generated(project_dir,view_index)
    if not src.is_absolute() and not src.is_file():
        src=project_dir/src
    if not src.is_file():
        raise FileNotFoundError(f'No existe el render de entrada: {src}')
    if preserve_existing is None:
        preserve_existing=_mode_from_name(src)

    with Image.open(project_dir/view['image_relative']) as img:
        photo_original=img.convert('RGB')
    with Image.open(src) as img:
        photo_generated=img.convert('RGB')
    with Image.open(project_dir/view['mask_relative']) as img:
        protected_mask=np.asarray(img.convert('L'),dtype=np.uint8)
    if photo_generated.size!=photo_original.size:
        raise ValueError('Las dimensiones de la salida no coinciden con la foto de calibración')
    base=np.asarray(photo_generated).copy()
    original=np.asarray(photo_original)
    working=_matching_work_mask(protected_mask,meta,view_index,bool(preserve_existing))
    allowed=(working>127)
    if not np.any(allowed):
        raise ValueError('Máscara editable vacía; revisa el modo y calibración.')
    if not np.array_equal(base[~allowed],original[~allowed]):
        # Garantía de fondo: las salidas v2.1 correctas ya cumplen esto; si una
        # salida externa no lo cumple, restauramos el original exactamente.
        base[~allowed]=original[~allowed]

    atlas_cfg=cfg.facade_config()
    atlases=make_atlases(meta,atlas_cfg)
    # Sólo creamos el mapa de materiales, pero no sustituimos el muro de SD.
    atlas_rendered, _, labels, diagnostics = render_one_view(
        photo_original, protected_mask, meta, view_index, atlases,atlas_cfg,
        preserve_existing=False,
    )
    atlas_rgb=np.asarray(atlas_rendered)
    if atlas_rgb.shape!=base.shape:
        raise AssertionError('Render 3D inválido')
    # El atlas sólo puede actuar dentro de la misma máscara que SD v2.1.
    walls=(labels==SEMANTICS['wall']) & allowed
    corrected=_warm_finish(base,walls,cfg.stucco_rgb,cfg.wall_tint)
    opacity=cfg.aerial_reduction if view_index<=2 else 1.0
    # Orden: detalles leves, marco oscuro, cristal reflectante.
    for label_name,alpha_strength in (
        ('architectural_detail',cfg.details_opacity*opacity),
        ('frame',cfg.frame_opacity*opacity),
        ('glass',cfg.glass_opacity*opacity),
    ):
        alpha=_alpha_for_label(labels,SEMANTICS[label_name],
                              alpha_strength,allowed,cfg.feather_sigma)
        corrected=_mix_rgb(corrected,atlas_rgb,alpha)

    # Compromiso fuerte de no modificar inmuebles/objetos externos.
    corrected[~allowed]=original[~allowed]
    if not np.array_equal(corrected[~allowed],original[~allowed]):
        raise AssertionError('Se modificó una región protegida')

    # Si preservamos tres pisos, comprobar que ni un píxel sea alterado dentro
    # de la parte fuera de la máscara de edición, incluyendo la silueta original.
    out_folder=project_dir/'generated_photofinish'
    out_folder.mkdir(parents=True,exist_ok=True)
    out_path=out_folder/(src.stem+'_fotofinish.png')
    Image.fromarray(corrected).save(out_path)
    audit={
        'source':str(src),'output':str(out_path),
        'view':int(view_index),'floors':int(meta['num_floors']),
        'preserve_existing_3_floors':bool(preserve_existing),
        'fov_horizontal_deg':view.get('fov_horizontal_deg'),
        'editable_pixels':int(np.count_nonzero(allowed)),
        'glass_pixels_in_edit':int(np.count_nonzero((labels==SEMANTICS['glass'])&allowed)),
        'frame_pixels_in_edit':int(np.count_nonzero((labels==SEMANTICS['frame'])&allowed)),
        'context_pixels_changed':0,
        'calibration_sha256':sha256(json_path.read_bytes()).hexdigest(),
        'shared_material_seed':int(cfg.material_seed),
        'finish':asdict(cfg),
        'visible_faces':diagnostics.get('faces_visible',[]),
        'limitations':[
          'Vidrios y marcos son renderizados sobre el resultado de difusion; no son foto 3D reconstruida.',
          'Los muros conservan la textura generada independientemente por cada vista.',
          'La precisión y exclusión de oclusiones dependen de las máscaras guardadas.',
        ],
    }
    (out_folder/(src.stem+'_fotofinish.json')).write_text(
        json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if save_debug:
        Image.fromarray(np.where(allowed,labels,0).astype(np.uint8),'L').save(
            out_folder/(src.stem+'_materiales.png'))
        rgbmask=np.array(base).copy()
        for lab,col in [(SEMANTICS['glass'],(0,210,230)),
                        (SEMANTICS['frame'],(255,210,0))]:
            idx=(labels==lab)&allowed
            rgbmask[idx]=np.clip(0.55*rgbmask[idx]+0.45*np.asarray(col),0,255).astype(np.uint8)
        Image.fromarray(rgbmask).save(out_folder/(src.stem+'_verificacion.png'))
    if show_preview:
        try:
            from IPython.display import display
            # Comparación del resultado completo, mismas dimensiones originales.
            a=Image.fromarray(base);b=Image.fromarray(corrected)
            a.thumbnail((800,650));b.thumbnail((800,650))
            canvas=Image.new('RGB',(a.width+b.width,max(a.height,b.height)+34),'#e7e7e7')
            canvas.paste(a,(0,34));canvas.paste(b,(a.width,34))
            d=ImageDraw.Draw(canvas)
            d.text((8,9),'Antes: v2.1',fill=(10,10,10))
            d.text((a.width+8,9),'Después: acabado arquitectónico',fill=(10,10,10))
            display(canvas)
        except ImportError:
            pass
    print(f'Vista {view_index} | acabado: {out_path.name} | '
          f'{audit["glass_pixels_in_edit"]} px de cristal | fuera de máscara: 0 px')
    return out_path,audit


def finish_previous_results(project_dir:str|Path,views:Sequence[int]=(3,4),
                            config:PhotoFinishConfig|None=None,
                            show_preview:bool=True)->list[tuple[Path,dict]]:
    """Procesa los PNG guardados más recientes de cada vista sin GPU."""
    outputs=[]
    for view in views:
        outputs.append(create_photo_finish(project_dir,view,config=config,
                                           preserve_existing=None,
                                           show_preview=show_preview))
    return outputs


def refine_only_walls_gpu(
    project_dir:str|Path,
    view_index:int,
    pipeline,
    finished_path:str|Path,
    config:PhotoFinishConfig|None=None,
    seed:int=1234,
    steps:int=32,
    guidance:float=5.0,
    control_scale:float=0.48,
    max_dim:int=768,
    show_preview:bool=True,
)->Path:
    """Opcional: retexturiza únicamente el PAÑO DE MURO con SD+ControlNet.

    Protege ventanas/marcos, vecinos, vehículos y silueta existente.
    `pipeline` es StableDiffusionControlNetInpaintPipeline ya cargada en CUDA.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('La segunda pasada de muros requiere GPU CUDA')
    cfg=(config or PhotoFinishConfig()).validate()
    project_dir=Path(project_dir)
    img_path=Path(finished_path)
    if not img_path.exists():
        raise FileNotFoundError(str(img_path))
    meta=json.loads((project_dir/'calibration_multiview.json').read_text(encoding='utf-8'))
    view=meta['views'][view_index-1]
    with Image.open(project_dir/view['image_relative']) as im:
        source=im.convert('RGB')
    with Image.open(project_dir/view['mask_relative']) as im:
        allowed_saved=np.array(im.convert('L'),dtype=np.uint8)
    with Image.open(img_path) as im:
        finished=im.convert('RGB')
    mode=_mode_from_name(img_path)
    mask=_matching_work_mask(allowed_saved,meta,view_index,mode)>127
    atlas_cfg=cfg.facade_config()
    atlases=make_atlases(meta,atlas_cfg)
    _,_,labels,_=render_one_view(source,allowed_saved,meta,view_index,
                                  atlases,atlas_cfg,preserve_existing=False)
    edit=((labels==SEMANTICS['wall'])&mask).astype(np.uint8)*255
    edit=cv2.erode(edit,np.ones((3,3),np.uint8))
    ys,xs=np.nonzero(edit)
    if len(xs)==0:
        raise ValueError('No existen paños de muro editables')
    H,W=edit.shape
    xmin,xmax=int(xs.min()),int(xs.max())
    ymin,ymax=int(ys.min()),int(ys.max())
    mx=max(55,int((xmax-xmin+1)*0.32));my=max(55,int((ymax-ymin+1)*0.20))
    roi=(max(0,xmin-mx),max(0,ymin-my),min(W,xmax+mx+1),min(H,ymax+my+1))
    crop=finished.crop(roi)
    crop_mask=Image.fromarray(edit).crop(roi)
    rw,rh=crop.size
    factor=min(1.5,float(max_dim)/max(rw,rh))
    ww=max(64,int(round(rw*factor/8))*8)
    hh=max(64,int(round(rh*factor/8))*8)
    work_photo=crop.resize((ww,hh),Image.Resampling.LANCZOS)
    work_mask=crop_mask.resize((ww,hh),Image.Resampling.NEAREST)
    rgb=np.asarray(crop,dtype=np.uint8)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,120,210)
    canny=Image.fromarray(cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)).resize((ww,hh),Image.Resampling.NEAREST)
    prompt=(
        f'Photorealistic finished {meta["num_floors"]}-story residential infill building in Mexico City, '
        'clean warm gray limestone stucco, realistic fine mineral wall microtexture, '
        'sunlit maintained architectural facade, natural ambient shadows, '
        'documentary street photography, subtle variations in the render, finished building'
    )
    negative=(
        'unfinished building, boarded windows, blank facade, exposed concrete, '
        'patchwork, repetitive grid, dirty wall, renovation, dilapidated, '
        'cartoon, painting, unreal facade, mosaic, deformed geometry'
    )
    generator=torch.Generator(device='cuda').manual_seed(int(seed))
    generated=pipeline(prompt=prompt,negative_prompt=negative,
                       image=work_photo,mask_image=work_mask,control_image=canny,
                       controlnet_conditioning_scale=float(control_scale),
                       num_inference_steps=int(steps),guidance_scale=float(guidance),
                       width=ww,height=hh,generator=generator).images[0]
    original=np.asarray(finished).copy()
    final=original.copy()
    x0,y0,x1,y1=roi
    patch=np.asarray(generated.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS))
    sub=final[y0:y1,x0:x1]
    submask=edit[y0:y1,x0:x1]>0
    a=cv2.GaussianBlur(submask.astype(np.float32),(0,0),0.8)*submask.astype(np.float32)
    sub[:]=(sub.astype(np.float32)*(1-a[...,None])+patch*a[...,None]).clip(0,255).astype(np.uint8)
    if not np.array_equal(final[edit==0],original[edit==0]):
        raise AssertionError('El refinamiento cambió un cristal, marco o píxel protegido')
    if not np.array_equal(final[~mask],np.asarray(source)[~mask]):
        raise AssertionError('El refinamiento cambió contexto ajeno a la máscara')
    folder=project_dir/'generated_photofinish'
    folder.mkdir(exist_ok=True,parents=True)
    out=folder/(img_path.stem+'_muros_refinados.png')
    Image.fromarray(final).save(out)
    (folder/(img_path.stem+'_muros_refinados.json')).write_text(
       json.dumps({'source':str(img_path),'output':str(out),'view':view_index,
                   'edited_wall_pixels':int(np.count_nonzero(edit)),
                   'exterior_pixels_changed':0,'seed':seed,'steps':steps,
                   'controlnet_conditioning_scale':control_scale,
                   'prompt':prompt,'negative_prompt':negative},ensure_ascii=False,indent=2),
       encoding='utf-8')
    if show_preview:
        try:
            from IPython.display import display
            preview=Image.fromarray(final);preview.thumbnail((1100,800));display(preview)
        except ImportError:pass
    print(f'Vista {view_index}: muros refinados -> {out}')
    return out
