"""GeneraBuild - capa de realismo v2.1.

Usa el JSON de calibracion de GeneraBuild_v2, sin repetir la calibracion.
Genera control Canny con ventanas en geometria 3D compartida; recorta el
inpainting alrededor del edificio; mantiene originales fuera de la mascara.

No hace reconstruccion 3D fisica ni fuerza texturas identicas entre vistas.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def project_points(P, xyz):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    ph = (np.asarray(P, dtype=np.float64) @ np.c_[xyz, np.ones(len(xyz))].T).T
    if not np.all(np.isfinite(ph)) or np.any(ph[:, 2] <= 1e-6):
        raise ValueError("Cannot project points behind the camera")
    return ph[:, :2] / ph[:, 2:3]


def _point_on_wall(face, u, v):
    """u: ancho de cara [0,1], v: altura [0,1]."""
    p0, p1, p2, p3 = face
    return (1-u)*(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3


def _line3(drawing, P, xyz0, xyz1, thickness=1):
    xy = np.rint(project_points(P, np.array([xyz0, xyz1]))).astype(np.int32)
    cv2.line(drawing, tuple(xy[0]), tuple(xy[1]), 255, thickness, cv2.LINE_AA)


def _quad3(drawing, P, xyz4, thickness=1):
    uv = np.rint(project_points(P, np.array(xyz4))).astype(np.int32)
    cv2.polylines(drawing, [uv], True, 255, thickness=thickness, lineType=cv2.LINE_AA)


def _architectural_lines_on_wall(drawing, P, face, floors, face_type,
                                  front_bays=3, side_bays=5,
                                  reveal_windows=True):
    bays = front_bays if face_type in ('front','back') else side_bays
    # Franjas horizontales muy discretas en forjados: no dibujar rejilla vertical.
    for k in range(1, floors):
        level = k / floors
        _line3(drawing, P, _point_on_wall(face, 0, level),
               _point_on_wall(face, 1, level), 1)
    if not reveal_windows:
        return

    for story in range(floors):
        # Los huecos del edificio tienen un mismo sistema de coordenadas 3D
        # y por tanto no dependen de la fotografia elegida.
        if story == 0:
            # Planta baja: huecos comerciales de mas altura.
            t0, t1 = 0.10 / floors, 0.88 / floors
        else:
            t0 = (story + 0.28) / floors
            t1 = (story + 0.83) / floors

        for b in range(bays):
            # Margenes en cada crujia; aberturas visiblemente separadas por muro.
            u0 = (b + (0.20 if story else 0.12)) / bays
            u1 = (b + (0.80 if story else 0.88)) / bays
            quad = [_point_on_wall(face,u0,t0), _point_on_wall(face,u1,t0),
                    _point_on_wall(face,u1,t1), _point_on_wall(face,u0,t1)]
            _quad3(drawing, P, quad, thickness=2)
            # Marco central/montante solo si la apertura es suficientemente ancha.
            if face_type in ('front','back') and story >= 1:
                uc = (u0+u1)/2
                _line3(drawing, P, _point_on_wall(face,uc,t0),
                       _point_on_wall(face,uc,t1), thickness=1)
            # Antepecho sin lineas decorativas excesivas.
            if story >= 1 and face_type in ('front','back'):
                _line3(drawing, P, _point_on_wall(face,u0,t0-0.045/floors),
                       _point_on_wall(face,u1,t0-0.045/floors), thickness=1)


def architectural_canny_realistic(original: Image.Image, mask: np.ndarray,
                                   metadata: dict, view_index: int,
                                   front_bays: int = 3,
                                   side_bays: int = 4) -> Image.Image:
    """Canny con fachada explicita, no una cuadricula de azulejos.

    Geometria de ventanas identica en las cuatro proyecciones, pero el modelo
    de difusion sigue siendo independiente por vista (sin textura 3D comun).
    """
    if view_index not in (1, 2, 3, 4):
        raise ValueError('view_index debe ser 1..4')
    img = np.asarray(original.convert('RGB'))
    H, W = img.shape[:2]
    if mask.shape != (H,W):
        raise ValueError('Mask must match image dimensions')
    P = np.asarray(metadata['views'][view_index-1]['P'], dtype=float)
    vertices = np.asarray(metadata['vertices_3d_m'], dtype=float)
    view = metadata['views'][view_index-1]
    R, _ = cv2.Rodrigues(np.asarray(view['rvec'], dtype=float).reshape(3,1))
    t = np.asarray(view['tvec'], dtype=float).reshape(3,1)
    C = (-R.T @ t).ravel()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    background_edges = cv2.Canny(gray, 100, 200)
    strokes = np.zeros((H,W), dtype=np.uint8)
    specs = [
        ((0,1,5,4), 'front',True),
        ((1,2,6,5), 'right',True),
        ((2,3,7,6), 'back',True),
        ((3,0,4,7), 'left',True),
        ((4,5,6,7), 'roof',False),
        ((0,3,2,1), 'base',False),
    ]
    visible = []
    for ids, label, wall in specs:
        face = vertices[list(ids)]
        outward_normal = np.cross(face[1]-face[0], face[2]-face[1])
        center = face.mean(axis=0)
        if np.dot(outward_normal, C-center) > 1e-8:
            visible.append((float(np.linalg.norm(C-center)), face, label, wall))
    visible.sort(key=lambda v:-v[0])
    for _, face, name, is_wall in visible:
        polygon = np.rint(project_points(P, face)).astype(np.int32)
        # Limpia el interior antes de dibujar una cara (profundidad painter).
        cv2.fillPoly(strokes, [polygon], 0)
        cv2.polylines(strokes, [polygon], True, 255, thickness=2, lineType=cv2.LINE_AA)
        if is_wall:
            _architectural_lines_on_wall(
                strokes, P, face, int(metadata['num_floors']), name,
                front_bays=front_bays, side_bays=side_bays,
            )
    edges = np.where(mask>127, strokes, background_edges).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def _projected_hull_mask(vertices, P, size):
    W,H=size
    uv = np.rint(project_points(P, vertices)).astype(np.int32)
    hull = cv2.convexHull(uv).reshape(-1,2)
    mask = np.zeros((H,W), dtype=np.uint8)
    cv2.fillConvexPoly(mask,hull,255)
    return mask


def working_mask_for_realism(saved_mask, metadata, view_index,
                             preserve_existing=True, transition_px=0):
    """Máscara verdadera de volumen añadido, sin restar el edificio original."""
    from gb21_extension_mask import extension_work_mask
    return extension_work_mask(saved_mask,metadata,view_index,preserve_existing)


def bbox_crop_for_inpaint(mask, image_size, margin=0.45):
    """ROI con suficiente contexto y aspecto ~3:4 para un edificio esbelto."""
    W,H=image_size
    ys,xs=np.nonzero(np.asarray(mask)>127)
    if len(xs)==0:
        raise ValueError('Empty mask after applying extension mode')
    x0,x1=int(xs.min()),int(xs.max())+1
    y0,y1=int(ys.min()),int(ys.max())+1
    bw,bh=x1-x0,y1-y0
    mx=max(64,int(round(bw*margin)))
    my=max(64,int(round(bh*margin*0.50)))
    cropw=max(bw+2*mx, int((bh+2*my)*0.70))
    croph=max(bh+2*my, int(cropw*0.80))
    cropw=min(W,int(cropw))
    croph=min(H,int(croph))
    cx=(x0+x1)/2
    cy=(y0+y1)/2
    xx=max(0,min(W-cropw,int(round(cx-cropw/2))))
    yy=max(0,min(H-croph,int(round(cy-croph/2))))
    return (xx, yy,xx+cropw,yy+croph)


def composite_unmodified_outside(original, generated_crop, mask, roi, feather_px=1.5):
    """Composicion que garantiza igualdad EXACTA fuera de la mascara."""
    original=np.asarray(original.convert('RGB'),dtype=np.uint8)
    x0,y0,x1,y1=roi
    working=original.copy()
    crop=working[y0:y1,x0:x1]
    gen=np.asarray(generated_crop.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS),dtype=np.uint8)
    crop_mask=np.asarray(mask[y0:y1,x0:x1],dtype=np.uint8)>127
    if feather_px>0:
        a=cv2.GaussianBlur(crop_mask.astype(np.float32),(0,0),float(feather_px))
        a*=crop_mask.astype(np.float32)  # no tocar UN solo pixel fuera
        crop[:]=(gen*a[...,None]+crop*(1-a[...,None])).clip(0,255).astype(np.uint8)
    else:
        crop[crop_mask]=gen[crop_mask]
    if np.any(working[np.asarray(mask)<=127]!=original[np.asarray(mask)<=127]):
        raise AssertionError('Se ha cambiado un pixel fuera de mascara')
    return Image.fromarray(working)


def generate_one_view_realista(
        view_index: int,
        project_dir: str|Path,
        pipeline,
        seed=1234,
        steps=40,
        guidance=5.5,
        control_scale=0.65,
        max_dim=768,
        preserve_existing=True,
        margin=0.45,
        feather_px=1.5,
        front_bays=3,
        side_bays=4,
        prompt=None,
        negative_prompt=None,
        show_preview=True):
    """Generacion con inpainting focalizado y condicionamiento arquitectonico.

    `pipeline` es StableDiffusionControlNetInpaintPipeline ya cargada en GPU.
    No carga modelos por si sola. Loguea configuracion y resultado final.
    """
    if view_index not in (1,2,3,4):
        raise ValueError('view_index debe ser 1..4')
    import torch
    project_dir=Path(project_dir)
    meta_path=project_dir/'calibration_multiview.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Calibration not found: {meta_path}')
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    v=meta['views'][view_index-1]
    image_path=project_dir/v['image_relative']
    with Image.open(image_path) as im: original=im.convert('RGB')
    with Image.open(project_dir/v['mask_relative']) as im: mask_saved=np.asarray(im.convert('L'))
    mask=working_mask_for_realism(mask_saved,meta,view_index,preserve_existing)
    if not np.any(mask):
        raise ValueError(f'Empty resulting mask in view {view_index}; try preserve_existing=False')
    control=architectural_canny_realistic(original,mask,meta,view_index,front_bays,side_bays)
    roi=bbox_crop_for_inpaint(mask,original.size,margin)
    cropped_img=original.crop(roi)
    cropped_control=control.crop(roi)
    cropped_mask=Image.fromarray(mask).crop(roi)
    rw,rh=cropped_img.size
    # Permitimos ampliacion de recortes pequenos para que ventanas tengan
    # suficientes pixeles; maximo 1.8x para evitar aumentos extremos.
    factor=min(1.8,float(max_dim)/max(rw,rh)) if max_dim else 1.
    ww=max(64,int(round(rw*factor/8))*8)
    hh=max(64,int(round(rh*factor/8))*8)
    work_img=cropped_img.resize((ww,hh),Image.Resampling.LANCZOS)
    work_mask=cropped_mask.resize((ww,hh),Image.Resampling.NEAREST)
    work_control=cropped_control.resize((ww,hh),Image.Resampling.NEAREST)
    aerial = view_index<=2
    if prompt is None:
        # Intencionalmente conciso: SD1.5 suele truncar prompts largos.
        base=(
            f'Photorealistic {meta["num_floors"]}-story Roma Norte infill building, '
            'warm off-white stucco and light gray concrete, '
            'clearly separated floors, evenly spaced recessed rectangular windows, '
            'dark slim aluminum window frames, realistic glass reflections, '
            'subtle balconies, simple parapet roof, natural daylight, '
            'materials and shadows matching nearby buildings, '
            'professional architectural documentary photograph'
        )
        prompt=base + (
            ', oblique aerial photogrammetry view'
            if aerial else ', realistic street-level perspective'
        )
    if negative_prompt is None:
        negative_prompt=(
            'mosaic, patchwork, checkerboard, colorful tiles, mural, '
            'windowless facade, blank wall, multicolored panels, '
            'repetitive squares, oversaturated, warped windows, '
            'cartoon, CGI, plastic, distorted geometry, blurry'
        )
    generator=torch.Generator(device='cuda').manual_seed(int(seed))
    t0=time.perf_counter()
    new_img=pipeline(
        prompt=prompt, negative_prompt=negative_prompt,
        image=work_img, mask_image=work_mask,
        control_image=work_control,
        controlnet_conditioning_scale=float(control_scale),
        num_inference_steps=int(steps),guidance_scale=float(guidance),
        width=ww,height=hh,generator=generator,
    ).images[0]
    elapsed=time.perf_counter()-t0
    # La textura 3D de Google Earth es menos nitida que Street View.
    # Saturacion levemente reducida evita bloques cromaticos intensos.
    new_img=ImageEnhance.Color(new_img).enhance(0.88 if aerial else 0.94)
    if aerial:
        new_img=Image.fromarray(cv2.GaussianBlur(np.asarray(new_img),(0,0),0.50))
    combined=composite_unmodified_outside(original,new_img,mask,roi,feather_px)

    folder=project_dir/'generated_realistic'
    folder.mkdir(parents=True,exist_ok=True)
    stem=f'Vista_{view_index}_{meta["num_floors"]}pisos_{"ampliacion" if preserve_existing else "completo"}_s{seed}'
    out=folder/f'{stem}.png'
    combined.save(out)
    Image.fromarray(mask).save(folder/f'{stem}_mask.png')
    control.save(folder/f'{stem}_canny.png')
    log={
      'view':view_index,'original':str(v['image_relative']),
      'result':str(out.relative_to(project_dir)),'floors':meta['num_floors'],
      'seed':int(seed),'steps':int(steps),'guidance_scale':float(guidance),
      'controlnet_conditioning_scale':float(control_scale),'roi_xyxy':list(roi),
      'working_resolution_wh':[ww,hh], 'feather_px':float(feather_px),
      'preserve_existing_3_floors':bool(preserve_existing),
      'facade_windows_front_bays':int(front_bays),
      'facade_windows_side_bays':int(side_bays),
      'elapsed_sec':round(elapsed,2),
      'editable_pixels':int(np.count_nonzero(mask)),
      'pixels_changed_outside_mask':0,
      'prompt':prompt,'negative_prompt':negative_prompt,
      'limitations':[
        'Las texturas de las diferentes vistas no estan acopladas en 3D; las ventanas si comparten geometria.',
        'La perspectiva depende de la exactitud de la calibracion original.',
        'Las oclusiones/areas protegidas dependen de la mascara validada manualmente.',
      ]
    }
    (folder/f'{stem}.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    if show_preview:
        from IPython.display import display
        preview=combined.copy();preview.thumbnail((1100,850))
        display(preview)
    print(f'Vista {view_index}: {out} | {elapsed:.1f}s | resolucion recorte {ww}x{hh}')
    return out
