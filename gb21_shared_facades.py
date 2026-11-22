"""GeneraBuild v2.1-foto: motor auxiliar de fachadas compartidas en coordenadas 3D.

Compatible con `calibration_multiview.json` de GeneraBuild v2/v2.1.
No recalibra cámaras. Cada fachada/azotea recibe exactamente una textura 2D
(en sistema métrico local), proyectada a las 4 vistas con homografía inducida
por la cámara P_i. El renderer no genera materiales diferentes por fotografía.

Dependencias CPU: numpy, opencv-python-headless (o cv2), Pillow.
GPU/diffusers: opcionales; solo para refinamiento de material sin tocar vidrio.

IMPORTANTE: El realismo de una textura procedimental es limitado. Las ventanas
comparten posición y textura, pero esto NO equivale a una reconstrucción física
ni garantiza realismo fotográfico del material generado.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from hashlib import sha256
from typing import Optional
import json
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

FACE_IDS = {
    "front": (0, 1, 5, 4),
    "right": (1, 2, 6, 5),
    "back": (2, 3, 7, 6),
    "left": (3, 0, 4, 7),
    "roof": (4, 5, 6, 7),
}
WALL_FACES = ("front", "right", "back", "left")
# Valor de pixel de la etiqueta semántica (PIL/L):
SEMANTICS = {"background": 0, "wall": 1, "glass": 2,
             "frame": 3, "architectural_detail": 4, "roof": 5}


@dataclass(frozen=True)
class FacadeConfig:
    """Parámetros ÚNICOS para crear las texturas de TODAS las vistas.

    W/D/H/floors se leen del JSON de calibración, no se editan aquí.
    """
    style: str = "roma_contemporaneo"
    front_windows: int = 3
    side_windows: int = 5
    render_px_per_m: int = 44
    material_seed: int = 2026
    stucco_rgb: tuple[int, int, int] = (222, 217, 205)
    glass_rgb: tuple[int, int, int] = (67, 101, 119)
    frame_rgb: tuple[int, int, int] = (40, 47, 48)
    window_width_fraction: float = 0.58
    glass_light: float = 0.80
    floor_band_cm: float = 12.0
    balconies: bool = False  # barandales proyectados en la textura plana, NO volumen físico
    aerial_softness_px: float = 0.45
    seam_feather_px: float = 0.8
    preserve_original_buffer_px: int = 2

    def validate(self):
        if self.style not in ("roma_contemporaneo", "roma_sobrio"):
            raise ValueError("Estilo no reconocido")
        if not (1 <= self.front_windows <= 8 and 0 <= self.side_windows <= 12):
            raise ValueError("Número de ventanas fuera de rango")
        if not (10 <= self.render_px_per_m <= 120):
            raise ValueError("render_px_per_m debe estar entre 10 y 120")
        if not (0.30 <= self.window_width_fraction <= 0.82):
            raise ValueError("window_width_fraction fuera de rango")
        if not (0.1 <= self.glass_light <= 1.4):
            raise ValueError("glass_light fuera de rango")
        for rgb in (self.stucco_rgb, self.glass_rgb, self.frame_rgb):
            if len(rgb) != 3 or not all(0 <= int(x) <= 255 for x in rgb):
                raise ValueError("RGB debe tener tres canales 0..255")
        return self

    def to_json(self):
        return asdict(self)


def _seed_for(name: str, base_seed: int) -> int:
    digest = sha256(f"{base_seed}:gb22:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _noisy_material(width: int, height: int, base_rgb, rng, amplitude=5.3):
    """Plaster no periódico con variación muy discreta a varias escalas."""
    coarse = rng.normal(size=(max(5, height//35), max(5, width//35))).astype("float32")
    medium = rng.normal(size=(max(5, height//9), max(5, width//9))).astype("float32")
    fine = rng.normal(size=(height, width)).astype("float32")
    n = (cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)*0.9
         +cv2.resize(medium, (width, height), interpolation=cv2.INTER_LINEAR)*0.4
         +fine*0.21)
    # Iluminación suave del paño, sin patrones de mosaico.
    light = np.linspace(1.9, -1.8, height, dtype=np.float32)[:, None]
    v = np.array(base_rgb, dtype=np.float32)[None, None, :] + (n*amplitude+light)[..., None]
    v = np.clip(v, 0, 255).astype("uint8")
    return v


def _pix_z(height_px: int, z_m: float, ppm: int):
    # En la textura la azotea está ARRIBA y la planta baja ABAJO.
    return int(np.clip(round(height_px - z_m*ppm), 0, height_px-1))


def _add_window(rgb, labels, x0, y0, x1, y1, cfg: FacadeConfig, rng,
                shop=False, balconied=False):
    """Hueco, jambas, marco, cristal y reflejos con etiqueta semántica."""
    H, W = labels.shape
    x0, x1 = int(max(0,x0)), int(min(W-1,x1))
    y0, y1 = int(max(0,y0)), int(min(H-1,y1))
    if x1-x0 < 11 or y1-y0 < 12:
        return
    frame = max(2, int(round(min(x1-x0, y1-y0)*0.062)))
    reveal = max(frame+2, int(round(frame*1.65)))
    # Sombra del vano en el muro y jambas ligeramente más oscuras.
    cv2.rectangle(rgb, (x0-reveal, y0-reveal), (x1+reveal, y1+reveal),
                  (159, 155, 145), -1, cv2.LINE_AA)
    cv2.rectangle(labels, (max(0,x0-reveal), max(0,y0-reveal)),
                  (min(W-1,x1+reveal),min(H-1,y1+reveal)), SEMANTICS["architectural_detail"], -1)
    cv2.rectangle(rgb, (x0-1, y0-1), (x1+1, y1+1), cfg.frame_rgb, -1)
    cv2.rectangle(labels, (x0-1, y0-1), (x1+1,y1+1), SEMANTICS["frame"], -1)
    xa, xb = x0+frame, x1-frame
    ya, yb = y0+frame, y1-frame
    if xb <= xa or yb <= ya:
        return
    gh, gw = yb-ya+1, xb-xa+1
    # El vidrio tiene transparencia óptica sugerida por gradientes + reflejos,
    # NO es un plano de concreto con un rectángulo pintado encima.
    base = np.array(cfg.glass_rgb, dtype=np.float32)
    rows = np.linspace(0,1,gh,dtype=np.float32)[:, None, None]
    cols = np.linspace(0,1,gw,dtype=np.float32)[None,:,None]
    sky = 1.18 - 0.25*rows + 0.15*np.sin(cols*2.5 + rows*1.2)
    patch = np.ones((gh,gw,3),dtype=np.float32)*base*float(cfg.glass_light)*sky
    patch += np.array([3,7,12],dtype=np.float32)[None,None,:]*cols
    patch += rng.normal(0,1.7,size=(gh,gw,1)).astype(np.float32)
    rgb[ya:yb+1,xa:xb+1] = np.clip(patch,0,255).astype("uint8")
    labels[ya:yb+1,xa:xb+1] = SEMANTICS["glass"]
    # Reflejo cielo: franja oblicua suave con variación, igual en las 4 vistas.
    for band in range(2):
        sx = xa + int(gw*(0.15+0.34*band))
        ex = min(xb,sx+max(2,int(gw*0.13)))
        ramp = np.linspace(0,1,max(1,ex-sx+1),dtype=np.float32)[None,:,None]
        rgb[ya:yb+1,sx:ex+1] = np.clip(
            rgb[ya:yb+1,sx:ex+1].astype(np.float32)
            + (8 if band==0 else 4)*np.sin(ramp*np.pi), 0,255
        ).astype("uint8")
    # Montante (carpintería), no reescribe toda la ventana como muro.
    if shop or gw > 24:
        cx = int(round((xa+xb)/2))
        m = max(2,frame//2)
        cv2.rectangle(rgb,(cx-m, ya),(cx+m,yb),cfg.frame_rgb,-1)
        cv2.rectangle(labels,(cx-m,ya),(cx+m,yb),SEMANTICS["frame"],-1)
    # Travesaño horizontal discreto.
    if not shop:
        cy=int(round(ya+gh*0.27))
        cv2.line(rgb,(xa,cy),(xb,cy),cfg.frame_rgb,max(2,frame//2),cv2.LINE_AA)
        cv2.line(labels,(xa,cy),(xb,cy),SEMANTICS["frame"],max(2,frame//2))
    # Alféizar + sombra inferior para sugerir profundidad.
    sill_y=min(H-1,y1+reveal+max(1,frame//2))
    cv2.rectangle(rgb,(max(0,x0-reveal-2), y1+reveal),
                  (min(W-1,x1+reveal+2),sill_y), (196,193,184),-1)
    cv2.rectangle(labels,(max(0,x0-reveal-2), y1+reveal),
                  (min(W-1,x1+reveal+2),sill_y), SEMANTICS["architectural_detail"],-1)
    if balconied:
        # Detalle superficial (sin geometría volumétrica): barandal oscuro.
        ybar=int(min(H-3,y1+reveal+int(0.16*(y1-y0))))
        cv2.line(rgb,(x0-reveal-6,ybar),(x1+reveal+6,ybar),(48,55,57),3,cv2.LINE_AA)
        cv2.line(labels,(max(0,x0-reveal-6),ybar),
                 (min(W-1,x1+reveal+6),ybar),SEMANTICS["architectural_detail"],3)
        for t in (0.20,0.50,0.80):
            xbar = int(round(x0+t*(x1-x0)))
            cv2.line(rgb,(xbar,ybar),(xbar,min(H-1,ybar+max(8,(y1-y0)//5))), (55,57,61),2)
            cv2.line(labels,(xbar,ybar),(xbar,min(H-1,ybar+max(8,(y1-y0)//5))),
                     SEMANTICS["architectural_detail"],2)


def _wall_atlas(face_name: str, span_m: float, total_h: float, floors: int,
                cfg: FacadeConfig) -> tuple[np.ndarray,np.ndarray]:
    ppm=int(cfg.render_px_per_m)
    width=max(48,int(round(span_m*ppm)))
    height=max(72,int(round(total_h*ppm)))
    rng=np.random.default_rng(_seed_for(face_name,cfg.material_seed))
    stucco=cfg.stucco_rgb if cfg.style=="roma_contemporaneo" else (231,222,210)
    rgb=_noisy_material(width,height,stucco,rng,5.5)
    sem=np.full((height,width),SEMANTICS["wall"],dtype=np.uint8)
    level_h=total_h/floors
    # Franjas entre losas, sin rejilla de azulejos ni divisiones verticales continuas.
    band_px=max(2,int(round(cfg.floor_band_cm/100.*ppm)))
    for k in range(1,floors):
        y=_pix_z(height,k*level_h,ppm)
        cv2.line(rgb,(0,y),(width-1,y),(189,186,178),band_px,cv2.LINE_AA)
        cv2.line(sem,(0,y),(width-1,y),SEMANTICS["architectural_detail"],band_px)
        cv2.line(rgb,(0,max(0,y-band_px)),(width-1,max(0,y-band_px)),(233,226,214),max(1,band_px//2))
    # Marcado de remate y zócalo.
    edge=max(3,int(round(0.10*ppm)))
    cv2.rectangle(rgb,(0,0),(width-1,edge),(196,195,184),-1)
    cv2.rectangle(sem,(0,0),(width-1,edge),SEMANTICS["architectural_detail"],-1)
    cv2.rectangle(rgb,(0,height-edge*2),(width-1,height-1),(178,177,166),-1)
    cv2.rectangle(sem,(0,height-edge*2),(width-1,height-1),SEMANTICS["architectural_detail"],-1)

    bays = int(cfg.front_windows if face_name in ("front","back") else cfg.side_windows)
    for story in range(floors):
        lower_z=story*level_h
        if story == 0:
            sill_z=lower_z+0.10
            head_z=lower_z+0.88*level_h
        else:
            sill_z=lower_z+0.28*level_h
            head_z=lower_z+0.83*level_h
        top=_pix_z(height,head_z,ppm)
        bottom=_pix_z(height,sill_z,ppm)
        for bay in range(bays):
            center=(bay+0.5)/bays
            fraction=cfg.window_width_fraction
            if story == 0: fraction=0.76
            u0=(center-fraction/(bays*2))
            u1=(center+fraction/(bays*2))
            a=int(round(width*u0));b=int(round(width*u1))
            # Solo algunos niveles con barandal, idéntico en todas las vistas.
            balcony=bool(cfg.balconies and face_name=="front" and story>1
                         and story%3==0 and bay==bays//2)
            _add_window(rgb,sem,a,top,b,bottom,cfg,rng,shop=(story==0),balconied=balcony)
    return rgb,sem


def _roof_atlas(W_m: float,D_m: float,cfg:FacadeConfig) -> tuple[np.ndarray,np.ndarray]:
    ppm=int(cfg.render_px_per_m)
    width=max(48,int(round(W_m*ppm)))
    height=max(48,int(round(D_m*ppm)))
    rng=np.random.default_rng(_seed_for("roof",cfg.material_seed))
    roof=_noisy_material(width,height,(176,172,161),rng,5.4)
    lbl=np.full((height,width),SEMANTICS["roof"],dtype=np.uint8)
    e=max(3,int(round(.20*ppm)))
    # Pretil de cubierta físicamente plano (texturizado, sin volumen 3D añadido).
    cv2.rectangle(roof,(0,0),(width-1,height-1),(207,203,192),e*2)
    cv2.rectangle(lbl,(0,0),(width-1,height-1),SEMANTICS["architectural_detail"],e*2)
    for s in (0.3,0.65):
        x=int(round(width*s))
        y=int(round(height*0.48))
        ww=max(7,int(round(.9*ppm)));hh=max(9,int(round(1.5*ppm)))
        if x+ww>=width-e*2 or y+hh>=height-e*2:continue
        cv2.rectangle(roof,(x,y),(x+ww,y+hh),(95,99,100),-1)
        cv2.rectangle(roof,(x+3,y+3),(x+ww-3,y+hh-3),(128,145,145),-1)
        cv2.rectangle(lbl,(x,y),(x+ww,y+hh),SEMANTICS["architectural_detail"],-1)
    return roof,lbl


def load_atlases_from_dir(directory:str|Path):
    """Carga atlas fotografiados o retocados por el usuario con UV FIJO.

    Debe existir atlas_<cara>.png y semantics_<cara>.png para cada cara.
    Edita las texturas sin alterar la disposición física de las ventanas;
    si la alteras, actualiza también la máscara semántica correspondiente.
    """
    directory=Path(directory)
    atlases={}
    for name in FACE_IDS:
        c=directory/f"atlas_{name}.png"
        m=directory/f"semantics_{name}.png"
        if not c.is_file() or not m.is_file():
            raise FileNotFoundError(f"Faltan atlas o etiquetas para {name} en {directory}")
        rgb=np.asarray(Image.open(c).convert("RGB"),dtype=np.uint8).copy()
        sem=np.asarray(Image.open(m).convert("L"),dtype=np.uint8).copy()
        if rgb.shape[:2]!=sem.shape or min(sem.shape)<8:
            raise ValueError(f"Dimensiones distintas para atlas y semántica de {name}")
        if np.any(~np.isin(np.unique(sem),list(SEMANTICS.values()))):
            raise ValueError(f"Etiquetas de material no reconocidas en {name}")
        atlases[name]=(rgb,sem)
    return atlases


def make_atlases(metadata:dict,cfg:FacadeConfig):
    """Produce textura física y etiquetas de material para cada cara, UNA sola vez."""
    cfg.validate()
    box=metadata.get("target_box_m")
    if box is None:
        # El JSON v2.1 puede conservar solo vertices_3d_m. Derivar
        # dimensiones de las aristas ortogonales sin recalibrar cámaras.
        pts=np.asarray(metadata["vertices_3d_m"],dtype=float)
        box={"W":float(np.linalg.norm(pts[1]-pts[0])),
             "D":float(np.linalg.norm(pts[3]-pts[0])),
             "H":float(np.linalg.norm(pts[4]-pts[0]))}
    W,D,H=(float(box[k]) for k in ("W","D","H"))
    floors=int(metadata["num_floors"])
    if not (W>0 and D>0 and H>0 and floors>=2):
        raise ValueError("Geometría inválida")
    atlases={}
    for name in WALL_FACES:
        atlases[name]=_wall_atlas(name,W if name in ("front","back") else D,H,floors,cfg)
    atlases["roof"]=_roof_atlas(W,D,cfg)
    return atlases


def _save_atlases(atlases:dict,folder:Path,cfg:FacadeConfig,metadata:dict)->dict:
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    checksums={}
    for face,(rgb,labels) in atlases.items():
        path=folder/f"atlas_{face}.png"
        Image.fromarray(rgb).save(path)
        Image.fromarray(labels).save(folder/f"semantics_{face}.png")
        checksums[face]=sha256(rgb.tobytes()+labels.tobytes()).hexdigest()
    manifest={"version":"GeneraBuild_v2.2","config":cfg.to_json(),
              "reference_geometry_m":metadata["reference_box_m"],
              "target_geometry_m":metadata["target_box_m"],
              "floors":int(metadata["num_floors"]),
              "atlas_sha256_rgb_plus_semantics":checksums,
              "semantic_labels":SEMANTICS,
              "note":"Un atlas por cara compartido entre todas las vistas."
              }
    (folder/"facade_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    return manifest


def _project(P:np.ndarray, xyz:np.ndarray)->np.ndarray:
    xyz=np.asarray(xyz,dtype=np.float64).reshape(-1,3)
    q=(np.asarray(P,dtype=np.float64) @ np.c_[xyz,np.ones(len(xyz))].T).T
    if not np.all(np.isfinite(q)) or np.any(q[:,2]<=1e-5):
        raise ValueError("Una cara está detrás de la cámara, o la calibración es inválida")
    return q[:,:2]/q[:,2,None]


def _camera_center(view:dict)->np.ndarray:
    if "camera_center_m" in view:
        return np.array(view["camera_center_m"],dtype=np.float64).reshape(3)
    R,_=cv2.Rodrigues(np.array(view["rvec"],dtype=np.float64).reshape(3,1))
    return (-R.T@np.array(view["tvec"],dtype=np.float64).reshape(3,1)).ravel()


def _mask_reference(metadata, view, shape, buffer_px=2):
    """Resguarda todos los píxeles en la silueta del edificio original 3 pisos."""
    h,w=shape
    ref=np.array(metadata["reference_vertices_3d_m"],dtype=np.float64)
    uv=np.rint(_project(view["P"],ref)).astype(np.int32)
    hull=cv2.convexHull(uv)
    mask=np.zeros((h,w),dtype=np.uint8)
    cv2.fillConvexPoly(mask,hull,255)
    if buffer_px>0:
        k=2*int(buffer_px)+1
        mask=cv2.dilate(mask,np.ones((k,k),dtype=np.uint8))
    return mask


def _warp_face_to_image(rgb_face,sem_face,uv,canvas_rgb,canvas_sem,
                        saved_mask,full_edited_mask):
    """Warp por ROI: textura de cara -> plano proyectado en la imagen.

    `saved_mask` restringe tanto las caras cercanas como lejanas.
    `full_edited_mask` acumula la zona cubierta de cualquier cara.
    """
    H,W=canvas_sem.shape
    minxy=np.floor(np.min(uv,axis=0)-2).astype(int)
    maxxy=np.ceil(np.max(uv,axis=0)+2).astype(int)
    x0=max(0,int(minxy[0]));y0=max(0,int(minxy[1]))
    x1=min(W,int(maxxy[0]+1));y1=min(H,int(maxxy[1]+1))
    if x1<=x0 or y1<=y0:
        return 0
    cwidth,cheight=(x1-x0,y1-y0)
    sh,sw=sem_face.shape
    # Esquina inferior izq / inferior der / superior der / superior izq.
    src=np.array([[0,sh-1],[sw-1,sh-1],[sw-1,0],[0,0]],dtype=np.float32)
    dst=(np.asarray(uv,dtype=np.float32)-[x0,y0]).astype(np.float32)
    Hmatrix=cv2.getPerspectiveTransform(src,dst)
    proj_rgb=cv2.warpPerspective(rgb_face,Hmatrix,(cwidth,cheight),
                                flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    proj_sem=cv2.warpPerspective(sem_face,Hmatrix,(cwidth,cheight),
                                flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    poly=np.rint(dst).astype(np.int32)
    in_face=np.zeros((cheight,cwidth),dtype=np.uint8)
    cv2.fillConvexPoly(in_face,poly,1)
    edit=(in_face>0)&(saved_mask[y0:y1,x0:x1]>127)
    if np.any(edit):
        crop_rgb=canvas_rgb[y0:y1,x0:x1]
        crop_sem=canvas_sem[y0:y1,x0:x1]
        crop_rgb[edit]=proj_rgb[edit]
        crop_sem[edit]=proj_sem[edit]
        full_edited_mask[y0:y1,x0:x1][edit]=255
    return int(np.count_nonzero(edit))


def render_one_view(original:Image.Image, saved_mask:np.ndarray, metadata:dict,
                    view_index:int,atlases:dict,cfg:FacadeConfig,
                    preserve_existing:bool=True):
    """Proyecta el MISMO atlas 3D a una vista y compone sobre foto intacta.

    Returns (PIL RGB, mask_trabajo uint8, etiquetas_en_imagen uint8, diagnostic).
    No modifica un solo píxel fuera de `mask_trabajo`.
    """
    cfg.validate()
    if not (1<=view_index<=len(metadata["views"])):
        raise ValueError("view_index fuera de rango")
    view=metadata["views"][view_index-1]
    base=np.asarray(original.convert("RGB"),dtype=np.uint8)
    h,w=base.shape[:2]
    saved_mask=np.asarray(saved_mask,dtype=np.uint8)
    if saved_mask.shape!=(h,w):
        raise ValueError("Máscara de protección con resolución diferente a imagen")
    from gb21_extension_mask import extension_work_mask
    # Para ampliación, renderizamos únicamente el prisma H0..H1;
    # en vistas aéreas ese prisma debe cubrir la cubierta original.
    work_mask=extension_work_mask(saved_mask,metadata,view_index,preserve_existing)
    if not np.any(work_mask):
        raise ValueError("Máscara de trabajo vacía; inspecciona la calibración, cambia el modo o revisa el área permitida")
    target=np.asarray(metadata["vertices_3d_m"],dtype=np.float64)
    P=np.asarray(view["P"],dtype=np.float64)
    C=_camera_center(view)
    canvas=base.copy()
    labels=np.zeros((h,w),dtype=np.uint8)
    coverage=np.zeros((h,w),dtype=np.uint8)
    visible=[]
    for name,ids in FACE_IDS.items():
        face=target[list(ids)]
        normal=np.cross(face[1]-face[0],face[2]-face[1])
        if float(np.dot(normal,C-face.mean(axis=0)))<=1e-7:
            continue
        # Distancia al centro de cámara: pintado de atrás hacia delante.
        depth=float(np.linalg.norm(face.mean(axis=0)-C))
        visible.append((depth,name,face))
    visible.sort(reverse=True,key=lambda item:item[0])
    pixels={}
    for _,name,face in visible:
        try:
            uv=_project(P,face)
        except ValueError:
            continue
        arr,sem=atlases[name]
        pixels[name]=_warp_face_to_image(arr,sem,uv,canvas,labels,work_mask,coverage)
    # Todas las escrituras anteriores se limitaron a máscara exacta; verificar.
    unchanged=bool(np.array_equal(canvas[work_mask==0],base[work_mask==0]))
    if not unchanged:
        raise AssertionError("Se cambió un píxel fuera de la máscara")
    if preserve_existing:
        # Conservamos los píxeles originales que SIGUEN SIENDO visibles.
        # En vista aérea no se debe proteger la cubierta antigua tapada
        # físicamente por la nueva torre, que cae dentro de work_mask.
        preserved=_mask_reference(metadata,view,(h,w),cfg.preserve_original_buffer_px)
        keep_old=(preserved>0)&(work_mask==0)
        if not np.array_equal(canvas[keep_old],base[keep_old]):
            raise AssertionError("Se alteró parte visible de la base original")
    if cfg.aerial_softness_px>0 and view_index<=2:
        # Sólo una pequeña suavización del material para integrarlo con Earth.
        smooth=cv2.GaussianBlur(canvas,(0,0),cfg.aerial_softness_px)
        canvas[coverage>0]=smooth[coverage>0]
        if not np.array_equal(canvas[work_mask==0],base[work_mask==0]):
            raise AssertionError("La suavización modificó el contexto")
    # Suavización interna opcional, con ALFA estrictamente cero fuera de work_mask.
    if cfg.seam_feather_px>0:
        alpha=cv2.GaussianBlur((coverage>0).astype(np.float32),(0,0),cfg.seam_feather_px)
        alpha=np.where((coverage>0)&(work_mask>0),alpha,0.)
        mixed=(canvas.astype(np.float32)*alpha[...,None]
               +base.astype(np.float32)*(1.-alpha[...,None]))
        canvas=mixed.clip(0,255).astype(np.uint8)
    if not np.array_equal(canvas[work_mask==0],base[work_mask==0]):
        raise AssertionError("Un píxel fuera de máscara fue cambiado")
    stats={
        "view":view_index,
        "fov_horizontal_deg":view.get("fov_horizontal_deg"),
        "faces_visible":[v[1] for v in visible],
        "projected_face_pixel_counts":pixels,
        "edit_mask_pixels":int(np.count_nonzero(work_mask)),
        "actually_rendered_pixels":int(np.count_nonzero(coverage)),
        "glass_pixels":int(np.count_nonzero(labels==SEMANTICS["glass"])),
        "frame_pixels":int(np.count_nonzero(labels==SEMANTICS["frame"])),
        "changed_pixels_outside_mask":0,
        "original_building_pixels_changed":0 if preserve_existing else None,
    }
    return Image.fromarray(canvas),work_mask,labels,stats


def diagnostic_overlay(photo:Image.Image,mask:np.ndarray,labels:np.ndarray):
    """Previsualización: máscara azul, cristal cian y marcos amarillos."""
    canvas=np.asarray(photo.convert("RGB")).copy()
    blue=mask>127
    canvas[blue]=(canvas[blue]*.75+np.array([45,140,240])*.25).astype(np.uint8)
    glass=labels==SEMANTICS["glass"]
    frame=labels==SEMANTICS["frame"]
    canvas[glass]=(canvas[glass]*0.55+np.array([35,240,240])*.45).astype(np.uint8)
    canvas[frame]=(canvas[frame]*.5+np.array([255,213,35])*.5).astype(np.uint8)
    return Image.fromarray(canvas)


def run_project(project_dir: str|Path,
                config:FacadeConfig=FacadeConfig(),
                preserve_existing:bool=True,
                indices=(1,2,3,4),
                name:str="v22",
                atlas_override_dir:Optional[str|Path]=None):
    """Genera UNA vez los atlas y los reutiliza para las cuatro cámaras.

    Exporta fotografía integrada, máscaras, semántica y diagnósticos. No GPU.
    """
    project_dir=Path(project_dir)
    meta_path=project_dir/"calibration_multiview.json"
    if not meta_path.exists():
        raise FileNotFoundError("No se encontró calibration_multiview.json; reutiliza el proyecto calibrado de v2.1")
    metadata=json.loads(meta_path.read_text(encoding="utf-8"))
    if len(metadata.get("views",[]))!=4:
        raise ValueError("Se requieren las cuatro vistas calibradas")
    cfg=config.validate()
    # Seguridad: ninguna operación de este módulo reescribe el JSON de calibración.
    hash_before=sha256(meta_path.read_bytes()).hexdigest()
    output_dir=project_dir/"generated_v22"/name
    atlas_dir=output_dir/"shared_atlases"
    if atlas_override_dir is not None:
        # El atlas editado prevalece, pero sigue siendo compartido por las 4 vistas.
        atlas=load_atlases_from_dir(atlas_override_dir)
    else:
        atlas=make_atlases(metadata,cfg)
    manifest=_save_atlases(atlas,atlas_dir,cfg,metadata)
    manifest['atlas_override_source']=(str(atlas_override_dir) if atlas_override_dir else None)
    (atlas_dir/'facade_manifest.json').write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    records=[]
    for i in indices:
        view=metadata["views"][i-1]
        im_path=project_dir/view["image_relative"]
        mask_path=project_dir/view["mask_relative"]
        if not im_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Vista {i}: falta imagen o máscara de la calibración")
        with Image.open(im_path) as src:
            original=src.convert("RGB")
        with Image.open(mask_path) as saved:
            saved_mask=np.asarray(saved.convert("L"),dtype=np.uint8)
        result,mask,labels,stats=render_one_view(
            original,saved_mask,metadata,i,atlas,cfg,preserve_existing)
        base=f"Vista_{i}_{metadata['num_floors']}pisos"
        result.save(output_dir/f"{base}_shared.png")
        Image.fromarray(mask).save(output_dir/f"{base}_mask.png")
        Image.fromarray(labels).save(output_dir/f"{base}_semantics.png")
        diagnostic_overlay(result,mask,labels).save(output_dir/f"{base}_diagnostico.png")
        Image.fromarray(cv2.Canny(np.asarray(result),80,170)).save(output_dir/f"{base}_canny.png")
        records.append({"index":i,"image_relative":str((output_dir/f"{base}_shared.png").relative_to(project_dir)),
                        "mask_relative":str((output_dir/f"{base}_mask.png").relative_to(project_dir)),
                        "semantic_relative":str((output_dir/f"{base}_semantics.png").relative_to(project_dir)),
                        "diagnostic_relative":str((output_dir/f"{base}_diagnostico.png").relative_to(project_dir)),
                        "canny_relative":str((output_dir/f"{base}_canny.png").relative_to(project_dir)),
                        "source_sha256":view.get("image_sha256"),
                        "rmse_px":view.get("reprojection_rmse_px"),
                        **stats})
    report={"generabuild_version":"v2.2","created_unix":time.time(),
            "calibration_sha256":hash_before,
            "facade_manifest":manifest,"preserve_original_3_floors":bool(preserve_existing),
            "facade_mode":"shared_3d_uv_textures",
            "views":records,
            "notes":[
              "Los cristales, marcos y materiales se proyectan de los mismos atlas 3D.",
              "Fuera de la máscara cada píxel original permanece idéntico.",
              "Si la máscara original invade a un vecino, el sistema NO puede detectarlo automáticamente.",
              "Las fachadas se representan como planos texturizados; balcones/pretiles dibujados no tienen volumen 3D.",
              "Una cámara con FOV efectivo correcto visualmente no garantiza precisión métrica real.",
            ]}
    report_file=output_dir/"reporte_v22.json"
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if sha256(meta_path.read_bytes()).hexdigest()!=hash_before:
        raise AssertionError("Se ha modificado inesperadamente la calibración original")
    return report_file,report


def make_montage(report:dict,project_dir:Path,output:Path|None=None,
                 max_wh=(760,445)):
    """Montaje 2×2 de las cuatro imágenes; útil para control visual."""
    project_dir=Path(project_dir)
    panels=[]
    for r in report["views"]:
        im=Image.open(project_dir/r["image_relative"]).convert("RGB")
        im.thumbnail(max_wh)
        panel=Image.new("RGB",(max_wh[0]+22,max_wh[1]+38),(19,28,39))
        panel.paste(im,((panel.width-im.width)//2,26))
        ImageDraw.Draw(panel).text((12,7),
            f"VISTA {r['index']} | FOV={r['fov_horizontal_deg']} | cristales={r['glass_pixels']}px",
            fill=(240,245,250))
        panels.append(panel)
    if not panels:
        raise ValueError("No hay vistas para mostrar")
    pW,pH=panels[0].size
    montage=Image.new("RGB",(2*pW,2*pH),(13,17,25))
    for n,p in enumerate(panels[:4]):
        montage.paste(p,((n%2)*pW,(n//2)*pH))
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        montage.save(output)
    return montage


def _roi_from_mask(mask:np.ndarray,margin:0.35):
    """Recorte alrededor de una región editable, limitado a la imagen."""
    ys,xs=np.nonzero(mask>127)
    if len(xs)==0:
        raise ValueError("Refinamiento: máscara vacía")
    H,W=mask.shape
    x0,x1=int(xs.min()),int(xs.max())+1
    y0,y1=int(ys.min()),int(ys.max())+1
    bW,bH=x1-x0,y1-y0
    mx=max(48,int(bW*margin));my=max(48,int(bH*margin))
    return (max(0,x0-mx),max(0,y0-my),min(W,x1+mx),min(H,y1+my))


def refine_walls_one_view(project_dir:Path,view_index:int,pipeline,
                          generation_name:str="v22",seed:int=1234,
                          steps:int=35,cfg_scale:float=5.0,control_scale:float=0.55,
                          max_dim:int=768,prompt:Optional[str]=None,
                          negative_prompt:Optional[str]=None):
    """Opcional CUDA: mejora material de MUROS, protegiendo ventanas y marcos.

    `pipeline` debe ser un StableDiffusionControlNetInpaintPipeline. Los colores
    y microtexturas de los muros PUEDEN variar por vista tras este paso;
    usa los resultados sin refinar para consistencia material estricta.
    """
    import torch
    project_dir=Path(project_dir)
    meta=json.loads((project_dir/"calibration_multiview.json").read_text(encoding="utf-8"))
    folder=project_dir/"generated_v22"/generation_name
    stem=f"Vista_{view_index}_{meta['num_floors']}pisos"
    original=Image.open(folder/f"{stem}_shared.png").convert("RGB")
    label=np.asarray(Image.open(folder/f"{stem}_semantics.png").convert("L"))
    # SÓLO paño de muro; no tocar vidrios/marcos/divisiones ni entorno urbano.
    editable=np.where(label==SEMANTICS["wall"],255,0).astype(np.uint8)
    # Erosión para que IA no invada bordes de carpintería/vecinos.
    editable=cv2.erode(editable,np.ones((3,3),np.uint8))
    if not np.any(editable):
        raise ValueError("La máscara de muros es vacía. Prueba el render sin refinamiento.")
    roi=_roi_from_mask(editable,.37)
    img_crop=original.crop(roi)
    mask_crop=Image.fromarray(editable).crop(roi)
    rw,rh=img_crop.size
    factor=min(1.4,max_dim/max(rw,rh))
    ww=max(64,int(round(rw*factor/8))*8)
    hh=max(64,int(round(rh*factor/8))*8)
    image_work=img_crop.resize((ww,hh),Image.Resampling.LANCZOS)
    mask_work=mask_crop.resize((ww,hh),Image.Resampling.NEAREST)
    edges=cv2.Canny(np.asarray(img_crop),85,170)
    edge_rgb=cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    control_work=Image.fromarray(edge_rgb).resize((ww,hh),Image.Resampling.LANCZOS)
    if prompt is None:
        prompt=(f"Photorealistic {meta['num_floors']}-story modern Mexico City infill building, "
                "warm cream plaster stucco wall microtexture, finely finished exterior wall, "
                "natural daylight, mild weathering, precise construction details, "
                "professional documentary street architecture photograph")
    if negative_prompt is None:
        negative_prompt=("unfinished building, boarded windows, concrete slabs, tiles, checkerboard, "
                         "mosaic, mural, colorful pattern, blur, CGI, distorted buildings")
    try:
        device="cuda" if torch.cuda.is_available() else "cpu"
        generator=torch.Generator(device=device).manual_seed(int(seed))
        output=pipeline(prompt=prompt,negative_prompt=negative_prompt,image=image_work,
                        mask_image=mask_work,control_image=control_work,
                        controlnet_conditioning_scale=control_scale,
                        num_inference_steps=int(steps),guidance_scale=float(cfg_scale),
                        width=ww,height=hh,generator=generator).images[0]
    except Exception as exc:
        raise RuntimeError("Falló el refinamiento; el render compartido sin IA sigue disponible") from exc
    # Componer SOLO la región de material muro definida por la semántica 3D.
    canvas=np.asarray(original).copy()
    x0,y0,x1,y1=roi
    refined=np.asarray(output.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS),dtype=np.uint8)
    roi_mask=editable[y0:y1,x0:x1]>127
    piece=canvas[y0:y1,x0:x1]
    piece[roi_mask]=refined[roi_mask]
    if np.any(canvas[label!=SEMANTICS["wall"]] != np.asarray(original)[label!=SEMANTICS["wall"]]):
        raise AssertionError("El refinamiento tocó una ventana o el contexto")
    path=folder/f"{stem}_refinado_paredes.png"
    Image.fromarray(canvas).save(path)
    return path
