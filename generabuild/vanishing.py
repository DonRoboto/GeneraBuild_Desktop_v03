"""Diagnóstico de fugas para imágenes urbanas.

OpenCV detecta segmentos; la geometría pinhole proyecta las direcciones 3D X,Y,Z
hacia puntos de fuga. Las familias de segmentos se asignan por compatibilidad
angular, NO constituyen una reconstrucción automática ni ground truth.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

AXES = ('X · fachada', 'Y · profundidad', 'Z · vertical')
RGB_COLORS = ((246, 172, 65), (74, 209, 195), (205, 139, 255))


def detect_segments(image: Image.Image | np.ndarray | str | Path, *,
                    max_side: int = 1400, canny_low: int = 65,
                    hough_threshold: int = 70, min_length_fraction: float = .055,
                    max_segments: int = 280) -> list[list[float]]:
    """HoughLinesP, coordenadas devueltas en resolución ORIGINAL.

    Conserva solo segmentos largos y evita enviar miles de líneas al navegador.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            rgb = np.asarray(im.convert('RGB'))
    elif isinstance(image, Image.Image):
        rgb = np.asarray(image.convert('RGB'))
    else:
        rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError('La imagen debe ser RGB/RGBA')
    h, w = rgb.shape[:2]
    if min(h, w) < 40:
        return []
    scale = min(1., float(max_side) / max(w, h))
    small = cv2.resize(rgb[:, :, :3], None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else rgb[:, :, :3]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(gray, int(canny_low), min(255, int(2.2 * canny_low)))
    length = max(22, int(max(small.shape[:2]) * float(min_length_fraction)))
    raw = cv2.HoughLinesP(edges, 1., np.pi / 180., threshold=int(hough_threshold),
                          minLineLength=length, maxLineGap=max(9, length // 5))
    if raw is None:
        return []
    segments = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        seg = np.array([x1, y1, x2, y2], dtype=float) / scale
        length2 = float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))
        if length2 >= 18:
            segments.append((length2, seg.tolist()))
    segments.sort(key=lambda s: s[0], reverse=True)
    return [x[1] for x in segments[:int(max_segments)]]


def camera_vanishing_points(camera: dict | None, size_wh: tuple[int, int]) -> list[dict]:
    """VP_i = K R e_i en coordenadas homogéneas; posible infinito.

    Se considera 'at_infinity' también un VP a más de 50 diagonales del centro
    para evitar posiciones de pantalla numéricamente imprácticas.
    """
    if not camera or camera.get('ok') is False:
        return []
    if 'rvec' not in camera or 'K' not in camera:
        return []
    rvec = np.asarray(camera['rvec'], np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    K = np.asarray(camera['K'], np.float64).reshape(3, 3)
    w, h = map(int, size_wh)
    diag = float(np.hypot(w, h))
    output = []
    for j, label in enumerate(AXES):
        homogeneous = K @ R[:, j]
        finite = abs(float(homogeneous[2])) > 1e-9
        xy = homogeneous[:2] / homogeneous[2] if finite else None
        if finite:
            finite = bool(np.all(np.isfinite(xy)) and np.linalg.norm(xy - [w/2, h/2]) < 50 * diag)
        direction = homogeneous[:2]
        if np.linalg.norm(direction) < 1e-10:
            direction = np.array([1., 0.])
        direction = direction / np.linalg.norm(direction)
        output.append({
            'axis': j, 'label': label,
            'xy': [float(xy[0]), float(xy[1])] if finite else None,
            'at_infinity': not finite,
            'direction': direction.tolist(),
        })
    return output


def _segment_axis_angle(seg: list[float], vp: dict) -> float:
    x1, y1, x2, y2 = map(float, seg)
    d = np.array([x2-x1, y2-y1])
    length = np.linalg.norm(d)
    if length < 1e-9:
        return 90.
    if vp['at_infinity']:
        radial = np.asarray(vp['direction'], float)
    else:
        middle = np.array([(x1+x2)/2, (y1+y2)/2])
        radial = np.array(vp['xy'], float) - middle
    size = np.linalg.norm(radial)
    if size < 1e-9:
        return 90.
    cross = abs(d[0]*radial[1] - d[1]*radial[0]) / (length * size)
    return float(np.rad2deg(np.arcsin(np.clip(cross, 0., 1.))))


def assign_segments(segments: list[list[float]], vps: list[dict], *,
                    tolerance_deg: float = 9., max_per_axis: int = 24) -> tuple[list[dict], list[dict]]:
    """Clasifica segmentos de Hough por su cercanía ANGULAR a las fugas del modelo.

    La clasificación depende de la cámara inicial y no valida independientemente
    su pose; es una ayuda visual y un diagnóstico de concordancia.
    """
    if not vps:
        raw = [{'xyxy': [float(x) for x in s], 'axis': -1, 'error_deg': None}
               for s in segments[:max_per_axis]]
        return raw, []
    grouped: dict[int, list] = {i: [] for i in range(len(vps))}
    for seg in segments:
        deviations = [_segment_axis_angle(seg, vp) for vp in vps]
        best = int(np.argmin(deviations))
        if deviations[best] <= tolerance_deg:
            length = float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))
            grouped[best].append((length, deviations[best], seg))
    classified = []
    stats = []
    for k, vp in enumerate(vps):
        group = sorted(grouped[k], key=lambda z: z[0], reverse=True)[:max_per_axis]
        for length, angle, seg in group:
            classified.append({'xyxy': [float(x) for x in seg],
                               'axis': k, 'error_deg': round(float(angle), 3)})
        angles = [x[1] for x in group]
        stats.append({
            'axis': vp['label'], 'n_segments': len(group),
            'median_angular_error_deg': round(float(np.median(angles)), 3) if angles else None,
            'vp_x': round(vp['xy'][0], 1) if vp['xy'] else None,
            'vp_y': round(vp['xy'][1], 1) if vp['xy'] else None,
            'vp_at_infinity': bool(vp['at_infinity']),
        })
    return classified, stats


def vanishing_diagnostics(image: Image.Image | np.ndarray | Path | str,
                          camera: dict | None, *, canny_low: int = 65,
                          tolerance_deg: float = 9., max_segments: int = 240,
                          max_per_axis: int = 24) -> dict:
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            size = im.size
    elif isinstance(image, Image.Image):
        size = image.size
    else:
        h,w = image.shape[:2];size=(w,h)
    segments = detect_segments(image, canny_low=canny_low,max_segments=max_segments)
    vps = camera_vanishing_points(camera,size)
    lines,stats=assign_segments(segments,vps,tolerance_deg=tolerance_deg,max_per_axis=max_per_axis)
    empirical = estimate_vanishing_candidates(segments,size)
    return {'image_size_wh': list(size), 'raw_line_count':len(segments),
            'vanishing_points':vps,'empirical_vps':empirical,'segments':lines, 'stats':stats,
            'note':'Fugas teóricas K·R; segmentos OpenCV (Canny + HoughLinesP) agrupados por ángulo. No equivale a puntos de fuga medidos independientemente.'}


def _extend_to_image(segment: list[float], width: int, height: int):
    x1,y1,x2,y2=map(float,segment)
    v=np.array([x2-x1,y2-y1],float)
    n=np.linalg.norm(v)
    if n<1e-9:return None
    v/=n
    cx,cy=(x1+x2)/2,(y1+y2)/2
    radius=3*np.hypot(width,height)
    p=tuple(np.rint([cx-radius*v[0],cy-radius*v[1]]).astype(int))
    q=tuple(np.rint([cx+radius*v[0],cy+radius*v[1]]).astype(int))
    ok,pa,pb=cv2.clipLine((0,0,width,height),p,q)
    return (pa,pb) if ok else None


def draw_vanishing_overlay(image: Image.Image | np.ndarray, diag: dict, *,
                           extend_lines: bool = True, original_opacity: float=.8) -> Image.Image:
    rgb=np.array(image.convert('RGB') if isinstance(image,Image.Image) else image,dtype=np.uint8).copy()
    h,w=rgb.shape[:2]
    layer=rgb.copy()
    extended_counts={0:0,1:0,2:0}
    for line in diag.get('segments',[]):
        k=line['axis'];color=RGB_COLORS[k] if k>=0 else (235,219,141)
        # OpenCV procesa BGR; aquí sobre array RGB (el orden permanece tal cual)
        coords=line['xyxy']
        if extend_lines and k>=0 and extended_counts[k]<5:
            extended_counts[k]+=1
            ext=_extend_to_image(coords,w,h)
            if ext:cv2.line(layer,*ext,color,1,cv2.LINE_AA)
        p,q=np.rint(coords[:2]).astype(int),np.rint(coords[2:]).astype(int)
        cv2.line(layer,tuple(p),tuple(q),color,3,cv2.LINE_AA)
    rgb=cv2.addWeighted(layer,float(original_opacity),rgb,1-float(original_opacity),0)
    for vp in diag.get('vanishing_points',[]):
        k=vp['axis'];color=RGB_COLORS[k]
        if vp['at_infinity']:
            continue
        x,y=vp['xy']
        if 0 <= x < w and 0 <= y < h:
            x,y=int(round(x)),int(round(y))
            cv2.drawMarker(rgb,(x,y),color,cv2.MARKER_CROSS,24,2,cv2.LINE_AA)
            cv2.putText(rgb,f'VP {"XYZ"[k]}',(min(w-80,x+9),max(16,y-11)),
                        cv2.FONT_HERSHEY_SIMPLEX,.53,color,2,cv2.LINE_AA)
    # VP obtenidos exclusivamente de las rectas detectadas (candidatos).
    for empirical in diag.get('empirical_vps', []):
        x,y=empirical['xy']
        if 0<=x<w and 0<=y<h:
            x,y=int(round(x)),int(round(y))
            cv2.circle(rgb,(x,y),11,(248,248,248),2,cv2.LINE_AA)
            cv2.putText(rgb,empirical['name'],(min(w-130,x+12),max(18,y-12)),
                        cv2.FONT_HERSHEY_SIMPLEX,.44,(250,250,250),2,cv2.LINE_AA)
    return Image.fromarray(rgb)


def estimate_vanishing_candidates(segments: list[list[float]], size_wh: tuple[int,int], *,
                                  n_candidates: int=3, trials: int=450,
                                  threshold_px: float=8., min_inliers: int=5,
                                  seed: int=2026) -> list[dict]:
    """VP geométricos mediante RANSAC sobre rectas Hough, SIN usar cámaras.

    Un VP solo es candidato; múltiples edificios, distorsiones y líneas espurias
    pueden causar soluciones incorrectas. Omite soluciones a infinito.
    """
    w,h=map(int,size_wh)
    diag=math.hypot(w,h)
    if len(segments)<min_inliers or diag<=0:
        return []
    origin=np.array([w*.5,h*.5],float)
    rows=[];lengths=[]
    for seg in segments:
        x1,y1,x2,y2=map(float,seg)
        a=np.r_[(np.array([x1,y1])-origin)/diag,1.]
        b=np.r_[(np.array([x2,y2])-origin)/diag,1.]
        line=np.cross(a,b)
        n=np.linalg.norm(line[:2])
        if n<1e-10:continue
        rows.append(line/n)
        lengths.append(float(np.hypot(x2-x1,y2-y1)))
    if len(rows)<min_inliers:return []
    L=np.asarray(rows,float)
    lengths=np.asarray(lengths,float)
    remaining=np.arange(len(L))
    rng=np.random.default_rng(seed)
    candidates=[]
    for k in range(int(n_candidates)):
        if len(remaining)<min_inliers:break
        best=None;bestscore=-float('inf')
        for attempt in range(int(trials)):
            i,j=rng.choice(remaining,size=2,replace=False)
            v=np.cross(L[i],L[j])
            if abs(v[2])<1e-5:continue
            v/=v[2]
            if not np.all(np.isfinite(v)) or np.linalg.norm(v[:2])>15:
                continue
            err=abs(L[remaining]@v)*diag
            inliers=remaining[err<float(threshold_px)]
            if len(inliers)<min_inliers:continue
            score=float(np.sum(np.sqrt(lengths[inliers])))
            if score>bestscore:
                bestscore=score;best=inliers
        if best is None:break
        # Reajuste en coordenadas centradas-normalizadas.
        try:
            _,_,vh=np.linalg.svd(L[best],full_matrices=False)
            v=vh[-1]
            if abs(v[2])<1e-5:break
            v=v/v[2]
            err=abs(L[remaining]@v)*diag
            inliers=remaining[err<float(threshold_px)]
            if len(inliers)<min_inliers:break
            vp=origin + diag*v[:2]
            if not np.all(np.isfinite(vp)):break
        except np.linalg.LinAlgError:
            break
        candidates.append({
            'name':f'VP observado {k+1}',
            'xy':[float(vp[0]),float(vp[1])],
            'support':len(inliers),
            'median_distance_px':float(np.median(err[err<float(threshold_px)])),
            'method':'HoughLinesP + RANSAC + SVD (sin cámaras)',
        })
        remaining=remaining[err>=float(threshold_px)]
    return candidates


def draw_manual_annotations_overlay(image: Image.Image | np.ndarray, guides: list[dict],
                                    observed: list[dict], camera_vps: list[dict],
                                    empirical_vps: list[dict] | None=None) -> Image.Image:
    """Superpone guías y observaciones guardadas en píxeles originales.

    No altera las cámaras. Las guías se generan desde los puntos de fuga
    actuales; las aristas medidas son observaciones independientes.
    """
    from .manual_lines import line_through_point_and_vp, normalize_line_annotations
    rgb = np.asarray(image.convert('RGB') if isinstance(image, Image.Image) else image,dtype=np.uint8).copy()
    h,w=rgb.shape[:2]
    guides,observed=normalize_line_annotations(guides,observed,(w,h))
    palette=[(255,188,99),(101,255,222),(219,173,255)]
    for guide in guides:
        source=guide.get('source','camera')
        if source.startswith('opencv_'):
            idx=int(source.split('_')[1]); pool=empirical_vps or []
            vp={'xy':pool[idx]['xy'],'at_infinity':False} if idx<len(pool) else None
        else:
            vp=next((v for v in camera_vps if v['axis']==guide['axis']),None)
        if vp is None:continue
        segment=line_through_point_and_vp(guide['point'],vp,(w,h))
        if segment is None:continue
        p=(int(round(segment[0])),int(round(segment[1])))
        q=(int(round(segment[2])),int(round(segment[3])))
        c=palette[guide['axis']]
        # Trazo discontinuo a lo largo de toda la imagen.
        n=max(2,int(np.hypot(q[0]-p[0],q[1]-p[1])/25))
        for j in range(0,n,2):
            a=(int(round(p[0]+j*(q[0]-p[0])/n)),int(round(p[1]+j*(q[1]-p[1])/n)))
            b=(int(round(p[0]+min(n,j+1)*(q[0]-p[0])/n)),
               int(round(p[1]+min(n,j+1)*(q[1]-p[1])/n)))
            cv2.line(rgb,a,b,c,2,cv2.LINE_AA)
        point=(int(round(guide['point']['x'])),int(round(guide['point']['y'])))
        cv2.circle(rgb,point,5,c,-1,cv2.LINE_AA)
    for item in observed:
        a=(int(round(item['p1']['x'])),int(round(item['p1']['y'])))
        b=(int(round(item['p2']['x'])),int(round(item['p2']['y'])))
        c=palette[item['axis']]
        cv2.line(rgb,a,b,(20,24,30),6,cv2.LINE_AA)
        cv2.line(rgb,a,b,c,3,cv2.LINE_AA)
        cv2.circle(rgb,a,5,(255,255,255),-1,cv2.LINE_AA)
        cv2.circle(rgb,b,5,(255,255,255),-1,cv2.LINE_AA)
    return Image.fromarray(rgb)
