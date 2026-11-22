"""Líneas de perspectiva manuales: guías dependientes de cámara y aristas observadas.

Convención de las direcciones del prisma: X fachada, Y profundidad, Z vertical.
Una guía por punto usa un VP ya calculado y NO es evidencia independiente.
Una arista observada se marca con dos extremos reales y SÍ puede emplearse
como restricción de orientación (opcional) al ajustar cámaras.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import shutil
from pathlib import Path

import numpy as np

AXES = ('X', 'Y', 'Z')


def _point(value: dict, size_wh: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, dict) or not {'x', 'y'} <= set(value):
        raise ValueError('Se espera un punto con coordenadas x, y')
    x, y = float(value['x']), float(value['y'])
    w, h = size_wh
    if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h):
        raise ValueError(f'Punto fuera de la imagen ({w}×{h})')
    return np.array([x, y], dtype=np.float64)


def normalize_line_annotations(guides: list, observed: list,
                               size_wh: tuple[int, int]) -> tuple[list, list]:
    """Valida y normaliza líneas, restringiendo su cardinalidad.

    Las anotaciones usan SIEMPRE píxeles de la fotografía original, no el canvas.
    """
    if not isinstance(guides, list) or not isinstance(observed, list):
        raise ValueError('Las líneas deben representarse como listas por vista')
    if len(guides) > 120 or len(observed) > 120:
        raise ValueError('No se permiten más de 120 anotaciones por tipo y vista')
    g, o = [], []
    for item in guides:
        axis = int(item['axis'])
        if axis not in (0, 1, 2):
            raise ValueError('Eje de guía fuera de X/Y/Z')
        p = _point(item['point'], size_wh)
        source=str(item.get('source','camera'))
        if source not in ('camera','opencv_0','opencv_1','opencv_2'):
            raise ValueError('Fuente del VP de guía inválida')
        g.append({'axis': axis, 'source': source,
                  'point': {'x': float(p[0]), 'y': float(p[1])}})
    for item in observed:
        axis = int(item['axis'])
        if axis not in (0, 1, 2):
            raise ValueError('Eje de arista fuera de X/Y/Z')
        p1 = _point(item['p1'], size_wh)
        p2 = _point(item['p2'], size_wh)
        if np.linalg.norm(p2-p1) < 8:
            raise ValueError('Una arista observada debe medir al menos 8 píxeles')
        o.append({'axis': axis,
                  'p1': {'x': float(p1[0]), 'y': float(p1[1])},
                  'p2': {'x': float(p2[0]), 'y': float(p2[1])}})
    return g, o


def line_through_point_and_vp(point: dict, vp: dict,
                              size_wh: tuple[int, int]) -> list[float] | None:
    """Devuelve extremos de una guía recortada al rectángulo de la imagen.

    Soporta VP fuera del encuadre y al infinito (dirección paralela).
    """
    p = _point(point, size_wh)
    if vp.get('at_infinity') or vp.get('xy') is None:
        d = np.asarray(vp.get('direction', []), dtype=float)
    else:
        d = np.asarray(vp['xy'], dtype=float) - p
    if d.shape != (2,) or not np.isfinite(d).all():
        return None
    n = np.linalg.norm(d)
    if n < 1e-10:
        return None
    d /= n
    w, h = size_wh
    xs, ys = max(0, w-1), max(0, h-1)
    samples = []
    if abs(d[0]) > 1e-10:
        for x in (0., float(xs)):
            t = (x-p[0])/d[0]
            y = p[1]+t*d[1]
            if -1e-7 <= y <= ys+1e-7:
                samples.append((t, (x, float(np.clip(y, 0, ys)))))
    if abs(d[1]) > 1e-10:
        for y in (0., float(ys)):
            t = (y-p[1])/d[1]
            x = p[0]+t*d[0]
            if -1e-7 <= x <= xs+1e-7:
                samples.append((t, (float(np.clip(x, 0, xs)), y)))
    if len(samples) < 2:
        return None
    samples.sort(key=lambda value: value[0])
    q1, q2 = samples[0][1], samples[-1][1]
    return [float(q1[0]), float(q1[1]), float(q2[0]), float(q2[1])]


def _segment_values(item: dict) -> np.ndarray:
    p1, p2 = item['p1'], item['p2']
    return np.array([p1['x'], p1['y'], p2['x'], p2['y']], dtype=np.float64)


def manual_axis_error_deg(item: dict, vp: dict) -> float:
    """Desviación angular entre arista observada y fuga de la cámara."""
    x1, y1, x2, y2 = _segment_values(item)
    d = np.array([x2-x1, y2-y1])
    if vp.get('at_infinity') or vp.get('xy') is None:
        radial = np.asarray(vp.get('direction', [0, 0]), float)
    else:
        radial = np.asarray(vp['xy'], float)-np.array([(x1+x2)/2, (y1+y2)/2])
    denominator = np.linalg.norm(d)*np.linalg.norm(radial)
    if denominator < 1e-8:
        return 90.
    return float(np.degrees(np.arcsin(np.clip(abs(np.linalg.det(np.vstack((d, radial))))/denominator, 0., 1.))))


def predicted_line_angle_residual_deg(segment: dict, K: np.ndarray, R: np.ndarray) -> float:
    """Desviación con signo para optimización numérica (sin derivadas discontinuas)."""
    axis = int(segment['axis'])
    x1, y1, x2, y2 = _segment_values(segment)
    d = np.array([x2-x1, y2-y1], dtype=float)
    mid = np.array([(x1+x2)/2, (y1+y2)/2], dtype=float)
    vh = np.asarray(K, float) @ np.asarray(R, float)[:,axis]
    radial = vh[:2] - vh[2]*mid  # funciona también para fugas en el infinito
    denom = np.linalg.norm(d)*np.linalg.norm(radial)
    if denom < 1e-10:
        return 90.
    sin_angle = (d[0]*radial[1]-d[1]*radial[0])/denom
    return float(np.degrees(np.arcsin(np.clip(sin_angle, -1., 1.))))


def fit_manual_vanishing(observed: list, size_wh: tuple[int, int], axis: int) -> dict:
    """VP estimado SOLO desde rectas observadas y etiquetadas manualmente.

    Se necesitan al menos dos líneas con orientaciones diferentes; de lo
    contrario la fuga finita es inestable (o efectivamente está en infinito).
    """
    segments = [item for item in observed if int(item['axis']) == int(axis)]
    result = {'axis': AXES[axis], 'n_lines': len(segments), 'status': 'insufficient'}
    if len(segments) < 2:
        return result
    w, h = size_wh
    diagonal = max(1., math.hypot(w, h))
    center = np.array([w/2, h/2], float)
    rows=[]
    directions=[]
    for item in segments:
        x1,y1,x2,y2 = _segment_values(item)
        d=np.array([x2-x1,y2-y1],float)
        d/=np.linalg.norm(d)
        directions.append(d)
        p=np.r_[(np.array([x1,y1])-center)/diagonal,1.]
        q=np.r_[(np.array([x2,y2])-center)/diagonal,1.]
        line=np.cross(p,q)
        rows.append(line/np.linalg.norm(line[:2]))
    # Máximo ángulo agudo entre pares: 0 significa todas paralelas.
    max_sin = max(abs(np.linalg.det(np.vstack((a,b)))) for j,a in enumerate(directions) for b in directions[j+1:])
    if max_sin < math.sin(math.radians(1.5)):
        result.update(status='near_parallel',
                      note='Rectas casi paralelas: punto de fuga finito no identificable')
        return result
    _,_,vh = np.linalg.svd(np.asarray(rows),full_matrices=False)
    v = vh[-1]
    if abs(v[2]) < 1e-5:
        result.update(status='at_infinity',
                      note='El mejor ajuste sitúa el punto de fuga en el infinito')
        return result
    v=v/v[2]
    xy=center+diagonal*v[:2]
    if not np.all(np.isfinite(xy)):
        result['status']='not_finite'
        return result
    residuals=np.abs(np.asarray(rows)@v)*diagonal
    result.update(status='ok',xy=[float(xy[0]),float(xy[1])],
                  median_line_distance_px=float(np.median(residuals)),
                  max_line_distance_px=float(np.max(residuals)))
    return result


def manual_line_report(observed: list, size_wh: tuple[int,int],
                       camera_vps: list[dict]) -> dict:
    """Reporte de rectas medidas, comparación con fugas teóricas y VP independientes."""
    _, normalized = normalize_line_annotations([], observed, size_wh)
    fits = [fit_manual_vanishing(normalized, size_wh, k) for k in range(3)]
    rows = []
    for k in range(3):
        matched=[line for line in normalized if line['axis']==k]
        vp=next((v for v in camera_vps if v['axis']==k), None)
        errors = [manual_axis_error_deg(item,vp) for item in matched] if vp else []
        rows.append({'axis':AXES[k], 'observed_segments':len(matched),
                     'mean_angle_to_camera_deg':float(np.mean(errors)) if errors else None,
                     'max_angle_to_camera_deg':float(np.max(errors)) if errors else None,
                     'manual_vp':fits[k]})
    return {'observed_total':len(normalized),'axes':rows,
            'note':'Las guías por punto no se usan aquí: dependen del VP de la propia cámara.'}


def save_manual_annotations(project_dir: Path, guides_by_view: list,
                            observed_by_view: list) -> dict:
    """Persistencia atómica sin modificar poses, FOV, máscaras ni vistas."""
    project_dir=Path(project_dir)
    path=project_dir/'calibration_multiview.json'
    if not path.is_file():
        raise FileNotFoundError('Primero guarda una calibración PnP del sitio')
    meta=json.loads(path.read_text(encoding='utf-8'))
    if len(meta.get('views',[]))!=4 or len(guides_by_view)!=4 or len(observed_by_view)!=4:
        raise ValueError('Se necesitan cuatro grupos de líneas')
    prepared=[]
    for i, view in enumerate(meta['views']):
        size=tuple(view['image_size_wh'])
        prepared.append(normalize_line_annotations(guides_by_view[i], observed_by_view[i], size))
    for view,(guides,observed) in zip(meta['views'],prepared):
        view['manual_vp_guides_px']=guides
        view['observed_axis_segments_px']=observed
    meta['manual_perspective_annotations_updated_utc']=datetime.now(timezone.utc).isoformat()
    backup=project_dir/f"calibration_backup_manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    shutil.copy2(path,backup)
    temp=path.with_name('calibration_multiview.json.tmp')
    temp.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)
    return meta
