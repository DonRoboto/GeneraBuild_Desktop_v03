"""GeneraBuild v2.1-FR2: máscara de ampliación volumétrica 3D.

El modo anterior restaba la silueta 2D COMPLETA del edificio original;
esto no es válido para cámaras aéreas: la torre nueva debe ocultar la
azotea antigua. Aquí se proyecta el prisma de la AMPLIACIÓN, formado por
la azotea original en z=H0 y la azotea nueva en z=H1.

Cambiar estos módulos no altera las cámaras, las fotos de entrada ni las
máscaras externas de protección. CPU, OpenCV, numpy solamente.
"""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np


def _vertices_valid(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    old = np.asarray(meta['reference_vertices_3d_m'], dtype=np.float64)
    new = np.asarray(meta['vertices_3d_m'], dtype=np.float64)
    if old.shape != (8, 3) or new.shape != (8, 3):
        raise ValueError('Se esperan 8 vertices de referencia y 8 de escenario (8x3)')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('La geometría contiene coordenadas no finitas')
    if not np.allclose(new[:4], old[:4], atol=1e-3, rtol=0):
        raise ValueError('La ampliacion no puede modificar la huella de la base')
    if not np.allclose(new[4:,:2], old[4:,:2], atol=1e-3, rtol=0):
        raise ValueError('La ampliacion no puede modificar el ancho/profundidad')
    if not np.allclose(old[4:,2], old[4,2], atol=1e-3, rtol=0):
        raise ValueError('Azotea original no horizontal')
    if not np.allclose(new[4:,2], new[4,2], atol=1e-3, rtol=0):
        raise ValueError('Azotea nueva no horizontal')
    if new[4,2] < old[4,2]-1e-3:
        raise ValueError('La altura solicitada es inferior a la original')
    return old, new


def project_vertices(P: Any, vertices: Any) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if P.shape != (3, 4) or not np.isfinite(P).all():
        raise ValueError('Matriz de proyeccion 3x4 invalida')
    homo = (P @ np.c_[V, np.ones(len(V))].T).T
    if not np.isfinite(homo).all() or np.any(homo[:, 2] <= 1e-7):
        raise ValueError('Vertices detras de la camara; revisar calibracion')
    uv = homo[:, :2] / homo[:, 2:3]
    if np.max(np.abs(uv)) > 1_000_000:
        raise ValueError('Proyeccion fuera de limites numericos')
    return uv


def polygon_silhouette(P: Any, vertices: Any, shape_hw: tuple[int,int]) -> np.ndarray:
    h, w = map(int, shape_hw)
    if h < 1 or w < 1:
        raise ValueError('Resolucion de imagen invalida')
    mask = np.zeros((h,w), dtype=np.uint8)
    uv = project_vertices(P, vertices)
    hull = cv2.convexHull(np.rint(uv).astype(np.int32))
    if len(hull) >= 3:
        cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_8)
    return mask


def extension_volume(meta: dict) -> np.ndarray:
    """Ocho esquinas de la losa original H0 y del nuevo techo H1.

    No se incluyen los niveles 0..H0 porque no forman parte del volumen nuevo.
    """
    old, new = _vertices_valid(meta)
    return np.vstack([old[4:8],new[4:8]])


def extension_work_mask(saved_mask: np.ndarray, metadata: dict, view_index: int,
                        preserve_existing: bool = True) -> np.ndarray:
    """Máscara editable correcta para ampliación, vista aérea o a nivel calle.

    La máscara editable externa sigue siendo obligatoria, para excluir
    edificios colindantes y oclusiones anotadas. En modo fachada completa
    no hay cambios respecto al comportamiento anterior.
    """
    allowed = np.asarray(saved_mask, dtype=np.uint8)
    if allowed.ndim != 2:
        raise ValueError('Se requiere mascara 2D uint8')
    allowed = np.where(allowed>127, 255, 0).astype(np.uint8)
    if not preserve_existing:
        return allowed
    if not 1 <= int(view_index) <= len(metadata['views']):
        raise ValueError('Indice de vista fuera de rango')
    V = extension_volume(metadata)
    if np.isclose(V[4,2],V[0,2],atol=1e-3):
        return np.zeros_like(allowed)  # Altura sin modificar.
    P = metadata['views'][view_index-1]['P']
    silhouette = polygon_silhouette(P,V,allowed.shape)
    return cv2.bitwise_and(allowed,silhouette)


def extension_mask_diagnostics(saved_mask: np.ndarray, metadata: dict,
                               view_index: int) -> dict:
    """Mide cuántos píxeles de la ampliación eliminaba la regla antigua."""
    allowed = np.where(np.asarray(saved_mask)>127,255,0).astype(np.uint8)
    old,_ = _vertices_valid(metadata)
    P = metadata['views'][view_index-1]['P']
    old_hull = polygon_silhouette(P,old,allowed.shape)
    extension = extension_work_mask(allowed,metadata,view_index,True)
    overlap = (old_hull>0)&(extension>0)
    legacy = (allowed>0)&(old_hull==0)
    return {
        'view':int(view_index),
        'editable_full_pixels':int(np.count_nonzero(allowed)),
        'extension_pixels':int(np.count_nonzero(extension)),
        'pixels_extension_hidden_by_legacy_mask':int(np.count_nonzero(overlap)),
        'legacy_mask_pixels':int(np.count_nonzero(legacy)),
        'fov_horizontal_deg':metadata['views'][view_index-1].get('fov_horizontal_deg'),
    }
