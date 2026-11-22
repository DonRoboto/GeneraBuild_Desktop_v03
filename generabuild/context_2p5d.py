"""Volúmenes urbanos 2.5D opcionales en coordenadas del edificio de referencia.

La geometría adicional se almacena fuera de calibration_multiview.json.
Las esquinas marcadas en una vista se interpretan en el plano Z=ground_z.
No reconstruye automáticamente alturas, cubiertas ni elementos oclusores no modelados.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import uuid

import cv2
import numpy as np

SCHEMA_VERSION = 1
CONTEXT_FILENAME = 'urban_context_2p5d.json'


def polygon_area(xy: np.ndarray) -> float:
    a = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return float(.5 * np.abs(np.dot(a[:, 0], np.roll(a[:, 1], -1)) -
                               np.dot(a[:, 1], np.roll(a[:, 0], -1))))


def _segments_intersect(a, b, c, d) -> bool:
    def cross(u, v): return float(u[0]*v[1]-u[1]*v[0])
    ab = b-a
    cd = d-c
    denom = cross(ab, cd)
    if abs(denom) < 1e-10:
        return False  # Los segmentos colineales quedan fuera de este chequeo; se rechazan longitudes pequeñas.
    t = cross(c-a, cd) / denom
    u = cross(c-a, ab) / denom
    return 1e-9 < t < 1-1e-9 and 1e-9 < u < 1-1e-9


def validate_footprint(xy, *, min_area: float = 0.50) -> np.ndarray:
    p = np.asarray(xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not (3 <= len(p) <= 16):
        raise ValueError('Footprint requires 3–16 XY vertices')
    if not np.isfinite(p).all() or (np.abs(p) > 1000).any():
        raise ValueError('Non-finite or out-of-range footprint coordinates')
    if min(np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1)) < .15:
        raise ValueError('Two footprint vertices are too close')
    if polygon_area(p) < min_area:
        raise ValueError('Footprint is too small or degenerate')
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            if abs(i-j) in (0, 1, len(p)-1): continue
            if _segments_intersect(p[i], p[(i+1)%len(p)], p[j], p[(j+1)%len(p)]):
                raise ValueError('Footprint intersects itself')
    # La rasterización 3D necesita una triangulación controlada. MVP: convexos.
    if not cv2.isContourConvex(p.astype(np.float32)):
        raise ValueError('Split concave footprints into two convex volumes for now')
    return p


def image_to_ground(P, uv, *, ground_z: float = 0.0) -> np.ndarray:
    """Retroproyecta puntos 2D al plano Z=ground_z usando H = [P0,P1,P2*z+P3].

    La profundidad no se deduce de la imagen: es una hipótesis explícita de plano.
    """
    projection = np.asarray(P, dtype=np.float64)
    coords = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        raise ValueError('Invalid 3×4 camera matrix')
    if not np.isfinite(coords).all(): raise ValueError('Non-finite 2D points')
    H = np.column_stack([projection[:, 0], projection[:, 1],
                         projection[:, 2]*float(ground_z)+projection[:, 3]])
    if np.linalg.cond(H) > 1e11:
        raise ValueError('Ground plane near the horizon or degenerate calibration')
    result = (np.linalg.inv(H) @ np.column_stack([coords, np.ones(len(coords))]).T).T
    if np.any(np.abs(result[:, 2]) < 1e-9):
        raise ValueError('Marked point is near the horizon; no stable ground intersection')
    xy = result[:, :2] / result[:, 2:3]
    if not np.isfinite(xy).all() or (np.abs(xy) > 1000).any():
        raise ValueError('Marked point is far from the site; mark a ground corner, not a roof edge')
    return xy


def world_to_image(P, xyz):
    p = np.asarray(P, float) @ np.column_stack([
        np.asarray(xyz, float).reshape(-1, 3),
        np.ones(len(np.asarray(xyz, float).reshape(-1, 3)))
    ]).T
    if not np.all(np.isfinite(p)) or np.any(p[2] <= 1e-7):
        raise ValueError('Volume is behind the camera')
    return (p[:2] / p[2:3]).T


@dataclass
class NeighborBuilding:
    id: str
    name: str
    footprint_xy_m: list[list[float]]
    height_m: float
    ground_z_m: float = 0.0
    color_rgb: tuple[int,int,int] = (39, 176, 143)
    enabled: bool = True
    occluder: bool = True
    annotations: dict = field(default_factory=dict)

    def validate(self) -> 'NeighborBuilding':
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', self.id):
            raise ValueError('Invalid neighbor building ID')
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name)>100:
            raise ValueError('Invalid neighbor name')
        validate_footprint(self.footprint_xy_m)
        if not 1.0 <= float(self.height_m) <= 150.0:
            raise ValueError('Neighbor height must be between 1 and 150 m')
        if abs(float(self.ground_z_m)) > 20:
            raise ValueError('Base elevation must be within the local coordinate system')
        if len(self.color_rgb)!=3 or any(not 0<=int(v)<=255 for v in self.color_rgb):
            raise ValueError('Invalid RGB color')
        return self

    @property
    def footprint(self) -> np.ndarray:
        return validate_footprint(self.footprint_xy_m)

    def vertices(self) -> np.ndarray:
        p = self.footprint
        low = np.column_stack([p, np.full(len(p), self.ground_z_m)])
        top = np.column_stack([p, np.full(len(p), self.ground_z_m + self.height_m)])
        return np.vstack([low, top])

    def triangles(self) -> list[np.ndarray]:
        v = self.vertices(); n = len(v)//2
        triangles=[]
        for i in range(1, n-1):
            triangles.append(v[[n, n+i, n+i+1]])
            triangles.append(v[[0, i+1, i]])
        for i in range(n):
            j=(i+1)%n
            triangles.extend([v[[i,j,n+j]], v[[i,n+j,n+i]]])
        return triangles

    def to_dict(self) -> dict:
        self.validate()
        return { 'id':self.id, 'name':self.name,
                 'footprint_xy_m':np.asarray(self.footprint_xy_m,float).tolist(),
                 'height_m':float(self.height_m), 'ground_z_m':float(self.ground_z_m),
                 'color_rgb':list(map(int,self.color_rgb)),
                 'enabled':bool(self.enabled),'occluder':bool(self.occluder),
                 'annotations':self.annotations }

    @classmethod
    def from_dict(cls, value: dict) -> 'NeighborBuilding':
        return cls(id=str(value['id']), name=str(value['name']),
                   footprint_xy_m=value['footprint_xy_m'], height_m=float(value['height_m']),
                   ground_z_m=float(value.get('ground_z_m',0)),
                   color_rgb=tuple(value.get('color_rgb',[39,176,143])),
                   enabled=bool(value.get('enabled',True)),
                   occluder=bool(value.get('occluder',True)),
                   annotations=value.get('annotations',{})).validate()


def new_neighbor(name: str, footprint_xy, height_m: float, *,
                 source_view: int|None=None, observed_px=None) -> NeighborBuilding:
    item = NeighborBuilding(id='neighbor_'+uuid.uuid4().hex[:10],
                            name=name, footprint_xy_m=validate_footprint(footprint_xy).tolist(),
                            height_m=float(height_m), annotations={
                                'source_view':source_view,
                                'marked_ground_corners_px': observed_px,
                                'assumption':'Corners lie on calibrated ground plane Z=0; height estimated manually.',
                            })
    return item.validate()


def load_context(site_folder:Path) -> list[NeighborBuilding]:
    path=Path(site_folder)/CONTEXT_FILENAME
    if not path.exists():return []
    d=json.loads(path.read_text(encoding='utf-8'))
    if d.get('schema_version')!=SCHEMA_VERSION:
        raise ValueError('Unsupported 2.5D context version')
    buildings=[NeighborBuilding.from_dict(x) for x in d.get('neighbors',[])]
    if len({x.id for x in buildings})!=len(buildings):raise ValueError('Duplicate building IDs')
    return buildings


def save_context(site_folder:Path, neighbors:list[NeighborBuilding], *,
                 calibration_sha256:str|None=None):
    path=Path(site_folder)/CONTEXT_FILENAME
    path.parent.mkdir(parents=True,exist_ok=True)
    if len({x.id for x in neighbors})!=len(neighbors):raise ValueError('Duplicate building IDs')
    data={'schema_version':SCHEMA_VERSION,
          'coordinate_system':'local_target_building_XY_ground_plane_Z_up_m',
          'updated_utc':datetime.now(timezone.utc).isoformat(),
          'calibration_sha256':calibration_sha256,
          'neighbors':[x.to_dict() for x in neighbors],
          'notes':('Context is optional and approximate. Original images and original calibration '
                   'are never changed; height and footprint are user-supplied.')}
    tmp=path.with_suffix('.json.tmp')
    try:
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
    return path
