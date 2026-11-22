"""GeneraBuild v2 - Núcleo geométrico y rasterización reproducible.
Sistema 3D: X=ancho de fachada, Y=profundidad, Z=altura.
Vértices 0..3: planta; 4..7: azotea, mismos (X,Y).
La geometría de REFERENCIA se usa para calibrar; la hipotética solo cambia H.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares


def box_vertices(W: float, D: float, H: float) -> np.ndarray:
    W, D, H = float(W), float(D), float(H)
    if not np.all(np.isfinite([W, D, H])) or min(W, D, H) <= 0:
        raise ValueError("W, D and H must be positive finite values")
    return np.array([
        [0, 0, 0], [W, 0, 0], [W, D, 0], [0, D, 0],
        [0, 0, H], [W, 0, H], [W, D, H], [0, D, H],
    ], dtype=np.float64)


def intrinsic_matrix(width: int, height: int, fov_horizontal_deg: float) -> np.ndarray:
    fov = float(fov_horizontal_deg)
    if not (15.0 <= fov <= 125.0):
        raise ValueError("Horizontal FOV must be between 15 and 125 degrees")
    f = (float(width) / 2) / np.tan(np.deg2rad(fov) / 2)
    return np.array([[f, 0.0, width / 2],
                     [0.0, f, height / 2],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def project_camera(P: np.ndarray, pts3: np.ndarray) -> np.ndarray:
    xyz = np.asarray(pts3, dtype=np.float64).reshape(-1, 3)
    homo = np.column_stack([xyz, np.ones(len(xyz))])
    p = (np.asarray(P, dtype=np.float64) @ homo.T).T
    if np.any(~np.isfinite(p)) or np.any(p[:, 2] <= 1e-6):
        raise ValueError("Attempted to project a point behind the camera")
    uv = p[:, :2] / p[:, 2, None]
    if np.max(np.abs(uv)) > 1e7:
        raise ValueError("Numerically unstable projection")
    return uv


def _depths(rvec: np.ndarray, tvec: np.ndarray, pts3: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return (R @ np.asarray(pts3).T + np.asarray(tvec).reshape(3, 1)).T[:, 2]


def calibrate_view(anchors_px: dict, width: int, height: int,
                   W0: float, D0: float, H0: float, fov_deg: float) -> dict:
    """PnP de una vista: intrínsecos aproximados, varias hipótesis y refinamiento.

    Precaución: RMSE calculado sobre anchors de entrenamiento NO es validación externa.
    """
    if len(anchors_px) < 4:
        return {"ok": False, "error": f"Se requieren al menos 4 puntos (hay {len(anchors_px)})"}
    try:
        ref3 = box_vertices(W0, D0, H0)
        ids = sorted([int(k) for k in anchors_px])
        if len(set(ids)) != len(ids) or any(k < 0 or k > 7 for k in ids):
            raise ValueError("Invalid vertex indices")
        obs = np.array([[float(anchors_px[str(k)]["x"]), float(anchors_px[str(k)]["y"])]
                        for k in ids], dtype=np.float64)
        if np.any(~np.isfinite(obs)) or np.any(obs[:, 0] < 0) or np.any(obs[:, 0] >= width) \
                or np.any(obs[:, 1] < 0) or np.any(obs[:, 1] >= height):
            raise ValueError("Marked points must lie inside the photograph")
        obj = np.ascontiguousarray(ref3[ids], dtype=np.float64)
        obs = np.ascontiguousarray(obs, dtype=np.float64)
        rank3 = np.linalg.matrix_rank(obj - obj.mean(axis=0), tol=1e-7)
        rank2 = np.linalg.matrix_rank(obs - obs.mean(axis=0), tol=1e-7)
        if rank3 < 2 or rank2 < 2:
            raise ValueError("Collinear points: mark corners spanning an area")

        planar = bool(rank3 == 2)
        K = intrinsic_matrix(width, height, fov_deg)
        dist = np.zeros((5, 1), dtype=np.float64)
        options = []
        flags = [cv2.SOLVEPNP_SQPNP]
        if planar:
            flags.append(cv2.SOLVEPNP_IPPE)
        if len(ids) >= 5:
            flags.append(cv2.SOLVEPNP_EPNP)
        for flag in flags:
            try:
                result = cv2.solvePnPGeneric(obj, obs, K, dist, flags=flag)
                if result[0]:
                    options.extend(zip(result[1], result[2]))
            except cv2.error:
                pass
        if not options:
            raise ValueError("OpenCV did not find a valid PnP solution")

        candidates = []
        for rv, tv in options:
            rv, tv = np.asarray(rv, dtype=np.float64).reshape(3, 1), np.asarray(tv, dtype=np.float64).reshape(3, 1)
            if np.min(_depths(rv, tv, ref3)) <= 1e-6:
                continue
            # Con 4 puntos también se puede refinar, pero la solución puede ser ambigua.
            def residual(x):
                pr, _ = cv2.projectPoints(obj, x[:3].reshape(3, 1), x[3:].reshape(3, 1), K, dist)
                return (pr.reshape(-1, 2) - obs).ravel()
            init = np.r_[rv.ravel(), tv.ravel()]
            opt = least_squares(residual, init, loss="soft_l1", f_scale=3.0, max_nfev=120)
            r1, t1 = opt.x[:3].reshape(3, 1), opt.x[3:].reshape(3, 1)
            if np.min(_depths(r1, t1, ref3)) <= 1e-6:
                continue
            errors = np.linalg.norm(residual(opt.x).reshape(-1, 2), axis=1)
            rmse = float(np.sqrt(np.mean(errors**2)))
            candidates.append((rmse, r1, t1, errors))
        if not candidates:
            raise ValueError("All estimated poses place points behind the camera")
        candidates.sort(key=lambda item: item[0])
        rmse, rvec, tvec, errors = candidates[0]
        R, _ = cv2.Rodrigues(rvec)
        P = K @ np.column_stack([R, tvec])
        C = (-R.T @ tvec).ravel()
        diag = float(np.hypot(width, height))
        nrmse = 100.0 * rmse / diag
        warnings = []
        if len(ids) == 4:
            warnings.append("Only four anchors: fitting RMSE does not validate the camera")
        if planar:
            warnings.append("Coplanar points: depth and pose weakly constrained; check another view")
        if nrmse > 0.5:
            warnings.append("RMSE >0.5% of diagonal: review vertices and FOV")
        return {
            "ok": True,
            "P": P.tolist(), "K": K.tolist(),
            "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
            "camera_center_m": C.tolist(),
            "rmse_px": rmse, "rmse_diagonal_pct": nrmse,
            "n_anchors": len(ids), "planar": planar,
            "per_vertex_error_px": {str(k): round(float(e), 3) for k, e in zip(ids, errors)},
            "warnings": warnings,
        }
    except (ValueError, KeyError, IndexError, cv2.error, np.linalg.LinAlgError) as e:
        return {"ok": False, "error": str(e)}


def _polygon_mask(size_wh: tuple[int, int], polygon: list | None) -> np.ndarray:
    width, height = size_wh
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon is None or len(polygon) < 3:
        return mask
    coords = np.array([[float(p["x"]), float(p["y"])] for p in polygon], dtype=np.float64)
    if not np.all(np.isfinite(coords)):
        raise ValueError("Polygon contains non-finite coordinates")
    coords = np.rint(coords).astype(np.int32)
    cv2.fillPoly(mask, [coords], 255)
    return mask


def mask_for_view(P: np.ndarray, vertices_target: np.ndarray,
                  size_wh: tuple[int, int], protected_polygons: list | None = None,
                  allowed_polygon: list | None = None) -> tuple[np.ndarray, dict]:
    """Máscara = silueta convexa proyectada, ∩ área autorizada, - áreas protegidas."""
    width, height = size_wh
    points_uv = project_camera(P, vertices_target)
    hull = cv2.convexHull(np.rint(points_uv).astype(np.int32)).reshape(-1, 2)
    raw = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(raw, hull, 255)
    mask = raw.copy()
    if allowed_polygon is not None and len(allowed_polygon) >= 3:
        permitted = _polygon_mask(size_wh, allowed_polygon)
        mask = cv2.bitwise_and(mask, permitted)
    for poly in protected_polygons or []:
        protect = _polygon_mask(size_wh, poly)
        mask[protect > 0] = 0
    raw_pixels = int(np.count_nonzero(raw))
    edit_pixels = int(np.count_nonzero(mask))
    if edit_pixels == 0:
        raise ValueError("Mask is empty: inspect camera pose, permitted area and protected polygons")
    return mask, {
        "silhouette_px": raw_pixels,
        "editable_px": edit_pixels,
        "fraction_of_image_editable": edit_pixels / (width * height),
        "fraction_silhouette_excluded": 1 - edit_pixels / max(1, raw_pixels),
        "silhouette_bbox_xywh": cv2.boundingRect(hull.astype(np.int32)),
    }


_FACE_SPECS = [
    ((0, 1, 5, 4), True, "front"),
    ((1, 2, 6, 5), True, "right"),
    ((2, 3, 7, 6), True, "back"),
    ((3, 0, 4, 7), True, "left"),
    ((4, 5, 6, 7), False, "roof"),
    ((0, 3, 2, 1), False, "base"),
]


def architectural_canny(image: Image.Image, mask: np.ndarray, P: np.ndarray,
                         vertices_target: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                         floors: int, bays_front: int = 4, bays_side: int = 6) -> Image.Image:
    """Solo caras visibles según orientación externa. Los trazos se recortan a la máscara."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if mask.shape != (h, w):
        raise ValueError("Image and mask dimensions differ")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    background_edges = cv2.Canny(gray, 100, 200)
    drawing = np.zeros((h, w), dtype=np.uint8)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    C = -R.T @ np.asarray(tvec, dtype=np.float64).reshape(3)
    visible = []
    for ids, wall, face_name in _FACE_SPECS:
        face_xyz = vertices_target[list(ids)]
        normal = np.cross(face_xyz[1] - face_xyz[0], face_xyz[2] - face_xyz[1])
        center = face_xyz.mean(axis=0)
        if float(normal @ (C - center)) > 1e-9:
            visible.append((float(np.linalg.norm(center - C)), ids, wall, face_name))
    # Pintar desde atrás hacia delante. En un prisma convexo no debería haber solapamiento espurio.
    visible.sort(key=lambda item: -item[0])
    for _, ids, wall, face_name in visible:
        face_xyz = vertices_target[list(ids)]
        face_uv = np.rint(project_camera(P, face_xyz)).astype(np.int32)
        cv2.fillPoly(drawing, [face_uv], 0)
        cv2.polylines(drawing, [face_uv], True, 255, thickness=2, lineType=cv2.LINE_AA)
        if wall:
            bottom_left, bottom_right, top_right, top_left = face_xyz
            for i in range(1, floors):
                t = i / floors
                pts = project_camera(P, np.array([
                    bottom_left * (1-t) + top_left * t,
                    bottom_right * (1-t) + top_right * t
                ]))
                a, b = np.rint(pts).astype(np.int32)
                cv2.line(drawing, tuple(a), tuple(b), 255, 1, cv2.LINE_AA)
            bays = bays_front if face_name in ("front", "back") else bays_side
            for i in range(1, bays):
                s = i / bays
                pts = project_camera(P, np.array([
                    bottom_left * (1-s) + bottom_right * s,
                    top_left * (1-s) + top_right * s,
                ]))
                a, b = np.rint(pts).astype(np.int32)
                cv2.line(drawing, tuple(a), tuple(b), 255, 1, cv2.LINE_AA)
    final = np.where(mask > 127, drawing, background_edges).astype(np.uint8)
    return Image.fromarray(np.repeat(final[:, :, None], 3, axis=2), mode="RGB")


def make_diagnostic_preview(image: Image.Image, mask: np.ndarray, P: np.ndarray,
                            reference_vertices: np.ndarray, target_vertices: np.ndarray,
                            annotations: dict, protected_polys: list,
                            allowed_poly: list | None) -> Image.Image:
    src = np.array(image.convert("RGB"))
    overlay = src.copy()
    # Verde = zona editable; las regiones rosa protegidas / fuera de área autorizada permanecen intactas.
    tint = np.array([40, 165, 90], dtype=np.uint8)
    overlay[mask > 0] = np.round(0.65 * overlay[mask > 0] + 0.35 * tint).astype(np.uint8)
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    for verts, color, thickness in [
        (reference_vertices, (65, 220, 255), 1),
        (target_vertices, (255, 175, 35), 3),
    ]:
        uv = project_camera(P, verts)
        for a, b in edges:
            cv2.line(overlay, tuple(np.rint(uv[a]).astype(int)),
                     tuple(np.rint(uv[b]).astype(int)), color, thickness, cv2.LINE_AA)
    uv_ref = project_camera(P, reference_vertices)
    for id_str, point in annotations.items():
        i = int(id_str)
        x, y = int(round(point["x"])), int(round(point["y"]))
        px, py = np.rint(uv_ref[i]).astype(int)
        cv2.line(overlay, (x, y), (px, py), (255, 60, 60), 2)
        cv2.circle(overlay, (x, y), 4, (255, 30, 30), -1)
        cv2.putText(overlay, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (255, 255, 255), 2, cv2.LINE_AA)
    for poly, color in [(p, (240, 70, 180)) for p in protected_polys] + (
        [(allowed_poly, (40, 220, 60))] if allowed_poly else []
    ):
        arr = np.rint(np.array([[p["x"], p["y"]] for p in poly])).astype(np.int32)
        if len(arr) > 2:
            cv2.polylines(overlay, [arr], True, color, 2)
    return Image.fromarray(overlay)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def project_image_path(project_dir: Path, view: dict) -> Path:
    candidate = project_dir / view["image_relative"]
    if not candidate.exists():
        original = Path(view.get("image_original_path", ""))
        if original.is_file() and sha256_file(original) == view.get("image_sha256"):
            return original
        raise FileNotFoundError(f"No image found for {view.get('name')}: {candidate}")
    if view.get("image_sha256") and sha256_file(candidate) != view["image_sha256"]:
        raise RuntimeError(f"Image changed after calibration: {candidate}")
    return candidate


def validate_project(project_dir: Path, meta: dict) -> list[dict]:
    """Comprueba que las máscaras guardadas coincidan exactamente con su reproyección."""
    rows = []
    target = np.asarray(meta["vertices_3d_m"], dtype=np.float64)
    for i, view in enumerate(meta["views"]):
        path = project_image_path(project_dir, view)
        with Image.open(path) as im:
            size = im.size
        with Image.open(project_dir / view["mask_relative"]) as m:
            mask = np.array(m.convert("L"))
        expected, stats = mask_for_view(
            np.asarray(view["P"]), target, size,
            view["protected_polygons_px"], view.get("allowed_polygon_px")
        )
        exact = bool(np.array_equal(mask, expected))
        rows.append({"view": i+1, "image": view["name"],
                     "rmse_px": view["reprojection_rmse_px"],
                     "rmse_diagonal_pct": view["reprojection_rmse_diagonal_pct"],
                     "anchor_count": len(view["anchors_px"]),
                     "mask_reproducible_exact": exact,
                     "editable_pixels": stats["editable_px"],
                     "fraction_editable": stats["fraction_of_image_editable"],
                     "warning": "; ".join(view["warnings"])})
    return rows


def cross_view_triangulation_diagnostics(meta: dict) -> list[dict]:
    """Comprueba correspondencia 3D de puntos observados en pares de imágenes.

    La referencia 3D es el prisma original asumido; el indicador NO equivale a
    medición topográfica ni prueba independiente si solo hay 4 correspondencias.
    """
    views = meta["views"]
    reference = np.asarray(meta["reference_vertices_3d_m"], dtype=np.float64)
    diagnostics = []
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            a, b = views[i], views[j]
            common = sorted(set(a["anchors_px"]) & set(b["anchors_px"]), key=int)
            if len(common) < 2:
                continue
            x1 = np.array([[a["anchors_px"][k]["x"], a["anchors_px"][k]["y"]] for k in common], dtype=np.float64)
            x2 = np.array([[b["anchors_px"][k]["x"], b["anchors_px"][k]["y"]] for k in common], dtype=np.float64)
            Xh = cv2.triangulatePoints(np.asarray(a["P"], dtype=np.float64),
                                      np.asarray(b["P"], dtype=np.float64), x1.T, x2.T)
            w = Xh[3]
            valid = np.abs(w) > 1e-10
            if not np.any(valid):
                continue
            X = (Xh[:3,valid]/w[None,valid]).T
            target = reference[[int(common[k]) for k in np.flatnonzero(valid)]]
            differences = np.linalg.norm(X-target, axis=1)
            diagnostics.append({
                "view_pair": f"{i+1}–{j+1}",
                "shared_anchors": int(valid.sum()),
                "median_3d_deviation_m": float(np.median(differences)),
                "max_3d_deviation_m": float(np.max(differences)),
            })
    return diagnostics
