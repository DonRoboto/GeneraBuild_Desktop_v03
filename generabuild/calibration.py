"""Calibración de cámaras y serialización FR2, extraídas sin dependencia de Colab.

La calibración se realiza con la geometría ORIGINAL, no con el escenario.
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from .manual_lines import normalize_line_annotations
from .geometry import (
    box_vertices, calibrate_view, mask_for_view, architectural_canny,
    make_diagnostic_preview, sha256_file,
)


def calibrate_all(payload: dict, paths: list[Path]) -> list[dict]:
    if len(paths) != 4:
        raise ValueError("Four photographs are required in view order 1–4")
    anchors, fovs = payload["views_anchors"], payload["fov_deg_by_view"]
    if len(anchors) != 4 or len(fovs) != 4:
        raise ValueError("Four annotation sets and FOVs are required")
    dims = tuple(float(payload[k]) for k in ("ref_W", "ref_D", "ref_H"))
    results = []
    for k, path in enumerate(paths):
        with Image.open(path) as img:
            w, h = img.size
        results.append(calibrate_view(anchors[k], w, h, *dims, float(fovs[k])))
    return results


def save_calibration(payload: dict, cameras: list[dict], paths: list[Path], project_dir: Path) -> dict:
    """Usa el mismo esquema calibration_multiview.json y las mismas funciones FR2.

    Exige confirmación explícita de inspección; no certifica validez física.
    """
    project_dir = Path(project_dir)
    if not payload.get("visual_inspection_confirmed"):
        raise ValueError("Confirm that you reviewed the four projections and protected regions")
    if len(paths) != 4 or len(cameras) != 4 or not all(c.get("ok") for c in cameras):
        raise ValueError("Calibrate all four views successfully before saving")
    floors = int(payload["num_floors"])
    hf = float(payload["floor_height_m"])
    w, d, h0 = (float(payload[k]) for k in ("ref_W", "ref_D", "ref_H"))
    if floors not in (8, 9, 10) or not 2 <= hf <= 5:
        raise ValueError("Target height: 8–10 floors, 2–5 m per floor")
    if floors * hf <= h0:
        raise ValueError("Target height must exceed the original building height")
    old, new = box_vertices(w, d, h0), box_vertices(w, d, floors * hf)
    guards = payload.get("protected_polygons_px", [[] for _ in range(4)])
    allowed = payload.get("allowed_polygons_px", [None] * 4)
    if not all(len(a) == 4 for a in (guards, allowed)):
        raise ValueError("Polygon settings are required for all four views")
    for dname in ("inputs", "masks", "canny", "previews"):
        (project_dir / dname).mkdir(parents=True, exist_ok=True)
    prepared=[]
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            image=im.convert("RGB")
        camera = cameras[i]
        P=np.asarray(camera["P"], dtype=np.float64)
        mask,stats=mask_for_view(P,new,image.size,guards[i],allowed[i])
        canny=architectural_canny(image,mask,P,new,camera["rvec"],camera["tvec"],floors)
        preview=make_diagnostic_preview(image,mask,P,old,new,payload["views_anchors"][i], guards[i], allowed[i])
        prepared.append((image,mask,canny,preview,stats))
    metadata={
        "format":"GeneraBuild_v2",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "coordinate_system":"X fachada derecha, Y profundidad, Z altura; dimensiones aproximadas",
        "camera_model":"P=K[R|t], FOV horizontal por vista, distorsión cero",
        "reference_box_m":{"W":w,"D":d,"H":h0},
        "target_box_m":{"W":w,"D":d,"H":floors*hf},
        "num_floors":floors,"floor_height_m":hf,
        "reference_vertices_3d_m":old.tolist(),"vertices_3d_m":new.tolist(),
        "views":[],
        "limitations":["Semiautomatic calibration; lengths and FOVs are approximate.",
                       "Fitting RMSE is not independent point validation.",
                       "Protected areas must be reviewed visually."],
    }
    manual_guides = payload.get('guide_lines', [[] for _ in range(4)])
    observed_lines = payload.get('observed_lines', [[] for _ in range(4)])
    if len(manual_guides)!=4 or len(observed_lines)!=4:
        raise ValueError('Manual annotation lists are required for all four views')
    for i, (path,cam,prep) in enumerate(zip(paths,cameras,prepared),start=1):
        im, mask, canny, preview, stats=prep
        ext = path.suffix.lower() if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") else ".png"
        inp=project_dir/"inputs"/f"Vista_{i}{ext}"
        if path.resolve() != inp.resolve():
            shutil.copy2(path,inp)
        stem=f"Vista_{i}"
        m=project_dir/"masks"/f"mask_{stem}.png"
        c=project_dir/"canny"/f"canny_{stem}.png"
        p=project_dir/"previews"/f"preview_{stem}.png"
        Image.fromarray(mask).save(m); canny.save(c);preview.save(p)
        guides, observed = normalize_line_annotations(manual_guides[i-1], observed_lines[i-1], im.size)
        metadata["views"].append({
            "index":i,"name":path.name,"image_original_path":str(inp.resolve()),
            "image_relative":inp.relative_to(project_dir).as_posix(),
            "image_sha256":sha256_file(inp),"image_size_wh":list(im.size),
            "mask_relative":m.relative_to(project_dir).as_posix(),
            "canny_relative":c.relative_to(project_dir).as_posix(),
            "preview_relative":p.relative_to(project_dir).as_posix(),
            "fov_horizontal_deg":float(payload["fov_deg_by_view"][i-1]),
            "K":cam["K"],"rvec":cam["rvec"],"tvec":cam["tvec"],
            "P":cam["P"],"camera_center_m":cam["camera_center_m"],
            "anchors_px":payload["views_anchors"][i-1],
            "reprojection_rmse_px":cam["rmse_px"],
            "reprojection_rmse_diagonal_pct":cam["rmse_diagonal_pct"],
            "per_vertex_error_px":cam["per_vertex_error_px"],
            "planar_anchors":cam["planar"],"warnings":cam["warnings"],
            "allowed_polygon_px":allowed[i-1],"protected_polygons_px":guards[i-1],
            "manual_vp_guides_px":guides,"observed_axis_segments_px":observed,
            "mask_stats":stats,
        })
    target=project_dir/"calibration_multiview.json"
    if target.exists():
        backup=project_dir/f"calibration_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}.json"
        shutil.copy2(target,backup)
    target.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    return metadata


def editor_state_from_metadata(meta:dict) -> dict:
    return {
        "anchors":[v.get("anchors_px",{}) for v in meta["views"]],
        "fovs":[v.get("fov_horizontal_deg",55) for v in meta["views"]],
        "protected":[v.get("protected_polygons_px",[]) for v in meta["views"]],
        "allowed":[v.get("allowed_polygon_px") for v in meta["views"]],
        "guide_lines":[v.get("manual_vp_guides_px",[]) for v in meta["views"]],
        "observed_lines":[v.get("observed_axis_segments_px",[]) for v in meta["views"]],
    }
