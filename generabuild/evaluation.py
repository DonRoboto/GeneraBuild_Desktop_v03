"""Evaluación reproducible. La métrica visual IoU requiere segmentación independiente."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from gb21_extension_mask import extension_work_mask
from .storage import load_metadata


def evaluate_run(run_dir:Path, final_images:dict[int,Path],config:dict)->dict:
    meta=load_metadata(run_dir)
    rows=[]
    for i,path in sorted(final_images.items()):
        view=meta["views"][i-1]
        original=np.asarray(Image.open(run_dir/view["image_relative"]).convert("RGB"))
        final=np.asarray(Image.open(path).convert("RGB"))
        if original.shape!=final.shape:raise ValueError("Se modificó la resolución")
        mask=np.asarray(Image.open(run_dir/view["mask_relative"]).convert("L"))
        work=extension_work_mask(mask,meta,i,config["mode"]=="ampliacion")>127
        changed=np.any(original!=final,axis=2)
        outside=int(np.count_nonzero(changed & ~work))
        permitted=int(np.count_nonzero(work))
        rows.append({
            "view":int(i),"floors":config["floors"],
            "pixels_outside_mask_changed":outside,
            "context_error_pct":round(100*outside/max(1,np.count_nonzero(~work)),9),
            "pixels_inside_mask_changed":int(np.count_nonzero(changed & work)),
            "work_mask_pixels":permitted,
            "reprojection_rmse_px":view.get("reprojection_rmse_px"),
            "reprojection_rmse_diagonal_pct":view.get("reprojection_rmse_diagonal_pct"),
            "image_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "result_relative":str(Path(path).relative_to(run_dir)),
        })
    report={
        "method":"GeneraBuild FR2 — Streamlit interface",
        "calibration_sha256":meta.get("source_calibration_sha256"),
        "config":config,"views":rows,
        "context_preservation_pass":all(r["pixels_outside_mask_changed"]==0 for r in rows),
        "interpretation":{
            "context":"Exactitud de píxeles FUERA de la máscara autorizada; no valida que la máscara esté bien delimitada.",
            "rmse":"Reproyección sobre puntos de calibración. No es error 3D independiente.",
            "silhouette_iou":"No implementado: requiere segmentación independiente de la imagen final; no comparar máscara consigo misma.",
        }
    }
    out=Path(run_dir)/"evaluation.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
