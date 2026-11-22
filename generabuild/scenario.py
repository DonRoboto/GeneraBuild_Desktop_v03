"""Escenarios aislados: no se modifica el JSON de calibración original."""
from __future__ import annotations
import json
import sys
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from importlib.metadata import version, PackageNotFoundError
from .geometry import (box_vertices, mask_for_view, make_diagnostic_preview,
                       architectural_canny, sha256_file, validate_project)
from .storage import load_metadata
from gb21_extension_mask import extension_work_mask


def config_from_metadata(meta:dict, *, floors:int=9, mode:str="ampliacion", seed:int=1234,
                         floor_height_m:float|None=None, views:list[int]|None=None,
                         steps:int=40,guidance:float=5.5,control_scale:float=0.65,
                         max_dim:int=768,front_windows:int=3,side_windows:int=4,
                         prompt:str|None=None,negative_prompt:str|None=None,
                         photo_finish:bool=True,roof_finish:bool=True,
                         roof_equipment:int=0, wall_tint:float=0.30)->dict:
    return {
        "floors":int(floors),"mode":str(mode),"seed":int(seed),
        "floor_height_m":float(floor_height_m if floor_height_m is not None else meta["floor_height_m"]),
        "views":list(views if views is not None else [1,2,3,4]),
        "steps":int(steps),"guidance":float(guidance),"control_scale":float(control_scale),
        "max_dim":int(max_dim),"front_windows":int(front_windows),"side_windows":int(side_windows),
        "prompt":prompt or None,"negative_prompt":negative_prompt or None,
        "postprocessing":{"photo_finish":bool(photo_finish),"roof_finish":bool(roof_finish),
                          "roof_equipment":int(roof_equipment),"wall_tint":float(wall_tint)},
    }


def validate_config(config:dict, meta:dict)->dict:
    if config["mode"] not in ("ampliacion","completo"):
        raise ValueError("mode must be ampliacion or completo")
    if not 8<=int(config["floors"])<=10 or not 2<=float(config["floor_height_m"])<=5:
        raise ValueError("Scenario must have 8–10 floors and floor height 2–5 m")
    if int(config["floors"])*float(config["floor_height_m"])<=float(meta["reference_box_m"]["H"]):
        raise ValueError("Target height must exceed the original building")
    if not config["views"] or any(i not in (1,2,3,4) for i in config["views"]):
        raise ValueError("Select at least one of four views")
    if not 1<=config["steps"]<=100 or not 256<=config["max_dim"]<=1536:
        raise ValueError("Diffusion parameters are out of range")
    if not 1<=config["front_windows"]<=8 or not 0<=config["side_windows"]<=12:
        raise ValueError("Invalid window count")
    return config


def prepare_run(source:Path,config:dict,run_id:str|None=None)->Path:
    """Copia imágenes y recalcula máscaras para la altura solicitada.

    El directorio source/ nunca se edita, incluyendo el archivo de calibración.
    """
    source=Path(source)
    canonical=source/"calibration_multiview.json"
    original_bytes=canonical.read_bytes()
    meta=load_metadata(source)
    config=validate_config(config,meta)
    if run_id is None:
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + config["mode"] + f"_{config['floors']}p"
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("Invalid run identifier")
    target=source/"runs"/run_id
    if target.exists():raise FileExistsError("Ya existe el identificador de ejecución")
    target.mkdir(parents=True)
    try:
        meta=json.loads(original_bytes)
        old=meta["reference_box_m"]
        h=float(config["floors"])*float(config["floor_height_m"])
        target_vertices=box_vertices(float(old["W"]),float(old["D"]),h)
        meta["num_floors"]=int(config["floors"])
        meta["floor_height_m"]=float(config["floor_height_m"])
        meta["target_box_m"]={"W":float(old["W"]),"D":float(old["D"]),"H":h}
        meta["vertices_3d_m"]=target_vertices.tolist()
        meta["scenario_mode"]=config["mode"]
        meta["source_calibration_sha256"]=sha256_file(canonical)
        meta["scenario_run_utc"]=datetime.now(timezone.utc).isoformat()
        reference=np.asarray(meta["reference_vertices_3d_m"],dtype=float)
        for i,view in enumerate(meta["views"],1):
            # Sanitizar rutas: los proyectos importados pueden contener rutas obsoletas.
            for f in ("image_relative","mask_relative","canny_relative","preview_relative"):
                if f not in view:continue
                rel=Path(view[f]);
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe calibration path in {f}")
            original=source/view["image_relative"]
            if not original.is_file(): raise FileNotFoundError(original)
            if view.get("image_sha256") and sha256_file(original)!=view["image_sha256"]:
                raise RuntimeError("Photograph checksum does not match calibrated input")
            dest=target/view["image_relative"]
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,dest)
            view["image_original_path"]=str(dest.resolve())
            with Image.open(dest) as im:img=im.convert("RGB")
            P=np.asarray(view["P"],float)
            mask,stats=mask_for_view(P,target_vertices,img.size,
                 view.get("protected_polygons_px",[]),view.get("allowed_polygon_px"))
            mpath=target/view["mask_relative"];mpath.parent.mkdir(exist_ok=True,parents=True)
            Image.fromarray(mask).save(mpath)
            canny=architectural_canny(img,mask,P,target_vertices,
                                      view["rvec"],view["tvec"],int(config["floors"]))
            cpath=target/view["canny_relative"]
            cpath.parent.mkdir(exist_ok=True,parents=True)
            canny.save(cpath)
            overlay=make_diagnostic_preview(img,mask,P,reference,target_vertices,
                    view.get("anchors_px",{}),view.get("protected_polygons_px",[]),view.get("allowed_polygon_px"))
            ppath=target/view["preview_relative"]
            ppath.parent.mkdir(exist_ok=True,parents=True)
            overlay.save(ppath)
            view["mask_stats"]=stats
        (target/"calibration_multiview.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        (target/"scenario_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
        def package_version(name):
            try: return version(name)
            except PackageNotFoundError: return None
        environment={
            "python":sys.version,"platform":platform.platform(),
            "packages":{k:package_version(k) for k in
                        ("numpy","scipy","opencv-python-headless","opencv-python","Pillow",
                         "torch","diffusers","transformers","accelerate","streamlit")},
            "models":{
                "base":"runwayml/stable-diffusion-inpainting",
                "controlnet":"lllyasviel/control_v11p_sd15_canny",
            },
            "note":"Versiones y repositorios utilizados; registrar commit de modelo/checkpoint para reproducción exacta entre máquinas.",
        }
        (target/"environment.json").write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding="utf-8")
        if canonical.read_bytes()!=original_bytes:
            raise RuntimeError("Original calibration JSON changed during the run")
        return target
    except Exception:
        shutil.rmtree(target,ignore_errors=True)
        raise


def preview_masks(run_dir:Path,mode:str) -> list[dict]:
    meta=load_metadata(run_dir)
    rows=[]
    for i,view in enumerate(meta["views"],1):
        with Image.open(run_dir/view["mask_relative"]) as im:
            full=np.asarray(im.convert("L"))
        work=extension_work_mask(full,meta,i,mode=="ampliacion")
        rows.append({"view":i,"total_editable":int(np.count_nonzero(full)),
                     "work_editable":int(np.count_nonzero(work)),
                     "working_mask":work})
    return rows
