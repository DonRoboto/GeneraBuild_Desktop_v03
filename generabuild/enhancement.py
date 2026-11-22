"""Acabado arquitectónico posterior a generación: módulos independientes CPU."""
from __future__ import annotations
from pathlib import Path
from gb21_photo_finish import PhotoFinishConfig, create_photo_finish
from gb21_roof_finish import RoofConfig, finish_existing_roofs


def apply_enhancements(run_dir:Path,generated:dict[int,Path],config:dict)->tuple[dict[int,Path],dict]:
    opts=config["postprocessing"]
    current=dict(generated)
    reports={}
    if opts.get("photo_finish",False):
        facade=PhotoFinishConfig(
            front_windows=int(config["front_windows"]),
            side_windows=int(config["side_windows"]),
            glass_opacity=.98,frame_opacity=.98,details_opacity=.28,
            wall_tint=float(opts.get("wall_tint",.30)),
        )
        for i,src in current.items():
            after,audit=create_photo_finish(run_dir,i,src,config=facade,
                            preserve_existing=config["mode"]=="ampliacion",
                            show_preview=False)
            current[i]=Path(after)
            reports[f"facade_{i}"]=audit
    if opts.get("roof_finish",False):
        aerial={i:path for i,path in current.items() if i in (1,2)}
        if aerial:
            roof=RoofConfig(seed=2026,roof_equipment=int(opts.get("roof_equipment",0)),
                            preserve_existing=config["mode"]=="ampliacion")
            result,_=finish_existing_roofs(run_dir,views=tuple(aerial),config=roof,
                                           source_paths=aerial,preview=False)
            for record in result:
                current[int(record["view"])]=Path(record["output"])
            reports["roof"]=result
    return current,reports
