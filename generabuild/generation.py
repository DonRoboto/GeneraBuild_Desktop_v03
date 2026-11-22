"""Motor GPU de FR2. La importación de Diffusers es diferida y opcional."""
from __future__ import annotations
from pathlib import Path


def cuda_available()->bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except (ImportError,OSError):
        return False


def load_pipeline(base_model:str="runwayml/stable-diffusion-inpainting",
                  controlnet_model:str="lllyasviel/control_v11p_sd15_canny"):
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
    except ImportError as e:
        raise RuntimeError("Faltan dependencias GPU. Instala requirements-gpu.txt y PyTorch CUDA") from e
    if not torch.cuda.is_available():
        raise RuntimeError("No hay CUDA disponible. CPU permite vista previa y evaluación, no la generación FR2")
    control = ControlNetModel.from_pretrained(controlnet_model,torch_dtype=torch.float16)
    pipeline=StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model,controlnet=control,torch_dtype=torch.float16)
    pipeline.to("cuda")
    pipeline.enable_attention_slicing()
    return pipeline


def generate_one(run_dir:Path,view:int,pipeline, config:dict)->Path:
    from .generation_fr2 import generate_one_view_realista
    return generate_one_view_realista(
        view_index=int(view),project_dir=run_dir,pipeline=pipeline,
        seed=int(config["seed"]),steps=int(config["steps"]),
        guidance=float(config["guidance"]),control_scale=float(config["control_scale"]),
        max_dim=int(config["max_dim"]),preserve_existing=config["mode"]=="ampliacion",
        front_bays=int(config["front_windows"]),side_bays=int(config["side_windows"]),
        prompt=config.get("prompt"),negative_prompt=config.get("negative_prompt"),
        show_preview=False,
    )
