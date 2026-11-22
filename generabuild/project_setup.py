"""Creación de proyectos Desktop sin calibración previa.

Los borradores conservan imágenes, anotaciones, FOV y máscaras en un JSON
independiente. Solo una calibración PnP aprobada produce calibration_multiview.json.
No se sobrescriben proyectos ni imágenes existentes.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .geometry import sha256_file
from .storage import WORKSPACE, slug

DRAFT_FILENAME = "project_draft.json"
SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}
MAX_INPUT_SIZE = 45 * 1024 * 1024


def _image_details(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Invalid photograph: {path}")
    if path.stat().st_size > MAX_INPUT_SIZE:
        raise ValueError(f"Photograph exceeds {MAX_INPUT_SIZE//(1024*1024)} MiB: {path.name}")
    try:
        with Image.open(path) as im:
            w, h = im.size
            if w < 200 or h < 150 or w*h > 120_000_000:
                raise ValueError(f"Unsupported image dimensions in {path.name}: {w}×{h}")
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB").load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not verify photograph: {path.name}") from exc
    return int(w), int(h)


def new_draft(site: str, image_paths: list[str | Path], *, ref_w: float = 10.,
              ref_d: float = 20., ref_h: float = 9., floor_height: float = 3.,
              floors: int = 9, fovs: list[float] | None = None,
              workspace: Path | None = None) -> Path:
    """Crea el proyecto de forma atómica; no exige cámaras calibradas."""
    site_id = slug(site)
    workspace = Path(workspace) if workspace is not None else WORKSPACE
    dest = workspace / site_id
    if dest.exists():
        raise FileExistsError(f"Project already exists at {dest}; choose a different identifier")
    if len(image_paths) != 4:
        raise ValueError("Select exactly four photographs")
    dims = (float(ref_w), float(ref_d), float(ref_h), float(floor_height))
    if min(dims) <= 0 or not all(__import__('math').isfinite(x) for x in dims):
        raise ValueError("Dimensions must be finite positive numbers")
    if not 8 <= int(floors) <= 10:
        raise ValueError("Scenario must have 8–10 floors")
    if int(floors) * dims[3] <= dims[2]:
        raise ValueError("The target height must exceed the original height")
    fovs = list(fovs if fovs is not None else [55., 55., 55., 55.])
    if len(fovs) != 4 or any(not 15 <= float(f) <= 125 for f in fovs):
        raise ValueError("Four horizontal FOVs between 15° and 125° are required")
    paths = [Path(p).expanduser().resolve() for p in image_paths]
    if len(set(paths)) != 4:
        raise ValueError('Each view must use a different photograph')
    sizes = [_image_details(p) for p in paths]
    workspace.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{site_id}_building_", dir=str(workspace)))
    try:
        (temp / "inputs").mkdir()
        views = []
        for i, (source, size, fov) in enumerate(zip(paths, sizes, fovs), 1):
            rel = Path("inputs") / f"Vista_{i}{source.suffix.lower()}"
            shutil.copy2(source, temp / rel)
            views.append({
                "index": i, "name": source.name, "image_relative": rel.as_posix(),
                "image_size_wh": list(size), "image_sha256": sha256_file(temp / rel),
                "fov_horizontal_deg": float(fov), "anchors_px": {},
                "allowed_polygon_px": None, "protected_polygons_px": [],
                "manual_vp_guides_px": [], "observed_axis_segments_px": [],
            })
        draft = {
            "format": "GeneraBuild_Desktop_Draft_v1", "site_id": site_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference_box_m": {"W": dims[0], "D": dims[1], "H": dims[2]},
            "floor_height_m": dims[3], "num_floors": int(floors), "views": views,
            "instructions": "Mark 2D–3D correspondences, calibrate and approve before generation.",
        }
        (temp / DRAFT_FILENAME).write_text(
            json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if dest.exists():
            raise FileExistsError(f"Site already exists: {dest}")
        os.rename(temp, dest)
        return dest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def load_draft(folder: Path, *, verify_hash: bool = True) -> dict:
    folder = Path(folder)
    path = folder / DRAFT_FILENAME
    draft = json.loads(path.read_text(encoding="utf-8"))
    if draft.get("format") != "GeneraBuild_Desktop_Draft_v1" or len(draft.get("views", [])) != 4:
        raise ValueError("Invalid draft: four views are required")
    for i, view in enumerate(draft["views"], 1):
        rel = Path(view["image_relative"])
        if not rel.parts or rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "inputs":
            raise ValueError(f"Unsafe image path in View {i}")
        image = folder / rel
        if not image.is_file():
            raise FileNotFoundError(image)
        if verify_hash and sha256_file(image) != view.get("image_sha256"):
            raise ValueError(f"Photograph View {i} changed after import")
    return draft


def save_draft(folder: Path, payload: dict) -> dict:
    """Persistencia atómica de anotaciones y dimensiones sin modificar las imágenes."""
    folder = Path(folder)
    draft = load_draft(folder)
    for key in ("reference_box_m", "floor_height_m", "num_floors"):
        if key in payload:
            draft[key] = payload[key]
    if "views" in payload:
        if len(payload["views"]) != 4:
            raise ValueError("Four views are required")
        editable = ("anchors_px", "fov_horizontal_deg", "allowed_polygon_px",
                    "protected_polygons_px", "manual_vp_guides_px", "observed_axis_segments_px")
        for target, source in zip(draft["views"], payload["views"]):
            for field in editable:
                if field in source:
                    target[field] = source[field]
    draft["updated_utc"] = datetime.now(timezone.utc).isoformat()
    # Comprobar serialización antes de escribir; no alterar el proyecto si falla.
    serialized = json.dumps(draft, ensure_ascii=False, indent=2, allow_nan=False)
    stage = folder / (DRAFT_FILENAME + ".tmp")
    stage.write_text(serialized, encoding="utf-8")
    stage.replace(folder / DRAFT_FILENAME)
    return draft
