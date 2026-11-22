"""Almacenamiento local seguro y paquetes importables desde Colab FR2."""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

WORKSPACE = Path(__file__).resolve().parents[1]/"workspace"/"sites"


def slug(name: str) -> str:
    value=re.sub(r"[^A-Za-z0-9_-]+","_", name.strip()).strip("_")[:55]
    if not value or value.startswith("."):
        raise ValueError("Invalid site identifier")
    return value


def site_path(site: str) -> Path:
    return WORKSPACE/slug(site)


def list_sites() -> list[str]:
    WORKSPACE.mkdir(parents=True,exist_ok=True)
    return sorted(p.name for p in WORKSPACE.iterdir() if p.is_dir() and (p/"calibration_multiview.json").is_file())


def load_metadata(directory: Path) -> dict:
    path=directory/"calibration_multiview.json"
    meta=json.loads(path.read_text(encoding="utf-8"))
    if len(meta.get("views",[]))!=4:
        raise ValueError("Project must contain four views")
    return meta


def image_paths(directory:Path,meta:dict|None=None)->list[Path]:
    if meta:
        paths=[directory/v["image_relative"] for v in meta["views"]]
    else:
        imgs=sorted((directory/"inputs").glob("Vista_*"))
        paths=[]
        for i in range(1,5):
            matches=[p for p in imgs if p.stem==f"Vista_{i}"]
            if len(matches)!=1: raise FileNotFoundError(f"Missing original photograph Vista_{i}")
            paths.append(matches[0])
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)
    return paths


def import_colab_zip(uploaded: bytes, site:str) -> Path:
    """Copia SOLO insumos, calibración, masks y vistas previas; no ejecuta código ajeno."""
    import io
    dest=site_path(site)
    if (dest/"calibration_multiview.json").exists():
        raise FileExistsError("Ya existe este sitio; elige otro identificador")
    if len(uploaded)>160*1024*1024:
        raise ValueError("ZIP is too large; maximum 160 MiB compressed")
    with zipfile.ZipFile(io.BytesIO(uploaded)) as archive:
        members=[m for m in archive.infolist() if not m.is_dir()]
        if len(members)>2000 or sum(m.file_size for m in members)>500*1024*1024:
            raise ValueError("ZIP is too large or contains too many files")
        clean={}
        for m in members:
            path=PurePosixPath(m.filename.replace('\\','/'))
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"Unsafe ZIP path: {m.filename}")
            if stat.S_ISLNK(m.external_attr>>16):
                raise ValueError("Symbolic links are not allowed in ZIP archives")
            if path.name=="calibration_multiview.json":
                clean[str(path)]=m
        if len(clean)!=1:
            raise ValueError("ZIP must contain exactly one calibration_multiview.json")
        key=PurePosixPath(next(iter(clean)))
        prefix=key.parent.parts
        dest.mkdir(parents=True,exist_ok=True)
        try:
            acceptable={"inputs","masks","previews","canny"}
            for member in members:
                parts=PurePosixPath(member.filename.replace('\\','/')).parts
                if parts[:len(prefix)]!=prefix: continue
                relative=parts[len(prefix):]
                if not relative or (len(relative)>1 and relative[0] not in acceptable):continue
                if len(relative)==1 and relative[0]!="calibration_multiview.json":continue
                suffix=PurePosixPath(*relative).suffix.lower()
                if suffix not in (".json",".jpg",".jpeg",".png",".webp",".tif",".tiff"):continue
                if len(relative)>1 and relative[0]=="inputs" and suffix not in (".png",".jpg",".jpeg",".webp",".tif",".tiff"):continue
                out=dest.joinpath(*relative)
                out.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(member) as src,open(out,'wb') as dst:
                    shutil.copyfileobj(src,dst,length=1<<20)
            meta=load_metadata(dest)
            from .geometry import project_image_path
            for v in meta["views"]:
                # No permitir que un JSON manipulado acceda a archivos externos.
                for field in ("image_relative","mask_relative","canny_relative","preview_relative"):
                    rel=PurePosixPath(v[field].replace('\\','/'))
                    if not rel.parts or rel.is_absolute() or ".." in rel.parts or rel.parts[0] not in acceptable:
                        raise ValueError(f"Unsafe project path: {field}")
                src=dest/v["image_relative"]
                if not src.is_file(): raise FileNotFoundError(src)
                if v.get("image_sha256") and hashlib.sha256(src.read_bytes()).hexdigest()!=v["image_sha256"]:
                    raise ValueError("Photograph checksum does not match the calibration")
            return dest
        except Exception:
            shutil.rmtree(dest,ignore_errors=True)
            raise


def save_four_uploads(uploads: list,site:str)->Path:
    if len(uploads)!=4:
        raise ValueError("Four photographs are required")
    dest=site_path(site)
    if (dest/"calibration_multiview.json").exists():
        raise FileExistsError("Este sitio ya tiene calibración; usa otro nombre")
    folder=dest/"inputs";folder.mkdir(parents=True,exist_ok=True)
    from PIL import Image
    import io
    for i,file in enumerate(uploads,1):
        data=file.getvalue()
        try:
            img=Image.open(io.BytesIO(data))
            img.verify()
        except Exception as exc:
            raise ValueError(f"View {i}: invalid image") from exc
        ext=Path(file.name).suffix.lower()
        if ext not in ('.png','.jpeg','.jpg','.webp'):
            raise ValueError(f"View {i}: unsupported image format")
        (folder/f"Vista_{i}{ext}").write_bytes(data)
    return dest
