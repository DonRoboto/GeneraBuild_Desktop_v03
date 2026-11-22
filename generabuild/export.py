from __future__ import annotations
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED


def zip_folder(folder:Path,*,max_files:int=1000)->bytes:
    """Exportación del escenario completo (incluye imágenes y configuración)."""
    output=BytesIO()
    with ZipFile(output,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        files=[p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()]
        if len(files)>max_files:raise ValueError('Demasiados archivos para exportar')
        for p in files:
            archive.write(p,arcname=p.relative_to(folder).as_posix())
    return output.getvalue()
