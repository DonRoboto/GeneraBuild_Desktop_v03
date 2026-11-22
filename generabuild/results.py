"""Visor y exportación de ejecuciones FR2 sin depender de Qt ni de CUDA.

No presenta una previsualización geométrica como resultado fotográfico: las
imágenes finales se resuelven únicamente desde el manifiesto de generación.
Todas las rutas se validan dentro del directorio de la ejecución.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

ResultMode = Literal['original', 'final', 'base', 'compare', 'preview']


@dataclass(frozen=True)
class RunRecord:
    path: Path
    run_id: str
    floors: int | None
    mode: str
    generated_views: tuple[int, ...]
    preview_views: tuple[int, ...]

    @property
    def label(self) -> str:
        height = f'{self.floors} floors' if self.floors is not None else 'floors n/a'
        status = (f'Generated {len(self.generated_views)}/4' if self.generated_views
                  else 'CPU preview only')
        display_mode = {'ampliacion':'Vertical extension', 'completo':'Full façade replacement'}.get(self.mode,self.mode)
        return f'{self.run_id}  ·  {height}  ·  {display_mode}  ·  {status}'


@dataclass
class ResultSet:
    run_dir: Path
    original: dict[int, Path] = field(default_factory=dict)
    final: dict[int, Path] = field(default_factory=dict)
    base: dict[int, Path] = field(default_factory=dict)
    preview: dict[int, Path] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def path_for(self, view: int, mode: ResultMode) -> Path | None:
        if mode == 'original':
            return self.original.get(view)
        if mode == 'base':
            return self.base.get(view)
        if mode == 'preview':
            return self.preview.get(view)
        if mode in ('final', 'compare'):
            return self.final.get(view)
        raise ValueError(f'Unknown result mode: {mode}')

    @property
    def available_views(self) -> tuple[int, ...]:
        return tuple(sorted(self.final))


def _read_json(file: Path, warnings: list[str]) -> dict:
    if not file.is_file():
        return {}
    try:
        data = json.loads(file.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('JSON root must be an object')
        return data
    except (OSError, ValueError, UnicodeError) as exc:
        warnings.append(f'Could not read {file.name}: {exc}')
        return {}


def _safe_image(root: Path, rel: object, warnings: list[str]) -> Path | None:
    """Impide que manifiestos antiguos o malformados escapen del proyecto."""
    if not isinstance(rel, str) or not rel.strip():
        return None
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts:
        warnings.append(f'Image path not permitted: {rel!r}')
        return None
    base = root.resolve()
    candidate = (base / p).resolve()
    if not candidate.is_relative_to(base):
        warnings.append(f'Path outside run: {rel!r}')
        return None
    if not candidate.is_file():
        warnings.append(f'Missing image: {rel!r}')
        return None
    return candidate


def _manifest_images(root: Path, manifest: dict, field_name: str,
                     warnings: list[str]) -> dict[int, Path]:
    results: dict[int, Path] = {}
    values = manifest.get(field_name, {})
    if not isinstance(values, dict):
        warnings.append(f'Invalid {field_name} field in run_manifest.json')
        return results
    for key, relative in values.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= 4:
            path = _safe_image(root, relative, warnings)
            if path is not None:
                results[index] = path
    return results


def load_result_set(run_dir: Path, site_dir: Path | None = None) -> ResultSet:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Run does not exist: {run_dir}')
    result = ResultSet(run_dir)
    result.config = _read_json(run_dir / 'scenario_config.json', result.warnings)
    result.evaluation = _read_json(run_dir / 'evaluation.json', result.warnings)
    calibration = _read_json(run_dir / 'calibration_multiview.json', result.warnings)
    views = calibration.get('views', [])
    for i in range(1, 5):
        view = views[i-1] if isinstance(views, list) and len(views) >= i else {}
        if not isinstance(view, dict):
            continue
        for base in (run_dir, Path(site_dir).resolve() if site_dir else None):
            if base is None:
                continue
            img = _safe_image(base, view.get('image_relative'), result.warnings)
            if img is not None:
                result.original[i] = img
                break
        preview = _safe_image(run_dir, view.get('preview_relative'), result.warnings)
        if preview is not None:
            result.preview[i] = preview

    manifest = _read_json(run_dir / 'run_manifest.json', result.warnings)
    result.final = _manifest_images(run_dir, manifest, 'final_images', result.warnings)
    result.base = _manifest_images(run_dir, manifest, 'base_images', result.warnings)
    # Fallback únicamente para ejecuciones con evaluación y manifiesto ausente.
    if not result.final and not (run_dir / 'run_manifest.json').is_file():
        for row in result.evaluation.get('views', []):
            if not isinstance(row, dict):
                continue
            try:
                i = int(row['view'])
            except (KeyError, ValueError, TypeError):
                continue
            if 1 <= i <= 4:
                img = _safe_image(run_dir, row.get('result_relative'), result.warnings)
                if img is not None:
                    result.final[i] = img
    return result


def list_runs(site_dir: Path) -> list[RunRecord]:
    """Encuentra ejecuciones guardadas; no confunde máscaras CPU con imagen final."""
    site_dir = Path(site_dir).resolve()
    folder = site_dir / 'runs'
    if not folder.is_dir():
        return []
    rows: list[RunRecord] = []
    for run in sorted(folder.iterdir(), key=lambda p: p.name, reverse=True):
        if not run.is_dir() or not (run / 'scenario_config.json').is_file():
            continue
        data = load_result_set(run, site_dir)
        try:
            floors = int(data.config['floors'])
        except (KeyError, TypeError, ValueError):
            floors = None
        rows.append(RunRecord(run, run.name, floors,
                              str(data.config.get('mode', '?')),
                              data.available_views, tuple(sorted(data.preview))))
    return rows


def compose_wipe(original: Image.Image, generated: Image.Image, percent: int) -> Image.Image:
    """Izquierda original y derecha generada; no mezcla píxeles."""
    if original.size != generated.size:
        raise ValueError('Invalid comparison: images have different resolutions')
    percent = max(0, min(100, int(percent)))
    before = original.convert('RGB')
    after = generated.convert('RGB')
    x = round(before.width * percent / 100)
    if x < before.width:
        before.paste(after.crop((x, 0, before.width, before.height)), (x, 0))
    return before


def export_montage(results: ResultSet, target: Path, mode: ResultMode = 'final',
                   split_percent: int = 50, cell_size: tuple[int, int] = (850, 470)) -> Path:
    """Exporta 2×2 sin modificar los PNG originales ni fingir resultados ausentes."""
    if mode not in {'original', 'final', 'base', 'compare', 'preview'}:
        raise ValueError(f'Invalid export mode: {mode}')
    w, h = cell_size
    if w < 200 or h < 150:
        raise ValueError('Montage cells are too small')
    margin, top = 16, 38
    background = Image.new('RGB', (2*w + 3*margin, 2*(h+top) + 3*margin), (16, 24, 36))
    draw = ImageDraw.Draw(background)
    font = ImageFont.load_default()
    for view in range(1, 5):
        col, row = (view-1) % 2, (view-1) // 2
        left = margin + col*(w+margin)
        y = margin + row*(h+top+margin)
        source = results.original.get(view)
        final = results.path_for(view, mode)
        valid = bool(final and final.is_file())
        text = f'VIEW {view} · {mode.upper()}' if valid else f'VIEW {view} · NO RESULT {mode.upper()}'
        draw.text((left+8, y+12), text, fill=(221, 233, 246), font=font)
        if source is not None and source.is_file():
            with Image.open(source) as image:
                reference = image.convert('RGB')
            if valid and mode == 'compare':
                with Image.open(final) as image:
                    generated = image.convert('RGB')
                shown = compose_wipe(reference, generated, split_percent)
            elif valid and mode != 'original':
                with Image.open(final) as image:
                    shown = image.convert('RGB')
            else:
                shown = reference
            shown.thumbnail((w, h), Image.Resampling.LANCZOS)
            x = left + (w-shown.width)//2
            yy = y + top + (h-shown.height)//2
            background.paste(shown, (x, yy))
        else:
            draw.text((left+8, y+top+16), 'Original photograph unavailable', fill=(235, 148, 148), font=font)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    background.save(target)
    return target
