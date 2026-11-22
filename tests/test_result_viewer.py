"""Pruebas del Result Viewer; no requieren PySide6 ni modelos generativos."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from generabuild.results import compose_wipe, export_montage, list_runs, load_result_set

ROOT = Path(__file__).resolve().parents[1]


def make_site(tmp_path: Path, generated=(1, 2, 3, 4), *, name='20261120T203000Z_ampliacion_9p'):
    site = tmp_path / 'urban_site'
    run = site / 'runs' / name
    run.mkdir(parents=True)
    views = []
    final = {}
    base = {}
    for i in range(1, 5):
        original = f'inputs/vista_{i}.png'
        preview = f'previews/vista_{i}.png'
        for rel, color in ((original, (20+i, 40+i, 60+i)),
                           (preview, (42, 50, 60))):
            path = run / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (48, 32), color).save(path)
        views.append({'image_relative': original, 'preview_relative': preview})
        if i in generated:
            rel = f'generated/vista_{i}.png'
            path = run / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (48, 32), (200, 30+i, 90)).save(path)
            final[str(i)] = rel
            b = f'base/vista_{i}.png'
            bp = run / b
            bp.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (48, 32), (120, 50+i, 40)).save(bp)
            base[str(i)] = b
    (run / 'calibration_multiview.json').write_text(
        json.dumps({'views': views}), encoding='utf-8')
    (run / 'scenario_config.json').write_text(
        json.dumps({'floors': 9, 'mode': 'ampliacion', 'seed': 1234}), encoding='utf-8')
    (run / 'evaluation.json').write_text(
        json.dumps({'context_preservation_pass': True, 'views': [
            {'view': 1, 'pixels_outside_mask_changed': 0}]}), encoding='utf-8')
    if generated:
        (run / 'run_manifest.json').write_text(
            json.dumps({'final_images': final, 'base_images': base}), encoding='utf-8')
    return site, run


def test_historial_descubre_ejecuciones_reales(tmp_path):
    site, run = make_site(tmp_path)
    rows = list_runs(site)
    assert len(rows) == 1
    assert rows[0].path == run
    assert rows[0].floors == 9
    assert rows[0].mode == 'ampliacion'
    assert rows[0].generated_views == (1, 2, 3, 4)
    assert 'Generated 4/4' in rows[0].label


def test_loader_original_final_base_y_preview(tmp_path):
    site, run = make_site(tmp_path)
    result = load_result_set(run, site)
    assert result.available_views == (1, 2, 3, 4)
    assert all(result.path_for(i, 'final').is_file() for i in range(1, 5))
    assert all(result.path_for(i, 'preview').is_file() for i in range(1, 5))
    assert result.evaluation['context_preservation_pass']


def test_preview_cpu_no_es_resultado_fotografico(tmp_path):
    site, run = make_site(tmp_path, generated=())
    rows = list_runs(site)
    assert rows[0].generated_views == ()
    assert 'CPU preview only' in rows[0].label
    result = load_result_set(run, site)
    assert not result.final
    assert len(result.preview) == 4


def test_ejecucion_parcial_no_inventa_resultados(tmp_path):
    site, run = make_site(tmp_path, generated=(1, 3))
    results = load_result_set(run, site)
    assert results.available_views == (1, 3)
    assert results.path_for(2, 'final') is None
    target = tmp_path / 'montaje.png'
    export_montage(results, target, 'final', cell_size=(250, 150))
    assert target.is_file()
    with Image.open(target) as im:
        assert im.size == (3*16+2*250, 3*16+2*(150+38))


def test_comparar_por_division_conserva_pixes_exactos():
    a = Image.new('RGB', (20, 10), (10, 20, 30))
    b = Image.new('RGB', (20, 10), (200, 210, 220))
    out = compose_wipe(a, b, 50)
    assert out.getpixel((9, 3)) == (10, 20, 30)
    assert out.getpixel((10, 3)) == (200, 210, 220)
    assert compose_wipe(a, b, 0).getpixel((0, 0)) == (200, 210, 220)
    assert compose_wipe(a, b, 100).getpixel((19, 0)) == (10, 20, 30)
    with pytest.raises(ValueError, match='resolutions'):
        compose_wipe(a, Image.new('RGB', (21, 10)), 50)


def test_montaje_comparativo_y_original_sin_gpu(tmp_path):
    site, run = make_site(tmp_path)
    result = load_result_set(run, site)
    for mode in ('compare', 'original', 'base', 'preview', 'final'):
        dst = tmp_path / f'montage_{mode}.png'
        assert export_montage(result, dst, mode, split_percent=30).is_file()
        with Image.open(dst) as im:
            assert im.size[0] > 1000


def test_manifiesto_no_acepta_rutas_que_salen_del_proyecto(tmp_path):
    site, run = make_site(tmp_path)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    manifest['final_images']['1'] = '../../secret.png'
    manifest['final_images']['2'] = '/tmp/secret.png'
    (run / 'run_manifest.json').write_text(json.dumps(manifest))
    result = load_result_set(run, site)
    assert 1 not in result.final and 2 not in result.final
    assert 3 in result.final and 4 in result.final
    assert any('not permitted' in x for x in result.warnings)


def test_resultados_qt_conectados_con_vistas_2x2_y_unica():
    main = (ROOT/'generabuild_desktop'/'main.py').read_text(encoding='utf-8')
    photos = (ROOT/'generabuild_desktop'/'photos.py').read_text(encoding='utf-8')
    tree = ast.parse(main)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert {'_render_result_view','refresh_result_runs','_result_run_changed',
            'save_results_montage','show_results','_result_split_changed'} <= methods
    assert "self.photo_layout_choice.addItem('▦  Four views (2×2)', 'grid')" in main
    assert 'panel.viewer.show_result(original,generated' in main
    assert 'class WipePixmapItem(QGraphicsPixmapItem)' in photos
    assert 'painter.setClipRect' in photos


def test_qt_result_view_if_pyside6_installed(tmp_path):
    pytest.importorskip('PySide6')
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from generabuild_desktop.main import MainWindow
    app = QApplication.instance() or QApplication([])
    site, run = make_site(tmp_path)
    # Para abrir con la interfaz se requiere metadata completa; verificar el
    # viewer sobre la demo que sí contiene una calibración válida.
    demo = ROOT/'workspace/sites/synthetic_demo'
    window = MainWindow(demo)
    try:
        window.tabs.setCurrentWidget(window.results_tab)
        assert window.result_run_picker.count() >= 1
        window.result_mode.setCurrentIndex(window.result_mode.findData('preview'))
        window._render_result_view()
        window._apply_photo_layout('single')
        assert window.photo_layout_mode == 'single'
        window._apply_photo_layout('grid')
        assert window.photo_layout_mode == 'grid'
    finally:
        window.close()
