"""Contratos estáticos mínimos de la interfaz Qt, no requieren PySide6.

Un smoke test GUI real necesita ejecutar PySide6 con QT_QPA_PLATFORM=offscreen.
"""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_qt_editor_contains_new_project_workflow():
    path = ROOT / 'generabuild_desktop' / 'main.py'
    source=path.read_text(encoding='utf-8')
    mod=ast.parse(source)
    main=next(n for n in mod.body if isinstance(n,ast.ClassDef) and n.name=='MainWindow')
    methods={n.name for n in main.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    required={
        'create_project','open_draft','open_site','cal_mark_click',
        'calibrate_selected','calibrate_four','cal_save_final','cal_save_draft',
        'cal_finish_polygon','cal_fov_changed','cal_geometry_changed',
        'on_photo_click','cal_detect_lines','redraw'
    }
    assert not required - methods
    assert 'NewProjectDialog' in source
    assert 'save_calibration(' in source
    assert 'refresh_overlays(' in source
    assert 'calibration_overlay=' in source


def test_all_qt_self_connect_callbacks_exist():
    tree=ast.parse((ROOT/'generabuild_desktop'/'main.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MainWindow')
    methods={n.name for n in cls.body if isinstance(n,ast.FunctionDef)}
    connected=set()
    for node in ast.walk(cls):
        if not isinstance(node,ast.Call):continue
        if not isinstance(node.func,ast.Attribute) or node.func.attr!='connect':continue
        for arg in node.args:
            if isinstance(arg,ast.Attribute) and isinstance(arg.value,ast.Name) and arg.value.id=='self':
                connected.add(arg.attr)
    assert not (connected - methods), f'Callbacks sin método: {connected-methods}'


def test_photo_view_overlay_signature():
    mod=ast.parse((ROOT/'generabuild_desktop'/'photos.py').read_text())
    cls=next(n for n in mod.body if isinstance(n,ast.ClassDef) and n.name=='PhotoView')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_overlays')
    assert 'calibration_overlay' in [arg.arg for arg in method.args.kwonlyargs]
