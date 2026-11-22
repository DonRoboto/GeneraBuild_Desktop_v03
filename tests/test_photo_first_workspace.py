"""UI structure contract: no separate 3D viewport, unchanged 3D math and photo editor."""
import ast
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_workspace_has_only_photo_and_control_panes():
    path=ROOT/'generabuild_desktop/main.py'
    source=path.read_text(encoding='utf8')
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MainWindow')
    make_ui=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_make_ui')
    splitter_adds=[n for n in ast.walk(make_ui) if isinstance(n,ast.Call) and
                   isinstance(n.func,ast.Attribute) and n.func.attr=='addWidget' and
                   isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='main_split']
    assert len(splitter_adds)==2
    assert 'Scene3D' not in source and 'scene3d' not in source
    assert 'self.main_split.setChildrenCollapsible(False)' in source
    assert 'self.main_split.setSizes([1090,460])' in source
    assert "self.tabs.setMinimumWidth(355)" in source
    assert 'result_maximize' not in source


def test_camera_generation_and_photo_controls_unchanged():
    source=(ROOT/'generabuild_desktop/main.py').read_text()
    assert 'calibrate_view(' in source
    assert 'build_camera_sync_proposal(' in source
    assert 'self._sync_candidate' in source
    assert 'refresh_overlays(' in source
    assert 'show_neighbors=self.show_context.isChecked() and not in_cal' in source
    assert 'GenerationWorker' in source
    assert 'self.scenario_floor_height.setValue(3.0)' in source
    assert "('Generated · Final image','final')" in source
    assert "'compare'" in source
    assert "self.photo_layout_choice.addItem('▦  Four views (2×2)', 'grid')" in source


def test_interactive_photo_first_qt_when_available():
    pytest.importorskip('PySide6')
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from generabuild_desktop.main import MainWindow
    app=QApplication.instance() or QApplication([])
    win=MainWindow()
    try:
        assert win.main_split.count()==2
        assert win.main_split.widget(0) is win.photo_grid
        assert win.main_split.widget(1) is win.tabs
        assert not hasattr(win,'scene3d')
        original=[id(p) for p in win.panels]
        win._apply_photo_layout('single')
        win.activate_view(4)
        assert win.photo_layout_mode=='single'
        win._apply_photo_layout('grid')
        assert win.photo_layout_mode=='grid'
        assert [id(p) for p in win.panels]==original
    finally:
        win.close()
