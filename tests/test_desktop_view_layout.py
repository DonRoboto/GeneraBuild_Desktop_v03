"""Regresión del selector 2×2 / única (motor independiente de Qt)."""
from pathlib import Path
import ast
import pytest

from generabuild_desktop.view_layout import plan_photo_layout

ROOT=Path(__file__).resolve().parents[1]


def test_grid_layout_four_distinct_cells():
    expected={1:(0,0,1,1),2:(0,1,1,1),3:(1,0,1,1),4:(1,1,1,1)}
    for selected in range(1,5):
        plan=plan_photo_layout('grid',selected)
        assert plan.positions==expected
        assert plan.visible==(1,2,3,4)


def test_single_layout_expands_each_selected_photo_to_two_by_two():
    for selected in range(1,5):
        plan=plan_photo_layout('single',selected)
        assert plan.positions=={selected:(0,0,2,2)}
        assert plan.visible==(selected,)


@pytest.mark.parametrize('mode,index',[('grid',0),('single',5),('unknown',1),('',4)])
def test_invalid_layout_rejected(mode,index):
    with pytest.raises(ValueError):
        plan_photo_layout(mode,index)


def test_ui_has_global_layout_tools_and_preserves_existing_photo_panels():
    main=(ROOT/'generabuild_desktop'/'main.py').read_text(encoding='utf-8')
    photos=(ROOT/'generabuild_desktop'/'photos.py').read_text(encoding='utf-8')
    mod=ast.parse(main)
    main_cls=next(n for n in mod.body if isinstance(n,ast.ClassDef) and n.name=='MainWindow')
    methods={m.name for m in main_cls.body if isinstance(m,ast.FunctionDef)}
    assert {'_apply_photo_layout','_sync_photo_layout_controls',
            'fit_visible_photos','toggle_photo_layout_from_view',
            '_photo_focus_picker_changed'} <= methods
    assert "self.photo_layout_choice.addItem('▦  Four views (2×2)', 'grid')" in main
    assert 'gl.removeWidget(panel)' in main
    assert 'gl.addWidget(panel,row,col,rowspan,colspan)' in main
    assert 'self._sync_mark_modes()' in main
    assert 'self.cal_focus_view(bool(in_cal and self.cal_focus.isChecked()))' not in main
    assert 'doubleClicked=Signal(int)' in photos
    assert 'not self.mark_mode' in photos


def test_photo_qt_interaction_if_available():
    pytest.importorskip('PySide6')
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from generabuild_desktop.main import MainWindow
    app=QApplication.instance() or QApplication([])
    win=MainWindow()
    try:
        original=[id(p) for p in win.panels]
        assert win.photo_layout_mode=='grid'
        assert all(p.isHidden() is False for p in win.panels)
        win._apply_photo_layout('single')
        assert win.photo_layout_mode=='single'
        assert sum(not p.isHidden() for p in win.panels)==1
        win.activate_view(3)
        assert not win.panels[2].isHidden()
        assert win.photo_layout_mode=='single'
        assert all(p.isHidden() for i,p in enumerate(win.panels) if i!=2)
        win._apply_photo_layout('grid')
        assert all(not p.isHidden() for p in win.panels)
        assert [id(p) for p in win.panels]==original
    finally:
        win.close()
