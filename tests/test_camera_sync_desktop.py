"""Regression tests for reversible four-camera synchronization and English UI."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

import numpy as np
import pytest

from generabuild.camera_sync import build_camera_sync_proposal, accept_camera_sync_proposal
from generabuild.geometry import sha256_file
from generabuild.storage import load_metadata

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'workspace/sites/synthetic_demo'


def demo_copy(tmp_path:Path) -> Path:
    target=tmp_path/'urban_test'
    shutil.copytree(DEMO,target)
    return target


def test_proposal_is_read_only_and_cameras_lock_fov(tmp_path):
    site=demo_copy(tmp_path)
    original=sha256_file(site/'calibration_multiview.json')
    proposal=build_camera_sync_proposal(site,max_nfev=65)
    report=proposal['proposal']['report']
    assert report['solver_success']
    assert len(proposal['proposal']['cameras'])==4
    assert report['n_views_with_depth_evidence']>=1
    assert report['fovs_locked']==[65,68,65,110]
    assert sha256_file(site/'calibration_multiview.json')==original
    assert not list(site.glob('calibration_backup_*.json'))


def test_explicit_acceptance_backs_up_and_preserves_FOV_and_masks(tmp_path):
    site=demo_copy(tmp_path)
    before=sha256_file(site/'calibration_multiview.json')
    proposal=build_camera_sync_proposal(site,max_nfev=65)
    with pytest.raises(ValueError,match='Inspect all four'):
        accept_camera_sync_proposal(site,proposal,reviewed=False)
    assert sha256_file(site/'calibration_multiview.json')==before
    result=accept_camera_sync_proposal(site,proposal,reviewed=True)
    meta=result['metadata']
    assert [v['fov_horizontal_deg'] for v in meta['views']]==[65,68,65,110]
    assert meta['reference_box_m']['H']==9
    assert len(result['backup_paths'])==1
    backup=Path(result['backup_paths'][0])
    assert sha256_file(backup)==before
    assert Path(result['audit_path']).is_file()
    assert all((site/v['mask_relative']).is_file() for v in meta['views'])
    assert all((site/v['preview_relative']).is_file() for v in meta['views'])
    for v in meta['views']:
        assert np.all(np.isfinite(np.asarray(v['P'],float)))
    assert not any('synchronized' in key for key in meta)  # same FR2 metadata schema


def test_refuses_stale_proposals_and_requires_depth_correspondences(tmp_path):
    site=demo_copy(tmp_path)
    candidate=build_camera_sync_proposal(site,max_nfev=65)
    source=(site/'calibration_multiview.json')
    source.write_text(source.read_text()+'\n',encoding='utf8')
    with pytest.raises(ValueError,match='changed after the proposal'):
        accept_camera_sync_proposal(site,candidate,reviewed=True)
    import json
    meta=json.loads(source.read_text())
    for v in meta['views']:
        v['anchors_px']={k:p for k,p in v['anchors_px'].items() if k in {'0','1','4','5'}}
    source.write_text(json.dumps(meta),encoding='utf8')
    with pytest.raises(ValueError, match='depth|profundidad'):
        build_camera_sync_proposal(site,max_nfev=20)


def test_english_desktop_ui_and_camera_performance_contract():
    main=(ROOT/'generabuild_desktop/main.py').read_text(encoding='utf8')
    parsed=ast.parse(main)
    main_class=next(node for node in parsed.body
                    if isinstance(node,ast.ClassDef) and node.name=='MainWindow')
    methods={n.name for n in main_class.body if isinstance(n,ast.FunctionDef)}
    assert {'propose_camera_sync','_camera_sync_ready','accept_camera_sync',
            'discard_camera_sync','_update_camera_performance'} <= methods
    assert "self.tabs.addTab(scroll,'Camera Performance')" in main
    assert "self.sync_preview.toggled.connect(self.redraw)" in main
    assert "self.photo_layout_choice.addItem('▦  Four views (2×2)', 'grid')" in main
    assert "self.sync_accept.setEnabled(False)" in main
    assert "self.sync_review.isChecked()" in main
    assert 'self._sync_candidate' in main
    assert 'self._sync_candidate[\'proposal\']' in main


def test_qt_camera_performance_when_installed(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication,QMessageBox
    from generabuild_desktop.main import MainWindow
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    app=QApplication.instance() or QApplication([])
    site=demo_copy(tmp_path)
    gui=MainWindow(site)
    try:
        gui.tabs.setCurrentWidget(gui.cameras_tab)
        assert gui.camera_metrics_table.rowCount()==4
        proposal=build_camera_sync_proposal(site,max_nfev=65)
        gui._camera_sync_ready(proposal)
        assert gui.sync_preview.isChecked()
        assert gui.sync_discard.isEnabled()
        gui.discard_camera_sync()
        assert gui._sync_candidate is None
        gui._camera_sync_ready(proposal)
        gui.sync_review.setChecked(True)
        monkeypatch.setattr(QMessageBox,'question',lambda *args,**kwargs: QMessageBox.StandardButton.Yes)
        gui.accept_camera_sync()
        assert gui._sync_candidate is None
        assert len(list(site.glob('calibration_backup_*.json')))==1
    finally:
        gui.close()
