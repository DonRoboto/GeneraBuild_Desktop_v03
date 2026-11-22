"""Regression tests: scenario heights use the explicit 3 m/floor assumption.

The original building geometry and the four calibrated cameras must not be edited.
"""
from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from generabuild.storage import load_metadata
from generabuild.scenario import config_from_metadata, prepare_run

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'workspace/sites/synthetic_demo'


def test_saved_five_meter_assumption_is_overridden_per_scenario(tmp_path):
    source = tmp_path / 'site'
    shutil.copytree(DEMO, source)
    calibration_file = source / 'calibration_multiview.json'
    saved = json.loads(calibration_file.read_text(encoding='utf8'))
    saved['floor_height_m'] = 5.0  # simulate a legacy or incorrectly saved assumption
    calibration_file.write_text(json.dumps(saved), encoding='utf8')
    original_bytes = calibration_file.read_bytes()
    config = config_from_metadata(saved, floors=9, floor_height_m=3.0, mode='ampliacion')
    assert config['floor_height_m'] == 3.0
    run = prepare_run(source, config)
    result = load_metadata(run)
    assert result['target_box_m']['H'] == pytest.approx(27.0)
    assert np.max(np.asarray(result['vertices_3d_m'])[:, 2]) == pytest.approx(27.0)
    assert result['reference_box_m']['H'] == pytest.approx(saved['reference_box_m']['H'])
    assert [v['P'] for v in result['views']] == [v['P'] for v in saved['views']]
    assert calibration_file.read_bytes() == original_bytes
    saved_config = json.loads((run / 'scenario_config.json').read_text())
    assert saved_config['floor_height_m'] == 3.0


def test_scenario_height_contract_in_desktop_main():
    main = (ROOT / 'generabuild_desktop/main.py').read_text(encoding='utf8')
    assert "self.scenario_floor_height.setValue(3.0)" in main
    assert "floor_height_m=self.scenario_floor_height.value()" in main
    assert "height=float(self.scenario_floor_height.value())*self.floors.value()" in main
    assert "self.floors.value()*float(self.scenario_floor_height.value())" in main
    assert "height=float(self.meta['floor_height_m'])*self.floors.value()" not in main


def test_desktop_shows_numeric_scenario_height_when_qt_installed(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from generabuild_desktop.main import MainWindow
    source = tmp_path / 'site'
    shutil.copytree(DEMO, source)
    calibration_file = source / 'calibration_multiview.json'
    saved = json.loads(calibration_file.read_text(encoding='utf8'))
    saved['floor_height_m'] = 5.0
    calibration_file.write_text(json.dumps(saved), encoding='utf8')
    app = QApplication.instance() or QApplication([])
    window = MainWindow(source)
    try:
        assert window.scenario_floor_height.value() == 3.0
        assert window._config()['floor_height_m'] == 3.0
        assert np.max(window.target_vertices()[:, 2]) == 27.0
        assert '27.00 m' in window.height_summary.text()
        assert 'saved calibration says 5.00' in window.height_summary.text()
        window.floors.setValue(10)
        assert np.max(window.target_vertices()[:, 2]) == 30.0
    finally:
        window.close()
