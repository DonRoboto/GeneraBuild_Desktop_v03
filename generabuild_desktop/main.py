"""GeneraBuild Studio Desktop v0.3 — scientific urban scenario editor.

Photo-first PySide6 interface with four calibrated photographs and an adjustable
controls panel. Optional 2.5D urban context, FR2 generation, and camera synchronization.
The 3D geometry core remains available for all perspective projections.
"""
from __future__ import annotations
import hashlib
import copy
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path
from PIL import Image

import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QUrl, QTimer, QLocale
from PySide6.QtGui import QAction, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QDialog,
    QGridLayout, QPushButton, QLabel, QGroupBox, QFileDialog, QMessageBox,
    QTabWidget, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QLineEdit,
    QListWidget, QListWidgetItem, QTextEdit, QInputDialog, QStatusBar,
    QScrollArea, QSlider, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from generabuild.storage import load_metadata, image_paths, import_colab_zip, site_path
from generabuild.geometry import box_vertices,sha256_file,calibrate_view
from generabuild.scenario import config_from_metadata, prepare_run
from generabuild.context_2p5d import (
    image_to_ground, validate_footprint, new_neighbor, load_context, save_context,
)
from generabuild.context_occlusion import apply_context_occlusions_to_run
from generabuild.calibration import save_calibration
from generabuild.project_setup import load_draft,save_draft,DRAFT_FILENAME
from generabuild.results import list_runs, load_result_set, export_montage
from generabuild.vanishing import detect_segments
from generabuild.camera_sync import build_camera_sync_proposal, accept_camera_sync_proposal
from .new_project_dialog import NewProjectDialog
from .photos import PhotoPanel
from .view_layout import plan_photo_layout

ROOT=Path(__file__).resolve().parents[1]


class GenerationWorker(QObject):
    log=Signal(str)
    completed=Signal(str)
    failed=Signal(str)
    def __init__(self,run_dir:Path,config:dict):
        super().__init__()
        self.run_dir=Path(run_dir);self.config=config

    @Slot()
    def run(self):
        try:
            from generabuild.generation import load_pipeline,generate_one
            from generabuild.enhancement import apply_enhancements
            from generabuild.evaluation import evaluate_run
            self.log.emit('Loading Stable Diffusion 1.5 + ControlNet models on CUDA…')
            pipe=load_pipeline()
            images={}
            for i in self.config['views']:
                self.log.emit(f'Generating View {i}…')
                images[i]=generate_one(self.run_dir,i,pipe,self.config)
            self.log.emit('Applying optional architectural enhancements…')
            final,reports=apply_enhancements(self.run_dir,images,self.config)
            (self.run_dir/'enhancement_reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            manifest={'base_images':{str(k):str(Path(v).relative_to(self.run_dir)) for k,v in images.items()},
                      'final_images':{str(k):str(Path(v).relative_to(self.run_dir)) for k,v in final.items()}}
            (self.run_dir/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
            report=evaluate_run(self.run_dir,final,self.config)
            self.log.emit(f"Context preservation: {'PASS' if report['context_preservation_pass'] else 'REVIEW'}")
            self.completed.emit(str(self.run_dir))
        except BaseException as exc:
            self.failed.emit(f'{exc}\n\n{traceback.format_exc(limit=12)}')


class CameraSyncWorker(QObject):
    """Compute joint camera refinement off the Qt GUI thread."""
    completed=Signal(object)
    failed=Signal(str)

    def __init__(self, site: Path, options: dict):
        super().__init__()
        self.site=Path(site)
        self.options=options

    @Slot()
    def run(self):
        try:
            self.completed.emit(build_camera_sync_proposal(self.site, **self.options))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self,site:Path|None=None):
        super().__init__()
        self.setWindowTitle('GeneraBuild Studio | Geometry-Constrained Urban Scenario Editor')
        self.resize(1570,915)
        self.setMinimumSize(1050,650)
        self.folder=None;self.meta=None;self.paths=[]
        self.neighbors=[];self.draft_xy=[];self.draft_uv=[]
        self.active_view=1;self.current_run=None
        self.photo_layout_mode="grid"
        self._applied_photo_layout=None
        self.display_override={};self._thread=None;self._worker=None
        self.result_set=None
        self._result_records=[]
        self.cal_anchors=[{} for _ in range(4)]
        self.cal_fovs=[55.]*4
        self.cal_allowed=[None]*4
        self.cal_protected=[[] for _ in range(4)]
        self.cal_guides=[[] for _ in range(4)]
        self.cal_observed=[[] for _ in range(4)]
        self.cal_cameras=[None]*4
        self.cal_detected=[[] for _ in range(4)]
        self.cal_poly_draft=[];self.cal_poly_view=None;self.cal_line_start=None
        self._cal_updating=False
        self._sync_candidate=None
        self._sync_thread=None
        self._sync_worker=None
        self._sync_source_site=None
        self._make_ui();self._make_menu();self._style()
        demo=ROOT/'workspace'/'sites'/'synthetic_demo'
        if site is None and demo.is_dir():site=demo
        if site is not None:
            try:self.open_site(Path(site))
            except Exception as exc:self.error(str(exc))

    def _style(self):
        self.setStyleSheet('''
        QMainWindow,QWidget{background:#101824;color:#d5e1f0;font-family:Sans;font-size:12px;}
        QGroupBox{border:1px solid #344459;border-radius:6px;margin-top:14px;padding:9px;}
        QGroupBox::title{color:#9fc3de;subcontrol-origin:margin;left:8px;}
        QPushButton{background:#263a52;border:1px solid #415d78;border-radius:5px;padding:7px 9px;}
        QPushButton:hover{background:#36516f;}
        QPushButton:checked{background:#13735d;border-color:#53ddb1;}
        QPushButton:disabled{color:#81909b;background:#182633;}
        QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#192a3e;border:1px solid #41556b;border-radius:4px;padding:5px;}
        QTabWidget::pane{border:1px solid #344459;}
        QTabBar::tab{background:#1c2b3f;padding:9px;border:1px solid #314458;}
        QTabBar::tab:selected{background:#284c66;}
        QListWidget,QTextEdit{background:#182538;border:1px solid #34475c;}
        QLabel#PhotoTitle{color:#54e5b3;font-size:12px;font-weight:bold;margin-left:7px;}
        QSplitter::handle{background:#283a50;}
        QSplitter::handle:hover{background:#48708c;}
        ''')

    def _make_ui(self):
        base=QWidget();self.setCentralWidget(base)
        root_layout=QVBoxLayout(base);root_layout.setContentsMargins(6,6,6,6);root_layout.setSpacing(5)
        topline=QHBoxLayout();root_layout.addLayout(topline)
        self.site_title=QLabel('GeneraBuild Studio  •  No project open');self.site_title.setFont(QFont('Sans',14,QFont.Weight.Bold))
        topline.addWidget(self.site_title,1)
        new_btn=QPushButton('＋ New project…');new_btn.clicked.connect(self.create_project);topline.addWidget(new_btn)
        open_btn=QPushButton('Open site…');open_btn.clicked.connect(self.choose_site);topline.addWidget(open_btn)
        zip_btn=QPushButton('Import Colab ZIP…');zip_btn.clicked.connect(self.import_zip);topline.addWidget(zip_btn)
        topline.addSpacing(12)
        topline.addWidget(QLabel('View layout'))
        self.photo_layout_choice=QComboBox()
        self.photo_layout_choice.addItem('▦  Four views (2×2)', 'grid')
        self.photo_layout_choice.addItem('▣  One enlarged view', 'single')
        self.photo_layout_choice.setToolTip('Switch between all four photographs and a single enlarged view')
        self.photo_layout_choice.currentIndexChanged.connect(self._photo_layout_choice_changed)
        topline.addWidget(self.photo_layout_choice)
        self.photo_focus_picker=QComboBox()
        for i in range(1,5):self.photo_focus_picker.addItem(f'View {i}',i)
        self.photo_focus_picker.setToolTip('Choose the photograph to edit or enlarge')
        self.photo_focus_picker.currentIndexChanged.connect(self._photo_focus_picker_changed)
        self.photo_focus_picker.setEnabled(False)
        topline.addWidget(self.photo_focus_picker)
        fit_btn=QPushButton('Fit image')
        fit_btn.setToolTip('Fit image to its panel without modifying source pixels')
        fit_btn.clicked.connect(self.fit_visible_photos)
        topline.addWidget(fit_btn)
        # Photo-first workspace: native images on the left and a resizable
        # inspector/control pane on the right. No separate 3D canvas is needed:
        # calibrated 3D geometry is still projected over the original photos.
        self.main_split=QSplitter(Qt.Orientation.Horizontal)
        self.main_split.setObjectName('WorkspaceSplitter')
        self.main_split.setChildrenCollapsible(False)
        self.main_split.setHandleWidth(8)
        root_layout.addWidget(self.main_split,1)
        grid=QWidget()
        grid.setObjectName('PhotoWorkspace')
        grid.setMinimumWidth(540)
        gl=QGridLayout(grid);gl.setContentsMargins(2,2,2,2);gl.setSpacing(5)
        gl.setRowStretch(0,1);gl.setRowStretch(1,1)
        gl.setColumnStretch(0,1);gl.setColumnStretch(1,1)
        self.photo_grid=grid
        self.photo_grid_layout=gl
        self.panels=[]
        for i in range(1,5):
            panel=PhotoPanel(i)
            panel.viewer.clicked.connect(self.on_photo_click)
            panel.viewer.activated.connect(self.activate_view)
            panel.viewer.doubleClicked.connect(self.toggle_photo_layout_from_view)
            gl.addWidget(panel,(i-1)//2,(i-1)%2)
            self.panels.append(panel)
        self.main_split.addWidget(grid)
        self.tabs=QTabWidget()
        self.tabs.setObjectName('ControlTabs')
        self.tabs.setMinimumWidth(355)
        self.main_split.addWidget(self.tabs)
        self.main_split.setStretchFactor(0,3)
        self.main_split.setStretchFactor(1,1)
        self.main_split.setSizes([1090,460])
        self._tab_calibration();self._tab_scenario();self._tab_neighborhood();self._tab_cameras();self._tab_results()
        self.tabs.currentChanged.connect(self.tab_changed)
        self._apply_photo_layout('grid',force=True)
        self.status=QStatusBar();self.setStatusBar(self.status)
        self.status.showMessage('Open a calibrated project. The optional 2.5D module does not modify photos or cameras.')

    def _make_menu(self):
        menu=self.menuBar().addMenu('File')
        def action(name,callback):
            a=QAction(name,self);a.triggered.connect(callback);menu.addAction(a)
        action('New project (4 photographs)…',self.create_project)
        action('Open project folder…',self.choose_site)
        action('Import Colab ZIP…',self.import_zip)
        action('Save 2.5D neighbors',self.persist_neighbors)
        action('Export run ZIP…',self.export_run)
        menu.addSeparator();action('Exit',self.close)
        view_menu=self.menuBar().addMenu('View')
        a=QAction('Four photographs (2×2)',self);a.setShortcut('Ctrl+G')
        a.triggered.connect(lambda: self._apply_photo_layout('grid'));view_menu.addAction(a)
        a=QAction('Single photograph',self);a.setShortcut('Ctrl+L')
        a.triggered.connect(lambda: self._apply_photo_layout('single'));view_menu.addAction(a)
        view_menu.addSeparator()
        for i in range(1,5):
            a=QAction(f'Select View {i}',self);a.setShortcut(f'Ctrl+{i}')
            a.triggered.connect(lambda checked=False,index=i: self.activate_view(index))
            view_menu.addAction(a)
        help_=self.menuBar().addMenu('Help')
        a=QAction('User guide',self);a.triggered.connect(self.show_help);help_.addAction(a)

    def _tab_calibration(self):
        """Interfaz nativa para calibrar un sitio nuevo o revisar uno existente."""
        scroll=QScrollArea();scroll.setWidgetResizable(True)
        page=QWidget();layout=QVBoxLayout(page)
        intro=QLabel(
            'NEW PROJECT · Select four photos with File → New project.\n'
            'Mark the SAME physical 3D corner (0–7) in every view where it is visible.\n'
            'Calibrate using the ORIGINAL building; inspect the prism before saving.'
        );intro.setWordWrap(True);layout.addWidget(intro)
        self.cal_status=QLabel('No project · choose New project…')
        self.cal_status.setWordWrap(True);layout.addWidget(self.cal_status)

        g=QGroupBox('ORIGINAL reference geometry (approximate)')
        grid=QGridLayout(g)
        self.cal_w=QDoubleSpinBox();self.cal_w.setRange(1,250);self.cal_w.setDecimals(2);self.cal_w.setSuffix(' m');self.cal_w.setValue(10)
        self.cal_d=QDoubleSpinBox();self.cal_d.setRange(1,400);self.cal_d.setDecimals(2);self.cal_d.setSuffix(' m');self.cal_d.setValue(20)
        self.cal_h=QDoubleSpinBox();self.cal_h.setRange(1,200);self.cal_h.setDecimals(2);self.cal_h.setSuffix(' m');self.cal_h.setValue(9)
        self.cal_floor_h=QDoubleSpinBox();self.cal_floor_h.setRange(2,5);self.cal_floor_h.setDecimals(2);self.cal_floor_h.setValue(3);self.cal_floor_h.setSuffix(' m')
        for row,(label,widget) in enumerate((('W — width along X',self.cal_w),('D — depth along Y',self.cal_d),('H₀ — height along Z',self.cal_h),('Floor height',self.cal_floor_h))):
            grid.addWidget(QLabel(label),row,0);grid.addWidget(widget,row,1)
            widget.valueChanged.connect(self.cal_geometry_changed)
        layout.addWidget(g)

        g=QGroupBox('View and 2D–3D correspondences')
        grid=QGridLayout(g)
        self.cal_view=QComboBox()
        for i in range(1,5):self.cal_view.addItem(f'View {i}',i)
        self.cal_view.currentIndexChanged.connect(self.cal_select_view)
        self.cal_fov=QDoubleSpinBox();self.cal_fov.setRange(15,125);self.cal_fov.setValue(55);self.cal_fov.setDecimals(1);self.cal_fov.setSuffix('°')
        self.cal_fov.valueChanged.connect(self.cal_fov_changed)
        grid.addWidget(QLabel('Photograph'),0,0);grid.addWidget(self.cal_view,0,1)
        grid.addWidget(QLabel('FOV horizontal'),1,0);grid.addWidget(self.cal_fov,1,1)
        self.cal_anchor=QComboBox()
        for i,desc in ((0,'0 · front-left ground'),(1,'1 · front-right ground'),
                       (4,'4 · front-left roof'),(5,'5 · front-right roof'),
                       (2,'2 · rear-right ground'),(3,'3 · rear-left ground'),
                       (6,'6 · rear-right roof'),(7,'7 · rear-left roof')):
            self.cal_anchor.addItem(desc,i)
        grid.addWidget(QLabel('Physical vertex'),2,0)
        grid.addWidget(self.cal_anchor,2,1)
        self.cal_mode=QComboBox()
        self.cal_mode.addItem('Mark / move vertex','anchor')
        self.cal_mode.addItem('Polygon: permitted edit area','allowed')
        self.cal_mode.addItem('Polygon: protect surrounding context','protected')
        self.cal_mode.addItem('Observed structural edge · 2 clicks','observed')
        self.cal_mode.addItem('Vanishing-point guide','guide')
        grid.addWidget(QLabel('Tool'),3,0);grid.addWidget(self.cal_mode,3,1)
        self.cal_axis=QComboBox()
        for i,desc in enumerate(('X · façade width','Y · depth','Z · vertical')):
            self.cal_axis.addItem(desc,i)
        grid.addWidget(QLabel('Edge / guide axis'),4,0);grid.addWidget(self.cal_axis,4,1)
        self.cal_mark=QPushButton('Activate tool (click photograph)')
        self.cal_mark.setCheckable(True)
        self.cal_mark.toggled.connect(self.toggle_cal_mark)
        grid.addWidget(self.cal_mark,5,0,1,2)
        layout.addWidget(g)

        tools=QGroupBox('Editing and diagnostics');tg=QGridLayout(tools)
        btn=QPushButton('Delete selected vertex');btn.clicked.connect(self.cal_delete_anchor);tg.addWidget(btn,0,0,1,2)
        btn=QPushButton('Undo last point');btn.clicked.connect(self.cal_undo);tg.addWidget(btn,1,0)
        btn=QPushButton('Finish polygon');btn.clicked.connect(self.cal_finish_polygon);tg.addWidget(btn,1,1)
        btn=QPushButton('Delete last protected region');btn.clicked.connect(self.cal_delete_protected);tg.addWidget(btn,2,0,1,2)
        btn=QPushButton('Detect line segments with OpenCV');btn.clicked.connect(self.cal_detect_lines);tg.addWidget(btn,3,0,1,2)
        self.cal_focus=QCheckBox('Enlarge one view (also available in the toolbar)')
        self.cal_focus.toggled.connect(self.cal_focus_view);tg.addWidget(self.cal_focus,4,0,1,2)
        layout.addWidget(tools)

        g=QGroupBox('OpenCV calibration — PnP');grid=QVBoxLayout(g)
        btn=QPushButton('Calibrate selected view');btn.clicked.connect(self.calibrate_selected)
        grid.addWidget(btn)
        btn=QPushButton('Calibrate all four cameras');btn.clicked.connect(self.calibrate_four)
        grid.addWidget(btn)
        self.cal_metrics=QLabel('No estimated cameras yet. Mark at least four identifiable physical corners per view.')
        self.cal_metrics.setWordWrap(True);grid.addWidget(self.cal_metrics)
        self.cal_approved=QCheckBox('I visually reviewed all four projections and protected areas')
        grid.addWidget(self.cal_approved)
        btn=QPushButton('SAVE CALIBRATION AND ENABLE GENERATION')
        btn.clicked.connect(self.cal_save_final)
        grid.addWidget(btn)
        btn=QPushButton('Save draft (uncalibrated)');btn.clicked.connect(self.cal_save_draft)
        grid.addWidget(btn)
        layout.addWidget(g)
        note=QLabel('A camera fitted with four coplanar points may estimate depth poorly. '
                    'Do not invent hidden corners. Fitting RMSE is NOT an independent accuracy measurement.')
        note.setWordWrap(True);layout.addWidget(note)
        layout.addStretch()
        scroll.setWidget(page)
        self.calibration_tab=scroll
        self.tabs.addTab(scroll,'Calibration')

    def _tab_scenario(self):
        page=QWidget();layout=QVBoxLayout(page)
        g=QGroupBox('Geometric model');form=QGridLayout(g)
        form.addWidget(QLabel('Target floors'),0,0)
        self.floors=QSpinBox();self.floors.setRange(8,10);self.floors.setValue(9);self.floors.valueChanged.connect(self.redraw)
        form.addWidget(self.floors,0,1)
        # Scenario assumptions are independent of the saved camera calibration.
        # For the Guanajuato 8 case study the agreed floor-to-floor height is 3 m.
        self.scenario_floor_height=QDoubleSpinBox()
        self.scenario_floor_height.setRange(2.0,5.0)
        self.scenario_floor_height.setDecimals(2)
        self.scenario_floor_height.setSingleStep(0.25)
        self.scenario_floor_height.setValue(3.0)
        self.scenario_floor_height.setSuffix(' m / floor')
        self.scenario_floor_height.setToolTip(
            'Default: 3.00 m per floor. This controls scenario height without '
            'changing the reference geometry or any saved camera calibration.')
        form.addWidget(QLabel('Assumed height per floor'),1,0)
        form.addWidget(self.scenario_floor_height,1,1)
        restore_floor_height=QPushButton('Reset to 3.00 m / floor')
        restore_floor_height.clicked.connect(lambda: self.scenario_floor_height.setValue(3.0))
        form.addWidget(restore_floor_height,2,0,1,2)
        form.addWidget(QLabel('Mode'),3,0)
        self.mode=QComboBox();self.mode.addItem('Vertical extension (3→N)','ampliacion');self.mode.addItem('Full façade replacement','completo');form.addWidget(self.mode,3,1)
        self.footprint_label=QLabel('Target footprint: uncalibrated');form.addWidget(self.footprint_label,4,0,1,2)
        self.height_summary=QLabel('Scenario: 9 floors × 3.00 m = 27.00 m')
        self.height_summary.setWordWrap(True)
        form.addWidget(self.height_summary,5,0,1,2)
        self.scenario_floor_height.valueChanged.connect(self.redraw)
        layout.addWidget(g)
        g2=QGroupBox('Generation');g2l=QGridLayout(g2)
        self.seed=QSpinBox();self.seed.setRange(0,2**31-1);self.seed.setValue(1234)
        self.steps=QSpinBox();self.steps.setRange(1,100);self.steps.setValue(40)
        self.cfg=QDoubleSpinBox();self.cfg.setRange(1,20);self.cfg.setValue(5.5);self.cfg.setSingleStep(.5)
        self.control=QDoubleSpinBox();self.control.setRange(0,2);self.control.setValue(.65);self.control.setSingleStep(.05)
        for row,(label,w) in enumerate([('Seed',self.seed),('Steps',self.steps),('CFG',self.cfg),('ControlNet',self.control)]):
            g2l.addWidget(QLabel(label),row,0);g2l.addWidget(w,row,1)
        layout.addWidget(g2)
        self.facade=QCheckBox('Optional architectural façade finishing');self.facade.setChecked(True)
        self.roof=QCheckBox('Optional rooftop finishing');self.roof.setChecked(True)
        layout.addWidget(self.facade);layout.addWidget(self.roof)
        self.views_check=[]
        h=QHBoxLayout();h.addWidget(QLabel('Views:'))
        for i in range(1,5):
            c=QCheckBox(str(i));c.setChecked(True);h.addWidget(c);self.views_check.append(c)
        layout.addLayout(h)
        btn=QPushButton('Prepare scenario · CPU');btn.clicked.connect(self.prepare_preview)
        layout.addWidget(btn)
        gpu=QPushButton('GENERAR · Stable Diffusion + ControlNet (GPU)');gpu.clicked.connect(self.start_generation)
        self.gpu_button=gpu;layout.addWidget(gpu)
        restore=QPushButton('Show original photographs');restore.clicked.connect(self.show_originals);layout.addWidget(restore)
        self.run_label=QLabel('No active run');self.run_label.setWordWrap(True);layout.addWidget(self.run_label)
        layout.addStretch();self.tabs.addTab(page,'Scenario')

    def _tab_neighborhood(self):
        page=QWidget();layout=QVBoxLayout(page)
        note=QLabel('Mark the neighbor’s GROUND footprint, not its roof. Clicks are back-projected to the calibrated Z=0 plane. Approximate, optional context volumes.')
        note.setWordWrap(True);layout.addWidget(note)
        self.view_active_label=QLabel('Active view: 1');layout.addWidget(self.view_active_label)
        self.name_neighbor=QLineEdit();self.name_neighbor.setPlaceholderText('e.g., adjacent tall building');layout.addWidget(self.name_neighbor)
        h=QHBoxLayout();h.addWidget(QLabel('Approximate height (m)'))
        self.h_neighbor=QDoubleSpinBox();self.h_neighbor.setRange(1,150);self.h_neighbor.setValue(15);self.h_neighbor.setSingleStep(3)
        h.addWidget(self.h_neighbor);layout.addLayout(h)
        self.mark=QPushButton('Mark footprint · activate');self.mark.setCheckable(True)
        self.mark.toggled.connect(self.toggle_mark);layout.addWidget(self.mark)
        h=QHBoxLayout()
        undo=QPushButton('Undo point');undo.clicked.connect(self.undo_point);h.addWidget(undo)
        clear=QPushButton('Cancel');clear.clicked.connect(self.clear_draft);h.addWidget(clear)
        layout.addLayout(h)
        save=QPushButton('Save neighbor volume');save.clicked.connect(self.finish_neighbor);layout.addWidget(save)
        self.draft_info=QLabel('Marked points: 0');layout.addWidget(self.draft_info)
        self.neighbor_list=QListWidget();self.neighbor_list.currentRowChanged.connect(self.select_neighbor)
        layout.addWidget(self.neighbor_list,1)
        h=QHBoxLayout()
        edit=QPushButton('Actualizar altura');edit.clicked.connect(self.update_neighbor_height);h.addWidget(edit)
        delete=QPushButton('Eliminar');delete.clicked.connect(self.delete_neighbor);h.addWidget(delete)
        layout.addLayout(h)
        self.show_context=QCheckBox('Show neighbor volumes');self.show_context.setChecked(True)
        self.show_context.toggled.connect(self.redraw);layout.addWidget(self.show_context)
        self.use_occlusion=QCheckBox('EXPERIMENTAL: use neighbors as generation occluders')
        self.use_occlusion.setChecked(False)
        layout.addWidget(self.use_occlusion)
        warning=QLabel('⚠ Occlusions may be incorrect when neighbor footprints or heights are approximate. Keep disabled for submission unless validated.')
        warning.setWordWrap(True);layout.addWidget(warning)
        self.neighbors_tab=page
        self.tabs.addTab(page,'2.5D Neighbors')

    def _tab_cameras(self):
        """Camera Performance: diagnostics and reversible four-camera synchronization."""
        page=QWidget()
        layout=QVBoxLayout(page)
        intro=QLabel(
            'Camera Performance displays the four calibrated views. Joint camera '
            'synchronization is OPTIONAL: it refines all four poses and shared width/depth '
            'while keeping the original reference height and each view’s horizontal FOV fixed. '
            'A calibration proposal is never saved automatically.'
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        g=QGroupBox('Current camera performance')
        gl=QVBoxLayout(g)
        self.camera_metrics_table=QTableWidget(4,5)
        self.camera_metrics_table.setHorizontalHeaderLabels([
            'View','Anchors','FOV (deg)','Current RMSE (px)','Joint RMSE (px)'])
        self.camera_metrics_table.verticalHeader().setVisible(False)
        self.camera_metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.camera_metrics_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.camera_metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.camera_metrics_table.setMaximumHeight(175)
        gl.addWidget(self.camera_metrics_table)
        self.camera_text=QTextEdit()
        self.camera_text.setReadOnly(True)
        self.camera_text.setMaximumHeight(135)
        gl.addWidget(self.camera_text)
        layout.addWidget(g)

        options=QGroupBox('Synchronize four cameras · Restricted bundle adjustment')
        ol=QGridLayout(options)
        self.sync_prior=QDoubleSpinBox()
        self.sync_prior.setRange(.01,.5);self.sync_prior.setDecimals(3)
        self.sync_prior.setSingleStep(.01);self.sync_prior.setValue(.08)
        self.sync_max_change=QDoubleSpinBox()
        self.sync_max_change.setRange(.01,.5);self.sync_max_change.setDecimals(2)
        self.sync_max_change.setSingleStep(.01);self.sync_max_change.setValue(.15)
        self.sync_observed=QCheckBox('Use manually observed structural lines')
        self.sync_observed.setChecked(True)
        ol.addWidget(QLabel('Geometry prior fraction'),0,0)
        ol.addWidget(self.sync_prior,0,1)
        ol.addWidget(QLabel('Maximum W/D change'),1,0)
        ol.addWidget(self.sync_max_change,1,1)
        ol.addWidget(self.sync_observed,2,0,1,2)
        self.sync_button=QPushButton('Synchronize cameras · Propose refinement')
        self.sync_button.clicked.connect(self.propose_camera_sync)
        ol.addWidget(self.sync_button,3,0,1,2)
        self.sync_progress=QLabel('Ready. Requires saved PnP cameras and measured rear corners in at least one view.')
        self.sync_progress.setWordWrap(True)
        ol.addWidget(self.sync_progress,4,0,1,2)
        layout.addWidget(options)

        comparison=QGroupBox('Review proposed calibration')
        cl=QVBoxLayout(comparison)
        self.sync_report=QTextEdit()
        self.sync_report.setReadOnly(True)
        self.sync_report.setMinimumHeight(175)
        self.sync_report.setText('No proposal yet. Current calibration is unchanged.')
        cl.addWidget(self.sync_report)
        self.sync_preview=QCheckBox('Preview synchronized geometry in all four photographs')
        self.sync_preview.setEnabled(False)
        self.sync_preview.toggled.connect(self.redraw)
        cl.addWidget(self.sync_preview)
        self.sync_review=QCheckBox('I reviewed all four reprojections and the protected regions')
        self.sync_review.setEnabled(False)
        cl.addWidget(self.sync_review)
        buttons=QHBoxLayout()
        self.sync_accept=QPushButton('Accept and save synchronization')
        self.sync_accept.setEnabled(False)
        self.sync_accept.clicked.connect(self.accept_camera_sync)
        buttons.addWidget(self.sync_accept)
        self.sync_discard=QPushButton('Discard proposal')
        self.sync_discard.setEnabled(False)
        self.sync_discard.clicked.connect(self.discard_camera_sync)
        buttons.addWidget(self.sync_discard)
        cl.addLayout(buttons)
        layout.addWidget(comparison)

        visibility=QGroupBox('Projection overlays')
        vl=QVBoxLayout(visibility)
        self.show_vps=QCheckBox('Show saved vanishing-point guides / observed lines')
        self.show_vps.setChecked(True)
        self.show_vps.toggled.connect(self.redraw)
        vl.addWidget(self.show_vps)
        self.show_box=QCheckBox('Show original and proposed building prisms')
        self.show_box.setChecked(True)
        self.show_box.toggled.connect(self.redraw)
        vl.addWidget(self.show_box)
        layout.addWidget(visibility)
        layout.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(page)
        self.cameras_tab=scroll
        self.tabs.addTab(scroll,'Camera Performance')

    def _update_camera_performance(self):
        if not hasattr(self,'camera_metrics_table'):
            return
        candidate=self._sync_candidate
        for i in range(4):
            cam=self.cal_cameras[i] if i<len(self.cal_cameras) else None
            before=(cam.get('rmse_px') if cam else None)
            after=(candidate['proposal']['cameras'][i]['rmse_px'] if candidate else None)
            cells=[f'View {i+1}',str(len(self.cal_anchors[i])),f'{self.cal_fovs[i]:.1f}',
                   f'{before:.3f}' if isinstance(before,(float,int)) else '—',
                   f'{after:.3f}' if isinstance(after,(float,int)) else '—']
            for j,value in enumerate(cells):
                self.camera_metrics_table.setItem(i,j,QTableWidgetItem(value))
        self.camera_text.setPlainText('\n\n'.join(
            f'VIEW {i+1}  |  Horizontal FOV: {self.cal_fovs[i]:g}°  |  '
            f'Anchors: {len(self.cal_anchors[i])}  |  '
            f'Camera: {"calibrated" if self.cal_cameras[i] else "not calibrated"}'
            for i in range(4)))

    def _reset_sync_proposal(self, reason: str | None = None):
        self._sync_candidate=None
        if not hasattr(self,'sync_preview'):
            return
        self.sync_preview.setChecked(False)
        self.sync_preview.setEnabled(False)
        self.sync_review.setChecked(False)
        self.sync_review.setEnabled(False)
        self.sync_accept.setEnabled(False)
        self.sync_discard.setEnabled(False)
        self.sync_report.setPlainText(reason or 'No proposal. Current calibration is unchanged.')
        self._update_camera_performance()

    def _editor_matches_saved_calibration(self) -> bool:
        """Prevent accepting a joint proposal over unsaved editor changes."""
        if self.meta is None or len(self.meta.get('views', []))!=4:
            return False
        dims=self.meta['reference_box_m']
        if any(abs(value-float(dims[key]))>1e-8 for value,key in
               ((self.cal_w.value(),'W'),(self.cal_d.value(),'D'),(self.cal_h.value(),'H'))):
            return False
        for i,view in enumerate(self.meta['views']):
            if (self.cal_anchors[i] != view.get('anchors_px',{}) or
                abs(self.cal_fovs[i]-float(view['fov_horizontal_deg']))>1e-8 or
                self.cal_observed[i] != view.get('observed_axis_segments_px',[]) or
                self.cal_allowed[i] != view.get('allowed_polygon_px') or
                self.cal_protected[i] != view.get('protected_polygons_px',[])):
                return False
        return True

    def propose_camera_sync(self):
        if self.folder is None or self.meta is None:
            return self.error('Save an initial four-view PnP calibration first.')
        if self._sync_thread is not None and self._sync_thread.isRunning():
            return self.error('Camera synchronization is already running.')
        if self._thread is not None and self._thread.isRunning():
            return self.error('Wait for the image generation job to finish first.')
        # The saved metadata and current editor state must agree: never silently
        # synchronize a stale calibration if annotations or FOVs were modified.
        if not all(self.cal_cameras):
            return self.error('Recalibrate and save all four cameras before synchronization.')
        if not self._editor_matches_saved_calibration():
            return self.error('Unsaved calibration annotations or polygons detected. Save first.')
        if any(self.cal_anchors[i] != self.meta['views'][i]['anchors_px'] or
               abs(self.cal_fovs[i]-float(self.meta['views'][i]['fov_horizontal_deg']))>1e-8
               for i in range(4)):
            return self.error('The editor has unsaved changes. Save them before synchronization.')
        dims=self.meta['reference_box_m']
        if any(abs(value-float(dims[key]))>1e-8 for value,key in
               ((self.cal_w.value(),'W'),(self.cal_d.value(),'D'),(self.cal_h.value(),'H'))):
            return self.error('Reference dimensions have unsaved changes. Save them first.')
        if self._sync_candidate is not None:
            self._reset_sync_proposal('Previous proposal discarded; computing a new one.')
        self.sync_button.setEnabled(False)
        self.sync_progress.setText('Running joint optimization; the current calibration remains unchanged…')
        self._sync_source_site=self.folder.resolve()
        options={
            'geometry_prior_fraction':float(self.sync_prior.value()),
            'allowed_dimension_change':float(self.sync_max_change.value()),
            'use_observed_lines':bool(self.sync_observed.isChecked()),
            'max_nfev':120,
        }
        self._sync_thread=QThread(self)
        self._sync_worker=CameraSyncWorker(self.folder,options)
        self._sync_worker.moveToThread(self._sync_thread)
        self._sync_thread.started.connect(self._sync_worker.run)
        self._sync_worker.completed.connect(self._camera_sync_ready)
        self._sync_worker.failed.connect(self._camera_sync_failed)
        self._sync_worker.completed.connect(self._sync_thread.quit)
        self._sync_worker.failed.connect(self._sync_thread.quit)
        self._sync_thread.finished.connect(self._sync_worker.deleteLater)
        self._sync_thread.finished.connect(self._sync_thread.deleteLater)
        self._sync_thread.finished.connect(self._camera_sync_thread_finished)
        self._sync_thread.start()

    def _camera_sync_thread_finished(self):
        self._sync_thread=None
        self._sync_worker=None
        self.sync_button.setEnabled(True)

    def _camera_sync_failed(self, message: str):
        self.sync_progress.setText('Synchronization was not applied.')
        self._reset_sync_proposal('Proposal failed: '+message)
        self.append_log('Camera synchronization: '+message)
        self.sync_button.setEnabled(True)

    def _camera_sync_ready(self, candidate: dict):
        if self.folder is None or self.folder.resolve()!=self._sync_source_site:
            self.append_log('Ignoring a camera proposal from a previously opened project.')
            return
        from generabuild.geometry import sha256_file as _hash_cal
        if _hash_cal(self.folder/'calibration_multiview.json') != candidate['source_sha256']:
            self._camera_sync_failed('Saved calibration changed while optimizing. Retry.')
            return
        if not self._editor_matches_saved_calibration():
            self._camera_sync_failed('Calibration was edited during optimization. Save changes and retry.')
            return
        self._sync_candidate=candidate
        report=candidate['proposal']['report']
        before=report['initial_pooled_rmse_px'];after=report['refined_pooled_rmse_px']
        delta=100*(after/before-1) if before>1e-9 else 0.
        lines=[
            'RESTRICTED JOINT BUNDLE ADJUSTMENT',
            f'Optimization converged: {report["solver_success"]}  |  Iterations: {report["nfev"]}',
            f'Pooled reprojection RMSE: {before:.3f} → {after:.3f} px ({delta:+.1f}%)',
            f'Observed correspondences: {report["total_observations"]}',
            f'Views with real depth evidence: {report["n_views_with_depth_evidence"]}',
            f'Width change: {report["width_change_pct"]:+.2f}%',
            f'Depth change: {report["depth_change_pct"]:+.2f}%',
            f'Original building height: FIXED at {candidate["proposal"]["reference_dimensions_m"]["H"]:.2f} m',
            'FOV by camera: '+', '.join(f'{x:.1f}°' for x in report['fovs_locked'])+' (LOCKED)',
            f'Observed manual lines: {report["observed_manual_segments"]}',
            '',
            'Important: fitting RMSE is not independent validation.',
            'The four projected building edges and surrounding buildings require visual review.',
            'Existing 2.5D neighbor volumes need revalidation if W or D changes.',
        ]
        if report['warnings']:
            lines.extend(['','WARNINGS:']+['• '+x for x in report['warnings']])
        self.sync_report.setPlainText('\n'.join(lines))
        can_accept=(report['solver_success'] and after <= before*1.05 +.1)
        self.sync_accept.setEnabled(can_accept)
        self.sync_review.setEnabled(can_accept)
        self.sync_discard.setEnabled(True)
        self.sync_preview.setEnabled(True)
        self.sync_preview.setChecked(True)
        self._update_camera_performance()
        self.sync_progress.setText('Proposal ready. Compare the four photographs, then accept or discard.')
        self.append_log(f'Joint calibration proposal: RMSE {before:.3f} → {after:.3f} px; saved cameras unchanged.')

    def accept_camera_sync(self):
        if self._sync_candidate is None:
            return self.error('There is no camera synchronization proposal to accept.')
        if not self._editor_matches_saved_calibration():
            return self.error('Unsaved calibration changes detected. Discard and synchronize again.')
        if not self.sync_review.isChecked():
            return self.error('Review the projected original building and protected regions first.')
        answer=QMessageBox.question(
            self,'Confirm camera synchronization',
            'This will update all four calibrated poses and the shared building width/depth.\n'
            'The original building height and each FOV will remain unchanged.\n'
            'The previous calibration JSON will be backed up; masks and previews will be regenerated.\n'
            'Any manually defined 2.5D neighbors should be revalidated.\n\n'
            'Save the synchronized cameras?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer!=QMessageBox.StandardButton.Yes:
            return
        try:
            result=accept_camera_sync_proposal(
                self.folder,self._sync_candidate,reviewed=True)
            audit=result['audit_path']
            self.open_site(self.folder)
            self._reset_sync_proposal('Synchronized calibration saved. Backup and audit report created.')
            self.tabs.setCurrentWidget(self.cameras_tab)
            self.sync_progress.setText('Synchronized calibration saved and masks regenerated.')
            self.append_log('Joint calibration accepted: '+audit)
        except Exception as exc:
            self.error(str(exc))

    def discard_camera_sync(self):
        self._reset_sync_proposal('Proposal discarded. Your saved cameras were not changed.')
        self.sync_progress.setText('Previous calibration retained.')
        self.redraw()

    def _tab_results(self):
        """Result Viewer: saved runs, before/after comparison, and four-view montage."""
        page=QWidget();layout=QVBoxLayout(page)
        heading=QLabel('RESULT VIEWER · Scenario history')
        heading.setFont(QFont('Sans',12,QFont.Weight.Bold));layout.addWidget(heading)
        g=QGroupBox('Saved run');box=QVBoxLayout(g)
        self.result_run_picker=QComboBox()
        self.result_run_picker.setMinimumWidth(220)
        self.result_run_picker.setToolTip('Select a saved run for the active project')
        self.result_run_picker.currentIndexChanged.connect(self._result_run_changed)
        box.addWidget(self.result_run_picker)
        refresh=QPushButton('⟳ Refresh run history')
        refresh.clicked.connect(self.refresh_result_runs)
        box.addWidget(refresh)
        layout.addWidget(g)

        g=QGroupBox('Result display');box=QVBoxLayout(g)
        self.result_mode=QComboBox()
        for title,value in (
            ('Generated · Final image','final'),
            ('Compare · Original | Generated','compare'),
            ('Original · Unmodified','original'),
            ('Base · Before enhancements','base'),
            ('Geometry preview · CPU','preview'),
        ):
            self.result_mode.addItem(title,value)
        self.result_mode.currentIndexChanged.connect(self._result_display_changed)
        box.addWidget(self.result_mode)
        self.result_split_label=QLabel('Comparison: 50% original | 50% generated')
        box.addWidget(self.result_split_label)
        self.result_split=QSlider(Qt.Orientation.Horizontal)
        self.result_split.setRange(0,100);self.result_split.setValue(50)
        self.result_split.setToolTip('Move divider; original on the left, generated on the right')
        self.result_split.valueChanged.connect(self._result_split_changed)
        box.addWidget(self.result_split)
        note=QLabel('The photo workspace always has maximum available space. '
                    'Drag the divider to resize the controls. Use View layout in the top toolbar '
                    'for 2×2 or a single enlarged photo. In Compare mode, move the reveal slider.')
        note.setWordWrap(True);box.addWidget(note)
        layout.addWidget(g)

        self.result_status=QLabel('Open a project and select a saved run.')
        self.result_status.setWordWrap(True);layout.addWidget(self.result_status)
        self.result_info=QTextEdit()
        self.result_info.setReadOnly(True);self.result_info.setMaximumHeight(135)
        layout.addWidget(self.result_info)
        h=QHBoxLayout()
        original=QPushButton('Show original');original.clicked.connect(self._result_show_original)
        h.addWidget(original)
        final=QPushButton('Show generated');final.clicked.connect(self.show_results)
        h.addWidget(final)
        layout.addLayout(h)
        h=QHBoxLayout()
        montage=QPushButton('Save 2×2 montage…')
        montage.clicked.connect(self.save_results_montage)
        h.addWidget(montage)
        folder=QPushButton('Open folder');folder.clicked.connect(self.open_result_folder)
        h.addWidget(folder)
        layout.addLayout(h)
        export=QPushButton('Export run ZIP…')
        export.clicked.connect(self.export_run);layout.addWidget(export)
        self.log=QTextEdit();self.log.setReadOnly(True)
        self.log.setMaximumHeight(175)
        layout.addWidget(QLabel('Run log'))
        layout.addWidget(self.log,1)
        self.results_tab=page
        self.tabs.addTab(page,'Results')
        self._result_display_changed()

    def refresh_result_runs(self, preferred: Path | None = None):
        """Read saved runs without modifying calibrations or source images."""
        previous = preferred or self.current_run
        try:
            self._result_records = list_runs(self.folder) if self.folder else []
        except Exception as exc:
            self._result_records=[]
            self.append_log(f'Could not list saved runs: {exc}')
        self.result_run_picker.blockSignals(True)
        self.result_run_picker.clear()
        for record in self._result_records:
            self.result_run_picker.addItem(record.label,str(record.path))
        desired = str(Path(previous).resolve()) if previous else None
        selected = self.result_run_picker.findData(desired) if desired else -1
        if selected >= 0:
            self.result_run_picker.setCurrentIndex(selected)
        elif self._result_records:
            self.result_run_picker.setCurrentIndex(0)
        self.result_run_picker.blockSignals(False)
        self._result_run_changed()

    def _result_run_changed(self,*_):
        selection = self.result_run_picker.currentData()
        self.result_set = None
        if selection and self.folder:
            try:
                run=Path(selection).resolve()
                # Solo se admiten ejecuciones que pertenecen al sitio actual.
                if run.parent != (self.folder.resolve()/'runs'):
                    raise ValueError('The selected run is outside the current project')
                self.current_run=run
                self.result_set=load_result_set(run,self.folder)
            except Exception as exc:
                self.append_log(f'Could not open run: {exc}')
        else:
            self.current_run=None
        if self.tabs.currentWidget() is self.results_tab:
            self._render_result_view()

    def _result_display_changed(self,*_):
        mode=self.result_mode.currentData()
        compare=mode=='compare'
        self.result_split.setEnabled(compare)
        self.result_split_label.setEnabled(compare)
        if self.tabs.currentWidget() is self.results_tab:
            self._render_result_view()

    def _result_split_changed(self,value:int):
        self.result_split_label.setText(
            f'Comparison: {value}% original | {100-value}% generated')
        if self.tabs.currentWidget() is self.results_tab and self.result_mode.currentData()=='compare':
            for panel in self.panels:
                panel.viewer.set_result_percent(value)

    def _result_show_original(self):
        self.result_mode.setCurrentIndex(self.result_mode.findData('original'))
        self.tabs.setCurrentWidget(self.results_tab)
        self._render_result_view()

    def _render_result_view(self):
        """Render results without calibration or 3D prism overlays."""
        if self.tabs.currentWidget() is not self.results_tab:return
        if not self.paths:
            self.result_status.setText('No photos available. Create or open a project.')
            return
        mode=str(self.result_mode.currentData() or 'final')
        results=self.result_set
        n=0
        unavailable=[]
        for i,panel in enumerate(self.panels,1):
            original=(results.original.get(i) if results else None) or self.paths[i-1]
            generated=(results.path_for(i,mode) if results else None)
            if mode=='original':generated=None
            if generated and mode!='original':n+=1
            elif mode!='original':unavailable.append(str(i))
            kind = ('ORIGINAL' if mode=='original' else
                    'COMPARISON' if mode=='compare' and generated else
                    'GENERATED' if mode=='final' and generated else
                    'BASE' if mode=='base' and generated else
                    'PREVIEW CPU' if mode=='preview' and generated else
                    'NO RESULT · ORIGINAL')
            panel.title.setText(f'VIEW {i} · {kind}')
            try:
                panel.viewer.show_result(original,generated,
                    mode='compare' if mode=='compare' and generated else
                         ('original' if generated is None else mode),
                    split_percent=self.result_split.value())
            except (OSError,ValueError) as exc:
                panel.viewer.show_result(original,mode='original')
                if generated and mode!='original':n=max(0,n-1)
                unavailable.append(str(i))
                self.append_log(f'View {i}: {exc}')
            panel.set_active(i==self.active_view)
        run_name=results.run_dir.name if results else 'no run'
        if mode=='original':
            msg=f'Original project photographs · run {run_name}'
        elif unavailable:
            msg=(f'{run_name} · {n}/4 views available in {mode}. '
                 f'Missing results: {", ".join(sorted(set(unavailable)))}; showing original photos.')
        else:
            msg=f'{run_name} · all four views available · {mode}'
        if mode=='preview':msg+=' · Geometry previews; these are NOT AI-generated photographs.'
        self.result_status.setText(msg)
        if results:
            cfg=results.config
            report=results.evaluation
            lines=[f'Scenario: {cfg.get("floors","?")} floors · mode {"Vertical extension" if cfg.get("mode")=="ampliacion" else "Full façade replacement" if cfg.get("mode")=="completo" else cfg.get("mode","?")}',
                   f'Seed: {cfg.get("seed","n/a")}',
                   'Views with final images: '+(', '.join(map(str,results.available_views)) or 'none')]
            if report:
                preserved=report.get('context_preservation_pass')
                lines.append('Context preservation: '+
                    ('PASS' if preserved is True else 'FAIL' if preserved is False else 'n/a'))
                for row in report.get('views',[]):
                    if isinstance(row,dict):
                        lines.append(f'View {row.get("view","?")}: '
                            f'{row.get("pixels_outside_mask_changed","?")} changed pixels outside mask')
            if results.warnings:
                lines.append(f'File warnings: {len(results.warnings)} (check run manifest)')
            self.result_info.setPlainText('\n'.join(lines))
        else:
            self.result_info.setPlainText('No saved runs. Prepare or generate a scenario in the Scenario tab.')

    def save_results_montage(self):
        if self.result_set is None:
            self.error('Select a saved run');return
        mode=str(self.result_mode.currentData() or 'final')
        if mode in ('final','compare') and not self.result_set.final:
            self.error('This run contains previews only. Generate photographs before exporting finals.');return
        path,_=QFileDialog.getSaveFileName(self,'Export four-view montage',
            str(self.result_set.run_dir / f'montaje_{mode}_2x2.png'), 'PNG (*.png)')
        if not path:return
        try:
            export_montage(self.result_set,Path(path),mode,self.result_split.value())
            self.append_log(f'2×2 montage exported: {path}')
        except Exception as exc:self.error(str(exc))

    def error(self,message):
        QMessageBox.critical(self,'GeneraBuild — Error',str(message))
        self.append_log('ERROR: '+str(message))

    def append_log(self,message):
        self.log.append(str(message))
        self.status.showMessage(str(message)[:200],8000)

    # ---- Creación de sitios y edición manual de calibración ----
    def create_project(self):
        dialog=NewProjectDialog(self)
        if dialog.exec()==QDialog.DialogCode.Accepted and dialog.result_folder:
            try:
                self.open_draft(dialog.result_folder)
                self.tabs.setCurrentWidget(self.calibration_tab)
                self.append_log('Project created. Annotate four views, calibrate, then save.')
            except Exception as exc:self.error(str(exc))

    def open_draft(self,folder:Path):
        folder=Path(folder).resolve()
        draft=load_draft(folder)
        paths=image_paths(folder)
        self._reset_sync_proposal()
        self.folder=folder;self.meta=None;self.paths=paths
        self.neighbors=[];self.current_run=None;self.draft_xy=[];self.draft_uv=[]
        self.display_override={}
        self.mark.setChecked(False)
        self.cal_focus.setChecked(False)
        self._apply_photo_layout('grid',force=True)
        self.cal_mark.setChecked(False)
        for panel,path in zip(self.panels,paths):panel.viewer.set_image(path)
        self.site_title.setText(f'GeneraBuild Studio / {folder.name}  [NOT CALIBRATED]')
        self.footprint_label.setText('Calibration pending · generation is not yet enabled')
        self.camera_text.setText('New project: cameras are missing. Open Calibration, '
                                'mark at least four points per view, then calibrate.')
        self._load_cal_state(draft)
        self.cal_approved.setChecked(False)
        self._refresh_neighbor_list();self.refresh_result_runs();self.redraw()
        self.tabs.setCurrentWidget(self.calibration_tab)
        self.append_log(f'New project: {folder} · four photographs copied · not yet calibrated')

    def _load_cal_state(self,record:dict):
        self._cal_updating=True
        try:
            ref=record['reference_box_m']
            for widget,value in ((self.cal_w,ref['W']),(self.cal_d,ref['D']),
                                 (self.cal_h,ref['H']),
                                 (self.cal_floor_h,record.get('floor_height_m',3))):
                widget.blockSignals(True);widget.setValue(float(value));widget.blockSignals(False)
            self.floors.blockSignals(True)
            self.floors.setValue(int(record.get('num_floors',9)))
            self.floors.blockSignals(False)
            views=record['views']
            self.cal_anchors=[copy.deepcopy(v.get('anchors_px',{})) for v in views]
            self.cal_fovs=[float(v.get('fov_horizontal_deg',55)) for v in views]
            self.cal_allowed=[copy.deepcopy(v.get('allowed_polygon_px')) for v in views]
            self.cal_protected=[copy.deepcopy(v.get('protected_polygons_px',[])) for v in views]
            self.cal_guides=[copy.deepcopy(v.get('manual_vp_guides_px',[])) for v in views]
            self.cal_observed=[copy.deepcopy(v.get('observed_axis_segments_px',[])) for v in views]
            self.cal_cameras=[]
            for v in views:
                if v.get('P') is not None:
                    self.cal_cameras.append({
                        'ok':True,'P':v['P'],'K':v['K'],'rvec':v['rvec'],
                        'tvec':v['tvec'],'camera_center_m':v.get('camera_center_m'),
                        'rmse_px':v.get('reprojection_rmse_px'),
                        'rmse_diagonal_pct':v.get('reprojection_rmse_diagonal_pct'),
                        'per_vertex_error_px':v.get('per_vertex_error_px',{}),
                        'planar':v.get('planar_anchors',False),
                        'warnings':v.get('warnings',[]),
                    })
                else:self.cal_cameras.append(None)
            self.cal_detected=[[] for _ in range(4)]
            self.cal_poly_draft=[];self.cal_poly_view=None;self.cal_line_start=None
            self.cal_view.setCurrentIndex(0)
            self.active_view=1
            self.cal_fov.blockSignals(True);self.cal_fov.setValue(self.cal_fovs[0]);self.cal_fov.blockSignals(False)
            self.cal_anchor.setCurrentIndex(0)
            self.cal_approved.setChecked(False)
        finally:self._cal_updating=False
        self._update_cal_status()

    def _update_cal_status(self):
        if not hasattr(self,'cal_status'):return
        i=self.active_view-1
        n=len(self.cal_anchors[i])
        cam=self.cal_cameras[i] if i<len(self.cal_cameras) else None
        label=f'View {i+1}: {n} anchors  · FOV {self.cal_fovs[i]:g}°'
        if cam and cam.get('ok'):
            rmse=cam.get('rmse_px')
            label+=f" · PnP: {'%.2f px' % rmse if isinstance(rmse,(int,float)) else 'loaded'}"
        else:label+=' · NOT CALIBRATED'
        self.cal_status.setText(label)
        labels=[]
        for k,c in enumerate(self.cal_cameras,1):
            err=c.get('rmse_px') if c else None
            labels.append(f"View {k}: {len(self.cal_anchors[k-1])} anchors, " +
                          (f'RMSE {float(err):.2f} px' if isinstance(err,(int,float))
                           else ('PnP loaded' if c and c.get('ok') else 'pending')))
        self.cal_metrics.setText('\n'.join(labels)+'\n'
                    'Review edges and depth visually; RMSE uses fitted anchors.')
        self._update_camera_performance()

    def _selected_cal_view(self)->int:
        return int(self.cal_view.currentData() or 1)-1

    def cal_select_view(self,_=None):
        if self._cal_updating:return
        i=self._selected_cal_view()
        self.active_view=i+1
        self.cal_fov.blockSignals(True);self.cal_fov.setValue(self.cal_fovs[i]);self.cal_fov.blockSignals(False)
        self._select_first_missing_anchor(i)
        self.cal_poly_draft=[];self.cal_poly_view=None;self.cal_line_start=None
        self._update_cal_status()
        self._apply_photo_layout(self.photo_layout_mode)
        self.redraw()

    def _select_first_missing_anchor(self,i:int):
        for n in (0,1,4,5,2,3,6,7):
            if str(n) not in self.cal_anchors[i]:
                j=self.cal_anchor.findData(n)
                if j>=0:self.cal_anchor.setCurrentIndex(j)
                return
        self.cal_anchor.setCurrentIndex(0)

    def _cal_invalidate(self,i:int|None=None):
        self._reset_sync_proposal("Calibration edited; previous synchronization proposal discarded.")
        if i is None:self.cal_cameras=[None]*4
        else:self.cal_cameras[i]=None
        self.cal_approved.setChecked(False)
        self._update_cal_status();self.redraw()

    def cal_geometry_changed(self,*_):
        if not self._cal_updating:self._cal_invalidate()

    def cal_fov_changed(self,value:float):
        if self._cal_updating:return
        i=self._selected_cal_view()
        self.cal_fovs[i]=float(value)
        self._cal_invalidate(i)

    def _photo_layout_choice_changed(self,_=None):
        mode=self.photo_layout_choice.currentData()
        if mode:self._apply_photo_layout(str(mode))

    def _photo_focus_picker_changed(self,_=None):
        selected=self.photo_focus_picker.currentData()
        if selected is not None:self.activate_view(int(selected))

    def _sync_photo_layout_controls(self):
        """Synchronize toolbar and calibration checkbox without recursive Qt signals."""
        desired=0 if self.photo_layout_mode=='grid' else 1
        for control,value in ((self.photo_layout_choice,desired),
                              (self.photo_focus_picker,self.active_view-1)):
            was_blocked=control.blockSignals(True)
            control.setCurrentIndex(value)
            control.blockSignals(was_blocked)
        self.photo_focus_picker.setEnabled(self.photo_layout_mode=='single')
        if hasattr(self,'cal_focus'):
            blocked=self.cal_focus.blockSignals(True)
            self.cal_focus.setChecked(self.photo_layout_mode=='single')
            self.cal_focus.blockSignals(blocked)

    def _apply_photo_layout(self,mode:str,*,force:bool=False):
        """Switch layout without recreating PhotoView instances or losing annotations.

        One view spans both rows and columns; the grid restores four panels.
        """
        plan=plan_photo_layout(mode,self.active_view)
        self.photo_layout_mode=mode
        if not force and plan==self._applied_photo_layout:
            self._sync_photo_layout_controls()
            return
        gl=self.photo_grid_layout
        for panel in self.panels:
            gl.removeWidget(panel)
            panel.hide()
        for index,(row,col,rowspan,colspan) in plan.positions.items():
            panel=self.panels[index-1]
            gl.addWidget(panel,row,col,rowspan,colspan)
            panel.show()
        self._applied_photo_layout=plan
        self._sync_photo_layout_controls()
        # Esperar a que Qt recalcule la cuadrícula antes de encajar las fotos.
        QTimer.singleShot(0,self.fit_visible_photos)

    def fit_visible_photos(self):
        for panel in self.panels:
            if panel.isVisible():panel.viewer.reset_view()

    def toggle_photo_layout_from_view(self,index:int):
        """Double-click toggles between 2×2 and single view outside annotation mode."""
        self.activate_view(index)
        mode='grid' if self.photo_layout_mode=='single' else 'single'
        self._apply_photo_layout(mode)

    def cal_focus_view(self,active:bool):
        # Alias para el checkbox heredado de la versión v0.2.
        self._apply_photo_layout('single' if active else 'grid')

    def tab_changed(self,_index):
        in_cal=self.tabs.currentWidget() is self.calibration_tab
        if not in_cal and self.cal_mark.isChecked():
            self.cal_mark.setChecked(False)
        if self.tabs.currentWidget() is not self.neighbors_tab and self.mark.isChecked():
            self.mark.setChecked(False)
        # Resultados utiliza fotografías limpias sin overlays geométricos.
        in_results=self.tabs.currentWidget() is self.results_tab
        if in_results:
            self.refresh_result_runs()
        self._sync_mark_modes()
        self.redraw()

    def _sync_mark_modes(self):
        for panel in self.panels:
            panel.viewer.set_mark_mode(self.cal_mark.isChecked() or self.mark.isChecked())

    def toggle_cal_mark(self,active:bool):
        if active:
            self.mark.setChecked(False)
            self.tabs.setCurrentWidget(self.calibration_tab)
        self.cal_mark.setText('TOOL ACTIVE · click photograph' if active
                              else 'Activate tool (click photograph)')
        self._sync_mark_modes()

    def on_photo_click(self,index:int,x:float,y:float):
        if self.cal_mark.isChecked() and self.cal_poly_draft and self.cal_poly_view!=index:
            self.append_log('Finish the polygon in its original photo or click Undo.')
            return
        if self.cal_mark.isChecked() and index != self.active_view:
            # No crear una ancla errónea durante el propio cambio de vista:
            # el selector de vértices cambia a la primera esquina pendiente.
            self.activate_view(index)
            self.append_log(f'View {index} activated. Confirm the vertex ID and click again.')
            return
        self.activate_view(index)
        if self.cal_mark.isChecked():self.cal_mark_click(index,x,y)
        elif self.mark.isChecked():self.add_ground_point(index,x,y)

    def cal_mark_click(self,index:int,x:float,y:float):
        if self.folder is None:return
        if self._sync_candidate is not None:
            self._reset_sync_proposal('Editor changed: previous camera proposal discarded.')
        i=index-1
        point={'x':round(float(x),3),'y':round(float(y),3)}
        mode=self.cal_mode.currentData()
        if mode=='anchor':
            anchor=int(self.cal_anchor.currentData())
            self.cal_anchors[i][str(anchor)]=point
            self._select_first_missing_anchor(i)
            self._cal_invalidate(i)
        elif mode in ('allowed','protected'):
            if self.cal_poly_view is not None and self.cal_poly_view!=index:
                self.error('Finish the polygon in the same photograph.');return
            self.cal_poly_view=index;self.cal_poly_draft.append(point)
            if len(self.cal_poly_draft)>90:
                self.cal_poly_draft.pop();self.error('Maximum 90 points per polygon');return
            self.cal_approved.setChecked(False)
        elif mode=='observed':
            if self.cal_line_start and self.cal_line_start[0]!=index:
                self.error('Mark both edge endpoints in the same view.');return
            if self.cal_line_start is None:
                self.cal_line_start=(index,point)
            else:
                from math import hypot
                first=self.cal_line_start[1]
                if hypot(point['x']-first['x'],point['y']-first['y'])<8:
                    self.error('The edge must be at least 8 pixels long');return
                self.cal_observed[i].append({'axis':int(self.cal_axis.currentData()),
                                              'p1':first,'p2':point})
                self.cal_line_start=None
        elif mode=='guide':
            self.cal_guides[i].append({'axis':int(self.cal_axis.currentData()),
                                      'source':'camera','point':point})
        self._update_cal_status();self.redraw()

    def cal_finish_polygon(self):
        mode=self.cal_mode.currentData()
        if mode not in ('allowed','protected'):
            self.error('Select Permitted area or Protect context first.');return
        if len(self.cal_poly_draft)<3:
            self.error('A polygon requires at least three vertices.');return
        i=self.cal_poly_view-1 if self.cal_poly_view is not None else self._selected_cal_view()
        if mode=='allowed':self.cal_allowed[i]=copy.deepcopy(self.cal_poly_draft)
        else:self.cal_protected[i].append(copy.deepcopy(self.cal_poly_draft))
        self.cal_poly_draft=[];self.cal_poly_view=None
        self.cal_approved.setChecked(False)
        self.redraw()

    def cal_undo(self):
        if self.cal_poly_draft:self.cal_poly_draft.pop()
        elif self.cal_line_start:self.cal_line_start=None
        else:
            self.append_log('No incomplete points to undo. Select a vertex to delete it.')
        self.redraw()

    def cal_delete_anchor(self):
        i=self._selected_cal_view()
        anchor=str(self.cal_anchor.currentData())
        if anchor in self.cal_anchors[i]:
            del self.cal_anchors[i][anchor]
            self._cal_invalidate(i)

    def cal_delete_protected(self):
        i=self._selected_cal_view()
        if self.cal_protected[i]:
            self.cal_protected[i].pop()
            self.cal_approved.setChecked(False);self.redraw()

    def cal_detect_lines(self):
        if not self.paths:return
        i=self._selected_cal_view()
        try:
            self.cal_detected[i]=detect_segments(self.paths[i],max_segments=100)
            self.append_log(f'View {i+1}: {len(self.cal_detected[i])} line segments detected by OpenCV. Diagnostic only, not annotations.')
            self.redraw()
        except Exception as exc:self.error(str(exc))

    def _calibrate_one(self,i:int):
        if len(self.cal_anchors[i])<4:
            self.cal_cameras[i]=None
            raise ValueError(f'View {i+1}: mark at least four observable physical corners')
        with Image.open(self.paths[i]) as im:w,h=im.size
        cam=calibrate_view(self.cal_anchors[i],w,h,
                          self.cal_w.value(),self.cal_d.value(),
                          self.cal_h.value(),self.cal_fovs[i])
        if not cam.get('ok'):
            self.cal_cameras[i]=None
            raise ValueError(f"View {i+1}: {cam.get('error','PnP failed')}")
        self.cal_cameras[i]=cam
        return cam

    def calibrate_selected(self):
        if not self.paths:return self.error('Create or open a project first')
        i=self._selected_cal_view()
        try:
            cam=self._calibrate_one(i)
            self.cal_approved.setChecked(False)
            self._update_cal_status();self.redraw()
            warning='; '.join(cam.get('warnings',[]))
            self.append_log(f"View {i+1}: PnP calibrated; RMSE={cam['rmse_px']:.2f} px. {warning}")
        except Exception as exc:self._update_cal_status();self.error(str(exc))

    def calibrate_four(self):
        if not self.paths:return self.error('Create or open a project first')
        failed=[]
        for i in range(4):
            try:self._calibrate_one(i)
            except Exception as exc:failed.append(str(exc))
        self.cal_approved.setChecked(False)
        self._update_cal_status();self.redraw()
        if failed:
            self.error('Incomplete calibration:\n'+'\n'.join(failed))
        else:
            scores=', '.join(f'{float(c["rmse_px"]):.2f}' for c in self.cal_cameras)
            self.append_log(f'Cuatro poses estimadas. RMSE de ajuste [px]: {scores}. Review projected geometry before saving.')

    def _cal_editor_data(self):
        return {
            'reference_box_m':{'W':self.cal_w.value(),'D':self.cal_d.value(),'H':self.cal_h.value()},
            'floor_height_m':self.cal_floor_h.value(),
            'num_floors':self.floors.value(),
            'views':[{
                'anchors_px':self.cal_anchors[i], 'fov_horizontal_deg':self.cal_fovs[i],
                'allowed_polygon_px':self.cal_allowed[i],
                'protected_polygons_px':self.cal_protected[i],
                'manual_vp_guides_px':self.cal_guides[i],
                'observed_axis_segments_px':self.cal_observed[i],
            } for i in range(4)],
        }

    def cal_save_draft(self):
        if self.folder is None:return self.error('Create a project first')
        if not (self.folder/DRAFT_FILENAME).exists():
            self.append_log('The project is already calibrated. To modify it, use Save calibration with confirmation.')
            return
        try:
            save_draft(self.folder,self._cal_editor_data())
            self.append_log('Draft saved; you can close the app without losing annotations.')
        except Exception as exc:self.error(str(exc))

    def cal_save_final(self):
        if self.folder is None:return self.error('Create or open a project first')
        if not self.cal_approved.isChecked():
            return self.error('Confirm that you reviewed all four projections and protected regions before saving')
        if not all(c and c.get('ok') for c in self.cal_cameras):
            return self.error('Calibrate all four views successfully with OpenCV')
        state=self._cal_editor_data()
        state.update({
            'ref_W':self.cal_w.value(),'ref_D':self.cal_d.value(),'ref_H':self.cal_h.value(),
            'floor_height_m':self.cal_floor_h.value(),'num_floors':self.floors.value(),
            'views_anchors':self.cal_anchors,'fov_deg_by_view':self.cal_fovs,
            'protected_polygons_px':self.cal_protected,'allowed_polygons_px':self.cal_allowed,
            'guide_lines':self.cal_guides,'observed_lines':self.cal_observed,
            'visual_inspection_confirmed':True,
        })
        try:
            if (self.folder/DRAFT_FILENAME).exists():
                save_draft(self.folder,self._cal_editor_data())
            metadata=save_calibration(state,self.cal_cameras,self.paths,self.folder)
            self.open_site(self.folder)
            self.tabs.setCurrentWidget(self.calibration_tab)
            self.append_log('CALIBRATION SAVED: all four cameras are ready for scenario generation.')
            return metadata
        except Exception as exc:self.error(str(exc))

    def choose_site(self):
        path=QFileDialog.getExistingDirectory(self,'Open GeneraBuild project or draft',str(ROOT/'workspace/sites'))
        if not path:return
        try:
            folder=Path(path)
            if (folder/'calibration_multiview.json').is_file():self.open_site(folder)
            elif (folder/DRAFT_FILENAME).is_file():self.open_draft(folder)
            else:raise ValueError('Select a folder containing calibration_multiview.json or project_draft.json')
        except Exception as exc:self.error(str(exc))

    def import_zip(self):
        filename,_=QFileDialog.getOpenFileName(self,'Import Colab calibration ZIP',str(Path.home()),'ZIP (*.zip)')
        if not filename:return
        name,okay=QInputDialog.getText(self,'Project identifier','Project folder name (e.g., guanajuato8):')
        if not okay or not name.strip():return
        try:
            folder=import_colab_zip(Path(filename).read_bytes(),name)
            self.open_site(folder)
        except Exception as exc:self.error(str(exc))

    def open_site(self,folder:Path):
        folder=Path(folder).resolve();metadata=load_metadata(folder);paths=image_paths(folder,metadata)
        for path,view in zip(paths,metadata['views']):
            if view.get('image_sha256') and sha256_file(path)!=view['image_sha256']:
                raise ValueError(f'The photograph {path.name} does not match the calibration checksum')
        self._reset_sync_proposal()
        self.folder=folder;self.meta=metadata;self.paths=paths
        self._load_cal_state(metadata)
        self.neighbors=load_context(folder)
        self.current_run=None;self.display_override={};self.draft_xy=[];self.draft_uv=[]
        self.mark.setChecked(False)
        self.cal_mark.setChecked(False)
        self.cal_focus.setChecked(False)
        self._apply_photo_layout('grid',force=True)
        for p,path in zip(self.panels,paths):p.viewer.set_image(path)
        self.site_title.setText(f'GeneraBuild Studio  /  {folder.name}'+('  [DEMO SINTÉTICA]' if folder.name=='synthetic_demo' else ''))
        ref=metadata['reference_box_m']
        self.footprint_label.setText(f"Fixed footprint: {ref['W']} × {ref['D']} m  |  H₀={ref['H']} m")
        for i,view in enumerate(metadata['views'],1):
            self.panels[i-1].title.setText(f'VIEW {i}  ·  FOV {view.get("fov_horizontal_deg",0):g}°')
        self._update_camera_performance()
        self._refresh_neighbor_list();self.refresh_result_runs();self.redraw()
        self.append_log(f'Site opened: {folder} | {len(self.neighbors)} 2.5D neighbors')

    def _refresh_scenario_height(self):
        """Show the actual proposed height; never silently inherit a stale 5 m assumption."""
        if not hasattr(self,'height_summary'):
            return
        floors=int(self.floors.value())
        meters=float(self.scenario_floor_height.value())
        total_height=floors*meters
        if self.meta is not None:
            original=float(self.meta['reference_box_m']['H'])
            stored=float(self.meta.get('floor_height_m',3.0))
        else:
            original=float(self.cal_h.value()) if hasattr(self,'cal_h') else 9.0
            stored=None
        message=(f'Scenario height: {floors} floors × {meters:.2f} m = '
                 f'{total_height:.2f} m   |   Original H₀: {original:.2f} m')
        if stored is not None and abs(stored-meters)>1e-6:
            message+=(f'\nNote: saved calibration says {stored:.2f} m/floor; '
                      f'this scenario deliberately uses {meters:.2f} m/floor. '
                      'The calibrated cameras and original geometry remain unchanged.')
        if total_height <= original:
            message+='\nWARNING: target height must exceed the original reference height.'
        self.height_summary.setText(message)

    def target_vertices(self):
        box=self.meta['reference_box_m']
        return box_vertices(float(box['W']),float(box['D']),
                            self.floors.value()*float(self.scenario_floor_height.value()))

    def redraw(self,*_):
        self._refresh_scenario_height()
        if not self.paths:return
        if self.tabs.currentWidget() is self.results_tab:
            self._render_result_view()
            return
        in_cal=self.tabs.currentWidget() is self.calibration_tab
        in_sync_preview=(self.tabs.currentWidget() is self.cameras_tab and
                         self._sync_candidate is not None and self.sync_preview.isChecked())
        if in_sync_preview:
            suggested=self._sync_candidate['proposal']['reference_dimensions_m']
            ref=suggested
            height=float(self.scenario_floor_height.value())*self.floors.value()
            shown=self.neighbors if self.show_context.isChecked() else []
        elif self.meta is not None and not in_cal:
            ref=self.meta['reference_box_m']
            height=float(self.scenario_floor_height.value())*self.floors.value()
            shown=self.neighbors if self.show_context.isChecked() else []
        else:
            ref={'W':self.cal_w.value(),'D':self.cal_d.value(),'H':self.cal_h.value()}
            height=float(self.scenario_floor_height.value())*self.floors.value()
            shown=[]
        reference=box_vertices(float(ref['W']),float(ref['D']),float(ref['H']))
        target=box_vertices(float(ref['W']),float(ref['D']),height)
        for i,panel in enumerate(self.panels):
            if in_sync_preview:
                camera=self._sync_candidate['proposal']['cameras'][i]
                view_data=dict(self.meta['views'][i])
                view_data.update({k:camera[k] for k in ('P','K','rvec','tvec')})
                view_data['_reference_vertices']=reference
            elif in_cal or self.meta is None:
                cam=self.cal_cameras[i]
                view_data=({'P':cam['P'],'_reference_vertices':reference}
                           if cam and cam.get('ok') else None)
                if view_data and cam:
                    view_data.update({k:cam[k] for k in ('K','rvec','tvec') if k in cam})
            else:
                view_data=dict(self.meta['views'][i])
                view_data['_reference_vertices']=self.meta['reference_vertices_3d_m']
            cal_overlay=None
            if in_cal or self.meta is None:
                cal_overlay={
                    'anchors':self.cal_anchors[i],
                    'selected_anchor':int(self.cal_anchor.currentData() or 0),
                    'allowed':self.cal_allowed[i],
                    'protected':self.cal_protected[i],
                    'guides':self.cal_guides[i],
                    'observed':self.cal_observed[i],
                    'detected_segments':self.cal_detected[i],
                    'camera':self.cal_cameras[i],
                    'poly_draft':self.cal_poly_draft if self.cal_poly_view==i+1 else [],
                    'line_start':self.cal_line_start[1] if self.cal_line_start and self.cal_line_start[0]==i+1 else None,
                }
            panel.viewer.refresh_overlays(
                view_data,target if view_data is not None else None,
                shown,self.draft_xy,
                show_neighbors=self.show_context.isChecked() and not in_cal,
                show_vanishing=self.show_vps.isChecked() and not in_cal,
                show_box=self.show_box.isChecked() or in_cal,
                calibration_overlay=cal_overlay,
            )
            panel.set_active(i+1==self.active_view)

    def activate_view(self,index:int):
        if self.cal_poly_draft and self.cal_poly_view and self.cal_poly_view!=index:
            # La selección explícita cancela una anotación incompleta.
            self.cal_poly_draft=[];self.cal_poly_view=None
        self.active_view=index
        self.view_active_label.setText(f'Active view: {index}')
        if self.cal_view.currentIndex()!=index-1:
            self.cal_view.blockSignals(True);self.cal_view.setCurrentIndex(index-1);self.cal_view.blockSignals(False)
            self.cal_fov.blockSignals(True);self.cal_fov.setValue(self.cal_fovs[index-1]);self.cal_fov.blockSignals(False)
            self._select_first_missing_anchor(index-1)
            self._update_cal_status()
        # Cambiar la vista activa actualiza el panel ampliado sin duplicar la foto.
        self._apply_photo_layout(self.photo_layout_mode)
        for i,p in enumerate(self.panels,1):p.set_active(i==index)

    def toggle_mark(self,active:bool):
        if active:
            self.cal_mark.setChecked(False)
            self.tabs.setCurrentWidget(self.neighbors_tab)
        self.mark.setText('Footprint tool ACTIVE · click photograph' if active
                          else 'Mark footprint · activate')
        self._sync_mark_modes()

    def add_ground_point(self,index,x,y):
        if not self.mark.isChecked() or self.meta is None:return
        try:
            P=np.asarray(self.meta['views'][index-1]['P'],float)
            pt=image_to_ground(P,[[x,y]])[0]
            if self.draft_uv and self.draft_uv[0]['source_view']!=index:
                raise ValueError('Complete a footprint in one view; do not mix clicks across photographs')
            if len(self.draft_xy)>=16:raise ValueError('Maximum 16 footprint vertices per building')
            self.draft_xy.append([float(pt[0]),float(pt[1])])
            self.draft_uv.append({'source_view':index,'x':x,'y':y})
            self.draft_info.setText(f'Marked points: {len(self.draft_xy)}')
            self.redraw()
        except Exception as exc:self.error(str(exc))

    def undo_point(self):
        if self.draft_xy:self.draft_xy.pop();self.draft_uv.pop()
        self.draft_info.setText(f'Marked points: {len(self.draft_xy)}');self.redraw()

    def clear_draft(self):
        self.draft_xy=[];self.draft_uv=[]
        self.draft_info.setText('Marked points: 0');self.mark.setChecked(False)
        self.redraw()

    def persist_neighbors(self):
        if self.folder is None:return
        save_context(self.folder,self.neighbors,
                     calibration_sha256=sha256_file(self.folder/'calibration_multiview.json'))
        self.append_log(f'Neighbors saved: {self.folder/"urban_context_2p5d.json"}')

    def finish_neighbor(self):
        if self.meta is None:return
        try:
            if len(self.draft_xy)<3:raise ValueError('Mark at least three ground-footprint corners')
            name=self.name_neighbor.text().strip() or f'Neighbor_{len(self.neighbors)+1:02d}'
            footprint=validate_footprint(self.draft_xy)
            source=self.draft_uv[0]['source_view']
            item=new_neighbor(name,footprint,self.h_neighbor.value(),source_view=source,
                              observed_px=[{'x':v['x'],'y':v['y']} for v in self.draft_uv])
            self.neighbors.append(item)
            self.persist_neighbors()
            self.clear_draft();self._refresh_neighbor_list();self.redraw()
            self.append_log(f'Added volume: {name} | {item.height_m:.1f} m | {len(item.footprint)} vertices')
        except Exception as exc:self.error(str(exc))

    def _refresh_neighbor_list(self):
        self.neighbor_list.blockSignals(True);self.neighbor_list.clear()
        for b in self.neighbors:self.neighbor_list.addItem(f'{b.name}  ·  {b.height_m:g} m')
        self.neighbor_list.blockSignals(False)

    def select_neighbor(self,index:int):
        if 0<=index<len(self.neighbors):
            b=self.neighbors[index]
            self.name_neighbor.setText(b.name);self.h_neighbor.setValue(b.height_m)

    def update_neighbor_height(self):
        i=self.neighbor_list.currentRow()
        if i<0 or i>=len(self.neighbors):return
        b=self.neighbors[i];b.height_m=self.h_neighbor.value()
        b.name=self.name_neighbor.text().strip() or b.name
        try:
            b.validate();self.persist_neighbors();self._refresh_neighbor_list();self.redraw()
        except Exception as exc:self.error(str(exc))

    def delete_neighbor(self):
        i=self.neighbor_list.currentRow()
        if 0<=i<len(self.neighbors):
            self.neighbors.pop(i);self.persist_neighbors();self._refresh_neighbor_list();self.redraw()

    def _config(self):
        views=[i for i,c in enumerate(self.views_check,1) if c.isChecked()]
        if not views:raise ValueError('Select at least one view')
        cfg=config_from_metadata(self.meta,floors=self.floors.value(),mode=self.mode.currentData(),
              floor_height_m=self.scenario_floor_height.value(),
              seed=self.seed.value(),views=views,steps=self.steps.value(),guidance=self.cfg.value(),
              control_scale=self.control.value(),photo_finish=self.facade.isChecked(),
              roof_finish=self.roof.isChecked())
        return cfg

    def _prepare(self):
        if self.folder is None:raise ValueError('Open a project with four calibrated cameras first')
        cfg=self._config()
        run=prepare_run(self.folder,cfg)
        if self.use_occlusion.isChecked():
            if not self.neighbors:
                raise ValueError('Neighbor occlusion is enabled but no 2.5D neighbors are saved')
            report=apply_context_occlusions_to_run(run,self.folder)
            self.append_log(f'Oclusiones 2.5D: {report.get("reason", report.get("views"))}')
        self.current_run=run
        self.run_label.setText(f'Latest run: {run.name}')
        self.refresh_result_runs(preferred=run)
        return run,cfg

    def prepare_preview(self):
        try:
            run,cfg=self._prepare()
            meta=load_metadata(run)
            self.display_override={}
            for i,view in enumerate(meta['views'],1):
                path=run/view['preview_relative']
                self.panels[i-1].viewer.set_image(path)
                self.display_override[i]=path
            self.redraw()
            self.append_log(f'CPU scenario prepared: {run.name}. Original photographs unchanged.')
        except Exception as exc:self.error(str(exc))

    def start_generation(self):
        if self._thread and self._thread.isRunning():
            self.error('A generation job is already running');return
        try:
            from generabuild.generation import cuda_available
            if not cuda_available():
                raise RuntimeError('PyTorch CUDA GPU not found. Use Prepare scenario · CPU or install requirements-gpu.txt.')
            run,cfg=self._prepare()
        except Exception as exc:self.error(str(exc));return
        self._thread=QThread(self)
        self._worker=GenerationWorker(run,cfg);self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.append_log)
        self._worker.completed.connect(self.generation_complete)
        self._worker.failed.connect(self.generation_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self.gpu_button.setEnabled(False)
        self.tabs.setCurrentWidget(self.results_tab)
        self._thread.start()

    def _thread_finished(self):
        self.gpu_button.setEnabled(True)
        self._thread=None;self._worker=None

    def generation_complete(self,path:str):
        self.current_run=Path(path)
        self.append_log('Generation completed. Review the four views and evaluation report.')
        self.refresh_result_runs(preferred=self.current_run)
        self.show_results()

    def generation_failed(self,message:str):
        self.append_log('Generation failed: '+message)
        QMessageBox.critical(self,'Generation failed',message[:6000])

    def show_results(self):
        """Open Result Viewer, keeping generated photos distinct from CPU previews."""
        if self.folder is None:return
        self.refresh_result_runs(preferred=self.current_run)
        self.result_mode.setCurrentIndex(self.result_mode.findData('final'))
        self.tabs.setCurrentWidget(self.results_tab)
        self._render_result_view()
        if self.result_set and not self.result_set.final:
            self.append_log('No final photographs yet. Select Geometry preview · CPU to inspect masks.')

    def show_originals(self):
        if not self.paths:return
        if self.tabs.currentWidget() is self.results_tab:
            self._result_show_original()
            return
        for p,path in zip(self.panels,self.paths):p.viewer.set_image(path)
        self.redraw()

    def open_result_folder(self):
        if self.current_run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_run)))

    def export_run(self):
        if self.current_run is None:self.error('Prepare or generate a scenario first');return
        filename,_=QFileDialog.getSaveFileName(self,'Export reproducible run',
                      str(self.current_run.parent/(self.current_run.name+'.zip')),'ZIP (*.zip)')
        if not filename:return
        try:
            with zipfile.ZipFile(filename,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in self.current_run.rglob('*'):
                    if p.is_file():z.write(p,str(Path(self.current_run.name)/p.relative_to(self.current_run)))
                context=self.folder/'urban_context_2p5d.json'
                if context.is_file():z.write(context,f'{self.current_run.name}/source_urban_context_2p5d.json')
            self.append_log('ZIP created: '+filename)
        except Exception as exc:self.error(str(exc))

    def show_help(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT/'README.md')))

    def closeEvent(self,event):
        if self._sync_thread is not None and self._sync_thread.isRunning():
            QMessageBox.warning(self,'Camera synchronization running',
                'Finish camera synchronization before closing the application.')
            event.ignore();return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self,'Generation in progress',
                 'A GPU job is running. Finish it before closing the application.')
            event.ignore();return
        event.accept()


def start():
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    app=QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName('GeneraBuild Studio')
    site=Path(sys.argv[1]) if len(sys.argv)>1 else None
    window=MainWindow(site)
    window.show()
    return app.exec()
