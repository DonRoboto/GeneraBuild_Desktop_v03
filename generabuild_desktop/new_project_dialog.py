"""Native Qt wizard to create a project from four photographs."""
from __future__ import annotations

from pathlib import Path
from PIL import Image
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QDoubleSpinBox, QSpinBox, QDialogButtonBox,
    QMessageBox, QGroupBox,
)
from generabuild.project_setup import new_draft
from generabuild.storage import WORKSPACE


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('GeneraBuild Studio — New project')
        self.setMinimumWidth(690)
        self.result_folder = None
        self.image_paths = [None] * 4
        root = QVBoxLayout(self)
        description = QLabel(
            'Select four photographs of the same site: views 1 and 2 aerial, '
            'views 3 and 4 at street level. Original images are copied unchanged.\n'
            'Next, mark the physical corners and calibrate each camera in the editor.'
        )
        description.setWordWrap(True)
        root.addWidget(description)

        info = QGroupBox('Study information')
        grid = QGridLayout(info)
        self.site = QLineEdit()
        self.site.setPlaceholderText('guanajuato8  (project folder identifier)')
        grid.addWidget(QLabel('Project ID'), 0, 0)
        grid.addWidget(self.site, 0, 1)
        grid.addWidget(QLabel('Location: '+str(WORKSPACE)), 1, 0, 1, 2)
        root.addWidget(info)

        group = QGroupBox('Original photographs')
        photo_grid = QGridLayout(group)
        self.labels=[]
        for i in range(4):
            photo_grid.addWidget(QLabel(f'View {i+1} — '+('aerial' if i<2 else 'street')),i,0)
            label=QLabel('No file selected')
            label.setWordWrap(True)
            photo_grid.addWidget(label,i,1)
            button=QPushButton('Browse…')
            button.clicked.connect(lambda checked=False, k=i:self.pick_image(k))
            photo_grid.addWidget(button,i,2)
            self.labels.append(label)
        root.addWidget(group)

        group = QGroupBox('Estimated reference dimensions')
        params = QGridLayout(group)
        self.width=QDoubleSpinBox();self.width.setRange(1,200);self.width.setValue(10);self.width.setSuffix(' m')
        self.depth=QDoubleSpinBox();self.depth.setRange(1,300);self.depth.setValue(20);self.depth.setSuffix(' m')
        self.height=QDoubleSpinBox();self.height.setRange(1,200);self.height.setValue(9);self.height.setSuffix(' m')
        self.floor_h=QDoubleSpinBox();self.floor_h.setRange(2,5);self.floor_h.setValue(3);self.floor_h.setSuffix(' m')
        self.floors=QSpinBox();self.floors.setRange(8,10);self.floors.setValue(9)
        for row,(name,widget) in enumerate([
            ('Façade width W',self.width),('Depth D',self.depth),
            ('Original height H₀',self.height),('Floor height',self.floor_h),
            ('Target floors',self.floors),
        ]):
            params.addWidget(QLabel(name),row//3*2,row%3*2)
            params.addWidget(widget,row//3*2,row%3*2+1)
        root.addWidget(group)
        note = QLabel('Set FOV separately for each photo after importing. '
                      'Dimensions are approximate assumptions, not survey measurements.')
        note.setWordWrap(True);root.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Create project and calibrate')
        buttons.accepted.connect(self.create)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def pick_image(self,i:int):
        path,_=QFileDialog.getOpenFileName(
            self,f'Select photograph View {i+1}',str(Path.home()),
            'Images (*.png *.jpg *.jpeg *.webp)')
        if not path:return
        try:
            with Image.open(path) as img:
                size=img.size
                img.verify()
            self.image_paths[i]=Path(path)
            self.labels[i].setText(f'{Path(path).name}   ({size[0]} × {size[1]} px)')
        except Exception as exc:
            QMessageBox.critical(self,'Invalid image',str(exc))

    def create(self):
        if not self.site.text().strip():
            QMessageBox.warning(self,'Project ID required','Enter a project identifier.')
            return
        if any(p is None for p in self.image_paths):
            QMessageBox.warning(self,'Missing photographs','Select all four original photographs.')
            return
        try:
            self.result_folder = new_draft(
                self.site.text(),self.image_paths,ref_w=self.width.value(),
                ref_d=self.depth.value(),ref_h=self.height.value(),
                floor_height=self.floor_h.value(),floors=self.floors.value()
            )
        except Exception as exc:
            QMessageBox.critical(self,'Could not create project',str(exc))
            return
        self.accept()
