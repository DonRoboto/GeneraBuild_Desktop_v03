"""Four calibrated photos with native-pixel annotation tools."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QPen, QBrush, QPolygonF, QPixmap, QFont
from PySide6.QtWidgets import QGraphicsView,QGraphicsScene,QGraphicsPixmapItem,QFrame,QVBoxLayout,QLabel
from generabuild.context_2p5d import world_to_image

BOX_EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def _stroke(color,width=2,dash=False):
    return QPen(QColor(color),width,Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine)


class WipePixmapItem(QGraphicsPixmapItem):
    """Muestra la imagen final a la derecha de una división vertical ajustable."""

    def __init__(self, pixmap: QPixmap, percent_original: int = 50):
        super().__init__(pixmap)
        self.percent_original = max(0, min(100, int(percent_original)))

    def set_percent(self, value: int):
        self.percent_original = max(0, min(100, int(value)))
        self.update()

    def paint(self, painter, option, widget=None):
        pixmap = self.pixmap()
        x = pixmap.width() * self.percent_original / 100.0
        painter.save()
        painter.setClipRect(QRectF(x, 0, pixmap.width() - x, pixmap.height()))
        super().paint(painter, option, widget)
        painter.restore()


class PhotoView(QGraphicsView):
    clicked=Signal(int,float,float)
    activated=Signal(int)
    doubleClicked=Signal(int)
    def __init__(self,index:int,parent=None):
        super().__init__(parent)
        self.index=index
        self._scene=QGraphicsScene(self)
        self.setScene(self._scene)
        self._path=None
        self._size=None
        self._first=True
        self.mark_mode=False
        self._result_signature=None
        self._result_wipe=None
        self._result_divider=None
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QBrush(QColor('#0b121c')))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumSize(270,170)

    def set_mark_mode(self,active:bool):
        self.mark_mode=bool(active)
        self.setDragMode(QGraphicsView.DragMode.NoDrag if active else
                         QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor)

    def set_image(self,path:Path):
        path=Path(path)
        if not path.exists():raise FileNotFoundError(path)
        if self._path!=path:
            self._path=path
            self._pixmap=QPixmap(str(path))
            if self._pixmap.isNull():raise ValueError(f'Unsupported image: {path}')
            self._size=(self._pixmap.width(),self._pixmap.height())
            self._first=True

    def reset_view(self):
        self.fitInView(self._scene.sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)

    def show_result(self, original: Path, generated: Path | None = None,
                    mode: str = 'original', split_percent: int = 50):
        """Presenta fotografías limpias, sin prismas ni anotaciones.

        mode='compare': izquierda original, derecha generada; el divisor
        se mueve sin recalcular imágenes ni perder el zoom existente.
        """
        original = Path(original).resolve()
        generated = Path(generated).resolve() if generated is not None else None
        if not original.is_file():
            raise FileNotFoundError(original)
        if generated is not None and not generated.is_file():
            raise FileNotFoundError(generated)
        signature = (original, generated, mode)
        if signature == self._result_signature:
            self.set_result_percent(split_percent)
            return
        orig_pixmap = QPixmap(str(original))
        if orig_pixmap.isNull():
            raise ValueError(f'Could not open original photo: {original}')
        newer = QPixmap(str(generated)) if generated is not None else None
        if newer is not None and newer.isNull():
            raise ValueError(f'Could not open generated result: {generated}')
        if newer is not None and newer.size() != orig_pixmap.size():
            raise ValueError(f'View {self.index}: final image resolution differs from original')
        self._scene.clear()
        self._result_wipe = None
        self._result_divider = None
        self._scene.addPixmap(orig_pixmap)
        if newer is not None:
            if mode == 'compare':
                self._result_wipe = WipePixmapItem(newer, split_percent)
                self._scene.addItem(self._result_wipe)
                x = orig_pixmap.width() * max(0, min(100, int(split_percent))) / 100.0
                self._result_divider = self._scene.addLine(
                    x, 0, x, orig_pixmap.height(), _stroke('#f2ce63', 2.4))
                self._result_divider.setZValue(3)
            else:
                self._scene.addPixmap(newer)
        previous_size = self._size
        self._size = (orig_pixmap.width(), orig_pixmap.height())
        self._scene.setSceneRect(0, 0, *self._size)
        self._result_signature = signature
        if previous_size != self._size or self._first:
            self.reset_view()
            self._first = False

    def set_result_percent(self, value: int):
        """Solo mueve el divisor; no vuelve a cargar ni dibujar los PNG."""
        percent = max(0, min(100, int(value)))
        if self._result_wipe is not None:
            self._result_wipe.set_percent(percent)
        if self._result_divider is not None and self._size:
            x = self._size[0] * percent / 100.0
            self._result_divider.setLine(x, 0, x, self._size[1])

    def refresh_overlays(self,metadata:dict|None,target:np.ndarray|None,
                         neighbors:list,draft_world:list,*,show_neighbors=True,
                         show_vanishing=True,show_box=True,calibration_overlay=None):
        self._result_signature=None
        self._result_wipe=None
        self._result_divider=None
        self._scene.clear()
        if self._path is None:return
        self._scene.addPixmap(self._pixmap)
        w,h=self._size
        self._scene.setSceneRect(0,0,w,h)
        if metadata is not None:
            P=np.asarray(metadata['P'],float)
            if show_box:
                try:
                    ref=np.asarray(metadata.get('_reference_vertices'),float)
                    if ref.shape==(8,3):self._draw_box(P,ref,'#66e4f0',1.4,True)
                    if target is not None:self._draw_box(P,target,'#f5d56a',2,False)
                except (ValueError,TypeError,IndexError):pass
            if show_neighbors:
                for b in neighbors:
                    if not b.enabled:continue
                    try:self._draw_neighbor(P,b)
                    except ValueError:continue
            if draft_world:
                try:
                    xyz=np.c_[np.asarray(draft_world,dtype=float),np.zeros(len(draft_world))]
                    uv=world_to_image(P,xyz)
                    self._draw_line_strip(uv,'#ff9b4e',2,closed=False)
                    for p in uv:self._dot(p,'#ffa04f',4.1)
                except ValueError:pass
            if show_vanishing:
                self._draw_reference_guides(metadata,P)
        if calibration_overlay is not None:
            self._draw_calibration_overlay(calibration_overlay)
        if self._first:
            self.reset_view()
            self._first=False

    def _draw_line_strip(self,uv,color,width=1.5,closed=False):
        uv=np.asarray(uv,float)
        if len(uv)<2:return
        for i in range(len(uv) if closed else len(uv)-1):
            a,b=uv[i],uv[(i+1)%len(uv)]
            self._scene.addLine(float(a[0]),float(a[1]),float(b[0]),float(b[1]),_stroke(color,width))

    def _dot(self,xy,color,r=3.5):
        x,y=float(xy[0]),float(xy[1]);self._scene.addEllipse(x-r,y-r,2*r,2*r,
                                                      QPen(QColor('#0c1118'),1),QBrush(QColor(color)))

    def _draw_box(self,P,verts,color,width,dashed):
        uv=world_to_image(P,verts)
        for i,j in BOX_EDGES:
            a,b=uv[i],uv[j]
            self._scene.addLine(float(a[0]),float(a[1]),float(b[0]),float(b[1]),_stroke(color,width,dashed))

    def _draw_neighbor(self,P,b):
        vertices=b.vertices();n=len(vertices)//2
        uv=world_to_image(P,vertices)
        roof=QPolygonF([QPointF(float(u),float(v)) for u,v in uv[n:]])
        rgba=QColor(*b.color_rgb);rgba.setAlpha(55)
        self._scene.addPolygon(roof,_stroke('#35d5a0',1.5),QBrush(rgba))
        self._draw_line_strip(uv[:n],'#35d5a0',1.4,True)
        self._draw_line_strip(uv[n:],'#35d5a0',1.8,True)
        for i in range(n):
            a,b=uv[i],uv[n+i]
            self._scene.addLine(float(a[0]),float(a[1]),float(b[0]),float(b[1]),_stroke('#35d5a0',1.6))

    def _draw_reference_guides(self,metadata,P):
        import cv2
        colors=['#f5a86a','#67dce7','#bd9bff']
        # Aristas observadas existentes, guardadas por el editor v0.5.
        for seg in metadata.get('observed_axis_segments_px',[]):
            try:
                p1,p2=seg['p1'],seg['p2']
                idx=int(seg.get('axis',0))%3
                self._scene.addLine(float(p1['x']),float(p1['y']),float(p2['x']),float(p2['y']),
                                    _stroke(colors[idx],2.4))
            except (KeyError,ValueError,TypeError):pass
        guides=metadata.get('manual_vp_guides_px',[])
        if not guides:return
        try:
            K=np.asarray(metadata['K'],float)
            R,_=cv2.Rodrigues(np.asarray(metadata['rvec'],float))
        except (KeyError,ValueError):return
        for guide in guides:
            try:
                axis=int(guide['axis']);src=guide['point']
                p=np.array([float(src['x']),float(src['y'])])
                d=K@R[:,axis]
                if abs(d[2])<1e-8:direction=d[:2]
                else:direction=d[:2]/d[2]-p
                direction=direction/(np.linalg.norm(direction)+1e-12)
                p1=p-3000*direction;p2=p+3000*direction
                self._scene.addLine(*map(float,[p1[0],p1[1],p2[0],p2[1]]),_stroke(colors[axis],1,True))
            except (KeyError,ValueError,TypeError,IndexError):continue

    def _draw_calibration_overlay(self, state: dict):
        """Anotaciones editables; coordenadas nativas de imagen, sin alterar píxeles."""
        # Los polígonos solo se dibujan, jamás editan la fotografía original.
        for poly,color in ([(state.get('allowed'), '#58e98c')]
                           + [(p, '#ff65ae') for p in state.get('protected', [])]):
            if not poly or len(poly) < 2:continue
            pts = [QPointF(float(p['x']),float(p['y'])) for p in poly]
            fill=QColor(color);fill.setAlpha(25)
            if len(pts)>2:
                self._scene.addPolygon(QPolygonF(pts), _stroke(color,2),QBrush(fill))
            else:
                self._draw_line_strip([[p.x(),p.y()] for p in pts],color,2)
        for seg in state.get('detected_segments', []):
            x1,y1,x2,y2=map(float,seg)
            self._scene.addLine(x1,y1,x2,y2,_stroke('#9ca9bb',.65))
        colors=['#ffa96b','#5be4e6','#c7a3ff']
        for seg in state.get('observed', []):
            try:
                c=colors[int(seg.get('axis',0))%3]
                a,b=seg['p1'],seg['p2']
                self._scene.addLine(float(a['x']),float(a['y']),float(b['x']),float(b['y']),_stroke(c,2.6))
            except (KeyError,TypeError,ValueError):pass
        for guide in state.get('guides', []):
            try:
                from generabuild.manual_lines import line_through_point_and_vp
                from generabuild.vanishing import camera_vanishing_points
                camera=state.get('camera')
                if not camera:continue
                axis=int(guide['axis'])
                vps=camera_vanishing_points(camera,self._size)
                clipped=line_through_point_and_vp(guide['point'],vps[axis],self._size)
                if clipped:
                    self._scene.addLine(*clipped,_stroke(colors[axis],1.3,True))
            except (IndexError,KeyError,TypeError,ValueError):pass
        for key,point in state.get('anchors',{}).items():
            try:
                k=int(key)
                x,y=float(point['x']),float(point['y'])
                color='#ffdf78' if k==state.get('selected_anchor') else '#ff9e56'
                self._dot((x,y),color,5)
                text=self._scene.addText(str(k),QFont('Sans',10,QFont.Weight.Bold))
                text.setDefaultTextColor(QColor(color));text.setPos(x+5,y-20)
            except (ValueError,TypeError,KeyError):pass
        drawing=state.get('poly_draft') or []
        if drawing:
            pts=[[p['x'],p['y']] for p in drawing]
            self._draw_line_strip(pts,'#fff17e',2.3)
            for pt in pts:self._dot(pt,'#fff17e',3.5)
        partial=state.get('line_start')
        if partial:self._dot([partial['x'],partial['y']],'#e7a3ff',6)

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:
            xy=self.mapToScene(event.position().toPoint())
            if self._size and 0<=xy.x()<self._size[0] and 0<=xy.y()<self._size[1] and self.mark_mode:
                # No activar la vista ANTES de emitir clicked: el controlador
                # necesita detectar un clic sobre otra vista y evitar asociar
                # una esquina equivocada durante el cambio de fotografía.
                self.clicked.emit(self.index,float(xy.x()),float(xy.y()))
                event.accept();return
            self.activated.emit(self.index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self,event):
        # El doble clic solo alterna el diseño si no se están dibujando anclas.
        # Durante la anotación, siempre debe prevalecer el editor de puntos.
        if event.button()==Qt.MouseButton.LeftButton and not self.mark_mode:
            self.doubleClicked.emit(self.index)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self,event):
        factor=1.16 if event.angleDelta().y()>0 else 1/1.16
        self.scale(factor,factor)
        event.accept()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self._first and self._scene.sceneRect().isValid() and not self._scene.sceneRect().isEmpty():
            self.reset_view()


class PhotoPanel(QFrame):
    def __init__(self,index:int,parent=None):
        super().__init__(parent)
        self.setObjectName('PhotoPanel')
        layout=QVBoxLayout(self);layout.setContentsMargins(2,2,2,2);layout.setSpacing(3)
        self.title=QLabel(f'VIEW {index}');self.title.setObjectName('PhotoTitle')
        layout.addWidget(self.title)
        self.viewer=PhotoView(index)
        layout.addWidget(self.viewer,1)
        self.setMinimumHeight(185)

    def set_active(self,active:bool):
        self.setStyleSheet('QFrame#PhotoPanel{border:2px solid '+('#3be1b2' if active else '#26384a')+';}')
