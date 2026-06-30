"""PySide6 widget that displays the software-rendered framebuffer."""

from __future__ import annotations
from typing import Literal
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from packed_data_structures.graph.overlay import EdgeLayer, GraphOverlay, NodeLayer
from packed_data_structures.overlays.database import OverlaidDB

from .camera import OrbitCamera, FlyCamera
from .framebuffer import Framebuffer
from .renderer import Renderer
from .scene import Scene3D
from .modules import ModuleRegistry


class GraphWidget(QWidget):
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        module_registry: ModuleRegistry | None = None,
        control_mode: Literal["orbit", "fly"] = "orbit",
        camera=None,
        fly_speed: float = 200.0,
        parent=None,
    ):
        super().__init__(parent)
        self.setMinimumSize(width, height)
        self.fb = Framebuffer(width, height)
        self.control_mode = (
            control_mode if control_mode in ("orbit", "fly") else "orbit"
        )
        if camera is not None:
            self.camera = camera
            self.control_mode = "fly" if isinstance(camera, FlyCamera) else "orbit"
        elif self.control_mode == "fly":
            self.camera = FlyCamera()
            self.camera.look_at(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        else:
            self.camera = OrbitCamera(radius=100.0)
        self.scene = Scene3D()
        self.modules = module_registry or ModuleRegistry()
        self.renderer = Renderer(self.scene, self.fb, self.modules)
        self.setMouseTracking(True)
        self._last_pos = None
        self._keys_down: set[int] = set()
        self._fly_speed = fly_speed
        self._last_move_time = None
        if self.control_mode == "fly":
            from PySide6.QtCore import QTimer

            self._move_timer = QTimer(self)
            self._move_timer.timeout.connect(self._tick_fly_move)
            self._move_timer.start(16)

        # --- FPS Tracking ---
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps = 0.0

    def set_module_registry(self, registry: ModuleRegistry) -> None:
        self.modules = registry
        self.renderer.modules = registry

    def register_graph(
        self,
        graph_id: str,
        db: OverlaidDB,
        overlay: GraphOverlay,
        # node_positions: np.ndarray,
        # node_records: np.ndarray,
        # edge_layers: dict[str, np.ndarray] | None = None,
        # edge_records: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.scene.register_graph(
            graph_id=graph_id,
            db=db,
            overlay=overlay,
        )

    def resizeEvent(self, event):
        size = event.size()
        self.fb.resize(size.width(), size.height())
        super().resizeEvent(event)

    def paintEvent(self, event):
        img_buf = self.renderer.render(self.camera)

        h, w, _ = img_buf.shape
        qimg = QImage(img_buf.data, w, h, QImage.Format.Format_RGBA8888)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.drawImage(0, 0, qimg)

        # --- 3. Calculate Stats ---
        # FPS Calculation
        self._frame_count += 1
        now = time.perf_counter()
        if now - self._last_fps_time >= 0.5:  # Update every 0.5s
            self._fps = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now

        # Graph Stats (Totals)
        total_nodes = 0
        total_edges = 0
        for entry in self.scene.graphs.values():
            for layer in entry.overlay.layers.values():
                if isinstance(layer, NodeLayer):
                    total_nodes += len(entry.db.get_table(layer))
                elif isinstance(layer, EdgeLayer):
                    total_edges += len(entry.db.get_table(layer))

        # Screen Stats (Visible/Culled)
        # We sum the lengths of the index arrays cached during this frame's render
        vis_nodes = sum(len(res.indices) for res in self.scene._node_cache.values())
        vis_edges = sum(len(res.indices) for res in self.scene._edge_cache.values())

        # --- 4. Draw HUD ---
        lines = [
            f"FPS: {self._fps:.1f}",
            f"Nodes: {vis_nodes} / {total_nodes}",
            f"Edges: {vis_edges} / {total_edges}",
        ]

        painter.setPen(Qt.GlobalColor.yellow)
        font = painter.font()
        font.setFamily("Monospace")
        font.setBold(True)
        painter.setFont(font)

        # Draw top-left corner
        y_pos = 20
        for line in lines:
            painter.drawText(10, y_pos, line)
            y_pos += 15

        painter.end()

    # --- Camera Controls ---
    def mousePressEvent(self, event):
        self._last_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            self._last_pos = event.position()
            super().mouseMoveEvent(event)
            return
        delta = event.position() - self._last_pos
        self._last_pos = event.position()
        buttons = event.buttons()
        if self.control_mode == "orbit":
            if buttons & Qt.MouseButton.LeftButton:
                # Rotate
                self.camera.rotate(delta.x() * -0.005, delta.y() * 0.005)
                self.update()
            elif buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
                # Pan
                self.camera.pan(delta.x(), -delta.y(), scale=1.0)
                self.update()
        else:
            if buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.LeftButton):
                self.camera.rotate(delta.x() * 0.003, delta.y() * 0.003)
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._last_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0  # 1 notch = 120
        if self.control_mode == "orbit":
            self.camera.dolly(-0.1 * delta)
        else:
            # Adjust fly speed
            self._fly_speed = max(1.0, self._fly_speed * (1.0 + 0.1 * delta))
        self.update()
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        if self.control_mode == "fly":
            if not event.isAutoRepeat():
                self._keys_down.add(int(event.key()))
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self.control_mode == "fly":
            if not event.isAutoRepeat():
                self._keys_down.discard(int(event.key()))
        super().keyReleaseEvent(event)

    def _tick_fly_move(self):
        if self.control_mode != "fly":
            return

        now = time.perf_counter()
        if self._last_move_time is None:
            self._last_move_time = now
            return
        dt = min(0.05, now - self._last_move_time)
        self._last_move_time = now
        if not self._keys_down:
            return
        cam: FlyCamera = self.camera  # type: ignore[assignment]
        speed = self._fly_speed
        if Qt.Key_Shift in self._keys_down:
            speed *= 3.0
        if Qt.Key_Control in self._keys_down:
            speed *= 0.3
        move_f = (Qt.Key_W in self._keys_down) - (Qt.Key_S in self._keys_down)
        move_r = (Qt.Key_D in self._keys_down) - (Qt.Key_A in self._keys_down)
        move_u = (Qt.Key_E in self._keys_down or Qt.Key_Space in self._keys_down) - (
            Qt.Key_Q in self._keys_down
        )
        if move_f == 0 and move_r == 0 and move_u == 0:
            return
        cam.move_local(
            float(move_r) * speed * dt,
            float(move_u) * speed * dt,
            float(move_f) * speed * dt,
        )
        self.update()
