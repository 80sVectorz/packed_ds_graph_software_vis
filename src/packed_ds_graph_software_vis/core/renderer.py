"""
Renderer orchestrator: manages the frame lifecycle, invokes Scene3D culling,
and dispatches to active render modules. Restarts the frame if graph versions
change mid-frame (single-buffer).
"""

from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from .camera import OrbitCamera
from .framebuffer import Framebuffer
from .modules import ModuleRegistry, RenderModule
from .scene import Scene3D


class Renderer:
    def __init__(
        self,
        scene: Scene3D,
        framebuffer: Framebuffer,
        modules: ModuleRegistry | None = None,
    ):
        self.scene = scene
        self.fb = framebuffer
        self.modules = modules or ModuleRegistry()

    def render(self, camera: OrbitCamera) -> np.ndarray:
        """Render a frame. If versions change mid-frame, restart until clean."""
        while True:
            self.fb.clear()
            self.scene.begin_frame(camera, self.fb.width, self.fb.height)

            # Declare requirements
            for mod in self.modules.active():
                mod.declare(self.scene)

            # Draw
            restart = False
            for mod in self.modules.active():
                if self.scene.frame.restart_needed:
                    restart = True
                    break
                mod.render(self.scene, self.fb)
                if self.scene.frame.restart_needed:
                    restart = True
                    break

            if restart:
                # Start over with latest versions
                continue
            return self.fb.color

