"""Optional grid render module for orientation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from packed_ds_graph_software_vis.core import kernels
from packed_ds_graph_software_vis.core.modules import RenderModule
from packed_ds_graph_software_vis.core.framebuffer import Framebuffer
from packed_ds_graph_software_vis.core.scene import Scene3D


def _clip_line(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int
) -> tuple[bool, int, int, int, int]:
    """Liang-Barsky clip against screen rect [0,w) x [0,h)."""
    dx = x1 - x0
    dy = y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0, w - 1 - x0, y0, h - 1 - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q, strict=False):
        if pi == 0:
            if qi < 0:
                return (False, 0, 0, 0, 0)
            continue
        t = qi / pi
        if pi < 0:
            if t > u2:
                return (False, 0, 0, 0, 0)
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return (False, 0, 0, 0, 0)
            if t < u2:
                u2 = t
    nx0 = int(round(x0 + u1 * dx))
    ny0 = int(round(y0 + u1 * dy))
    nx1 = int(round(x0 + u2 * dx))
    ny1 = int(round(y0 + u2 * dy))
    return (True, nx0, ny0, nx1, ny1)


def _clip_line_clip_space(
    pa: np.ndarray, pb: np.ndarray, guard_band: float = 0.0, w_min: float = 1e-4
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Clip a homogeneous segment to the view volume with optional guard band.

    guard_band expands the canonical frustum by a multiplier (e.g., 0.05 keeps
    lines that are up to 5% outside the formal bounds to avoid harsh cut-offs).
    """
    g = 1.0 + max(0.0, guard_band)
    d = pb - pa
    t0, t1 = 0.0, 1.0
    planes = (
        (d[0] + g * d[3], pa[0] + g * pa[3]),  # x + w*g >= 0
        (-d[0] + g * d[3], -pa[0] + g * pa[3]),  # -x + w*g >= 0
        (d[1] + g * d[3], pa[1] + g * pa[3]),  # y + w*g >= 0
        (-d[1] + g * d[3], -pa[1] + g * pa[3]),  # -y + w*g >= 0
        (d[2] + g * d[3], pa[2] + g * pa[3]),  # z + w*g >= 0 (near)
        (-d[2] + g * d[3], -pa[2] + g * pa[3]),  # -z + w*g >= 0 (far)
        (d[3], pa[3] - w_min),  # w >= w_min to avoid divide-by-near-zero/behind-camera
    )
    for denom, numer in planes:
        if denom == 0.0:
            if numer < 0.0:
                return (False, pa, pb)
            continue
        t = -numer / denom
        if denom > 0.0:
            if t > t1:
                return (False, pa, pb)
            if t > t0:
                t0 = t
        else:
            if t < t0:
                return (False, pa, pb)
            if t < t1:
                t1 = t
    if t1 < t0:
        return (False, pa, pb)
    return (True, pa + d * t0, pa + d * t1)


@dataclass
class GridModule(RenderModule):
    """Draws a simple XY-plane grid projected into screen space using current camera."""

    extent: float = 1000.0
    spacing: float = 100.0
    guard_band: float = 0.05  # fraction to expand frustum to avoid harsh edge culling
    color: tuple[int, int, int, int] = (80, 80, 80, 255)
    name: str = "grid"
    active: bool = True

    def declare(self, scene: Scene3D) -> None:
        # Grid does not require graph data.
        return

    def render(self, scene: Scene3D, framebuffer: Framebuffer) -> None:
        if not self.active:
            return
        camera = scene.frame.camera
        if camera is None:
            return
        aspect = framebuffer.width / max(1, framebuffer.height)
        # Use higher precision for projection to reduce subpixel jitter on the grid.
        mvp = camera.view_proj(aspect).astype(np.float64, copy=False)
        w = float(framebuffer.width)
        h = float(framebuffer.height)
        max_x = framebuffer.width - 1
        max_y = framebuffer.height - 1

        # Build grid lines in world space on z=0 plane
        xs = np.arange(-self.extent, self.extent + self.spacing, self.spacing)
        ys = np.arange(-self.extent, self.extent + self.spacing, self.spacing)
        segments = []
        depths = []

        def add_segment(a: np.ndarray, b: np.ndarray) -> None:
            pa = mvp @ a
            pb = mvp @ b
            ok, pa, pb = _clip_line_clip_space(pa, pb, self.guard_band)
            if not ok or pa[3] == 0.0 or pb[3] == 0.0:
                return
            pa_ndc = pa[:3] / pa[3]
            pb_ndc = pb[:3] / pb[3]
            sx0 = int(np.rint((pa_ndc[0] * 0.5 + 0.5) * w))
            sy0 = int(np.rint((1.0 - (pa_ndc[1] * 0.5 + 0.5)) * h))
            sx1 = int(np.rint((pb_ndc[0] * 0.5 + 0.5) * w))
            sy1 = int(np.rint((1.0 - (pb_ndc[1] * 0.5 + 0.5)) * h))
            ok_clip, cx0, cy0, cx1, cy1 = _clip_line(
                sx0, sy0, sx1, sy1, framebuffer.width, framebuffer.height
            )
            if not ok_clip:
                return
            segments.append([[cx0, cy0], [cx1, cy1]])
            depths.append([pa_ndc[2], pb_ndc[2]])

        for x in xs:
            add_segment(
                np.array([x, -self.extent, 0.0, 1.0], dtype=np.float64),
                np.array([x, self.extent, 0.0, 1.0], dtype=np.float64),
            )
        for y in ys:
            add_segment(
                np.array([-self.extent, y, 0.0, 1.0], dtype=np.float64),
                np.array([self.extent, y, 0.0, 1.0], dtype=np.float64),
            )

        if not segments:
            return

        segs = np.asarray(segments, dtype=np.int32)
        dps = np.asarray(depths, dtype=np.float32)
        kernels.draw_edge_segments(
            framebuffer.color,
            framebuffer.depth,
            segs,
            dps,
            self.color,
        )
