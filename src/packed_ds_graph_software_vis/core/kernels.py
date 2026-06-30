"""Numba-accelerated software rasterization kernels.

These are intentionally simple and rely on direct reads from your PackedArrays.
"""

from __future__ import annotations

import numpy as np
import numba as nb


@nb.njit
def project_points(pos: np.ndarray, mvp: np.ndarray, width: int, height: int):
    """Project points.

    Args:
        pos ((N,3) float32): Array of 3D positions
        mvp ((4,4) float32): projection matrix
        width: Frame width
        height: Frame height

    Returns:
      screen_xy (N,2 int32), depth (N float32), inv_w (N float32)
    """
    n = pos.shape[0]
    xy = np.empty((n, 2), np.int32)
    depth = np.empty(n, np.float32)
    inv_w = np.empty(n, np.float32)

    for i in range(n):
        x, y, z = pos[i, 0], pos[i, 1], pos[i, 2]
        vx = mvp[0, 0] * x + mvp[0, 1] * y + mvp[0, 2] * z + mvp[0, 3]
        vy = mvp[1, 0] * x + mvp[1, 1] * y + mvp[1, 2] * z + mvp[1, 3]
        vz = mvp[2, 0] * x + mvp[2, 1] * y + mvp[2, 2] * z + mvp[2, 3]
        vw = mvp[3, 0] * x + mvp[3, 1] * y + mvp[3, 2] * z + mvp[3, 3]

        # Avoid divide by zero
        iw = 1.0 / vw if vw != 0 else 0.0
        inv_w[i] = iw

        ndc_x = vx * iw
        ndc_y = vy * iw

        # Screen coordinates
        xy[i, 0] = int((ndc_x * 0.5 + 0.5) * width)
        xy[i, 1] = int((1.0 - (ndc_y * 0.5 + 0.5)) * height)

        # Depth buffer value (NDC z)
        depth[i] = vz * iw

    return xy, depth, inv_w


@nb.njit
def draw_edges(
    frame: np.ndarray,
    depth_buf: np.ndarray,
    pts: np.ndarray,
    depths: np.ndarray,
    edges: np.ndarray,
    rgba: tuple[np.uint8, np.uint8, np.uint8, np.uint8],
):
    """Rasterize edges with a simple Bresenham line and depth test (single color).

    Args:
        frame: The frame buffer
        depth_buf: The depth buffer
        pts: Array of points
        depths: Array of point depths
        edges ((M,2) int32): Array of edge endpoints
        rgba: Edge color as uint8 RGBATuple
    """
    h, w, _ = frame.shape
    r, g, b, a = rgba
    for e in range(edges.shape[0]):
        a_idx = edges[e, 0]
        b_idx = edges[e, 1]
        x0, y0 = pts[a_idx, 0], pts[a_idx, 1]
        x1, y1 = pts[b_idx, 0], pts[b_idx, 1]
        z0, z1 = depths[a_idx], depths[b_idx]

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0

        for _ in range(32768):  # safety cap for pathological inputs
            if 0 <= x < w and 0 <= y < h:
                t = 0.0
                if (x1 - x0) != 0:
                    t = (x - x0) / (x1 - x0)
                z = z0 + t * (z1 - z0)
                if z < depth_buf[y, x]:
                    depth_buf[y, x] = z
                    frame[y, x, 0] = r
                    frame[y, x, 1] = g
                    frame[y, x, 2] = b
                    frame[y, x, 3] = a
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy


@nb.njit
def draw_edges_colors(
    frame: np.ndarray,
    depth_buf: np.ndarray,
    pts: np.ndarray,
    depths: np.ndarray,
    edges: np.ndarray,
    colors: np.ndarray,
):
    """Rasterize edges where each edge has its own RGBA color.

    Args:
        colors: (M,4) uint8 per-edge color aligned with edges.
    """
    h, w, _ = frame.shape
    for e in range(edges.shape[0]):
        a_idx = edges[e, 0]
        b_idx = edges[e, 1]
        x0, y0 = pts[a_idx, 0], pts[a_idx, 1]
        x1, y1 = pts[b_idx, 0], pts[b_idx, 1]
        z0, z1 = depths[a_idx], depths[b_idx]
        r = colors[e, 0]
        g = colors[e, 1]
        b = colors[e, 2]
        a = colors[e, 3]

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0

        for _ in range(32768):
            if 0 <= x < w and 0 <= y < h:
                t = 0.0
                if (x1 - x0) != 0:
                    t = (x - x0) / (x1 - x0)
                z = z0 + t * (z1 - z0)
                if z < depth_buf[y, x]:
                    depth_buf[y, x] = z
                    frame[y, x, 0] = r
                    frame[y, x, 1] = g
                    frame[y, x, 2] = b
                    frame[y, x, 3] = a
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy


@nb.njit
def draw_nodes(
    frame: np.ndarray,
    depth_buf: np.ndarray,
    pts: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    radius: int,
):
    """Draw filled disks for nodes with depth test.

    Args:
        colors: (N,4) uint8 per node
    """
    h, w, _ = frame.shape
    r2 = radius * radius
    for i in range(pts.shape[0]):
        cx, cy = pts[i, 0], pts[i, 1]
        z = depths[i]
        col = colors[i]
        for dy in range(-radius, radius + 1):
            y = cy + dy
            if y < 0 or y >= h:
                continue
            for dx in range(-radius, radius + 1):
                x = cx + dx
                if x < 0 or x >= w:
                    continue
                if dx * dx + dy * dy > r2:
                    continue
                if z < depth_buf[y, x]:
                    depth_buf[y, x] = z
                    frame[y, x, 0] = col[0]
                    frame[y, x, 1] = col[1]
                    frame[y, x, 2] = col[2]
                    frame[y, x, 3] = col[3]


@nb.njit
def draw_nodes_variable_radius(
    frame: np.ndarray,
    depth_buf: np.ndarray,
    pts: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    depth_interpolate: bool = False,
):
    """Draw filled disks for nodes with depth test and per-node radius."""
    if depth_interpolate:
        z_min = np.min(depths)
        z_max = np.max(depths)
        if z_min != z_max:
            z_t = (depths - z_min) / (z_max - z_min)
        else:
            depth_interpolate = False

    h, w, _ = frame.shape
    for i in range(pts.shape[0]):
        r = int(radii[i])
        if r <= 0:
            continue
        cx, cy = pts[i, 0], pts[i, 1]
        z = depths[i]
        col = colors[i]
        if depth_interpolate:
            col[3] *= 1 - z_t[i]

        # Circle drawing using Jesko's midpoint circle algorithm.
        dy = 0
        dx = r
        t1 = r // 16
        cx_l0 = min(w - 1, max(0, cx - dx))
        cx_r0 = min(w - 1, max(0, cx + dx))

        while dx >= dy:
            cx_l1 = min(w - 1, max(0, cx - dy))
            cx_r1 = min(w - 1, max(0, cx + dy))

            for x in range(cx_l0, cx_r0):
                y0 = min(h - 1, max(0, cy - dy))
                y1 = min(h - 1, max(0, cy + dy))
                if z < depth_buf[y0, x]:
                    depth_buf[y0, x] = z
                    frame[y0, x] = col
                if z < depth_buf[y1, x]:
                    depth_buf[y1, x] = z
                    frame[y1, x] = col

            for x in range(cx_l1, cx_r1):
                y0 = min(h - 1, max(0, cy - dx))
                y1 = min(h - 1, max(0, cy + dx))
                if z < depth_buf[y0, x]:
                    depth_buf[y0, x] = z
                    frame[y0, x] = col
                if z < depth_buf[y1, x]:
                    depth_buf[y1, x] = z
                    frame[y1, x] = col

            dy += 1
            t1 = t1 + dy
            t2 = t1 - dx
            if t2 >= 0:
                t1 = t2
                dx = dx - 1
                if dx < dy:
                    break

                cx_l0 = min(w - 1, max(0, cx - dx))
                cx_r0 = min(w - 1, max(0, cx + dx))


@nb.njit
def draw_edge_segments(
    frame: np.ndarray,
    depth_buf: np.ndarray,
    segments: np.ndarray,
    depths: np.ndarray,
    rgba: tuple[np.uint8, np.uint8, np.uint8, np.uint8],
):
    """Draw precomputed edge segments (screen-space endpoints).
    segments: (K,2,2) int32 of screen coords
    depths: (K,2) float32 depths
    """
    h, w, _ = frame.shape
    r, g, b, a = rgba
    for i in range(segments.shape[0]):
        x0, y0 = int(segments[i, 0, 0]), int(segments[i, 0, 1])
        x1, y1 = int(segments[i, 1, 0]), int(segments[i, 1, 1])
        z0, z1 = depths[i, 0], depths[i, 1]

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0

        for _ in range(32768):
            if 0 <= x < w and 0 <= y < h:
                t = 0.0
                if (x1 - x0) != 0:
                    t = (x - x0) / (x1 - x0)
                z = z0 + t * (z1 - z0)
                if z < depth_buf[y, x]:
                    depth_buf[y, x] = z
                    frame[y, x, :] = rgba
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy


@nb.njit
def draw_edge_segments_colors(
    frame: np.ndarray,
    depth_buf: np.ndarray,
    segments: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    depth_interpolate: bool = False,
) -> None:
    """Draw edge segments with per-edge colors.

    Args:
        frame: (H,W,4) uint8. The color frame buffer
        depth_buf: (H,W,1). The depth frame buffer
        segments: (K,2,2) int32
        depths: (K,2) float32
        colors: (K,4) uint8
        depth_interpolate: Depth based alpha fading. Default False.
    """
    h, w, _ = frame.shape

    if depth_interpolate:
        z_min = np.min(depths)
        z_max = np.max(depths)
        if z_min == z_max:
            depth_interpolate = False

    for i in range(segments.shape[0]):
        x0, y0 = int(segments[i, 0, 0]), int(segments[i, 0, 1])
        x1, y1 = int(segments[i, 1, 0]), int(segments[i, 1, 1])
        z0, z1 = depths[i, 0], depths[i, 1]

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0

        for _ in range(32768):
            if 0 <= x < w and 0 <= y < h:
                t = 0.0
                if (x1 - x0) != 0:
                    t = (x - x0) / (x1 - x0)
                z = z0 + t * (z1 - z0)
                if z < depth_buf[y, x]:
                    depth_buf[y, x] = z
                    frame[y, x, :] = colors[i]
                    if depth_interpolate:
                        frame[y, x, 3] *= 1 - ((z - z_min) / (z_max - z_min))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy


@nb.njit
def draw_triangles(
    frame: np.ndarray,
    depth_buf: np.ndarray,
    vertices: np.ndarray,
    depths: np.ndarray,
    rgba: tuple[np.uint8, np.uint8, np.uint8, np.uint8],
):
    """Rasterize filled triangles.

    Args:
        vertices: (K, 3, 2) int32 - 3 vertices per triangle (x,y)
        depths: (K, 3 or 1) float32 - depth for each vertex or each triangle.
        rgba: Solid color
    """
    h, w, _ = frame.shape
    r, g, b, a = rgba

    for i in range(vertices.shape[0]):
        x0, y0 = vertices[i, 0, 0], vertices[i, 0, 1]
        x1, y1 = vertices[i, 1, 0], vertices[i, 1, 1]
        x2, y2 = vertices[i, 2, 0], vertices[i, 2, 1]

        if depths.shape[1] == 3:
            z0, z1, z2 = depths[i, 0], depths[i, 1], depths[i, 2]
        else:
            z0 = z1 = z2 = depths[i, 0]

        # Bounding box
        min_x = max(0, min(x0, min(x1, x2)))
        max_x = min(w - 1, max(x0, max(x1, x2)))
        min_y = max(0, min(y0, min(y1, y2)))
        max_y = min(h - 1, max(y0, max(y1, y2)))

        if min_x > max_x or min_y > max_y:
            continue

        # Barycentric coordinates denominator
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

        if det == 0:
            continue

        inv_det = 1.0 / det

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Barycentric weights
                w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) * inv_det
                w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) * inv_det
                w2 = 1.0 - w0 - w1

                # If inside triangle (handles both winding orders)
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    z = w0 * z0 + w1 * z1 + w2 * z2
                    if z < depth_buf[y, x]:
                        depth_buf[y, x] = z
                        frame[y, x, :] = rgba
