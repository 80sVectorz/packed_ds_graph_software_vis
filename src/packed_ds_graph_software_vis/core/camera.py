"""Simple camera helpers for the software-rendered graph view.

The goal is to avoid depending on any external scene graph and keep all math
NumPy-friendly so Numba can jit the hot paths. This module only holds small
matrix utilities and camera state.
"""

from __future__ import annotations

import math
import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a right-handed view matrix."""
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    u = up / (np.linalg.norm(up) + 1e-9)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(
    fov_y_radians: float, aspect: float, z_near: float, z_far: float
) -> np.ndarray:
    """Basic perspective matrix."""
    f = 1.0 / math.tan(fov_y_radians / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


class OrbitCamera:
    """Lightweight orbit camera: maintains target, radius, yaw, pitch.

    Emits view-projection matrix for the renderer.
    """

    def __init__(
        self,
        target: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 10.0,
        yaw: float = 0.3,
        pitch: float = 0.8,
        fov_y_radians: float = math.radians(60.0),
        z_near: float = 0.01,
        z_far: float = 1000.0,
    ):
        self.target = np.array(target, dtype=np.float32)
        self.radius = float(radius)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov_y = float(fov_y_radians)
        self.z_near = float(z_near)
        self.z_far = float(z_far)

    def view_proj(self, aspect: float) -> np.ndarray:
        eye = self._eye_position()
        view = look_at(eye, self.target, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        proj = perspective(self.fov_y, aspect, self.z_near, self.z_far)
        return proj @ view

    def _eye_position(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        x = self.target[0] + self.radius * cp * cy
        y = self.target[1] + self.radius * cp * sy
        z = self.target[2] + self.radius * sp
        return np.array([x, y, z], dtype=np.float32)

    def dolly(self, delta: float) -> None:
        self.radius = max(0.1, self.radius * (1.0 + delta))

    def rotate(self, delta_yaw: float, delta_pitch: float) -> None:
        self.yaw += delta_yaw
        self.pitch = np.clip(
            self.pitch + delta_pitch, -math.pi / 2 + 0.05, math.pi / 2 - 0.05
        )

    def pan(self, dx: float, dy: float, scale: float = 1.0) -> None:
        """Pan the camera target along its right/up vectors.

        dx, dy are screen-space deltas; scale adjusts sensitivity.
        """
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        forward = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
        right = np.array([-sy, cy, 0.0], dtype=np.float32)
        up = np.cross(right, forward)

        pan_world = (-dx * right + dy * up) * (self.radius * 0.002 * scale)
        self.target += pan_world


class FlyCamera:
    """Free-fly camera with yaw/pitch orientation (Unity-style)."""

    def __init__(
        self,
        position: np.ndarray | tuple[float, float, float] = (0.0, -200.0, 120.0),
        yaw: float = 0.0,
        pitch: float = -0.15,
        fov_y_radians: float = math.radians(60.0),
        z_near: float = 0.01,
        z_far: float = 1000.0,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov_y = float(fov_y_radians)
        self.z_near = float(z_near)
        self.z_far = float(z_far)

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        forward = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
        forward /= np.linalg.norm(forward) + 1e-9
        right = np.array([-sy, cy, 0.0], dtype=np.float32)
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-9
        return forward, right, up

    def rotate(self, delta_yaw: float, delta_pitch: float) -> None:
        self.yaw += delta_yaw
        self.pitch = np.clip(
            self.pitch + delta_pitch, -math.pi / 2 + 0.01, math.pi / 2 - 0.01
        )

    def move_local(self, right_amt: float, up_amt: float, forward_amt: float) -> None:
        forward, right, up = self._basis()
        delta = right * right_amt + up * up_amt + forward * forward_amt
        self.position += delta.astype(np.float32)

    def look_at(self, target: np.ndarray | tuple[float, float, float]) -> None:
        """Orient the camera so its forward vector points toward target."""
        tgt = np.asarray(target, dtype=np.float32)
        dir_vec = tgt - self.position
        dir_xy = math.hypot(float(dir_vec[0]), float(dir_vec[1]))
        self.yaw = (
            math.atan2(float(dir_vec[1]), float(dir_vec[0])) if dir_xy != 0 else 0.0
        )
        self.pitch = math.atan2(float(dir_vec[2]), dir_xy)
        self.pitch = np.clip(self.pitch, -math.pi / 2 + 0.01, math.pi / 2 - 0.01)

    def view_matrix(self) -> np.ndarray:
        forward, right, up = self._basis()
        eye = self.position
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -forward
        m[0, 3] = -np.dot(right, eye)
        m[1, 3] = -np.dot(up, eye)
        m[2, 3] = np.dot(forward, eye)
        return m

    def view_proj(self, aspect: float) -> np.ndarray:
        view = self.view_matrix()
        proj = perspective(self.fov_y, aspect, self.z_near, self.z_far)
        return proj @ view
