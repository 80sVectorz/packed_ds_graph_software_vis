"""
CPU-side framebuffer helpers.

All drawing kernels mutate the provided np.uint8 color buffer and float32 depth
buffer in-place; this class only manages allocation and clearing.
"""

from __future__ import annotations

import numpy as np


class Framebuffer:
    def __init__(self, width: int, height: int):
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.color = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.depth = np.full((self.height, self.width), np.inf, dtype=np.float32)

    def clear(self, color: tuple[int, int, int, int] = (10, 10, 12, 255)) -> None:
        self.color[...] = color
        self.depth[...] = np.inf

