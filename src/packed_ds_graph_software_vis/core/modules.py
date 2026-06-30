"""Render module interfaces and basic implementations for nodes and edges.

Modules declare which graphs/layers they need. Scene3D handles culling and
returns compact index arrays (no full-size masks). Modules consume the culled
data and draw via kernel helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from collections.abc import Sequence
import numpy as np

from packed_data_structures.schemas import ColSchemaLike, SupportsGetTableSchema
from packed_data_structures.table import PackedArrayTable

from . import kernels
from .framebuffer import Framebuffer
from .scene import Scene3D, NodeCullResult, EdgeCullResult


class RenderModule:
    name: str
    active: bool = True

    def declare(self, scene: Scene3D) -> None:
        """Tell the scene which layers/graphs are needed."""
        raise NotImplementedError

    def render(self, scene: Scene3D, framebuffer: Framebuffer) -> None:
        raise NotImplementedError


type NodeColorFn = Callable[[PackedArrayTable, np.ndarray], np.ndarray]
type EdgeColorFn = Callable[
    [PackedArrayTable | None, np.ndarray], np.ndarray | tuple[int, int, int, int]
]


def _default_node_colors(table: PackedArrayTable, indices: np.ndarray) -> np.ndarray:
    n = len(indices)
    return np.tile(np.array([200, 210, 230, 255], dtype=np.uint8), (n, 1))


def _default_edge_colors(
    table: PackedArrayTable | None, indices: np.ndarray
) -> tuple[int, int, int, int]:
    return (140, 140, 150, 180)


@dataclass
class NodeModule(RenderModule):
    graph_id: str
    table: SupportsGetTableSchema
    pos_col: ColSchemaLike
    lod: str | None = None
    color_fn: NodeColorFn = _default_node_colors
    radius: int = 3
    name: str = "nodes"
    active: bool = True

    def declare(self, scene: Scene3D) -> None:
        scene.require_nodes(self.graph_id, self.table, self.pos_col, self.lod)

    def render(self, scene: Scene3D, framebuffer: Framebuffer) -> None:
        if not self.active:
            return
        cull: NodeCullResult = scene.cull_nodes(
            self.graph_id, self.table, self.pos_col, self.lod
        )
        if len(cull.indices) == 0:
            return

        db = scene.get_graph_db(self.graph_id)
        table = db.get_table(self.table)

        colors = self.color_fn(table, cull.indices).astype(np.uint8, copy=False)

        radii = np.clip(
            self.radius * cull.scales * 500.0,
            self.radius * 0.5,
            self.radius * 4.0,
        ).astype(np.int32)

        kernels.draw_nodes_variable_radius(
            framebuffer.color,
            framebuffer.depth,
            cull.projected,
            cull.depths,
            colors,
            radii,
        )


@dataclass
class EdgeModule(RenderModule):
    graph_id: str
    table: SupportsGetTableSchema
    src_col: ColSchemaLike
    dst_col: ColSchemaLike

    node_table: SupportsGetTableSchema
    node_pos_col: ColSchemaLike

    lod: str | None = None
    color_fn: EdgeColorFn = _default_edge_colors
    name: str = "edges"
    active: bool = True

    def declare(self, scene: Scene3D) -> None:
        scene.require_edges(
            self.graph_id,
            self.table,
            self.src_col,
            self.dst_col,
            self.node_table,
            self.node_pos_col,
            self.lod,
        )

    def render(self, scene: Scene3D, framebuffer: Framebuffer) -> None:
        if not self.active:
            return

        cull: EdgeCullResult = scene.cull_edges(
            self.graph_id,
            self.table,
            self.src_col,
            self.dst_col,
            self.node_table,
            self.node_pos_col,
            self.lod,
        )

        if len(cull.indices) == 0:
            return

        db = scene.get_graph_db(self.graph_id)
        table = db.get_table(self.table)

        colors = self.color_fn(table, cull.indices)

        if isinstance(colors, np.ndarray):
            kernels.draw_edge_segments_colors(
                framebuffer.color,
                framebuffer.depth,
                cull.endpoints,
                cull.depths,
                colors.astype(np.uint8, copy=False),
            )
        else:
            rgba = (
                colors
                if colors is not None
                else _default_edge_colors(table, cull.indices)
            )
            kernels.draw_edge_segments(
                framebuffer.color,
                framebuffer.depth,
                cull.endpoints,
                cull.depths,
                rgba,
            )


class ModuleRegistry:
    """Simple holder for modules with toggle support."""

    def __init__(self, modules: Sequence[RenderModule] | None = None):
        self._modules: dict[str, RenderModule] = {}
        for m in modules or []:
            self.add(m)

    def add(self, module: RenderModule) -> None:
        self._modules[module.name] = module

    def enable(self, name: str) -> None:
        if name in self._modules:
            self._modules[name].active = True

    def disable(self, name: str) -> None:
        if name in self._modules:
            self._modules[name].active = False

    def toggle(self, name: str) -> None:
        if name in self._modules:
            self._modules[name].active = not self._modules[name].active

    def active(self) -> list[RenderModule]:
        return [m for m in self._modules.values() if getattr(m, "active", False)]

    def all(self) -> list[RenderModule]:
        return list(self._modules.values())
