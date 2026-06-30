"""Scene3D: central manager for graphs, culling, and per-frame caches.

Design goals:
- Operate directly on PackedArray objects and their numpy
  views (DMA-friendly).
- Track which graphs/layers are requested by active modules; only compute culling
  for requested items.
- Provide culled index arrays (no full-size masks) for nodes/edges per frame.
- Handle per-frame restart if graph versions change mid-frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import DTypeLike

from packed_data_structures.graph.overlay import GraphOverlay
from packed_data_structures.overlays.database import OverlaidDB
from packed_data_structures.schemas import SupportsGetTableSchema, ColSchemaLike

from . import camera as cam
from . import kernels


@dataclass(slots=True)
class GraphEntry:
    db: OverlaidDB
    overlay: GraphOverlay


@dataclass(slots=True)
class NodeCullResult:
    indices: np.ndarray  # Indices matching table's index_spec.dtype
    projected: np.ndarray  # (N,2) screen coords aligned with indices
    depths: np.ndarray  # (N,) depths aligned with indices
    scales: np.ndarray  # (N,) perspective scales (inv_w)


@dataclass(slots=True)
class EdgeCullResult:
    indices: np.ndarray  # Indices matching table's index_spec.dtype
    endpoints: np.ndarray  # (K,2,2) screen coords for endpoints in culled order
    depths: np.ndarray  # (K,2) depth per endpoint
    scales: np.ndarray  # (K,2) scale per endpoint


@dataclass(slots=True)
class FrameState:
    frame_stamp: int = 0
    camera: cam.OrbitCamera | None = None
    width: int = 1
    height: int = 1
    graph_versions: dict[str, int] = field(default_factory=dict)
    restart_needed: bool = False


class Scene3D:
    def __init__(
        self,
        lod_bands: dict[str, tuple[float, float]] | None = None,
    ):
        self.graphs: dict[str, GraphEntry] = {}
        self.frame = FrameState()
        self._node_cache: dict[tuple, NodeCullResult] = {}
        self._edge_cache: dict[tuple, EdgeCullResult] = {}

        self._frame_counter = 0
        self._lod_bands = lod_bands or {
            "near": (0.0, 500.0),
            "med": (500.0, 1500.0),
            "far": (1500.0, float("inf")),
        }

    # --- Registration ---
    def register_graph(
        self, graph_id: str, db: OverlaidDB, overlay: GraphOverlay
    ) -> None:
        self.graphs[graph_id] = GraphEntry(db=db, overlay=overlay)

    # --- Frame lifecycle ---
    def begin_frame(self, camera: cam.OrbitCamera, width: int, height: int) -> None:
        self._frame_counter += 1
        self.frame = FrameState(
            frame_stamp=self._frame_counter,
            camera=camera,
            width=width,
            height=height,
            graph_versions={
                gid: self._graph_version(entry.db) for gid, entry in self.graphs.items()
            },
            restart_needed=False,
        )
        self._node_cache.clear()
        self._edge_cache.clear()

    def mark_restart(self) -> None:
        self.frame.restart_needed = True

    # --- Accessors ---
    def get_graph_db(self, graph_id: str) -> OverlaidDB:
        return self.graphs[graph_id].db

    # --- Requirements tracking ---

    def require_nodes(
        self,
        graph_id: str,
        table: SupportsGetTableSchema,
        pos_col: ColSchemaLike,
        lod: str | None = None,
    ) -> None:
        """Declare intent to use these nodes.
        Triggers culling immediately to warm the cache.
        """
        self.cull_nodes(graph_id, table, pos_col, lod)

    def require_edges(
        self,
        graph_id: str,
        table: SupportsGetTableSchema,
        src_col: ColSchemaLike,
        dst_col: ColSchemaLike,
        node_table: SupportsGetTableSchema,
        node_pos_col: ColSchemaLike,
        lod: str | None = None,
    ) -> None:
        """Declare intent to use these edges.
        Triggers culling immediately to warm the cache.
        """
        self.cull_edges(
            graph_id, table, src_col, dst_col, node_table, node_pos_col, lod
        )

    # --- Culling helpers ---

    def cull_nodes(
        self,
        graph_id: str,
        table: SupportsGetTableSchema,
        pos_col: ColSchemaLike,
        lod: str | None = None,
    ) -> NodeCullResult:
        entry = self.graphs[graph_id]
        current_version = self._graph_version(entry.db)

        key = (
            self.frame.frame_stamp,
            graph_id,
            table,
            pos_col,
            lod,
            current_version,
        )
        cached = self._node_cache.get(key)
        if cached is not None:
            return cached

        self._check_version(entry.db, graph_id)

        camera = self.frame.camera
        assert camera is not None
        aspect = self.frame.width / max(1, self.frame.height)
        mvp = camera.view_proj(aspect)

        # Retrieve Data
        table_obj = entry.db.get_table(table)
        positions = table_obj[pos_col].view

        # Retrieve the correct index dtype from the table schema
        index_dtype = table_obj.schema.index_spec.dtype

        if positions.ndim != 2 or positions.shape[1] < 3:
            raise ValueError(f"Position column must be (N,3+); got {positions.shape}")

        projected_all, depths_all, inv_w_all = kernels.project_points(
            positions.astype(np.float32, copy=False),
            mvp.astype(np.float32, copy=False),
            self.frame.width,
            self.frame.height,
        )

        mask = inv_w_all > 0
        margin = 250
        if self.frame.width > 0 and self.frame.height > 0:
            w_bound = self.frame.width + margin
            h_bound = self.frame.height + margin
            mask &= (projected_all[:, 0] >= -margin) & (projected_all[:, 0] < w_bound)
            mask &= (projected_all[:, 1] >= -margin) & (projected_all[:, 1] < h_bound)

        if lod is not None:
            band = self._lod_bands.get(lod)
            if band is not None:
                dist2 = self._node_distances_sq(positions, camera)
                mask &= (dist2 >= band[0] * band[0]) & (dist2 < band[1] * band[1])

        # Use the table's index dtype
        indices = np.nonzero(mask)[0].astype(index_dtype, copy=False)

        projected = projected_all[mask].astype(np.int32, copy=False)
        depths = depths_all[mask].astype(np.float32, copy=False)
        scales = inv_w_all[mask].astype(np.float32, copy=False)

        result = NodeCullResult(
            indices=indices, projected=projected, depths=depths, scales=scales
        )
        self._node_cache[key] = result
        return result

    def cull_edges(
        self,
        graph_id: str,
        table: SupportsGetTableSchema,
        src_col: ColSchemaLike,
        dst_col: ColSchemaLike,
        node_table: SupportsGetTableSchema,
        node_pos_col: ColSchemaLike,
        lod: str | None = None,
    ) -> EdgeCullResult:
        entry = self.graphs[graph_id]
        current_version = self._graph_version(entry.db)

        key = (
            self.frame.frame_stamp,
            graph_id,
            table,
            src_col,
            dst_col,
            lod,
            current_version,
        )
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached

        self._check_version(entry.db, graph_id)

        edge_tbl = entry.db.get_table(table)
        edge_count = len(edge_tbl)

        # Retrieve the correct index dtype from the table schema
        index_dtype = edge_tbl.schema.index_spec.dtype

        if edge_count == 0:
            return self._empty_edge_result(key, dtype=index_dtype)

        # Cull nodes first
        node_cull = self.cull_nodes(graph_id, node_table, node_pos_col, lod)

        if len(node_cull.indices) == 0:
            return self._empty_edge_result(key, dtype=index_dtype)

        idx_to_local = {int(idx): i for i, idx in enumerate(node_cull.indices.tolist())}

        edge_indices: list[int] = []
        endpoints: list[list[np.ndarray]] = []
        depths: list[list[float]] = []
        scales: list[list[float]] = []

        screen_w = self.frame.width
        screen_h = self.frame.height

        edge_from = edge_tbl[src_col].view
        edge_to = edge_tbl[dst_col].view
        missing_val = edge_tbl.schema.index_spec.missing

        for i in range(edge_count):
            a = int(edge_from[i])
            b = int(edge_to[i])

            if a == missing_val or b == missing_val:
                continue

            la = idx_to_local.get(a)
            lb = idx_to_local.get(b)

            if la is None or lb is None:
                continue

            pa = node_cull.projected[la]
            pb = node_cull.projected[lb]

            min_x = min(pa[0], pb[0])
            max_x = max(pa[0], pb[0])
            if max_x < 0 or min_x >= screen_w:
                continue

            min_y = min(pa[1], pb[1])
            max_y = max(pa[1], pb[1])
            if max_y < 0 or min_y >= screen_h:
                continue

            edge_indices.append(i)
            endpoints.append([pa, pb])
            depths.append([node_cull.depths[la], node_cull.depths[lb]])
            scales.append([node_cull.scales[la], node_cull.scales[lb]])

        if edge_indices:
            result = EdgeCullResult(
                indices=np.asarray(edge_indices, dtype=index_dtype),
                endpoints=np.asarray(endpoints, dtype=np.int32),
                depths=np.asarray(depths, dtype=np.float32),
                scales=np.asarray(scales, dtype=np.float32),
            )
        else:
            result = self._empty_edge_result(None, dtype=index_dtype)

        self._edge_cache[key] = result
        return result

    def _empty_edge_result(self, key, dtype: DTypeLike) -> EdgeCullResult:
        res = EdgeCullResult(
            indices=np.zeros((0,), dtype=dtype),
            endpoints=np.zeros((0, 2, 2), dtype=np.int32),
            depths=np.zeros((0, 2), dtype=np.float32),
            scales=np.zeros((0, 2), dtype=np.float32),
        )
        if key:
            self._edge_cache[key] = res
        return res

    def _graph_version(self, db: OverlaidDB) -> int:
        return db.last_dirty_timestamp

    def _check_version(self, db: OverlaidDB, graph_id: str) -> None:
        version_now = self._graph_version(db)
        if version_now != self.frame.graph_versions.get(graph_id):
            self.mark_restart()
            self.frame.graph_versions[graph_id] = version_now

    def _node_distances_sq(
        self, positions: np.ndarray, camera: cam.OrbitCamera
    ) -> np.ndarray:
        if isinstance(camera, cam.OrbitCamera):
            eye = camera._eye_position()
        else:
            eye = np.asarray(camera.position, dtype=np.float32)
        diff = positions[:, :3] - eye[None, :3]
        return np.sum(diff * diff, axis=1, dtype=np.float32)
