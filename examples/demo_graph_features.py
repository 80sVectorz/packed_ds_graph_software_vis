"""Dynamic graph demo: random edits over time using OverlayedDB API.
Showcases declarative feature-based graph components.
"""

from __future__ import annotations
from typing import cast

import math
import sys
import os

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from packed_data_structures.schemas import DataColSchema
from packed_data_structures.overlays.registry import SchemaRegistry
from packed_data_structures.overlays.database import OverlaidDB
from packed_data_structures.graph.overlay import (
    GraphFeature,
    NodeLayer,
    EdgeLayer,
    GraphOverlay,
)

# Vis Imports
from packed_ds_graph_software_vis.core.layout_engine import (
    ForceDirectedLayoutPhysicsConfig,
    GraphLayoutEngine,
)
from packed_ds_graph_software_vis.core import GraphWidget, ModuleRegistry
from vis_modules.modules import (
    ExampleArrowEdgeModule,
    ExampleNodeIdModule,
)
from vis_modules.grid_module import GridModule

RNG = np.random.default_rng(42)


# --- Physics Feature ---


class NodePhysicsFeature[T: np.generic](GraphFeature):
    """Exposes static schema keys for physics data.

    Type hints provide perfect autocomplete, while __init_subclass__ automatically
    injects the parameterized DataColSchema singletons into the subclass.
    """

    pos: DataColSchema[T, int]
    vel: DataColSchema[T, int]
    fixed: DataColSchema[np.uint8]

    def __init_subclass__(
        cls, prefix: str = "viz", dtype: type[T] | None = None, **kwargs
    ):
        super().__init_subclass__(**kwargs)
        dtype = cast(type[T], dtype or np.float32)
        cls.pos = DataColSchema(f"{prefix}_pos", dtype, shape=(3,), default=0.0)
        cls.vel = DataColSchema(f"{prefix}_vel", dtype, shape=(3,), default=0.0)
        cls.fixed = DataColSchema(f"{prefix}_fixed", np.uint8, default=0)

    def on_schema(self, registry: SchemaRegistry, layer_name: str):
        registry.add_column(layer_name, self.pos)
        registry.add_column(layer_name, self.vel)
        registry.add_column(layer_name, self.fixed)


# --- Weighted Edge Feature ---


class WeightedEdgeFeature[T: np.generic](GraphFeature):
    weight: DataColSchema[T]

    def __init_subclass__(
        cls,
        col_name: str = "weight",
        default_weight: float = 1.0,
        dtype: type[T] | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        dtype = cast(type[T], dtype or np.float32)
        cls.weight = DataColSchema(col_name, dtype, default=default_weight)

    def on_schema(self, registry: SchemaRegistry, layer_name: str):
        registry.add_column(layer_name, self.weight)


# --- Signal Feature (Logic + Data) ---


class SignalFeature(GraphFeature):
    spiked: DataColSchema[np.uint8]

    def __init_subclass__(cls, col_name: str = "spiked", **kwargs):
        super().__init_subclass__(**kwargs)
        cls.spiked = DataColSchema(col_name, np.uint8, default=0)

    def on_schema(self, registry: SchemaRegistry, layer_name: str):
        registry.add_column(layer_name, self.spiked)

    @classmethod
    def propagate(
        cls,
        db: OverlaidDB,
        node_layer: NodeLayer,
        edge_layer: EdgeLayer,
        rng: np.random.Generator,
        weight_class: type[WeightedEdgeFeature],
    ):
        nodes = db.get_table(node_layer)
        edges = db.get_table(edge_layer)

        vw_weights = edges[weight_class.weight].view
        vw_edges_spiked = edges[cls.spiked].view
        vw_nodes_spiked = nodes[cls.spiked].view

        vw_from = edges[edge_layer.src].view
        vw_to = edges[edge_layer.tgt].view

        spiking_initial = np.nonzero(vw_nodes_spiked)[0]
        transmission_chance = vw_weights < rng.uniform(0.0, 1.0, len(edges))

        vw_edges_spiked[:] = vw_nodes_spiked[vw_from] & transmission_chance

        spiked_edges = np.nonzero(vw_edges_spiked)[0]
        vw_nodes_spiked[:] = 0
        vw_nodes_spiked[vw_to[spiked_edges]] = 1
        vw_nodes_spiked[spiking_initial] = 0

        vw_weights *= 0.995
        vw_weights[vw_edges_spiked != 0] *= 1.2
        vw_weights[:] = np.clip(vw_weights, 0.01, 2)

    @classmethod
    def trigger_spike(cls, db: OverlaidDB, node_layer: NodeLayer, node_idx: int):
        db.get_table(node_layer)[cls.spiked].view[node_idx] = 1


# --- Explicit Declarative Subclasses ---
# We can optionally specify the generic typing bracket to strictly type the schema!
class DemoPhysics(NodePhysicsFeature[np.float32], prefix="viz", dtype=np.float32):
    pass


class DemoSignal(SignalFeature, col_name="spiked"):
    pass


class DemoWeight(
    WeightedEdgeFeature[np.float32],
    col_name="weight",
    default_weight=1.0,
    dtype=np.float32,
):
    pass


# --- 2. Initialization ---


def initialize_graph[T_idx: np.generic = np.uint64, T_counts: np.generic = np.uint32](
    n_start: int,
    index_dtype: type[T_idx] | None = None,
    counts_dtype: type[T_counts] | None = None,
) -> tuple[
    OverlaidDB,
    GraphOverlay[T_idx],
    NodeLayer[T_idx],
    EdgeLayer[T_idx, T_counts],
    np.random.Generator,
]:
    if index_dtype is None:
        index_dtype = cast(type[T_idx], np.uint64)
    if counts_dtype is None:
        counts_dtype = cast(type[T_counts], np.uint32)

    graph = GraphOverlay(index_dtype=index_dtype)

    nodes = graph.add_node_layer("nodes", features=[DemoPhysics(), DemoSignal()])
    edges = graph.add_edge_layer(
        "edges",
        source=nodes,
        target=nodes,
        features=[DemoWeight(), DemoSignal()],
        track_counts=True,
        counts_dtype=counts_dtype,
    )

    db = OverlaidDB(graph)

    node_indices = []
    with db.transaction():
        for i in range(n_start):
            idx = nodes.add_entry({DemoPhysics.fixed: 1 if i == 0 else 0})
            node_indices.append(idx)

    vw_pos = db.get_table(nodes)[DemoPhysics.pos].view
    for i, idx in enumerate(node_indices):
        theta = 2 * math.pi * (i / max(1, len(node_indices)))
        vw_pos[idx, 0] = math.cos(theta) * 200.0
        vw_pos[idx, 1] = math.sin(theta) * 200.0

    return db, graph, nodes, edges, RNG


# --- Edit Logic ---


def create_node_with_connections(
    db: OverlaidDB, nodes: NodeLayer, edges: EdgeLayer, rng
):
    if len(db.get_table(nodes)) == 0:
        return

    count = len(db.get_table(nodes))
    anchor = int(rng.integers(0, count))

    with db.transaction():
        vw_pos = db.get_table(nodes)[DemoPhysics.pos].view
        anchor_pos = vw_pos[anchor]
        new_pos = anchor_pos + rng.normal(loc=0, scale=10, size=3)

        idx = nodes.add_entry(
            {
                DemoPhysics.fixed: int(rng.uniform() < 0.01),
                DemoPhysics.pos: new_pos,
            }
        )

        edges.add_entry(
            {
                edges.src: anchor,
                edges.tgt: idx,
                DemoWeight.weight: np.float64(rng.random()),
            }
        )

        num_edges = int(rng.integers(1, min(4, max(2, count))))
        targets = rng.choice(np.arange(count), size=num_edges, replace=False)
        for t in targets:
            if t == idx:
                continue
            edges.add_entry(
                {
                    edges.src: idx,
                    edges.tgt: int(t),
                    DemoWeight.weight: np.float64(rng.random()),
                }
            )


def delete_node(db, nodes: NodeLayer, rng):
    count = len(db.get_table(nodes))
    if count == 0:
        return
    victim = int(rng.integers(0, count))
    if victim != 0:
        with db.transaction():
            db.get_table(nodes).del_entry(victim)


def merge_nodes(db, edges: EdgeLayer, nodes: NodeLayer, rng):
    count = len(db.get_table(nodes))
    if count < 2:
        return
    survivor, victim = rng.choice(np.arange(count), size=2, replace=False)
    if victim == 0:
        return

    touching = edges.get_touching(int(victim))

    tbl_edges = db.get_table(edges)
    vw_src = tbl_edges[edges.src].view
    vw_dst = tbl_edges[edges.tgt].view
    vw_weight = tbl_edges[DemoWeight.weight].view

    with db.transaction():
        for edge_idx in touching:
            e_from = int(vw_src[edge_idx])
            e_to = int(vw_dst[edge_idx])
            w = float(vw_weight[edge_idx])

            new_from = survivor if e_from == victim else e_from
            new_to = survivor if e_to == victim else e_to

            if new_from != new_to:
                edges.add_entry(
                    {
                        edges.src: new_from,
                        edges.tgt: new_to,
                        DemoWeight.weight: w,
                    }
                )

        db.get_table(nodes).del_entry(victim)


def split_node(db, nodes: NodeLayer, edges: EdgeLayer, rng):
    count = len(db.get_table(nodes))
    if count == 0:
        return
    node = int(rng.integers(0, count))
    if node == 0:
        return

    vw_pos = db.get_table(nodes)[DemoPhysics.pos].view
    new_pos = vw_pos[node] + rng.normal(loc=0, scale=30, size=3)

    touching = edges.get_touching(node)
    if not touching:
        return

    tbl_edges = db.get_table(edges)
    vw_src = tbl_edges[edges.src].view
    vw_dst = tbl_edges[edges.tgt].view
    vw_weight = tbl_edges[DemoWeight.weight].view

    with db.transaction():
        new_idx = nodes.add_entry({DemoPhysics.pos: new_pos})

        move_indices = rng.choice(
            touching, size=max(1, len(touching) // 2), replace=False
        )

        for edge_idx in move_indices:
            e_from = int(vw_src[edge_idx])
            e_to = int(vw_dst[edge_idx])
            w = float(vw_weight[edge_idx])

            new_from = new_idx if e_from == node else e_from
            new_to = new_idx if e_to == node else e_to

            if new_from != new_to:
                edges.add_entry(
                    {
                        edges.src: new_from,
                        edges.tgt: new_to,
                        DemoWeight.weight: w,
                    }
                )
                tbl_edges.del_entry(edge_idx)


def main():
    app = QApplication([])
    db, _, nodes, edges, rng = initialize_graph(1, np.uint32)

    tbl_nodes = db.get_table(nodes)
    tbl_edges = db.get_table(edges)

    col_out_degree = tbl_nodes[edges.src.adj_count]
    col_in_degree = tbl_nodes[edges.tgt.adj_count]

    layout = GraphLayoutEngine(
        positions=tbl_nodes[DemoPhysics.pos].arr,
        velocities=tbl_nodes[DemoPhysics.vel].arr,
        fixed_mask=tbl_nodes[DemoPhysics.fixed].arr,
        node_in_degree=col_in_degree.arr,
        node_out_degree=col_out_degree.arr,
        edges_src=tbl_edges[edges.src].arr,
        edges_tgt=tbl_edges[edges.tgt].arr,
        sim_config=ForceDirectedLayoutPhysicsConfig(
            repulsion_strength=1000.0,
            attraction_strength=0.05,
            damping=0.95,
            noise_strength=20,
        ),
    )

    modules = ModuleRegistry(
        [
            GridModule(extent=2000.0, spacing=200.0, color=(60, 60, 60, 255)),
            ExampleArrowEdgeModule(
                graph_id="demo",
                table=edges,
                src_col=edges.src,
                tgt_col=edges.tgt,
                node_table=nodes,
                node_pos_col=DemoPhysics.pos,
                weight_col=DemoWeight.weight,
                spiked_col=DemoSignal.spiked,
                color=(255, 255, 255, 255),
                head_size=3,
                node_radius=4.5,
            ),
            ExampleNodeIdModule(
                graph_id="demo",
                table=nodes,
                pos_col=DemoPhysics.pos,
                spiked_col=DemoSignal.spiked,
                radius=4,
            ),
        ]
    )

    widget = GraphWidget(module_registry=modules, control_mode="fly")
    widget.register_graph(
        "demo",
        db,
        nodes.overlay,
    )
    widget.show()

    edit_count = 0

    def edit_once():
        nonlocal edit_count
        ops = [
            lambda: create_node_with_connections(db, nodes, edges, rng),
            lambda: create_node_with_connections(db, nodes, edges, rng),
            lambda: create_node_with_connections(db, nodes, edges, rng),
            lambda: create_node_with_connections(db, nodes, edges, rng),
            lambda: create_node_with_connections(db, nodes, edges, rng),
            lambda: merge_nodes(db, edges, nodes, rng),
            lambda: merge_nodes(db, edges, nodes, rng),
            lambda: split_node(db, nodes, edges, rng),
            lambda: split_node(db, nodes, edges, rng),
            lambda: delete_node(db, nodes, rng),
        ]
        edit_count += 1
        for _ in range(5):
            rng.choice(ops)()  # type: ignore
        schedule_next_edit()

    edit_timer = QTimer()
    edit_timer.setSingleShot(True)

    spike_trigger_timer = QTimer()
    spike_trigger_timer.setSingleShot(True)

    _edit_connected = False

    def schedule_next_edit():
        nonlocal _edit_connected
        interval = int(rng.integers(1, 1000))
        edit_timer.setInterval(interval)
        if _edit_connected:
            edit_timer.timeout.disconnect()
        edit_timer.timeout.connect(edit_once)
        _edit_connected = True
        edit_timer.start()

    _spike_connected = False

    def schedule_next_spike():
        nonlocal _spike_connected
        interval_ms = int(RNG.integers(500, 5000))
        spike_trigger_timer.setInterval(interval_ms)
        if _spike_connected:
            spike_trigger_timer.timeout.disconnect()

        spike_trigger_timer.timeout.connect(
            lambda: (
                DemoSignal.trigger_spike(
                    db, nodes, int(rng.integers(len(db.get_table(nodes))))
                ),
                schedule_next_spike(),
            )
        )
        _spike_connected = True
        spike_trigger_timer.start()

    spike_update_timer = QTimer()
    spike_update_timer.timeout.connect(
        lambda: DemoSignal.propagate(db, nodes, edges, rng, DemoWeight)
    )
    spike_update_timer.start(5)

    layout_timer = QTimer()
    # layout engine holds views to float32 arrays natively, so we just step it
    layout_timer.timeout.connect(lambda: (layout.step(), widget.update()))
    layout_timer.start(5)

    schedule_next_edit()
    schedule_next_spike()
    app.exec()


if __name__ == "__main__":
    main()
