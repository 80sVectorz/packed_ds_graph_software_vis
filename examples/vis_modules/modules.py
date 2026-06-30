"""Example render modules for demos."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from packed_data_structures.schemas import ColSchemaLike, SupportsGetTableSchema
from packed_ds_graph_software_vis.core import kernels
from packed_ds_graph_software_vis.core.color_utils import hsva_array_to_rgba
from packed_ds_graph_software_vis.core.modules import RenderModule
from packed_ds_graph_software_vis.core.scene import Scene3D
from packed_ds_graph_software_vis.core.framebuffer import Framebuffer

# --- Tiny bitmap font -------------------------------------------------------
_FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "001", "001", "001"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "-": ["000", "000", "111", "000", "000"],
}


def _raster_text_center(
    frame: np.ndarray,
    cx: int,
    cy: int,
    text: str,
    box_size: int,
    color: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> None:
    if box_size <= 2 or len(text) == 0:
        return
    font_w, font_h = 3, 5
    spacing = 1
    target_h = max(1, box_size // 2)
    scale = max(
        1,
        min(
            target_h // font_h,
            box_size // (font_w * len(text) + spacing * (len(text) - 1) + 1),
        ),
    )

    text_w = len(text) * font_w * scale + spacing * scale * max(0, len(text) - 1)
    text_h = font_h * scale
    start_x = cx - text_w // 2
    start_y = cy - text_h // 2
    h, w, _ = frame.shape

    r, g, b, a = color
    for idx, ch in enumerate(text):
        glyph = _FONT.get(ch)
        if glyph is None:
            continue
        gx = start_x + idx * (font_w * scale + spacing * scale)
        for row in range(font_h):
            for col in range(font_w):
                if glyph[row][col] != "1":
                    continue
                px = gx + col * scale
                py = start_y + row * scale
                for sy in range(scale):
                    y = py + sy
                    if y < 0 or y >= h:
                        continue
                    for sx in range(scale):
                        x = px + sx
                        if x < 0 or x >= w:
                            continue
                        frame[y, x, 0] = r
                        frame[y, x, 1] = g
                        frame[y, x, 2] = b
                        frame[y, x, 3] = a


@dataclass
class ExampleNodeIdModule(RenderModule):
    graph_id: str
    table: SupportsGetTableSchema
    pos_col: ColSchemaLike
    spiked_col: ColSchemaLike | None = None

    radius: int = 8
    name: str = "example_node_ids"
    active: bool = True

    def declare(self, scene: Scene3D) -> None:
        scene.require_nodes(self.graph_id, self.table, self.pos_col)

    def render(self, scene: Scene3D, framebuffer: Framebuffer) -> None:
        if not self.active:
            return

        cull = scene.cull_nodes(self.graph_id, self.table, self.pos_col)
        if len(cull.indices) == 0:
            return

        db = scene.get_graph_db(self.graph_id)
        table_obj = db.get_table(self.table)

        colors = np.tile(
            np.array([255, 0, 0, 255], dtype=np.uint8), (len(cull.indices), 1)
        )

        if self.spiked_col:
            spiked = table_obj[self.spiked_col].view[cull.indices]
            colors[spiked != 0, :] = 255

        ref_dist = 500.0
        radii = (self.radius * cull.scales * ref_dist).astype(np.int32)
        radii = np.maximum(radii, 1)

        kernels.draw_nodes_variable_radius(
            framebuffer.color,
            framebuffer.depth,
            cull.projected,
            cull.depths,
            colors,
            radii,
            True,
        )


@dataclass
class ExampleArrowEdgeModule(RenderModule):
    graph_id: str
    table: SupportsGetTableSchema
    src_col: ColSchemaLike
    tgt_col: ColSchemaLike

    node_table: SupportsGetTableSchema
    node_pos_col: ColSchemaLike

    weight_col: ColSchemaLike | None = None
    spiked_col: ColSchemaLike | None = None

    color: tuple[int, int, int, int] = (60, 60, 60, 255)
    head_size: int = 8
    node_radius: float = 8.0
    name: str = "example_arrow_edges"
    active: bool = True

    def declare(self, scene: Scene3D) -> None:
        scene.require_edges(
            self.graph_id,
            self.table,
            self.src_col,
            self.tgt_col,
            self.node_table,
            self.node_pos_col,
        )

    def render(self, scene: Scene3D, framebuffer: Framebuffer) -> None:
        if not self.active:
            return

        cull = scene.cull_edges(
            self.graph_id,
            self.table,
            self.src_col,
            self.tgt_col,
            self.node_table,
            self.node_pos_col,
        )
        if len(cull.indices) == 0:
            return

        db = scene.get_graph_db(self.graph_id)
        edge_tbl = db.get_table(self.table)

        from_indices = edge_tbl[self.src_col].view
        to_indices = edge_tbl[self.tgt_col].view

        # Group edges connecting the same two nodes (regardless of direction)
        # to apply a visual offset so they don't overlap.
        grouped_edges = defaultdict(list)
        for i, edge_idx in enumerate(cull.indices):
            source_index = int(from_indices[edge_idx])
            target_index = int(to_indices[edge_idx])

            # Create a canonical key by sorting the indices.
            # This ensures A->B and B->A fall into the same group.
            pair_key = tuple(sorted((source_index, target_index)))
            grouped_edges[pair_key].append(i)

        # We work on a copy of endpoints to apply visual offsets and clipping
        final_endpoints = cull.endpoints.copy()
        reference_distance = 500.0  # Matches NodeModule scaling

        # Apply Offsets to overlapping edges
        for pair_key, edge_list_indices in grouped_edges.items():
            count = len(edge_list_indices)
            if count <= 1:
                continue

            # Sort indices by edge ID to ensure stable ordering between frames (prevents flickering)
            edge_list_indices.sort(key=lambda idx: cull.indices[idx])

            # Retrieve scale for the group to ensure the gap shrinks with distance.
            # We use the average scale of the endpoints of the first edge in the group.
            first_cull_idx = edge_list_indices[0]
            scale_source = cull.scales[first_cull_idx, 0]
            scale_target = cull.scales[first_cull_idx, 1]
            average_scale = (scale_source + scale_target) * 0.5

            # Base spacing in reference units (approx pixels at reference distance)
            base_spacing = 10.0
            perspective_spacing = base_spacing * average_scale * reference_distance

            for rank, list_index in enumerate(edge_list_indices):
                # Check direction relative to the canonical key
                edge_idx = cull.indices[list_index]
                source_index = int(from_indices[edge_idx])
                target_index = int(to_indices[edge_idx])

                # If source > target, this edge runs 'backwards' relative to our sorted key
                is_reversed_direction = source_index > target_index

                # Get current screen coordinates
                screen_start = final_endpoints[list_index, 0]
                screen_end = final_endpoints[list_index, 1]

                delta_x = float(screen_end[0] - screen_start[0])
                delta_y = float(screen_end[1] - screen_start[1])

                # Normalize direction. We want the offset vector to always be calculated
                # relative to the canonical "min -> max" direction so parallel edges
                # shift in the same visual direction.
                if is_reversed_direction:
                    delta_x = -delta_x
                    delta_y = -delta_y

                segment_length = (delta_x * delta_x + delta_y * delta_y) ** 0.5
                if segment_length < 1e-3:
                    continue

                # Calculate perpendicular vector (-y, x)
                perpendicular_x = -delta_y / segment_length
                perpendicular_y = delta_x / segment_length

                # Calculate offset amount.
                # Centers the group around the main line (e.g. -1.5, -0.5, 0.5, 1.5)
                offset_multiplier = rank - (count - 1) / 2.0

                offset_x = int(
                    perpendicular_x * offset_multiplier * perspective_spacing
                )
                offset_y = int(
                    perpendicular_y * offset_multiplier * perspective_spacing
                )

                # Apply offset to both start and end points
                final_endpoints[list_index, 0, 0] += offset_x
                final_endpoints[list_index, 0, 1] += offset_y
                final_endpoints[list_index, 1, 0] += offset_x
                final_endpoints[list_index, 1, 1] += offset_y

        screen_starts = final_endpoints[:, 0]
        screen_ends = final_endpoints[:, 1]

        # Perspective scaling factors for the start and end nodes
        source_scales = cull.scales[:, 0]
        target_scales = cull.scales[:, 1]

        # Calculate the visual radius of the nodes in screen pixels
        source_radii_pixels = self.node_radius * source_scales * reference_distance
        target_radii_pixels = self.node_radius * target_scales * reference_distance

        # Clamp to minimums (1.0px) to avoid errors with tiny/far-away nodes
        source_radii_pixels = np.clip(source_radii_pixels, 1.0, None)
        target_radii_pixels = np.clip(target_radii_pixels, 1.0, None)

        delta_x = (screen_ends[:, 0] - screen_starts[:, 0]).astype(float)
        delta_y = (screen_ends[:, 1] - screen_starts[:, 1]).astype(float)
        segment_lengths = (delta_x * delta_x + delta_y * delta_y) ** 0.5

        # Cull edges that are visually shorter than the combined radii of their nodes.
        total_clipping = source_radii_pixels + target_radii_pixels
        visible_edges = np.nonzero(
            (segment_lengths > 1e-3) | (segment_lengths > total_clipping)
        )[0]
        final_cull_indices = cull.indices[visible_edges]

        final_endpoints = final_endpoints[visible_edges]
        segment_lengths = segment_lengths[visible_edges]
        screen_starts = screen_starts[visible_edges].astype(int)
        screen_ends = screen_ends[visible_edges].astype(int)

        # Unit direction vector
        unit_dir_x = delta_x[visible_edges] / segment_lengths
        unit_dir_y = delta_y[visible_edges] / segment_lengths

        # Clip Start Point (Source): Move 'out' from the center
        # Subtract 0.5 for sub-pixel overlap avoidance
        start_retract = source_radii_pixels[visible_edges] - 0.5
        final_endpoints[:, 0, 0] = (
            screen_starts[:, 0] + unit_dir_x * start_retract
        ).astype(int)
        final_endpoints[:, 0, 1] = (
            screen_starts[:, 1] + unit_dir_y * start_retract
        ).astype(int)

        # Clip End Point (Target): Move 'back' from the center
        end_retract = target_radii_pixels[visible_edges] - 0.5
        final_endpoints[:, 1, 0] = (
            screen_ends[:, 0] - unit_dir_x * end_retract
        ).astype(int)
        final_endpoints[:, 1, 1] = (
            screen_ends[:, 1] - unit_dir_y * end_retract
        ).astype(int)

        # Clip Segments to Node Perimeters (Double-sided)
        # for i in range(len(cull.indices)):
        #     screen_start = final_endpoints[i, 0]
        #     screen_end = final_endpoints[i, 1]

        #     # Perspective scaling factors for the start and end nodes
        #     source_scale = cull.scales[i, 0]
        #     target_scale = cull.scales[i, 1]

        #     # Calculate the visual radius of the nodes in screen pixels
        #     source_radius_pixels = self.node_radius * source_scale * reference_distance
        #     target_radius_pixels = self.node_radius * target_scale * reference_distance

        #     # Clamp to minimums (1.0px) to avoid errors with tiny/far-away nodes
        #     source_radius_pixels = max(source_radius_pixels, 1.0)
        #     target_radius_pixels = max(target_radius_pixels, 1.0)

        #     delta_x = float(screen_end[0] - screen_start[0])
        #     delta_y = float(screen_end[1] - screen_start[1])
        #     segment_length = (delta_x * delta_x + delta_y * delta_y) ** 0.5

        #     # If the edge is shorter than the combined radii, it's inside the nodes; hide it.
        #     total_clipping = source_radius_pixels + target_radius_pixels
        #     if segment_length < 1e-3 or segment_length <= total_clipping:
        #         final_endpoints[i, 0] = screen_start  # Collapse segment
        #         final_endpoints[i, 1] = screen_start
        #         continue

        #     # Unit direction vector
        #     unit_dir_x = delta_x / segment_length
        #     unit_dir_y = delta_y / segment_length

        #     # Clip Start Point (Source): Move 'out' from the center
        #     # Subtract 0.5 for sub-pixel overlap avoidance
        #     start_retract = source_radius_pixels - 0.5
        #     final_endpoints[i, 0, 0] = int(screen_start[0] + unit_dir_x * start_retract)
        #     final_endpoints[i, 0, 1] = int(screen_start[1] + unit_dir_y * start_retract)

        #     # Clip End Point (Target): Move 'back' from the center
        #     end_retract = target_radius_pixels - 0.5
        #     final_endpoints[i, 1, 0] = int(screen_end[0] - unit_dir_x * end_retract)
        #     final_endpoints[i, 1, 1] = int(screen_end[1] - unit_dir_y * end_retract)

        # avg_depths = np.min(cull.depths, axis=1)
        # depths_t = 1 - (avg_depths - cull.depths.min()) / (
        #     cull.depths.max() - cull.depths.min()
        # )

        if len(visible_edges) == 0:
            return

        colors_hsv = np.repeat(
            ((1.0, 0.7, 0.9, 1.0),), len(final_endpoints), axis=0
        ).astype(np.float64)

        node_ent_ids = from_indices[final_cull_indices]

        if self.weight_col is not None:
            colors_hsv[:, 2] = np.clip(
                edge_tbl[self.weight_col].view[final_cull_indices], 0, 1
            )  # node_ent_ids % 100 / 100

        colors_hsv[:, 0] = node_ent_ids % 100 / 100

        colors_rgba = hsva_array_to_rgba(colors_hsv)

        if self.spiked_col is not None:
            colors_rgba[edge_tbl[self.spiked_col][final_cull_indices] != 0] = 255
        # colors_hsv[:, 3] = depths_t

        # Draw Lines
        kernels.draw_edge_segments_colors(
            framebuffer.color,
            framebuffer.depth,
            final_endpoints,
            cull.depths[visible_edges],
            colors_rgba,
            depth_interpolate=True,
        )

        # Draw Filled Arrowheads
        segment_endpoints = final_endpoints

        deltas = segment_endpoints[:, 1] - segment_endpoints[:, 0]

        segment_lengths = (deltas[:, 0] ** 2 + deltas[:, 1] ** 2) ** 0.5

        unit_dirs = deltas / (segment_lengths[:, np.newaxis] + 1e-9)

        # Perpendicular vector for arrow width
        perps = unit_dirs[:, (1, 0)] * (-1, 1)

        # Calculate arrow size based on perspective scale at the target
        target_scale = cull.scales[visible_edges, 1]
        scale_factor = target_scale * reference_distance
        arrow_sizes = float(self.head_size) * scale_factor

        # Skip drawing if arrow is too small to see
        visible_arrows = np.nonzero(arrow_sizes >= 2.0)[0]
        arrow_sizes = arrow_sizes[visible_arrows]
        perps = perps[visible_arrows]
        unit_dirs = unit_dirs[visible_arrows, :]

        # The arrow tip is exactly at the end of the line (which is at the node perimeter)
        arrow_tips = segment_endpoints[visible_arrows, 1]

        # The base of the arrow is pulled back along the line
        arrow_bases = arrow_tips - unit_dirs * arrow_sizes[:, np.newaxis]

        # Calculate the left and right corners of the arrowhead
        # 0.6 is the width-to-length ratio
        width = arrow_sizes * 0.6
        perps = (
            perps * width[:, np.newaxis]
        )  # np.stack((perp_x * width, perp_y * width), axis=1)
        arrows_left = arrow_bases + perps
        arrows_right = arrow_bases - perps

        # Create final vertex and depth buffers
        arrow_tris = np.stack((arrow_tips, arrows_left, arrows_right), axis=1)

        # Use the target's depth for all vertices to avoid self-occlusion artifacts
        arrow_depths = np.array(
            cull.depths[visible_edges[visible_arrows], 1], dtype=np.float32
        )

        # arrow_depths.append([target_depth, target_depth, target_depth])

        # for i in range(len(cull.indices)):
        #     segment_endpoints = final_endpoints[i]
        #     depth_values = cull.depths[i]
        #     scale_factors = cull.scales[i]

        #     x0 = float(segment_endpoints[0, 0])
        #     y0 = float(segment_endpoints[0, 1])
        #     x1 = float(segment_endpoints[1, 0])
        #     y1 = float(segment_endpoints[1, 1])

        #     # Check for collapsed segments
        #     if x0 == x1 and y0 == y1:
        #         continue

        #     delta_x = x1 - x0
        #     delta_y = y1 - y0
        #     segment_length = (delta_x * delta_x + delta_y * delta_y) ** 0.5

        #     if segment_length < 1e-3:
        #         continue

        #     unit_dir_x = delta_x / segment_length
        #     unit_dir_y = delta_y / segment_length

        #     # Perpendicular vector for arrow width
        #     perp_x = -unit_dir_y
        #     perp_y = unit_dir_x

        #     # Calculate arrow size based on perspective scale at the target
        #     target_scale = scale_factors[1]
        #     scale_factor = target_scale * reference_distance
        #     arrow_size = float(self.head_size) * scale_factor

        #     # Skip drawing if arrow is too small to see
        #     if arrow_size < 2.0:
        #         continue

        #     # The arrow tip is exactly at the end of the line (which is at the node perimeter)
        #     tip_x = x1
        #     tip_y = y1

        #     # The base of the arrow is pulled back along the line
        #     base_x = tip_x - unit_dir_x * arrow_size
        #     base_y = tip_y - unit_dir_y * arrow_size

        #     # Calculate the left and right corners of the arrowhead
        #     # 0.6 is the width-to-length ratio
        #     width = arrow_size * 0.6
        #     left_x = base_x + perp_x * width
        #     left_y = base_y + perp_y * width
        #     right_x = base_x - perp_x * width
        #     right_y = base_y - perp_y * width

        #     # Add triangle vertices
        #     arrow_triangles.append(
        #         [[tip_x, tip_y], [left_x, left_y], [right_x, right_y]]
        #     )

        #     # Use the target's depth for all vertices to avoid self-occlusion artifacts
        #     target_depth = depth_values[1]
        #     arrow_depths.append([target_depth, target_depth, target_depth])

        if len(arrow_tris):
            kernels.draw_triangles(
                framebuffer.color,
                framebuffer.depth,
                arrow_tris,
                arrow_depths[:, np.newaxis],
                self.color,
            )
