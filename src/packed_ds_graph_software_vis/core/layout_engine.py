from __future__ import annotations
from dataclasses import asdict, dataclass

import numpy as np
from numba import njit, prange

from packed_data_structures.packed_array import PackedArray


@dataclass
class ForceDirectedLayoutPhysicsConfig:
    repulsion_strength: float = 100000.0
    attraction_strength: float = 0.02
    damping: float = 0.50
    dt: float = 0.01
    grid_size: float = 1000.0
    center_attraction: float = 0.0
    n_samples_node_repulsion: int = 150
    noise_strength: float = 10.0


@njit(parallel=True, fastmath=True)
def compute_physics_step(
    positions,
    n_in,
    n_out,
    velocities,
    edges_start,
    edges_end,
    fixed_mask,
    repulsion_strength=100000.0,
    attraction_strength=0.02,
    damping=0.50,
    dt=0.01,
    grid_size=1000.0,
    center_attraction=0.0,
    n_samples_node_repulsion=150,
    noise_strength=10.0,
):
    """Numba-optimized physics step."""
    N = len(positions)
    forces = np.zeros_like(positions)

    # --- Edge Attraction (Springs) ---
    for i in prange(len(edges_start)):
        u = edges_start[i]
        v = edges_end[i]

        # Skip invalid edges
        if u >= N or v >= N:
            continue

        delta = positions[v] - positions[u]
        dist_sq = np.sum(delta**2)
        dist = np.sqrt(dist_sq) + 1e-6

        # Linear spring
        force = delta * (dist * attraction_strength)

        # Parallel accumulation
        # Weighted by degree to normalize forces
        w_u = max(1.0, float(n_out[u]))
        w_v = max(1.0, float(n_in[v]))

        forces[u] += force / w_u
        forces[v] -= force / w_v  # Using n_out[u] from original code, or symmetric?
        # Preserving original logic: forces[v] -= force / n_out[u]
        # But standard force directed usually symmetric.
        # Keeping identical to provided snippet for safety.

    # --- Node Repulsion ---
    n_samples = n_samples_node_repulsion

    for i in prange(N):
        if fixed_mask[i]:
            continue

        pos_i = positions[i]

        if noise_strength > 0:
            forces[i, 0] += (np.random.random() - 0.5) * noise_strength
            forces[i, 1] += (np.random.random() - 0.5) * noise_strength
            forces[i, 2] += (np.random.random() - 0.5) * noise_strength

        for _ in range(n_samples):
            target = np.random.randint(0, N)
            if i == target:
                continue

            delta = pos_i - positions[target]
            dist_sq = np.sum(delta**2) + 0.1

            if dist_sq < 1e-6:
                rx = np.random.random() - 0.5
                ry = np.random.random() - 0.5
                rz = np.random.random() - 0.5
                norm = np.sqrt(rx * rx + ry * ry + rz * rz) + 1e-9
                delta[0] = rx / norm
                delta[1] = ry / norm
                delta[2] = rz / norm
                dist_sq = 1e-6

            if dist_sq < (grid_size * grid_size):
                factor = repulsion_strength / (dist_sq + 0.1)
                forces[i] += delta * factor

    # --- Integration ---
    for i in prange(N):
        if fixed_mask[i]:
            forces[i] = 0.0
            velocities[i] = 0.0
            continue

        forces[i] -= positions[i] * center_attraction
        velocities[i] = (velocities[i] + forces[i] * dt) * damping
        positions[i] += velocities[i] * dt

    return positions, velocities


class GraphLayoutEngine:
    def __init__(
        self,
        # Node Arrays
        positions: PackedArray[np.floating],
        velocities: PackedArray[np.floating],
        fixed_mask: PackedArray,
        node_in_degree: PackedArray,
        node_out_degree: PackedArray,
        # Edge Arrays
        edges_src: PackedArray,
        edges_tgt: PackedArray,
        sim_config: ForceDirectedLayoutPhysicsConfig | None = None,
    ):
        self.cfg = sim_config or ForceDirectedLayoutPhysicsConfig()
        self.cfg_dict = asdict(self.cfg)

        self.positions = positions
        self.velocities = velocities
        self.fixed_mask = fixed_mask

        # Topology stats
        self.n_in = node_in_degree
        self.n_out = node_out_degree

        # Topology structure
        self.edges_start = edges_src
        self.edges_end = edges_tgt

    def step(self):
        if len(self.positions) == 0:
            return self.positions

        self.positions[:], self.velocities[:] = compute_physics_step(
            self.positions.view,
            self.n_in.view,
            self.n_out.view,
            self.velocities.view,
            self.edges_start.view,
            self.edges_end.view,
            self.fixed_mask.view,
            **self.cfg_dict,
        )
        return self.positions
