# Packed Data Structures Graph Software Visualizer

A lightweight, Numba-accelerated software renderer designed to visualize graphs directly from `packed_data_structures`. 

## Features
- **Software Rendering**: No OpenGL/GPU setup required.
- **Numba Accelerated**: Fast projection and rasterization kernels.
- **Direct Memory Access**: Works with `PackedArray` views avoiding data duplication.
- **Dynamic & Interactive**: Includes orbit/fly cameras, real-time node/edge updates.