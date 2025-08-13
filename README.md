# Julia CUDA Black Hole — Program Overview

This document explains, step‑by‑step, how the Julia/CUDA implementation of the black hole ray‑tracing/geodesic integrator works, how data flows from host → device → host, the shape of outputs, and quick notes to run and validate results.

## Quick checklist
- Describe program modules and responsibilities — Done
- Explain per‑pixel kernel flow and ODE integration — Done
- Show output formats and how to consume them — Done
- Note validation and tuning steps to match the original C++/GLSL pipeline — Done

## Project entry points
- `src/main.jl` — driver and scene builder; prepares camera, black hole and object data, allocates device buffers, launches the CUDA kernel, and collects results.
- `src/BlackHole.jl` — black hole data structure and Schwarzschild radius helper.
- `src/Camera.jl` — camera state and helper `camera_basis` that returns camera position and orientation basis (forward, right, up) used to generate pixel rays.
- `src/Objects.jl` — object (sphere) data structures and `to_device_soa` which converts CPU objects into device-friendly Structure‑of‑Arrays (SoA) CuArrays.
- `src/GPUKernels.jl` — CUDA kernel(s) and device helper functions: RK4 integrator, geodesic RHS, initialization and per‑pixel kernel `geodesic_kernel!`.

## High‑level data flow
1. Host builds the scene: black hole parameters (mass → r_s), camera parameters, and a list of scene objects (positions, radii, colors, masses).
2. Host converts objects to device SoA (CuArrays) via `to_device_soa`.
3. Host allocates per‑pixel device buffers for outputs (hit type, hit index, hit position x/y/z and color RGBA buffers).
4. Host launches the kernel `geodesic_kernel!` with a 2D grid of threads (thread per pixel), passing camera basis, fov/aspect, object buffers, disk radii and integration parameters.
5. Each device thread integrates its ray along the affine parameter using RK4 and writes the resulting hit record (type, position) and color to the device output arrays.
6. Host `synchronize()` then copies these device arrays back to CPU memory for saving, postprocessing, or visualization.

## Per‑pixel kernel behavior (step‑by‑step)
1. Pixel → NDC: compute normalized device coords from pixel coordinates.
2. Ray direction in camera space: use `tanHalfFov` and `aspect` to build a camera‑space direction vector.
3. World‑space direction: combine camera basis (right, up, forward) to produce the ray in world coordinates and normalize.
4. Initialize spherical state: convert camera position to spherical coordinates (r, θ, φ) and set initial derivatives (dr, dθ, dφ) by projecting the world direction onto the spherical basis vectors (e_r, e_θ, e_φ).
5. Compute conserved energy E from the null constraint so the time component is consistent with null geodesics (this follows the GLSL approach and is needed if you later shade by time dilation or do frequency shifts).
6. Integration loop (RK4): advance the six first‑order state y = (r, θ, φ, dr, dθ, dφ) using a 4‑stage RK4 step. The right‑hand side computes the second derivatives using the Schwarzschild metric expressions:
  - d²r/dλ² = - (r_s/(2 r²)) f (dt/dλ)² + (r_s/(2 r² f)) (dr/dλ)² + r (dθ/dλ)² + r sin²θ (dφ/dλ)²
  - d²θ/dλ² = -2 (dr/dλ)(dθ/dλ)/r + sinθ cosθ (dφ/dλ)²
  - d²φ/dλ² = -2 (dr/dλ)(dφ/dλ)/r - 2 cotθ (dθ/dλ)(dφ/dλ)
  where f(r) = 1 - r_s / r and dt/dλ = E / f.
7. After each step, convert the spherical (r,θ,φ) state to Cartesian (x,y,z) for intersection tests.
8. Intersection/termination checks (in the same order as the original GLSL code):
  - Horizon: if r <= r_s (plus a tiny epsilon) mark black hole hit.
  - Disk: detect equatorial plane crossing by sign change in y (previous y * current y < 0) and check radial bounds [disk_r1, disk_r2]. If hit, compute a disk color and break.
  - Object (sphere) intersection: test (x−cx)²+(y−cy)²+(z−cz)² ≤ radius². If true, record object index and do lambertian‑style shading similar to the C++ shader.
  - Escape: if r exceeds a large escape radius, treat as miss and stop integrating.
9. When loop ends, write hit kind, hit position and color into the thread’s slot in the output buffers.

## Output formats produced by `launch_geodesics` (what the host returns)
- `hit_type` :: Vector{UInt8} length = W*H — 0 = miss, 1 = horizon, 2 = disk, 3 = object
- `hit_idx`  :: Vector{Int32} length = W*H — object index (or -1)
- `hit_x, hit_y, hit_z` :: Vector{Float32} length = W*H — world coordinates of the recorded point
- `col_r, col_g, col_b, col_a` :: Vector{Float32} length = W*H — per‑pixel RGBA color matching the GLSL shading logic

These arrays are returned as normal Julia arrays (CPU memory) after the kernel finishes.

## How to create an image from the outputs
1. Convert color channels into an array with shape `(height, width, 4)` and clamp to [0,1].
2. Optionally apply gamma correction (sRGB) and convert to UInt8 0..255.
3. Save as PNG using `FileIO`/`PNGFiles` or any image library, or render directly with GLMakie/Plots.

Example: if `res` is the returned named tuple from `launch_geodesics`, build the image with
```julia
H, W = res.height, res.width
img = reshape(res.col_r, H, W) # repeat for g,b,a and combine to H×W×4
```

## Notes on numerical parity with original C++/GLSL implementation
- Integration scheme: uses full RK4 (k1..k4) for the same 6‑state system — this matches the GLSL math when implemented correctly.
- Step size and steps: the GLSL compute shader uses a large fixed affine step (D_LAMBDA) and many steps (60k). The Julia kernel exposes `h` and `maxSteps` so you can match those to obtain near-identical sampling.
- Precision: the original uses 32‑bit floats in the shader. The Julia kernel uses Float32 by default to match arithmetic. If you require higher fidelity, switch to Float64 and reduce `h` accordingly.
- Scaling: original code uses physical meters (very large numbers). For numerical stability consider rescaling inputs to geometric units (divide lengths by `r_s`) before integration and rescale outputs back for visualization.

## Validation & tests to confirm parity
1. Mirror symmetry: launch two symmetric rays and confirm symmetric hit positions.
2. Photon sphere test: initialize a near‑tangential ray near r = 1.5 * r_s and check that it orbits briefly (unstable photon sphere behaviour).
3. Convergence test: run with `h` and with `h/2`; ensure final hit positions converge (difference below tolerance).
4. Compare a small set of pixels with the original compute shader output (if available) to tune `h` and `maxSteps`.

## Quick run notes
- Kernel launch grid: threads (16,16) and blocks (cld(width,16), cld(height,16)) — same tiling as the GLSL `local_size_x = 16` and dispatch groups.
- Tune `maxSteps` and `step_scale` based on visual fidelity and performance; reduce when testing and increase for final renders.
- Outputs are ready to be saved as images (PNG) or as structured data (JLD2/HDF5/NPY) for later analysis/visualization.

## Where to find the code
- Kernel and integration helpers: `src/GPUKernels.jl`
- Scene & launch wrapper: `src/main.jl`
- Camera utilities: `src/Camera.jl`
- Black hole math: `src/BlackHole.jl`
- Object data and device conversion: `src/Objects.jl`

## Next steps you might want me to implement
- Add an automated PNG/JLD2 writer inside `main.jl` to save each rendered frame.
- Add an adaptive stepper that reduces step near the horizon for improved silhouette without global step increase.
- Add a small CPU verification script that integrates a handful of rays and compares results to the GPU kernel for sanity checks.

---

This file documents the current Julia/CUDA implementation and acts as a reference while tuning and validating the kernel to match (or exceed) the original C++/GLSL outputs.

# Black Hole Simulation in Julia

This project is a 3D simulation of a black hole and surrounding objects, implemented in Julia using CUDA.jl for GPU acceleration. The simulation visualizes gravitational interactions and allows for camera manipulation to explore the scene.

## Project Structure

- **src/**: Contains the main source code for the simulation.
  - **main.jl**: Entry point for the application, initializes the simulation and orchestrates rendering.
  - **BlackHole.jl**: Defines the `BlackHole` struct with properties and methods for gravitational calculations.
  - **Camera.jl**: Manages the camera's position and movement, handling user input for navigation.
  - **Objects.jl**: Holds positional and renderable data for objects in the simulation.
  - **GPUKernels.jl**: Contains GPU kernel functions for parallel computations using CUDA.jl.
  - **Renderer.jl**: Handles the rendering of the simulation, drawing objects and displaying the output.

- **test/**: Contains test cases to ensure the functionality of the simulation components.
  - **runtests.jl**: Script to run the test cases.

- **scripts/**: Contains scripts for running the simulation.
  - **run.jl**: Script to set up the environment and start the simulation.

- **Project.toml**: Configuration file specifying project dependencies.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. **Install Julia**: Ensure you have Julia installed on your system. You can download it from [the official Julia website](https://julialang.org/downloads/).

2. **Install CUDA.jl**: This project requires CUDA.jl for GPU acceleration. You can install it by running the following command in the Julia REPL:
   ```julia
   using Pkg
   Pkg.add("CUDA")
   ```

3. **Clone the Repository**: Clone this repository to your local machine using Git:
   ```bash
   git clone <repository-url>
   cd black_hole_jl
   ```

4. **Run the Simulation**: You can run the simulation by executing the `run.jl` script:
   ```bash
   julia scripts/run.jl
   ```

## Usage

- Use the mouse to navigate around the black hole and observe the gravitational effects on surrounding objects.
- Press the 'G' key to toggle gravity on and off during the simulation.

## Overview

This simulation provides a visual representation of the complex interactions around a black hole, allowing users to explore the effects of gravity in a 3D space. The use of CUDA.jl enables efficient computation, making it possible to simulate a large number of objects and their interactions in real-time.