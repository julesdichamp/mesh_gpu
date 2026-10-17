# mesh_gpu

Exercises on mesh operations, starting on CPU and moving to GPU. Meshes are read
with [libMeshb](https://github.com/LoicMarechal/libMeshb), visualized with
[ViZiR 4](https://pyamg.saclay.inria.fr/vizir4.html), and taken from the
[unit-square meshes](https://zenodo.org/records/14968025) dataset.

## Exercise 1 (CPU): edges and their neighbours

`source/extract_edges.cpp` reads a 2D triangular `.meshb` mesh and, for every edge,
finds the two triangles sharing it. Orientation picks the sides: an edge is keyed
by its two vertex ids `(low, high)`, and — because the meshes are counter-
clockwise -- the triangle that walks the edge `low -> high` is the **left**
neighbour, the one walking `high -> low` is the **right**. Boundary edges have a
`0` on their missing side.

Outputs (written into the directory passed with `-o`, never into the sources):
- `edges_neighbours.csv` -- `v1,v2,left_tri,right_tri` per edge.
- `edges.meshb` -- the edges as a mesh, openable in ViZiR.

## Exercises 2 & 3 (GPU): triangle and median-cell areas

Two quantities, each computed in parallel on the GPU and summed over the mesh:

- **Triangle areas** -- one thread per triangle, area from the 2D cross product.
- **Median-cell areas** -- the median (barycentric-dual) cell of a vertex is, in
  each incident triangle, the quad (vertex, the two incident edge midpoints, the
  triangle centroid). Each triangle gives a portion to its three vertices,
  scattered per vertex with atomic adds. Because the medians meet at the centroid,
  that portion is exactly `triangle_area / 3`. The median programs take
  `--method geometric|third` to compute it either way:
  - `geometric` (default) builds each dual-cell polygon explicitly, so the check
    below is an independent test (and the approach generalizes to Voronoi duals,
    where the share is *not* a third).
  - `third` uses the `area/3` identity directly (then the check is true by
    construction).

Each triangle's three vertex portions sum to its area, so integrated over the mesh
**Σ triangle areas == Σ median-cell areas** (both equal the domain area) -- that is
the check. Each program prints its total; compare the two.

Provided twice with identical math, once per framework:
- CUDA -- `source/cuda/area_triangles.cu`, `source/cuda/area_median_cells.cu`
- Kokkos -- `source/kokkos/area_triangles.cpp`, `source/kokkos/area_median_cells.cpp`

Validated here on CPU via the Kokkos Serial backend: on `unit_square_132` both
totals are exactly `1`; on `unit_square_11k` they agree to ~1e-14. (The CUDA
sources compile with `nvcc`; running them needs an NVIDIA GPU -- see below.)

## Build and run the project

The project is supported via a `Docker` image, with two Kokkos backends chosen at
build time with `--build-arg KOKKOS_BACKEND=`:

- **GPU image** (default) -- Kokkos on CUDA; set the arch to match your card
  (V100=70, T4=75, A100=80, RTX30=86, L4=89, H100=90):
  ```sh
  docker build -t mesh_gpu --build-arg CUDA_ARCH=75 .
  ```
- **CPU image** -- Kokkos on Serial/OpenMP, for a machine **without** a GPU. The
  `*_kokkos` programs then run on the CPU (no `libcuda` needed), which is handy for
  local testing. (The raw `*_cuda` programs still require an NVIDIA GPU to run.)
  ```sh
  docker build -t mesh_gpu_cpu --build-arg KOKKOS_BACKEND=cpu .
  ```

To launch the docker image with everything setup to visualize using ViZir *from* the Docker image, run:
```sh
xhost +local:docker
docker run -it --rm \
    -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e LIBGL_ALWAYS_SOFTWARE=0 --device /dev/dri \
    -v "$PWD":/workspace \
    mesh_gpu
```

Assuming you launched the `docker run` command one level up from the `source`
directory, build out-of-source (pass the same arch as above so the CUDA targets
match your GPU):
```sh
cd /workspace
cmake -S source -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

The build adapts to what is available: `*_cuda` targets build only when `nvcc` is
present, `*_kokkos` only when Kokkos is found. Run exercise 1 (CPU):
```sh
./build/extract_edges data/your_mesh.meshb -o output
```

Run the area programs and compare the two totals (they must match):
```sh
./build/area_triangles_cuda       data/your_mesh.meshb
./build/area_median_cells_cuda    data/your_mesh.meshb                  # geometric (default)
./build/area_median_cells_cuda    data/your_mesh.meshb --method third   # area/3 shortcut
./build/area_triangles_kokkos     data/your_mesh.meshb
./build/area_median_cells_kokkos  data/your_mesh.meshb --method geometric
```

### Running on a GPU

The GPU programs need a real NVIDIA GPU, which this image uses via `--gpus all`
(and the NVIDIA Container Toolkit on the host). With no local GPU, the simplest
option is a **cloud GPU VM that has Docker + the NVIDIA Container Toolkit** --
e.g. Lambda Cloud, Paperspace, Vast.ai, or AWS `g4dn`/`g5` -- where the workflow
is unchanged:
```sh
# on the GPU VM (nvidia-smi works there)
docker build -t mesh_gpu --build-arg CUDA_ARCH=<your_gpu_sm> .
docker run --gpus all -it --rm -v "$PWD":/workspace mesh_gpu
```
For a free check, **Google Colab** (NVIDIA T4, `sm_75`) works too, but it has no
Docker: install libMeshb + Kokkos in the notebook and compile the sources
directly. Match `CUDA_ARCH`/`-DCMAKE_CUDA_ARCHITECTURES` to the VM's GPU.

## Visualize with ViZiR

To vizualize the original mesh:
```sh
vizir4 -in path/to/your/mesh.meshb     # the original mesh
```
