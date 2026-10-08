# mesh_gpu

Exercises on mesh operations, starting on CPU and moving to GPU. Meshes are read
with [libMeshb](https://github.com/LoicMarechal/libMeshb), visualized with
[ViZiR 4](https://pyamg.saclay.inria.fr/vizir4.html), and taken from the
[unit-square meshes](https://zenodo.org/records/14968025) dataset.

## Exercise 1 (CPU): edges and their neighbours

`src/extract_edges.cpp` reads a 2D triangular `.meshb` mesh and, for every edge,
finds the two triangles sharing it. Orientation picks the sides: an edge is keyed
by its two vertex ids `(low, high)`, and — because the meshes are counter-
clockwise -- the triangle that walks the edge `low -> high` is the **left**
neighbour, the one walking `high -> low` is the **right**. Boundary edges have a
`0` on their missing side.

Outputs (written into the directory passed with `-o`, never into the sources):
- `edges_neighbours.csv` -- `v1,v2,left_tri,right_tri` per edge.
- `edges.meshb` -- the edges as a mesh, openable in ViZiR.

## Build and run the project

The project is supported via a `Docker` image. To build it, simply run:
```sh
docker build -t mesh_gpu .
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

Assuming you launched the `docker run` command one level up from the `source` directory:
```sh
cd /workspace
mkdir -p build && cd build
cmake ../source/ && make
./build/extract_edges data/your_mesh.meshb -o output
```

## Visualize with ViZiR

To vizualize the original mesh:
```sh
vizir4 -in path/to/your/mesh.meshb     # the original mesh
```
