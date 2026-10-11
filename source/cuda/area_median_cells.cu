// Exercise 2 (GPU, CUDA): total area of all median (barycentric-dual) cells.
// The median cell of a vertex is, inside each incident triangle, the quad bounded
// by the vertex, the two incident edge midpoints, and the triangle centroid. Each
// triangle therefore gives a portion to each of its 3 vertices; we compute those
// portions geometrically and scatter them per vertex with atomicAdd, then reduce.
// Summed over the mesh this must equal the total triangle area (same domain area).

#include <cstdio>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "mesh_io.hpp"

// Signed area of triangle (p, q, r); CCW is positive.
__device__ inline double tri_area(double px, double py, double qx, double qy,
                                  double rx, double ry) {
    return 0.5 * ((qx - px) * (ry - py) - (rx - px) * (qy - py));
}

__global__ void median_scatter(const double* x, const double* y, const int* tri,
                               int n_tri, double* varea) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;

    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    double ax = x[a], ay = y[a], bx = x[b], by = y[b], cx = x[c], cy = y[c];

    double gx = (ax + bx + cx) / 3.0, gy = (ay + by + cy) / 3.0;  // centroid
    double mab_x = 0.5 * (ax + bx), mab_y = 0.5 * (ay + by);      // edge midpoints
    double mbc_x = 0.5 * (bx + cx), mbc_y = 0.5 * (by + cy);
    double mca_x = 0.5 * (cx + ax), mca_y = 0.5 * (cy + ay);

    // Each vertex's quad = two triangles (vertex, midpoint, centroid, midpoint).
    double area_a = tri_area(ax, ay, mab_x, mab_y, gx, gy) + tri_area(ax, ay, gx, gy, mca_x, mca_y);
    double area_b = tri_area(bx, by, mbc_x, mbc_y, gx, gy) + tri_area(bx, by, gx, gy, mab_x, mab_y);
    double area_c = tri_area(cx, cy, mca_x, mca_y, gx, gy) + tri_area(cx, cy, gx, gy, mbc_x, mbc_y);

    atomicAdd(&varea[a], area_a);
    atomicAdd(&varea[b], area_b);
    atomicAdd(&varea[c], area_c);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <mesh.meshb>\n", argv[0]);
        return 1;
    }

    mesh_io::Mesh mesh = mesh_io::read_mesh(argv[1]);
    const int n_ver = static_cast<int>(mesh.coords.size());
    const int n_tri = static_cast<int>(mesh.tris.size());

    std::vector<double> hx(n_ver), hy(n_ver);
    std::vector<int> htri(3 * n_tri);
    for (int i = 0; i < n_ver; ++i) { hx[i] = mesh.coords[i][0]; hy[i] = mesh.coords[i][1]; }
    for (int t = 0; t < n_tri; ++t)
        for (int k = 0; k < 3; ++k) htri[3 * t + k] = mesh.tris[t][k] - 1;  // 1-based -> 0-based

    thrust::device_vector<double> dx(hx), dy(hy), dvarea(n_ver, 0.0);
    thrust::device_vector<int> dtri(htri);

    int block = 256, grid = (n_tri + block - 1) / block;
    median_scatter<<<grid, block>>>(thrust::raw_pointer_cast(dx.data()),
                                    thrust::raw_pointer_cast(dy.data()),
                                    thrust::raw_pointer_cast(dtri.data()), n_tri,
                                    thrust::raw_pointer_cast(dvarea.data()));
    cudaDeviceSynchronize();

    double total = thrust::reduce(dvarea.begin(), dvarea.end(), 0.0);
    std::printf("vertices: %d, total median-cell area = %.15g\n", n_ver, total);
    return 0;
}
