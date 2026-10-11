// Exercise 2 (GPU, CUDA): total area of all triangles in a 2D mesh.
// One thread per triangle computes its area via the 2D cross product; a Thrust
// reduction sums them. Triangles are CCW, so the signed area is already positive.

#include <cstdio>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "mesh_io.hpp"

__global__ void triangle_areas(const double* x, const double* y, const int* tri,
                               int n_tri, double* area) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;

    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];

    // 0.5 * cross(B-A, C-A)
    double abx = x[b] - x[a], aby = y[b] - y[a];
    double acx = x[c] - x[a], acy = y[c] - y[a];
    area[t] = 0.5 * (abx * acy - acx * aby);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <mesh.meshb>\n", argv[0]);
        return 1;
    }

    mesh_io::Mesh mesh = mesh_io::read_mesh(argv[1]);
    const int n_ver = static_cast<int>(mesh.coords.size());
    const int n_tri = static_cast<int>(mesh.tris.size());

    // Flatten to plain arrays (structure-of-arrays) for the device.
    std::vector<double> hx(n_ver), hy(n_ver);
    std::vector<int> htri(3 * n_tri);
    for (int i = 0; i < n_ver; ++i) { hx[i] = mesh.coords[i][0]; hy[i] = mesh.coords[i][1]; }
    for (int t = 0; t < n_tri; ++t)
        for (int k = 0; k < 3; ++k) htri[3 * t + k] = mesh.tris[t][k] - 1;  // 1-based -> 0-based

    thrust::device_vector<double> dx(hx), dy(hy), darea(n_tri);
    thrust::device_vector<int> dtri(htri);

    int block = 256, grid = (n_tri + block - 1) / block;
    triangle_areas<<<grid, block>>>(thrust::raw_pointer_cast(dx.data()),
                                    thrust::raw_pointer_cast(dy.data()),
                                    thrust::raw_pointer_cast(dtri.data()), n_tri,
                                    thrust::raw_pointer_cast(darea.data()));
    cudaDeviceSynchronize();

    double total = thrust::reduce(darea.begin(), darea.end(), 0.0);
    std::printf("triangles: %d, total area = %.15g\n", n_tri, total);
    return 0;
}
