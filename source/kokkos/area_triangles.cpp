// Exercise 3 (Kokkos): total area of all triangles in a 2D mesh.
// Same computation as the CUDA version, but portable: Kokkos runs the parallel
// reduction on whichever backend it was built with (CUDA on a GPU, Serial/OpenMP
// on a CPU). Build backend is chosen when Kokkos is configured, not here.

#include <cstdio>

#include <Kokkos_Core.hpp>

#include "mesh_io.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <mesh.meshb>\n", argv[0]);
        return 1;
    }

    mesh_io::Mesh mesh = mesh_io::read_mesh(argv[1]);
    const int n_ver = static_cast<int>(mesh.coords.size());
    const int n_tri = static_cast<int>(mesh.tris.size());

    Kokkos::initialize(argc, argv);
    {
        // Device views + host mirrors to upload the mesh.
        Kokkos::View<double*> x("x", n_ver), y("y", n_ver);
        Kokkos::View<int*> tri("tri", 3 * n_tri);
        auto hx = Kokkos::create_mirror_view(x);
        auto hy = Kokkos::create_mirror_view(y);
        auto htri = Kokkos::create_mirror_view(tri);

        for (int i = 0; i < n_ver; ++i) { hx(i) = mesh.coords[i][0]; hy(i) = mesh.coords[i][1]; }
        for (int t = 0; t < n_tri; ++t)
            for (int k = 0; k < 3; ++k) htri(3 * t + k) = mesh.tris[t][k] - 1;  // 1-based -> 0-based

        Kokkos::deep_copy(x, hx);
        Kokkos::deep_copy(y, hy);
        Kokkos::deep_copy(tri, htri);

        // One work-item per triangle; reduce the signed (CCW-positive) areas.
        double total = 0.0;
        Kokkos::parallel_reduce("triangle_areas", n_tri,
            KOKKOS_LAMBDA(int t, double& sum) {
                int a = tri(3 * t), b = tri(3 * t + 1), c = tri(3 * t + 2);
                double abx = x(b) - x(a), aby = y(b) - y(a);
                double acx = x(c) - x(a), acy = y(c) - y(a);
                sum += 0.5 * (abx * acy - acx * aby);
            }, total);

        std::printf("triangles: %d, total area = %.15g\n", n_tri, total);
    }
    Kokkos::finalize();
    return 0;
}
