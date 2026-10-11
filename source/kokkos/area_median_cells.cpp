// Exercise 3 (Kokkos): total area of all median (barycentric-dual) cells.
// Same geometry as the CUDA version: each triangle contributes a quad (vertex,
// edge midpoint, centroid, edge midpoint) to each of its 3 vertices, scattered
// with Kokkos::atomic_add, then reduced. The total must match the triangle area.

#include <cstdio>

#include <Kokkos_Core.hpp>

#include "mesh_io.hpp"

// Signed area of triangle (p, q, r); CCW is positive.
KOKKOS_INLINE_FUNCTION double tri_area(double px, double py, double qx, double qy,
                                       double rx, double ry) {
    return 0.5 * ((qx - px) * (ry - py) - (rx - px) * (qy - py));
}

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

        Kokkos::View<double*> varea("varea", n_ver);  // zero-initialized

        // Scatter each vertex's median-cell portion per triangle.
        Kokkos::parallel_for("median_scatter", n_tri, KOKKOS_LAMBDA(int t) {
            int a = tri(3 * t), b = tri(3 * t + 1), c = tri(3 * t + 2);
            double ax = x(a), ay = y(a), bx = x(b), by = y(b), cx = x(c), cy = y(c);

            double gx = (ax + bx + cx) / 3.0, gy = (ay + by + cy) / 3.0;
            double mab_x = 0.5 * (ax + bx), mab_y = 0.5 * (ay + by);
            double mbc_x = 0.5 * (bx + cx), mbc_y = 0.5 * (by + cy);
            double mca_x = 0.5 * (cx + ax), mca_y = 0.5 * (cy + ay);

            double area_a = tri_area(ax, ay, mab_x, mab_y, gx, gy) + tri_area(ax, ay, gx, gy, mca_x, mca_y);
            double area_b = tri_area(bx, by, mbc_x, mbc_y, gx, gy) + tri_area(bx, by, gx, gy, mab_x, mab_y);
            double area_c = tri_area(cx, cy, mca_x, mca_y, gx, gy) + tri_area(cx, cy, gx, gy, mbc_x, mbc_y);

            Kokkos::atomic_add(&varea(a), area_a);
            Kokkos::atomic_add(&varea(b), area_b);
            Kokkos::atomic_add(&varea(c), area_c);
        });

        double total = 0.0;
        Kokkos::parallel_reduce("sum_varea", n_ver,
            KOKKOS_LAMBDA(int i, double& s) { s += varea(i); }, total);

        std::printf("vertices: %d, total median-cell area = %.15g\n", n_ver, total);
    }
    Kokkos::finalize();
    return 0;
}
