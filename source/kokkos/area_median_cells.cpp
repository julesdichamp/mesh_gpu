// Exercise 3 (Kokkos): total area of all median (barycentric-dual) cells.
// The median cell of a vertex is, inside each incident triangle, the quad bounded
// by the vertex, the two incident edge midpoints and the triangle centroid. Summed
// over the mesh this equals the total triangle area, which is the check.
//
// Because the three medians meet at the centroid, each triangle contributes exactly
// one third of its area to each of its vertices. We offer that computation two ways,
// selected with --method:
//   geometric : build each vertex's quad explicitly (two sub-triangles) -- an
//               independent computation, so the "== triangle area" check is a real
//               test (and it generalizes to Voronoi duals, where the share != 1/3).
//   third     : use the identity directly, triangle_area / 3 per vertex.
// Same math as the CUDA version; Kokkos runs on whatever backend it was built with.

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>
#include <cxxopts.hpp>

#include "mesh_io.h"

// Signed area of triangle (p, q, r); CCW is positive.
KOKKOS_INLINE_FUNCTION double tri_area(double px, double py, double qx, double qy,
                                       double rx, double ry) {
    return 0.5 * ((qx - px) * (ry - py) - (rx - px) * (qy - py));
}

// Geometric method: for each triangle, split each vertex's quad into two
// sub-triangles (vertex, edge midpoint, centroid) and scatter their areas.
void scatter_geometric(Kokkos::View<double*> x, Kokkos::View<double*> y,
                       Kokkos::View<int*> tri, int n_tri, Kokkos::View<double*> varea) {
    Kokkos::parallel_for("scatter_geometric", n_tri, KOKKOS_LAMBDA(int t) {
        int a = tri(3 * t), b = tri(3 * t + 1), c = tri(3 * t + 2);
        double ax = x(a), ay = y(a), bx = x(b), by = y(b), cx = x(c), cy = y(c);

        double gx = (ax + bx + cx) / 3.0, gy = (ay + by + cy) / 3.0;  // centroid
        double mab_x = 0.5 * (ax + bx), mab_y = 0.5 * (ay + by);      // edge midpoints
        double mbc_x = 0.5 * (bx + cx), mbc_y = 0.5 * (by + cy);
        double mca_x = 0.5 * (cx + ax), mca_y = 0.5 * (cy + ay);

        Kokkos::atomic_add(&varea(a), tri_area(ax, ay, mab_x, mab_y, gx, gy) + tri_area(ax, ay, gx, gy, mca_x, mca_y));
        Kokkos::atomic_add(&varea(b), tri_area(bx, by, mbc_x, mbc_y, gx, gy) + tri_area(bx, by, gx, gy, mab_x, mab_y));
        Kokkos::atomic_add(&varea(c), tri_area(cx, cy, mca_x, mca_y, gx, gy) + tri_area(cx, cy, gx, gy, mbc_x, mbc_y));
    });
}

// A/3 method: for each triangle, scatter a third of its area to each vertex (the
// centroid identity, no polygon construction).
void scatter_third(Kokkos::View<double*> x, Kokkos::View<double*> y,
                   Kokkos::View<int*> tri, int n_tri, Kokkos::View<double*> varea) {
    Kokkos::parallel_for("scatter_third", n_tri, KOKKOS_LAMBDA(int t) {
        int a = tri(3 * t), b = tri(3 * t + 1), c = tri(3 * t + 2);
        double third = tri_area(x(a), y(a), x(b), y(b), x(c), y(c)) / 3.0;

        Kokkos::atomic_add(&varea(a), third);
        Kokkos::atomic_add(&varea(b), third);
        Kokkos::atomic_add(&varea(c), third);
    });
}

int main(int argc, char** argv) {
    try {
        // ---- Command line ---------------------------------------------------
        cxxopts::Options options("area_median_cells",
            "Total area of the median (barycentric-dual) cells of a 2D mesh.");

        options.add_options()
            ("i,input", "Input mesh (.meshb)", cxxopts::value<std::string>())
            ("m,method", "Computation method: geometric or third",
                cxxopts::value<std::string>()->default_value("geometric"))
            ("o,output-dir", "Directory for the per-vertex median-cell areas CSV",
                cxxopts::value<std::string>()->default_value("output"))
            ("h,help", "Print usage");

        options.parse_positional({"input"});
        options.positional_help("<input.meshb>");

        const auto args = options.parse(argc, argv);

        if (args.count("help") || !args.count("input")) {
            std::cout << options.help() << "\n";
            return args.count("input") ? 0 : 1;
        }

        const std::string input  = args["input"].as<std::string>();
        const std::string method = args["method"].as<std::string>();

        if (method != "geometric" && method != "third") {
            throw std::runtime_error("Unknown --method '" + method + "' (use geometric or third)");
        }

        // ---- Read the mesh (shared libMeshb reader) -------------------------
        const mesh_io::Mesh mesh = mesh_io::read_mesh(input);
        const int n_ver = static_cast<int>(mesh.coordinates.size());
        const int n_tri = static_cast<int>(mesh.triangles.size());

        Kokkos::initialize(argc, argv);
        {
            // ---- Upload geometry/connectivity to the device -----------------
            Kokkos::View<double*> x("x", n_ver), y("y", n_ver);
            Kokkos::View<int*> tri("tri", 3 * n_tri);
            auto hx = Kokkos::create_mirror_view(x);
            auto hy = Kokkos::create_mirror_view(y);
            auto htri = Kokkos::create_mirror_view(tri);

            for (int i = 0; i < n_ver; ++i) {
                hx(i) = mesh.coordinates[i][0];
                hy(i) = mesh.coordinates[i][1];
            }
            for (int t = 0; t < n_tri; ++t) {
                for (int k = 0; k < 3; ++k) {
                    htri(3 * t + k) = mesh.triangles[t][k] - 1;  // 1-based -> 0-based
                }
            }

            Kokkos::deep_copy(x, hx);
            Kokkos::deep_copy(y, hy);
            Kokkos::deep_copy(tri, htri);

            // ---- Scatter each triangle's thirds, then reduce ----------------
            Kokkos::View<double*> varea("varea", n_ver);  // zero-initialized
            if (method == "geometric") {
                scatter_geometric(x, y, tri, n_tri, varea);
            } else {
                scatter_third(x, y, tri, n_tri, varea);
            }

            double total = 0.0;
            Kokkos::parallel_reduce("sum_varea", n_ver,
                KOKKOS_LAMBDA(int i, double& s) { s += varea(i); }, total);

            // ---- Per-vertex CSV (1-based vertex ids) -------------------------
            const std::filesystem::path outdir = args["output-dir"].as<std::string>();
            std::filesystem::create_directories(outdir);

            auto h_varea = Kokkos::create_mirror_view(varea);
            Kokkos::deep_copy(h_varea, varea);

            std::ofstream csv(outdir / "median_cell_areas.csv");
            csv << std::setprecision(17);  // full double precision
            csv << "vertex,area\n";
            for (int i = 0; i < n_ver; ++i) {
                csv << (i + 1) << "," << h_varea(i) << "\n";
            }
            std::cout << "Wrote " << (outdir / "median_cell_areas.csv") << "\n";

            // ---- Report -----------------------------------------------------
            std::printf("method: %s, vertices: %d, total median-cell area = %.15g\n",
                        method.c_str(), n_ver, total);
        }
        Kokkos::finalize();
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
