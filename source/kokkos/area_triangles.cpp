// Exercise 3 (Kokkos): total area of all triangles in a 2D mesh.
// Same computation as the CUDA version, but portable: Kokkos runs on whichever
// backend it was built with (CUDA on a GPU, Serial/OpenMP on a CPU). A single
// parallel_reduce both stores each triangle's area in a View (for the CSV written
// with --output-dir: triangle,v1,v2,v3,area, 1-based) and accumulates the total.
// The triangles are CCW, so each signed area is positive.

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

int main(int argc, char** argv) {
    try {
        // ---- Command line ---------------------------------------------------
        cxxopts::Options options("area_triangles",
            "Total area of all triangles of a 2D mesh.");

        options.add_options()
            ("i,input", "Input mesh (.meshb)", cxxopts::value<std::string>())
            ("o,output-dir", "Directory for the per-triangle areas CSV",
                cxxopts::value<std::string>()->default_value("output"))
            ("h,help", "Print usage");

        options.parse_positional({"input"});
        options.positional_help("<input.meshb>");

        const auto args = options.parse(argc, argv);

        if (args.count("help") || !args.count("input")) {
            std::cout << options.help() << "\n";
            return args.count("input") ? 0 : 1;
        }

        const std::string input = args["input"].as<std::string>();

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

            // ---- Per-triangle areas, then reduce ----------------------------
            Kokkos::View<double*> area("area", n_tri);
            Kokkos::parallel_for("triangle_areas", n_tri, KOKKOS_LAMBDA(int t) {
                int a = tri(3 * t), b = tri(3 * t + 1), c = tri(3 * t + 2);
                double abx = x(b) - x(a), aby = y(b) - y(a);
                double acx = x(c) - x(a), acy = y(c) - y(a);
                area(t) = 0.5 * (abx * acy - acx * aby);
            });

            double total = 0.0;
            Kokkos::parallel_reduce("sum_area", n_tri,
                KOKKOS_LAMBDA(int t, double& sum) { sum += area(t); }, total);


                auto h_area = Kokkos::create_mirror_view(area);
                Kokkos::deep_copy(h_area, area);

            // ---- Per-triangle CSV (1-based ids, as in the mesh) -------------
            const std::filesystem::path outdir = args["output-dir"].as<std::string>();
            std::filesystem::create_directories(outdir);

            auto h_area = Kokkos::create_mirror_view(area);
            Kokkos::deep_copy(h_area, area);

            std::ofstream csv(outdir / "triangle_areas.csv");
            csv << std::setprecision(17);  // full double precision
            csv << "triangle,v1,v2,v3,area\n";
            for (int t = 0; t < n_tri; ++t) {
                csv << (t + 1) << ","
                    << mesh.triangles[t][0] << "," << mesh.triangles[t][1] << ","
                    << mesh.triangles[t][2] << "," << h_area(t) << "\n";
            }
            std::cout << "Wrote " << (outdir / "triangle_areas.csv") << "\n";

            // ---- Report -----------------------------------------------------
            std::printf("triangles: %d, total area = %.15g\n", n_tri, total);
        }
        Kokkos::finalize();
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
