// Exercise 2 (GPU, CUDA): total area of all triangles in a 2D mesh.
// One thread per triangle computes its area via the 2D cross product; a Thrust
// reduction sums them. The triangles are counter-clockwise, so each signed area is
// already positive.

#include <cassert>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cxxopts.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "mesh_io.h"

// One thread per triangle: store its area via the 2D cross product. 1D kernel.
__global__ void triangle_areas(const double* x, const double* y, const int* tri,
                               int n_tri, double* area) {
    assert(blockDim.y == 1 && blockDim.z == 1);  // launched as a 1D grid of 1D blocks
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) {
        return;
    }

    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    double abx = x[b] - x[a], aby = y[b] - y[a];
    double acx = x[c] - x[a], acy = y[c] - y[a];
    area[t] = 0.5 * (abx * acy - acx * aby);
}

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

        // ---- Upload geometry/connectivity to the device ---------------------
        // Flatten to structure-of-arrays; vertex ids go 1-based -> 0-based.
        std::vector<double> hx(n_ver), hy(n_ver);
        std::vector<int> htri(3 * n_tri);
        for (int i = 0; i < n_ver; ++i) {
            hx[i] = mesh.coordinates[i][0];
            hy[i] = mesh.coordinates[i][1];
        }
        for (int t = 0; t < n_tri; ++t) {
            for (int k = 0; k < 3; ++k) {
                htri[3 * t + k] = mesh.triangles[t][k] - 1; // 1-based -> 0-based
            }
        }

        thrust::device_vector<double> dx(hx), dy(hy), darea(n_tri);
        thrust::device_vector<int> dtri(htri);

        // ---- Compute per-triangle areas, then reduce ------------------------
        int block = 256, grid = (n_tri + block - 1) / block;
        triangle_areas<<<grid, block>>>(thrust::raw_pointer_cast(dx.data()),
                                        thrust::raw_pointer_cast(dy.data()),
                                        thrust::raw_pointer_cast(dtri.data()), n_tri,
                                        thrust::raw_pointer_cast(darea.data()));
        cudaDeviceSynchronize();

        double total = thrust::reduce(darea.begin(), darea.end(), 0.0);

        // ---- Per-triangle CSV (1-based ids, as in the mesh) -----------------
        const std::filesystem::path outdir = args["output-dir"].as<std::string>();
        std::filesystem::create_directories(outdir);

        const thrust::host_vector<double> harea = darea;  // copy device -> host
        std::ofstream csv(outdir / "triangle_areas.csv");
        csv << std::setprecision(17);  // full double precision
        csv << "triangle,v1,v2,v3,area\n";
        for (int t = 0; t < n_tri; ++t) {
            csv << (t + 1) << ","
                << mesh.triangles[t][0] << "," << mesh.triangles[t][1] << ","
                << mesh.triangles[t][2] << "," << harea[t] << "\n";
        }
        std::cout << "Wrote " << (outdir / "triangle_areas.csv") << "\n";

        // ---- Report ---------------------------------------------------------
        std::printf("triangles: %d, total area = %.15g\n", n_tri, total);
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
