// Exercise 2 (GPU, CUDA): total area of all median (barycentric-dual) cells.
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

// Signed area of triangle (p, q, r); CCW is positive.
__device__ inline double tri_area(double px, double py, double qx, double qy,
                                  double rx, double ry) {
    return 0.5 * ((qx - px) * (ry - py) - (rx - px) * (qy - py));
}

// Geometric method: one thread per triangle; each vertex's quad is split into two
// sub-triangles (vertex, edge midpoint, centroid) whose areas are scattered to it.
__global__ void scatter_geometric(const double* x, const double* y, const int* tri,
                                  int n_tri, double* varea) {
    assert(blockDim.y == 1 && blockDim.z == 1);  // launched as a 1D grid of 1D blocks
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) {
        return;
    }

    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    double ax = x[a], ay = y[a], bx = x[b], by = y[b], cx = x[c], cy = y[c];

    double gx = (ax + bx + cx) / 3.0, gy = (ay + by + cy) / 3.0;  // centroid
    double mab_x = 0.5 * (ax + bx), mab_y = 0.5 * (ay + by);      // edge midpoints
    double mbc_x = 0.5 * (bx + cx), mbc_y = 0.5 * (by + cy);
    double mca_x = 0.5 * (cx + ax), mca_y = 0.5 * (cy + ay);

    atomicAdd(&varea[a], tri_area(ax, ay, mab_x, mab_y, gx, gy) + tri_area(ax, ay, gx, gy, mca_x, mca_y));
    atomicAdd(&varea[b], tri_area(bx, by, mbc_x, mbc_y, gx, gy) + tri_area(bx, by, gx, gy, mab_x, mab_y));
    atomicAdd(&varea[c], tri_area(cx, cy, mca_x, mca_y, gx, gy) + tri_area(cx, cy, gx, gy, mbc_x, mbc_y));
}

// A/3 method: one thread per triangle; scatter a third of the triangle area to each
// of its three vertices (the centroid identity, no polygon construction).
__global__ void scatter_third(const double* x, const double* y, const int* tri,
                              int n_tri, double* varea) {
    assert(blockDim.y == 1 && blockDim.z == 1);  // launched as a 1D grid of 1D blocks
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) {
        return;
    }

    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    double third = tri_area(x[a], y[a], x[b], y[b], x[c], y[c]) / 3.0;

    atomicAdd(&varea[a], third);
    atomicAdd(&varea[b], third);
    atomicAdd(&varea[c], third);
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

        thrust::device_vector<double> dx(hx), dy(hy), dvarea(n_ver, 0.0);
        thrust::device_vector<int> dtri(htri);

        // ---- Scatter each triangle's thirds, then reduce --------------------
        const double* px = thrust::raw_pointer_cast(dx.data());
        const double* py = thrust::raw_pointer_cast(dy.data());
        const int*    pt = thrust::raw_pointer_cast(dtri.data());
        double*       pv = thrust::raw_pointer_cast(dvarea.data());

        int block = 256, grid = (n_tri + block - 1) / block;
        if (method == "geometric") {
            scatter_geometric<<<grid, block>>>(px, py, pt, n_tri, pv);
        } else {
            scatter_third<<<grid, block>>>(px, py, pt, n_tri, pv);
        }
        cudaDeviceSynchronize();

        double total = thrust::reduce(dvarea.begin(), dvarea.end(), 0.0);

        // ---- Per-vertex CSV (1-based vertex ids) ----------------------------
        const std::filesystem::path outdir = args["output-dir"].as<std::string>();
        std::filesystem::create_directories(outdir);

        const thrust::host_vector<double> hvarea = dvarea;  // copy device -> host
        std::ofstream csv(outdir / "median_cell_areas.csv");
        csv << std::setprecision(17);  // full double precision
        csv << "vertex,area\n";
        for (int i = 0; i < n_ver; ++i) {
            csv << (i + 1) << "," << hvarea[i] << "\n";
        }
        std::cout << "Wrote " << (outdir / "median_cell_areas.csv") << "\n";

        // ---- Report ---------------------------------------------------------
        std::printf("method: %s, vertices: %d, total median-cell area = %.15g\n",
                    method.c_str(), n_ver, total);
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
