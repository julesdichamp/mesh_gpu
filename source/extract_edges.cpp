// Exercise 1 (CPU): extract every edge of a 2D triangular mesh together with its
// left and right neighbouring triangle. The mesh is read through mesh_io.hpp.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include <cxxopts.hpp>
#include <libmeshb8.h>

#include "mesh_io.hpp"

using mesh_io::VertexCoords;
using mesh_io::Connectivity;

// The two triangles sharing an edge. "left"/"right" are defined relative to the
// edge directed from its lower to its higher vertex index: with the counter-
// clockwise triangles that INRIA meshes use, the triangle walking the edge
// low->high lies to its left, the one walking high->low lies to its right.
// Ids are 1-based; -1 means "no neighbour" (the edge is on the boundary).
struct Neighbours {
    int left  = -1;
    int right = -1;
};

// The map of edge (sorted vertex pair) to its triangle neighbours.
using EdgeNeighbors = std::map<std::pair<int, int>, Neighbours>;

// Build the edge -> (left, right) table from the triangle connectivity. Pure:
// it only reads the input array and returns the result.
auto extract_edge_neighbours(const Connectivity& triangles) -> EdgeNeighbors {
    EdgeNeighbors edges;

    for (std::size_t t = 0; t < triangles.size(); ++t) {
        const int tri_id = static_cast<int>(t) + 1;  // 1-based, as in the file

        // Walk the three edges in the triangle's (counter-clockwise) order.
        for (int e = 0; e < 3; ++e) {
            const int a = triangles[t][e];
            const int b = triangles[t][(e + 1) % 3];

            // Sort endpoints so both adjacent triangles share one key.
            const auto key = std::minmax(a, b);

            // a<b => this triangle walks the edge low->high, so it is the left.
            if (a < b)
                edges[{key.first, key.second}].left = tri_id;
            else
                edges[{key.first, key.second}].right = tri_id;
        }
    }

    return edges;
}

// Write the neighbour table as CSV: one row per edge.
void write_edges_csv(const std::filesystem::path& path, const EdgeNeighbors& edges) {
    std::ofstream csv(path);
    csv << "v1,v2,left_tri,right_tri\n";

    for (const auto& [verts, nb] : edges)
        csv << verts.first << "," << verts.second << ","
            << nb.left << "," << nb.right << "\n";
}

// Write the extracted edges as a .meshb (vertices + edges) for ViZiR.
void write_edges_mesh(const std::filesystem::path& path, const VertexCoords& coords,
                      const EdgeNeighbors& edges, int ver) {
    const int dim = 2;  // 2D exercise
    int64_t out = GmfOpenMesh(path.c_str(), GmfWrite, ver, dim);

    if (!out)
        throw std::runtime_error("Cannot write mesh file: " + path.string());

    GmfSetKwd(out, GmfVertices, static_cast<int64_t>(coords.size()));
    for (const auto& c : coords)
        GmfSetLin(out, GmfVertices, c[0], c[1], 0);

    GmfSetKwd(out, GmfEdges, static_cast<int64_t>(edges.size()));
    for (const auto& [verts, nb] : edges)
        GmfSetLin(out, GmfEdges, verts.first, verts.second, 0);

    GmfCloseMesh(out);
}

int main(int argc, char** argv) {
    try {
        // ---- Command line ---------------------------------------------------
        cxxopts::Options options("extract_edges",
            "Extract mesh edges with their left/right neighbouring triangles.");

        options.add_options()
            ("i,input", "Input mesh (.meshb)", cxxopts::value<std::string>())
            ("o,output-dir", "Directory for generated outputs",
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
        const std::filesystem::path outdir = args["output-dir"].as<std::string>();

        // Keep outputs out of the source tree.
        std::filesystem::create_directories(outdir);

        // ---- Read the mesh (shared libMeshb reader) -------------------------
        const mesh_io::Mesh mesh = mesh_io::read_mesh(input);

        std::cout << "Mesh " << input << " (version " << mesh.ver << ", 2D): "
                  << mesh.coords.size() << " vertices, " << mesh.tris.size()
                  << " triangles\n";

        // ---- Extract the edge neighbours ------------------------------------
        const auto edges = extract_edge_neighbours(mesh.tris);

        // ---- Write the outputs ----------------------------------------------
        const std::filesystem::path csv_path  = outdir / "edges_neighbours.csv";
        const std::filesystem::path mesh_path = outdir / "edges.meshb";

        write_edges_csv(csv_path, edges);
        write_edges_mesh(mesh_path, mesh.coords, edges, mesh.ver);

        // ---- Summary --------------------------------------------------------
        const auto boundary = std::count_if(edges.begin(), edges.end(),
            [](const auto& e) { return e.second.left == -1 || e.second.right == -1; });

        std::cout << "Extracted " << edges.size() << " unique edges ("
                  << boundary << " boundary, " << edges.size() - boundary
                  << " interior)\n";
        std::cout << "Wrote " << csv_path << " and " << mesh_path << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
