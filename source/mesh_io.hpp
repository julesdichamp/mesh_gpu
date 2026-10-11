// Shared libMeshb reader for 2D triangular .meshb meshes. All exercises read the
// mesh through here (same GmfGotoKwd + GmfGetLin as the original extract_edges),
// so the I/O lives in one place. Vertex ids are kept 1-based, exactly as stored in
// the file; the GPU exercises subtract 1 when they index device arrays.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <libmeshb8.h>

namespace mesh_io {

// A single vertex position (x, y)
using Position = std::array<double, 2>;
// All vertex positions (coords[i] is vertex i+1)
using VertexCoords = std::vector<Position>;
// One triangle's three vertex indices (1-based, as in the file)
using Triangle = std::array<int, 3>;
// The triangle connectivity.
using Connectivity = std::vector<Triangle>;

struct Mesh {
    int ver = 0;           // GMF version (kept so meshes can be written back)
    VertexCoords coords;
    Connectivity tris;
};

// Open a 2D .meshb, read its vertices and triangles, and close it. Throws if the
// file cannot be opened or is not 2D (the exercises are 2D only).
inline Mesh read_mesh(const std::string& path) {
    int ver = 0, dim = 0;
    int64_t m = GmfOpenMesh(path.c_str(), GmfRead, &ver, &dim);

    if (!m)
        throw std::runtime_error("Cannot open mesh file: " + path);

    if (dim != 2)
        throw std::runtime_error("Expected a 2D mesh, got dim=" + std::to_string(dim));

    const int64_t n_ver = GmfStatKwd(m, GmfVertices);
    const int64_t n_tri = GmfStatKwd(m, GmfTriangles);

    Mesh mesh;
    mesh.ver = ver;
    mesh.coords.resize(n_ver);
    mesh.tris.resize(n_tri);

    // Vertices: (x, y, ref) -- the reference tag is read but unused.
    GmfGotoKwd(m, GmfVertices);
    for (int64_t i = 0; i < n_ver; ++i) {
        int ref = 0;
        GmfGetLin(m, GmfVertices, &mesh.coords[i][0], &mesh.coords[i][1], &ref);
    }

    // Triangles: three 1-based vertex ids + ref tag (ignored).
    GmfGotoKwd(m, GmfTriangles);
    for (int64_t i = 0; i < n_tri; ++i) {
        int ref = 0;
        GmfGetLin(m, GmfTriangles, &mesh.tris[i][0], &mesh.tris[i][1], &mesh.tris[i][2], &ref);
    }

    GmfCloseMesh(m);
    return mesh;
}

}  // namespace mesh_io
