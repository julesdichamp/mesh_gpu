cmake_minimum_required(VERSION 3.16)
project(mesh_gpu_ex1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# libMeshb provides the imported target "Meshb.8"; cxxopts provides
# "cxxopts::cxxopts". In Docker both live in system paths; for a local build
# point CMake at their prefix with -DCMAKE_PREFIX_PATH=<prefix>.
find_package(libMeshb REQUIRED)
find_package(cxxopts REQUIRED)

add_executable(extract_edges src/extract_edges.cpp)

target_link_libraries(extract_edges PRIVATE Meshb.8 cxxopts::cxxopts)
