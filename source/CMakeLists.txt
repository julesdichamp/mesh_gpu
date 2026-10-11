cmake_minimum_required(VERSION 3.18)
project(mesh_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# libMeshb provides the imported target "Meshb.8" (carries its include dir).
find_package(libMeshb REQUIRED)

# --- Exercise 1 (CPU): edges and their neighbours -------------------------
find_package(cxxopts REQUIRED)            # target cxxopts::cxxopts
add_executable(extract_edges extract_edges.cpp)
target_link_libraries(extract_edges PRIVATE Meshb.8 cxxopts::cxxopts)

# --- Exercises 2-3 (GPU): triangle and median-cell areas ------------------
# Same two programs in CUDA and in Kokkos; mesh_io.hpp (this dir) is shared.
set(AREA_PROGS area_triangles area_median_cells)

# CUDA targets, only if a CUDA compiler is available.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # Must be set BEFORE enable_language(CUDA), which otherwise defaults to sm_52 --
    # too old for double-precision atomicAdd (needs sm_60+). Default to NVIDIA T4
    # (sm_75); override with -DCMAKE_CUDA_ARCHITECTURES=<sm> for your GPU
    # (V100=70, T4=75, A100=80, A10/RTX30=86, L4=89, H100=90).
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    enable_language(CUDA)
    foreach(p ${AREA_PROGS})
        add_executable(${p}_cuda cuda/${p}.cu)
        target_include_directories(${p}_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${p}_cuda PRIVATE Meshb.8)
        set_target_properties(${p}_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    endforeach()
    message(STATUS "CUDA found: building *_cuda targets (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "CUDA not found: skipping *_cuda targets")
endif()

# Kokkos targets, only if Kokkos is installed (find_package provides Kokkos::kokkos).
# The SAME .cpp runs on whatever backend Kokkos was built with. If that backend is
# CUDA, the source must be compiled by nvcc, so we mark it as a CUDA source; on a
# CPU backend (Serial/OpenMP) it compiles as ordinary C++.
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    foreach(p ${AREA_PROGS})
        add_executable(${p}_kokkos kokkos/${p}.cpp)
        target_include_directories(${p}_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${p}_kokkos PRIVATE Meshb.8 Kokkos::kokkos)
        if(Kokkos_ENABLE_CUDA)
            enable_language(CUDA)
            set_source_files_properties(kokkos/${p}.cpp PROPERTIES LANGUAGE CUDA)
            set_target_properties(${p}_kokkos PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
            # KOKKOS_LAMBDA puts __host__ __device__ on lambdas -> nvcc needs these.
            target_compile_options(${p}_kokkos PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
        endif()
    endforeach()
    message(STATUS "Kokkos found: building *_kokkos targets (CUDA backend: ${Kokkos_ENABLE_CUDA})")
else()
    message(STATUS "Kokkos not found: skipping *_kokkos targets")
endif()
