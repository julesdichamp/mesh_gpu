# Base on the CUDA "devel" image so the SAME image serves the CPU exercises now
# and the GPU ones later. It is Ubuntu 24.04 under the hood: on a laptop without
# an NVIDIA GPU it simply runs CPU-only; on a GPU host you add `--gpus all`.
# 12.6.3 is a conservative, broadly driver-compatible CUDA 12.x; bump it to match
# your GPU host's driver when you get there.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Qt (which ViZiR is built on) uses the X11 MIT-SHM extension by default, but
# shared memory does not cross the container boundary over a forwarded X socket
# and the window then silently fails to appear. Disabling it is the standard fix.
ENV QT_X11_NO_MITSHM=1

# ViZiR needs HARDWARE OpenGL 4. Software Mesa (llvmpipe/softpipe) does not render
# its geometry (blank window or crash), so a real GPU must be reachable: pass the
# host GPU with `--device /dev/dri` (Intel/AMD) or `--gpus all` (NVIDIA). Without
# one, build/run the exercises here but visualize with ViZiR on the host instead.
# (We deliberately do NOT force LIBGL_ALWAYS_SOFTWARE, which would block the GPU.)

# --- System packages -------------------------------------------------------
# Build toolchain + the OpenGL / X11 / XCB runtime libraries that the ViZiR Qt
# application needs to open a window over X11 forwarding. mesa-utils + the DRI
# driver give software OpenGL so ViZiR still renders when there is no GPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates curl libcxxopts-dev cmake-curses-gui \
    libgl1 libglu1-mesa libegl1 libgl1-mesa-dri mesa-utils xauth libxcb-cursor0 x11-apps \
    libglib2.0-0t64 \
    libx11-6 libx11-xcb1 libxcb1 libxcb-glx0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render0 libxcb-render-util0 \
    libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 \
    libxcb-xkb1 libxcb-util1 libxkbcommon0 libxkbcommon-x11-0 \
    libxrender1 libxi6 libsm6 libice6 libfontconfig1 libfreetype6 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# --- libMeshb (mesh I/O) ---------------------------------------------------
# Build Loic Marechal's libMeshb from source and install it into /usr/local
# so CMake finds libmeshb8.h and libMeshb.8.a system-wide.
RUN git clone --depth 1 https://github.com/LoicMarechal/libMeshb /tmp/libMeshb && \
    cmake -S /tmp/libMeshb -B /tmp/libMeshb/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/libMeshb/build -j"$(nproc)" && \
    cmake --install /tmp/libMeshb/build && \
    rm -rf /tmp/libMeshb

# --- Kokkos (performance portability, for the Kokkos exercises) -------------
# Two backends via `--build-arg KOKKOS_BACKEND=`:
#   cuda (default): Serial + CUDA, for running on an NVIDIA GPU. Set the arch for
#       YOUR card with `--build-arg CUDA_ARCH=<sm>` (V100=70, T4=75, A100=80,
#       A10/RTX30=86, L4/L40=89, H100=90). Needs nvcc (present in the CUDA base),
#       but not a physical GPU to build.
#   cpu: Serial + OpenMP, for running the *_kokkos programs on a machine WITHOUT a
#       GPU (no libcuda dependency) -- handy for local testing.
#
# Kokkos 4.7.x + C++17 on purpose: Kokkos 5.x requires C++20, which nvcc 12.6
# cannot compile against libstdc++ (host-STL errors); 4.x at C++17 compiles clean.
ARG KOKKOS_BACKEND=cuda
ARG CUDA_ARCH=75
RUN git clone --depth 1 --branch 4.7.04 https://github.com/kokkos/kokkos /tmp/kokkos && \
    if [ "$KOKKOS_BACKEND" = "cuda" ]; then \
        # Kokkos needs its own arch flag (-DKokkos_ARCH_<NAME>=ON); it ignores
        # CMAKE_CUDA_ARCHITECTURES and its GPU auto-detect fails with no GPU present.
        case "${CUDA_ARCH}" in \
            70) KARCH=VOLTA70 ;; \
            75) KARCH=TURING75 ;; \
            80) KARCH=AMPERE80 ;; \
            86) KARCH=AMPERE86 ;; \
            89) KARCH=ADA89 ;; \
            90) KARCH=HOPPER90 ;; \
            *) echo "Unsupported CUDA_ARCH=${CUDA_ARCH} (use 70/75/80/86/89/90)"; exit 1 ;; \
        esac; \
        KOKKOS_FLAGS="-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_${KARCH}=ON"; \
    elif [ "$KOKKOS_BACKEND" = "cpu" ]; then \
        KOKKOS_FLAGS="-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON"; \
    else \
        echo "KOKKOS_BACKEND must be 'cuda' or 'cpu'"; exit 1; \
    fi && \
    cmake -S /tmp/kokkos -B /tmp/kokkos/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_CXX_STANDARD=17 ${KOKKOS_FLAGS} && \
    cmake --build /tmp/kokkos/build -j"$(nproc)" && \
    cmake --install /tmp/kokkos/build && \
    rm -rf /tmp/kokkos

# --- ViZiR 4 (visualization) -----------------------------------------------
# Unpack the precompiled INRIA binary into /opt and expose its launcher.
# NB: vizir4.sh derives its install dir from $0, so it must be called by its real
# absolute path. A symlink would make it mis-resolve that dir (wrong LD_LIBRARY_PATH
# and infinite re-exec), so install a thin wrapper instead.
RUN curl -sL https://pyamg.saclay.inria.fr/download/vizir/exes/Linux/vizir4.2025.01.22.linux.tar.gz \
        -o /tmp/vizir.tar.gz && \
    tar xzf /tmp/vizir.tar.gz -C /opt && \
    # The bundle ships an old libfreetype (pre-FT_Done_MM_Var) that the modern system
    # libfontconfig binds against, breaking the xcb plugin. Drop the stale bundled
    # freetype/fontconfig so the consistent system ones (libfreetype6) are used.
    rm -f /opt/vizir4.2025.01.22.linux/lib/libfreetype.so* \
          /opt/vizir4.2025.01.22.linux/lib/libfontconfig.so* && \
    printf '#!/bin/sh\nexec /opt/vizir4.2025.01.22.linux/vizir4.sh "$@"\n' > /usr/local/bin/vizir4 && \
    chmod +x /usr/local/bin/vizir4 && \
    rm /tmp/vizir.tar.gz

# Land the user in an interactive shell to run things themselves.
CMD ["/bin/bash"]
