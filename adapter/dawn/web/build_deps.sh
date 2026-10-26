#!/usr/bin/env bash
# The web viewer's wasm dependencies, once, into OV_WEB_DEPS (default ~/ov-web-deps): glslang (from Dawn's third_party
# copy) and web/deps (emdawnwebgpu, Tint, SPIRV-Tools), all with -pthread (a threaded wasm links only objects built for
# shared memory). build.sh runs it; the Pages workflow caches OV_WEB_DEPS on this file and the two pins next to it.
#
# Pins: emsdk_version.txt (EMSDK, default ~/emsdk, must have it active) and dawn_revision.txt. Without a Dawn checkout
# at OV_DAWN_SRC (default $OV_WEB_DEPS/dawn) the pinned revision is fetched there with its dependencies.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
deps="${OV_WEB_DEPS:-$HOME/ov-web-deps}"
dawn_src="${OV_DAWN_SRC:-$deps/dawn}"
jobs="${JOBS:-$(nproc)}"
# shellcheck disable=SC1091
source "${EMSDK:-$HOME/emsdk}/emsdk_env.sh" > /dev/null 2>&1
command -v emcmake > /dev/null || { echo "build_deps.sh: no emcmake (set EMSDK)" >&2; exit 1; }
emsdk_version="$(cat "$here/emsdk_version.txt")"
emcc --version | head -1 | grep -qF " $emsdk_version " \
    || echo "build_deps.sh: warning: emcc is not $emsdk_version (emsdk_version.txt): $(emcc --version | head -1)" >&2

revision="$(cat "$here/dawn_revision.txt")"
if [ ! -f "$dawn_src/CMakeLists.txt" ]; then
    echo "== Dawn $revision"
    git init -q "$dawn_src"
    git -C "$dawn_src" fetch -q --depth 1 https://dawn.googlesource.com/dawn "$revision"
    git -C "$dawn_src" checkout -q FETCH_HEAD
fi
if [ "$(git -C "$dawn_src" rev-parse HEAD 2> /dev/null)" != "$revision" ]; then
    echo "build_deps.sh: warning: $dawn_src is not at $revision (dawn_revision.txt)" >&2
elif [ ! -f "$dawn_src/.ov_dependencies" ]; then
    # Until a fetch has finished (the marker): an interrupted one is fetched again.
    echo "== Dawn's dependencies"
    (cd "$dawn_src" && python3 tools/fetch_dawn_dependencies.py)
    touch "$dawn_src/.ov_dependencies"
fi

threads=(-DCMAKE_C_FLAGS=-pthread -DCMAKE_CXX_FLAGS=-pthread)
glslang="$deps/glslang-wasm-threads"
if [ ! -f "$glslang/lib/libglslang.a" ]; then
    echo "== glslang (wasm)"
    emcmake cmake -S "$dawn_src/third_party/glslang/src" -B "$glslang-build" -G Ninja -DCMAKE_BUILD_TYPE=Release "${threads[@]}" \
        -DCMAKE_INSTALL_PREFIX="$glslang" -DENABLE_OPT=OFF -DGLSLANG_TESTS=OFF -DENABLE_GLSLANG_BINARIES=OFF \
        -DGLSLANG_ENABLE_INSTALL=ON -DENABLE_HLSL=OFF -DBUILD_SHARED_LIBS=OFF > /dev/null
    ninja -C "$glslang-build" -j "$jobs" install > /dev/null
fi

tint="$deps/tint-wasm-threads"
if [ ! -f "$tint/emdawnwebgpu_pkg/emdawnwebgpu.port.py" ]; then
    echo "== emdawnwebgpu, Tint, SPIRV-Tools (wasm, several minutes)"
    emcmake cmake -S "$here/deps" -B "$tint" -G Ninja -DCMAKE_BUILD_TYPE=Release "${threads[@]}" -DDAWN_SRC="$dawn_src" \
        -DTINT_BUILD_SPV_READER=ON -DTINT_BUILD_WGSL_WRITER=ON -DTINT_BUILD_WGSL_READER=ON -DTINT_BUILD_CMD_TOOLS=OFF \
        -DTINT_BUILD_TESTS=OFF -DTINT_BUILD_GLSL_WRITER=OFF -DTINT_BUILD_HLSL_WRITER=OFF -DTINT_BUILD_MSL_WRITER=OFF \
        -DTINT_BUILD_SPV_WRITER=OFF -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_TESTS=OFF > /dev/null
    # The package also copies a C++ module interface the Emscripten generator doesn't write: an empty one.
    mkdir -p "$tint/dawn/gen/include/dawn" && touch "$tint/dawn/gen/include/dawn/webgpu.ixx"
    libs=$(ninja -C "$tint" -t targets all | grep -oE "^dawn/src/tint/libtint_lang_(spirv_reader|wgsl_writer)[a-z_]*\.a" | tr '\n' ' ')
    # shellcheck disable=SC2086
    ninja -C "$tint" -j "$jobs" emdawnwebgpu_pkg dawn/src/tint/libtint_api.a $libs SPIRV-Tools-opt > /dev/null
fi
