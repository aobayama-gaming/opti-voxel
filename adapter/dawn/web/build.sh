#!/usr/bin/env bash
# The web viewer: adapter/dawn/out-web/ov_dawn.{html,js,wasm}, served by web/serve.py.
#
#   adapter/dawn/web/build.sh            the dependencies once (into OV_WEB_DEPS), then the viewer
#
# Needs emsdk (EMSDK, default ~/emsdk; Emscripten 4.0.11 tested) and a Dawn checkout with its dependencies
# (python3 tools/fetch_dawn_dependencies.py in it): OV_DAWN_SRC, default $OV_WEB_DEPS/dawn. OV_WEB_DEPS (default
# ~/ov-web-deps) gets glslang (built from Dawn's third_party copy) and web/deps (emdawnwebgpu, Tint, SPIRV-Tools).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
adapter="$(dirname "$here")"
deps="${OV_WEB_DEPS:-$HOME/ov-web-deps}"
dawn_src="${OV_DAWN_SRC:-$deps/dawn}"
jobs="${JOBS:-$(nproc)}"
# shellcheck disable=SC1091
source "${EMSDK:-$HOME/emsdk}/emsdk_env.sh" > /dev/null 2>&1
command -v emcmake > /dev/null || { echo "build.sh: no emcmake (set EMSDK)" >&2; exit 1; }
[ -f "$dawn_src/CMakeLists.txt" ] || { echo "build.sh: no Dawn checkout at $dawn_src (set OV_DAWN_SRC)" >&2; exit 1; }

if [ ! -f "$deps/glslang-wasm/lib/libglslang.a" ]; then
    echo "== glslang (wasm)"
    emcmake cmake -S "$dawn_src/third_party/glslang/src" -B "$deps/glslang-wasm-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$deps/glslang-wasm" -DENABLE_OPT=OFF -DGLSLANG_TESTS=OFF -DENABLE_GLSLANG_BINARIES=OFF \
        -DGLSLANG_ENABLE_INSTALL=ON -DENABLE_HLSL=OFF -DBUILD_SHARED_LIBS=OFF > /dev/null
    ninja -C "$deps/glslang-wasm-build" -j "$jobs" install > /dev/null
fi

tint="$deps/tint-wasm"
if [ ! -f "$tint/emdawnwebgpu_pkg/emdawnwebgpu.port.py" ]; then
    echo "== emdawnwebgpu, Tint, SPIRV-Tools (wasm, several minutes)"
    emcmake cmake -S "$here/deps" -B "$tint" -G Ninja -DCMAKE_BUILD_TYPE=Release -DDAWN_SRC="$dawn_src" \
        -DTINT_BUILD_SPV_READER=ON -DTINT_BUILD_WGSL_WRITER=ON -DTINT_BUILD_WGSL_READER=ON -DTINT_BUILD_CMD_TOOLS=OFF \
        -DTINT_BUILD_TESTS=OFF -DTINT_BUILD_GLSL_WRITER=OFF -DTINT_BUILD_HLSL_WRITER=OFF -DTINT_BUILD_MSL_WRITER=OFF \
        -DTINT_BUILD_SPV_WRITER=OFF -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_TESTS=OFF > /dev/null
    # The package also copies a C++ module interface the Emscripten generator doesn't write: an empty one.
    mkdir -p "$tint/dawn/gen/include/dawn" && touch "$tint/dawn/gen/include/dawn/webgpu.ixx"
    libs=$(ninja -C "$tint" -t targets all | grep -oE "^dawn/src/tint/libtint_lang_(spirv_reader|wgsl_writer)[a-z_]*\.a" | tr '\n' ' ')
    # shellcheck disable=SC2086
    ninja -C "$tint" -j "$jobs" emdawnwebgpu_pkg dawn/src/tint/libtint_api.a $libs SPIRV-Tools-opt > /dev/null
fi

echo "== ov_dawn (wasm)"
emcmake cmake -S "$adapter" -B "$adapter/out-web" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOV_WEB_DEPS="$deps" \
    -DOV_DAWN_SRC="$dawn_src" > /dev/null
ninja -C "$adapter/out-web" -j "$jobs" ov_dawn
echo "built $adapter/out-web/ov_dawn.html: python3 $here/serve.py, then http://127.0.0.1:8095/ov_dawn.html"
