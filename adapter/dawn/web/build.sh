#!/usr/bin/env bash
# The web viewer: adapter/dawn/out-web/ov_dawn.{html,js,wasm,data}, the landing page (index.html) and the service worker
# that isolates it on a host without COOP + COEP headers, served by web/serve.py.
#
#   adapter/dawn/web/build.sh            the dependencies once (build_deps.sh: emsdk, Dawn, OV_WEB_DEPS), then the viewer
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
adapter="$(dirname "$here")"
"$here/build_deps.sh"
deps="${OV_WEB_DEPS:-$HOME/ov-web-deps}"
dawn_src="${OV_DAWN_SRC:-$deps/dawn}"
# shellcheck disable=SC1091
source "${EMSDK:-$HOME/emsdk}/emsdk_env.sh" > /dev/null 2>&1

echo "== ov_dawn (wasm)"
emcmake cmake -S "$adapter" -B "$adapter/out-web" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOV_WEB_DEPS="$deps" \
    -DOV_DAWN_SRC="$dawn_src" -DOV_TINT_BUILD="$deps/tint-wasm-threads" -DOV_GLSLANG_DIR="$deps/glslang-wasm-threads" > /dev/null
ninja -C "$adapter/out-web" -j "${JOBS:-$(nproc)}" ov_dawn
echo "built $adapter/out-web: python3 $here/serve.py, then http://127.0.0.1:8095/ (the scenes) or /ov_dawn.html"
