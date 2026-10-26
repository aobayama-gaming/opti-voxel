cmake_minimum_required(VERSION 3.20)
project(optivoxel_dawn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(EMSCRIPTEN)
    # The web build runs the core's threads as Web Workers (pthreads): every object, the dependencies' too (web/build.sh).
    add_compile_options(-pthread)
endif()

# The core both adapters build: src/core_files.txt is the one list (tools/check_core_purity.sh reads it too).
set(OV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${OV_ROOT}/src/core_files.txt")
file(STRINGS "${OV_ROOT}/src/core_files.txt" OV_CORE_LIST REGEX "\\.cpp$")
list(TRANSFORM OV_CORE_LIST PREPEND "${OV_ROOT}/")
add_library(ov_core STATIC ${OV_CORE_LIST})
target_include_directories(ov_core PUBLIC "${OV_ROOT}/src")
# -ffp-contract=off: the CPU mirrors of the GPU float math round like its `precise` code (see SConstruct).
target_compile_options(ov_core PRIVATE -Wall -Wextra -Werror -ffp-contract=off)

set(OV_DAWN_SOURCES src/main.cpp src/dawn_context.cpp src/dawn_device.cpp src/dawn_page_host.cpp src/dawn_upscaler.cpp src/fsr3.cpp
        src/glsl_compiler.cpp src/shader_module.cpp src/viewer.cpp)
set(OV_GLSLANG_LIBS glslang glslang-default-resource-limits MachineIndependent GenericCodeGen OSDependent SPIRV)

if(EMSCRIPTEN)
    # The web build: the browser's WebGPU through emdawnwebgpu, with glslang and Tint (SPIR-V to WGSL) in the wasm.
    # web/build.sh builds these dependencies (web/deps) and this.
    set(OV_WEB_DEPS "$ENV{HOME}/ov-web-deps" CACHE PATH "web/build.sh's dependency prefix")
    set(OV_DAWN_SRC "${OV_WEB_DEPS}/dawn" CACHE PATH "Dawn source (Tint's headers)")
    set(OV_TINT_BUILD "${OV_WEB_DEPS}/tint-wasm-threads" CACHE PATH "web/deps built with emcmake: emdawnwebgpu_pkg, Tint, SPIRV-Tools")
    set(OV_GLSLANG_DIR "${OV_WEB_DEPS}/glslang-wasm-threads" CACHE PATH "glslang built with emcmake and installed")
    set(OV_EMDAWNWEBGPU_PORT "${OV_TINT_BUILD}/emdawnwebgpu_pkg/emdawnwebgpu.port.py")
    if(NOT EXISTS "${OV_EMDAWNWEBGPU_PORT}")
        message(STATUS "No emdawnwebgpu at ${OV_EMDAWNWEBGPU_PORT}: building the core only")
        return()
    endif()
    add_executable(ov_dawn ${OV_DAWN_SOURCES} src/web_canvas.cpp)
    # The site next to the page: the service worker that isolates it on a host without COOP + COEP headers (shell.html),
    # and the landing page.
    foreach(OV_SITE_FILE coi-serviceworker.js index.html)
        configure_file("web/${OV_SITE_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/${OV_SITE_FILE}" COPYONLY)
    endforeach()
    # LINK_DEPENDS: the page is linked from the shell, which CMake doesn't track as an input.
    set_target_properties(ov_dawn PROPERTIES CXX_STANDARD 20 SUFFIX ".html" LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html")
    target_compile_options(ov_dawn PRIVATE -Wall -Wextra -Werror "--use-port=${OV_EMDAWNWEBGPU_PORT}")
    set_source_files_properties(src/viewer.cpp PROPERTIES COMPILE_OPTIONS -Wno-deprecated-declarations)
    target_include_directories(ov_dawn SYSTEM PRIVATE "${OV_GLSLANG_DIR}/include" "${OV_DAWN_SRC}" "${OV_DAWN_SRC}/include")
    # Tint's headers read its build switches (and Dawn's utils, which it includes, their marker).
    target_compile_definitions(ov_dawn PRIVATE DAWN_HAVE_UTILS_BUILD_CONFIG TINT_BUILD_SPV_READER=1 TINT_BUILD_WGSL_WRITER=1 TINT_BUILD_WGSL_READER=1 TINT_BUILD_IR_BINARY=1
            TINT_BUILD_SPV_WRITER=0 TINT_BUILD_GLSL_WRITER=0 TINT_BUILD_HLSL_WRITER=0 TINT_BUILD_MSL_WRITER=0 TINT_BUILD_NULL_WRITER=0
            TINT_BUILD_GLSL_VALIDATOR=0 TINT_ENABLE_IR_DUMPING=1 TINT_ENABLE_IR_VALIDATION_ASSERTS=1 TINT_BUILD_IS_LINUX=1 TINT_BUILD_IS_MAC=0
            TINT_BUILD_IS_WIN=0)
    list(TRANSFORM OV_GLSLANG_LIBS REPLACE "(.+)" "${OV_GLSLANG_DIR}/lib/lib\\1.a")
    file(GLOB_RECURSE OV_TINT_LIBS "${OV_TINT_BUILD}/dawn/src/tint/libtint_*.a" "${OV_TINT_BUILD}/dawn/third_party/abseil/*.a"
            "${OV_TINT_BUILD}/dawn/src/utils/*.a")
    # JSPI: the engine's blocking waits (device creation, error scopes, readbacks) suspend the wasm while the browser runs,
    # on the main thread and in a worker (the discoverer's and the field's local devices are the worker's own). Threads:
    # the workers are started with the page (a pthread only starts once the main thread yields): the pool's (one per core but
    # the main thread's, at least one) and the viewer's two long-lived workers (discoverer, field); one more starts late. Only a cross-origin isolated page (web/serve.py's headers)
    # gets SharedArrayBuffer. FS: a --shot's PNG is read back from the page (Module.FS.readFile); the Godot project's
    # climate worlds are preloaded at /climate (ov_dawn.data: ?climate=/climate/world_video.json&lods=25), its grade at /grade_lut.rgb.
    target_link_options(ov_dawn PRIVATE "SHELL:--preload-file ${OV_ROOT}/adapter/godot/project/climate@/climate"
            "SHELL:--preload-file ${OV_ROOT}/adapter/godot/project/grade_lut.rgb@/grade_lut.rgb" "--use-port=${OV_EMDAWNWEBGPU_PORT}" -pthread -sJSPI -Wno-experimental
            -Wno-pthreads-mem-growth "-sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency+2" -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=4GB
            -sSTACK_SIZE=8MB -sENVIRONMENT=web,worker -sEXIT_RUNTIME=0 -sEXPORTED_RUNTIME_METHODS=FS "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html")
    target_link_libraries(ov_dawn PRIVATE ov_core ${OV_GLSLANG_LIBS} -Wl,--start-group ${OV_TINT_LIBS}
            "${OV_TINT_BUILD}/spirv-tools/source/opt/libSPIRV-Tools-opt.a" "${OV_TINT_BUILD}/spirv-tools/source/libSPIRV-Tools.a" -Wl,--end-group)
    return()
endif()

# Dawn (github.com/google/dawn), built and installed natively: `cmake --install <dawn build> --prefix <prefix>`.
# Without it only the core builds (CI: the core compiles and links outside Godot).
set(OV_DAWN_DIR "$ENV{HOME}/dawn-install" CACHE PATH "Dawn install prefix")
set(OV_GLSLANG_DIR "$ENV{HOME}/glslang-install" CACHE PATH "glslang install prefix (the engine's GLSL to SPIR-V)")
if(NOT EXISTS "${OV_DAWN_DIR}/include/dawn/webgpu_cpp.h")
    message(STATUS "No Dawn at OV_DAWN_DIR=${OV_DAWN_DIR}: building the core only")
    return()
endif()

add_executable(ov_dawn ${OV_DAWN_SOURCES} src/x11_window.cpp)
set_target_properties(ov_dawn PROPERTIES CXX_STANDARD 20) # webgpu_cpp.h needs C++20; the core stays C++17 like the Godot build
target_compile_options(ov_dawn PRIVATE -Wall -Wextra -Werror)
# The viewer includes the World's headers, whose std::atomic_load / atomic_store of a shared_ptr (the instance snapshot)
# C++20 deprecates; the core is C++17, where they are the way.
set_source_files_properties(src/viewer.cpp PROPERTIES COMPILE_OPTIONS -Wno-deprecated-declarations)
target_include_directories(ov_dawn SYSTEM PRIVATE "${OV_DAWN_DIR}/include" "${OV_GLSLANG_DIR}/include")
list(TRANSFORM OV_GLSLANG_LIBS REPLACE "(.+)" "${OV_GLSLANG_DIR}/lib64/lib\\1.a")
target_link_libraries(ov_dawn PRIVATE ov_core "${OV_DAWN_DIR}/lib64/libwebgpu_dawn.a" ${OV_GLSLANG_LIBS} -l:libvulkan.so.1 X11 dl pthread)
