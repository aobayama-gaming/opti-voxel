cmake_minimum_required(VERSION 3.20)
project(optivoxel_dawn CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The engine-free core, built as a library both adapters link: the Godot GDExtension builds the same files
# through SConstruct, the Dawn host through this file.
set(OV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(OV_CORE_SOURCES
    ${OV_ROOT}/src/ov/executor.cpp
    ${OV_ROOT}/src/base_surface.cpp
    ${OV_ROOT}/src/cascade.cpp
    ${OV_ROOT}/src/cascade_glsl.cpp
    ${OV_ROOT}/src/cascade_shaped_glsl.cpp
    ${OV_ROOT}/src/chunk_math.cpp
    ${OV_ROOT}/src/trace.cpp
)

add_library(optivoxel_core STATIC ${OV_CORE_SOURCES})
target_include_directories(optivoxel_core PUBLIC ${OV_ROOT}/src ${OV_ROOT}/thirdparty/doctest)
target_compile_options(optivoxel_core PRIVATE -O2 -Wall -Wextra -Werror -ffp-contract=off)
target_compile_definitions(optivoxel_core PUBLIC OPTI_VOXEL_CORE_ONLY)

# Dawn: a native build of github.com/google/dawn. Point OV_DAWN_DIR at its install prefix
# (`cmake --install <dawn>/out --prefix <prefix>`); the headers and libwebgpu_dawn.a are there.
set(OV_DAWN_DIR "$ENV{HOME}/dawn-install" CACHE PATH "Dawn install prefix")
set(OV_GLSLANG_DIR "$ENV{HOME}/glslang-install" CACHE PATH "glslang install prefix")

add_executable(optivoxel_dawn
    src/main.cpp
    src/glsl.cpp
    src/gpu.cpp
    src/terrain_sdf.cpp
    src/cpu_mesher.cpp
    src/terrain.cpp
    src/renderer.cpp
    src/fsr3.cpp
)
target_compile_options(optivoxel_dawn PRIVATE -O2 -Wall -Wextra)
target_include_directories(optivoxel_dawn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(optivoxel_dawn SYSTEM PRIVATE
    ${OV_DAWN_DIR}/include
    ${OV_GLSLANG_DIR}/include
)
target_link_libraries(optivoxel_dawn PRIVATE
    optivoxel_core
    ${OV_DAWN_DIR}/lib64/libwebgpu_dawn.a
    ${OV_GLSLANG_DIR}/lib64/libglslang.a
    ${OV_GLSLANG_DIR}/lib64/libglslang-default-resource-limits.a
    ${OV_GLSLANG_DIR}/lib64/libMachineIndependent.a
    ${OV_GLSLANG_DIR}/lib64/libGenericCodeGen.a
    ${OV_GLSLANG_DIR}/lib64/libOSDependent.a
    ${OV_GLSLANG_DIR}/lib64/libSPIRV.a
    -l:libvulkan.so.1
    X11
    pthread
    dl
)

install(TARGETS optivoxel_dawn RUNTIME DESTINATION bin)
