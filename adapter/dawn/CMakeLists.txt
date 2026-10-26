cmake_minimum_required(VERSION 3.20)
project(optivoxel_dawn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
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

# Dawn (github.com/google/dawn), built and installed natively: `cmake --install <dawn build> --prefix <prefix>`.
# Without it only the core builds (CI: the core compiles and links outside Godot).
set(OV_DAWN_DIR "$ENV{HOME}/dawn-install" CACHE PATH "Dawn install prefix")
set(OV_GLSLANG_DIR "$ENV{HOME}/glslang-install" CACHE PATH "glslang install prefix (the engine's GLSL to SPIR-V)")
if(NOT EXISTS "${OV_DAWN_DIR}/include/dawn/webgpu_cpp.h")
    message(STATUS "No Dawn at OV_DAWN_DIR=${OV_DAWN_DIR}: building the core only")
    return()
endif()

add_executable(ov_dawn src/main.cpp src/dawn_context.cpp src/dawn_device.cpp src/glsl_compiler.cpp)
set_target_properties(ov_dawn PROPERTIES CXX_STANDARD 20) # webgpu_cpp.h needs C++20; the core stays C++17 like the Godot build
target_compile_options(ov_dawn PRIVATE -Wall -Wextra -Werror)
target_include_directories(ov_dawn SYSTEM PRIVATE "${OV_DAWN_DIR}/include" "${OV_GLSLANG_DIR}/include")
set(OV_GLSLANG_LIBS glslang glslang-default-resource-limits MachineIndependent GenericCodeGen OSDependent SPIRV)
list(TRANSFORM OV_GLSLANG_LIBS REPLACE "(.+)" "${OV_GLSLANG_DIR}/lib64/lib\\1.a")
target_link_libraries(ov_dawn PRIVATE ov_core "${OV_DAWN_DIR}/lib64/libwebgpu_dawn.a" ${OV_GLSLANG_LIBS} -l:libvulkan.so.1 dl pthread)
