#!/usr/bin/env python
import os
import sys

from methods import print_error


libname = "opti_voxel"
# Two adapters of one engine: the core in src/, the Godot adapter (GDExtension + project) in adapter/godot/,
# the Dawn adapter in adapter/dawn/ (CMake).
projectdir = "adapter/godot/project"

localEnv = Environment(tools=["default"], PLATFORM="")

# Build profiles can be used to decrease compile times.
# You can either specify "disabled_classes", OR
# explicitly specify "enabled_classes" which disables all other classes.
# Modify the example file as needed and uncomment the line below or
# manually specify the build_profile parameter when running SCons.

# localEnv["build_profile"] = "build_profile.json"

customs = ["custom.py"]
customs = [os.path.abspath(path) for path in customs]

opts = Variables(customs, ARGUMENTS)
opts.Add(BoolVariable("werror", "Treat compiler warnings in src/ and adapter/ as errors (werror=no to iterate past them)", True))
opts.Update(localEnv)

Help(opts.GenerateHelpText(localEnv))

env = localEnv.Clone()

if not (os.path.isdir("godot-cpp") and os.listdir("godot-cpp")):
    print_error("""godot-cpp is not available within this folder, as Git submodules haven't been initialized.
Run the following command to download godot-cpp:

    git submodule update --init --recursive""")
    sys.exit(1)

env = SConscript("godot-cpp/SConstruct", {"env": env, "customs": customs})

# Our sources get their own environment: warnings are ours to fix, godot-cpp keeps its own flags.
# Its headers are also passed as -isystem so their warnings don't drown ours (GCC then ignores the
# matching -I). They stay in CPPPATH: that is what SCons scans for header dependencies, and the
# generated bindings must exist before our sources compile.
src_env = env.Clone()
if not src_env.get("is_msvc", False):
    for include in src_env["CPPPATH"]:
        src_env.Append(CCFLAGS=["-isystem", src_env.Dir(include).abspath])
    src_env.Append(CCFLAGS=["-Wall", "-Wextra"])
    if localEnv["werror"]:
        src_env.Append(CCFLAGS=["-Werror"])
env = src_env

env.Append(CPPPATH=["src/", "adapter/godot/src/"])
# The CPU mirrors of the GPU float math (ChunkMath::sample_pos, exact_noise.hpp, the terrain SDF) must
# round exactly like the GPU's `precise` code: no FMA contraction, whatever the target (aarch64 and
# -march=native have FMA). MSVC does not contract unless asked to.
if not env.get("is_msvc", False):
    env.Append(CCFLAGS=["-ffp-contract=off"])
# tools/cpu_profile.sh builds with debug_symbols=yes (unstripped): frame pointers let `perf record -g` unwind our frames.
if env["debug_symbols"] and not env.get("is_msvc", False):
    env.Append(CCFLAGS=["-fno-omit-frame-pointer"])
sources = Glob("src/*.cpp") + Glob("src/ov/*.cpp") + Glob("adapter/godot/src/*.cpp")

# Unit tests (src/tests/, doctest) ship in debug builds only; tools/unit_test.sh runs them.
if env["target"] != "template_release":
    env.Append(CPPDEFINES=["OPTI_VOXEL_TESTS"])
    env.Append(CPPPATH=["thirdparty/doctest"])
    if not env.get("is_msvc", False):
        env.Append(CCFLAGS=["-isystem", env.Dir("thirdparty/doctest").abspath])
    sources += Glob("src/tests/*.cpp")

# .dev doesn't inhibit compatibility, so we don't need to key it.
# .universal just means "compatible with all relevant arches" so we don't need to key it.
suffix = env['suffix'].replace(".dev", "").replace(".universal", "")

lib_filename = "{}{}{}{}".format(env.subst('$SHLIBPREFIX'), libname, suffix, env.subst('$SHLIBSUFFIX'))

library = env.SharedLibrary(
    "bin/{}/{}".format(env['platform'], lib_filename),
    source=sources,
)

copy = env.Install("{}/bin/{}/".format(projectdir, env["platform"]), library)

default_args = [library, copy]
Default(*default_args)
