#!/usr/bin/env python
import os
import sys

from methods import print_error


libname = "opti_voxel"
projectdir = "project"

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
opts.Add(BoolVariable("werror", "Treat compiler warnings in src/ as errors (werror=no to iterate past them)", True))
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
# Its headers are included as system headers so their warnings don't drown ours.
src_env = env.Clone()
if not src_env.get("is_msvc", False):
    godot_includes = src_env["CPPPATH"]
    src_env["CPPPATH"] = []
    for include in godot_includes:
        src_env.Append(CCFLAGS=["-isystem", src_env.Dir(include).abspath])
    src_env.Append(CCFLAGS=["-Wall", "-Wextra"])
    if localEnv["werror"]:
        src_env.Append(CCFLAGS=["-Werror"])
env = src_env

env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp")

# Unit tests (src/tests/, doctest) ship in debug builds only; tools/unit_test.sh runs them.
if env["target"] != "template_release":
    env.Append(CPPDEFINES=["OPTI_VOXEL_TESTS"])
    if env.get("is_msvc", False):
        env.Append(CPPPATH=["thirdparty/doctest"])
    else:
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
