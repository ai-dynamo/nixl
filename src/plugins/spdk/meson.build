# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

spdk_sources = [
    'spdk_backend.cpp',
    'spdk_backend.h',
    'spdk_progress_engine.cpp',
    'spdk_progress_engine.h',
    'spdk_plugin.cpp',
]

# This plugin supplies its own implementation of SPDK's env interface (memory,
# threads, memory-registration map, and stubbed PCI) instead of linking
# spdk_env_dpdk. That removes the DPDK runtime entirely: no EAL bring-up, no
# hugepage requirement, and no PCI probing (local PCI devices are unsupported).
# These are plain C sources, built as a standalone library so the C++ plugin's
# warning flags and dependency compile args (which are C++-only) do not reach
# them; _GNU_SOURCE is needed for the CPU affinity / pthread_*_np helpers.
spdk_env_sources = [
    'spdk_env/env.c',
    'spdk_env/init.c',
    'spdk_env/threads.c',
    'spdk_env/memory.c',
    'spdk_env/mem_map.c',
    'spdk_env/pci.c',
]

# NOTE: the SPDK libraries are linked via spdk_link_args below (wrapped in
# --whole-archive), not as meson dependencies. SPDK registers bdev modules,
# accel modules, subsystems and RPC methods through link-time constructors that
# carry no referenced symbol, so a normal dependency link drops them and the
# runtime silently lacks mempool ops / accel modules / bdev create RPCs.
spdk_plugin_deps = [
    nixl_infra,
    nixl_common_dep,
    absl_log_dep,
]

spdk_include_path = get_option('spdk_include_path')
spdk_library_path = get_option('spdk_library_path')

# Every pkg-config query below goes through this: when an explicit library path
# was given, its prefix owns the .pc files, otherwise the default search path
# applies. Defined here because the include-directory lookup already needs it.
spdk_pc_env = {}
if spdk_library_path != ''
    spdk_pc_env = {'PKG_CONFIG_PATH': spdk_library_path + '/pkgconfig'}
endif

spdk_inc_dirs = [nixl_inc_dirs, utils_inc_dirs]
if spdk_include_path != ''
    spdk_inc_dirs += include_directories(spdk_include_path, is_system: true)
else
    # Fall back to pkg-config so a discoverable SPDK needs no *_path options at
    # all. Without this the headers were only ever found when they sat in a
    # default system include directory: any other prefix configured cleanly and
    # then failed the compile on spdk/stdinc.h.
    spdk_pc_cflags = run_command('pkg-config', '--cflags-only-I', spdk_pc_components,
        env: spdk_pc_env, check: true)
    spdk_pc_incs = []
    foreach _inc : spdk_pc_cflags.stdout().split()
        if _inc.startswith('-I') and not spdk_pc_incs.contains(_inc.substring(2))
            spdk_pc_incs += _inc.substring(2)
            spdk_inc_dirs += include_directories(_inc.substring(2), is_system: true)
        endif
    endforeach
endif

# Build the DPDK-free env implementation as its own C-only static library.
spdk_env_lib = static_library('nixl_spdk_env',
    spdk_env_sources,
    include_directories: spdk_inc_dirs,
    c_args: ['-D_GNU_SOURCE'],
    pic: true,
    install: false)

spdk_link_args = []
if spdk_library_path != ''
    spdk_link_args += ['-L' + spdk_library_path]
endif

# Resolve the SPDK library closure via pkg-config. spdk_pc_components is defined
# in the top-level meson.build, which gates this subdir on the same list being
# present. spdk_bdev_malloc pulls the malloc bdev module (a minimal default so
# the runtime can create a bdev from JSON); spdk_event_{bdev,accel} pull the
# bdev/accel subsystem and module registrations; spdk_init provides
# spdk_subsystem_load_config.
spdk_pc = run_command('pkg-config', '--libs', spdk_pc_components,
    env: spdk_pc_env, check: true)

# The directory holding the SPDK archives. Without spdk_library_path it comes
# from pkg-config, because the -lspdk_* tokens below have to be rewritten to
# full .a paths in every configuration, not just the explicit-path one (see the
# reordering note below).
spdk_resolved_libdir = spdk_library_path
if spdk_resolved_libdir == ''
    spdk_pc_libdirs = run_command('pkg-config', '--libs-only-L', spdk_pc_components,
        env: spdk_pc_env, check: true)
    foreach _dir : spdk_pc_libdirs.stdout().split()
        if _dir.startswith('-L') and spdk_resolved_libdir == ''
            spdk_resolved_libdir = _dir.substring(2)
        endif
    endforeach
endif

# Drop spdk_env_dpdk and the DPDK runtime from the closure: the spdk_env/
# sources above provide the spdk/env.h implementation instead. Filtering out
# env_dpdk avoids duplicate env symbols, and filtering the rte_* libraries and
# the DPDK library path keeps the EAL (and its PCI bus/mempool drivers) from
# being linked back in.
# Resolve the SPDK libs to full .a paths rather than -l flags. Meson reorders
# bare -l library flags out of a raw -Wl,--whole-archive group (which would break
# it and drop the constructor-only registrations - e.g. the software accel module
# - leaving g_modules_opc unpopulated and crashing spdk_accel_initialize). Full
# archive paths are kept in place. This matters especially for the baked-in
# (static_plugins) build, where the closure rides on the dependency and meson is
# free to reorder it. Non-spdk (system) entries are left untouched.
spdk_libs = []
foreach _lib : spdk_pc.stdout().split()
    if _lib == '-lspdk_env_dpdk' or _lib.startswith('-lrte_') or _lib.contains('/dpdk/')
        continue
    elif _lib.startswith('-lspdk_') and spdk_resolved_libdir != ''
        spdk_libs += spdk_resolved_libdir / ('lib' + _lib.substring(2) + '.a')
    else
        spdk_libs += _lib
    endif
endforeach

# Whole-archive the SPDK static libs so their constructor-based self-registration
# (bdev modules, accel modules, subsystems, RPC methods) is retained.
spdk_link_args += ['-Wl,--whole-archive'] + spdk_libs + ['-Wl,--no-whole-archive']

# The system and ISA-L libraries the SPDK archives depend on. Taken from
# pkg-config rather than hardcoded: which ones SPDK needs depends on how it was
# configured, so a fixed list breaks against a build with, say, fuse disabled.
# These belong after --no-whole-archive; they carry no constructor registrations
# and whole-archiving them would drag in every object they contain.
spdk_syslibs_pc = run_command('pkg-config', '--libs', 'spdk_syslibs',
    env: spdk_pc_env, check: true)
spdk_link_args += spdk_syslibs_pc.stdout().split()

compile_defs = []
compile_flags = []

if 'SPDK' in static_plugins
    # A static_library has no link step, so link_args on it are dropped. The SPDK
    # library closure (spdk_link_args, whole-archived) must instead ride on the
    # dependency so it reaches whatever links this plugin in (libnixl and the unit
    # tests) when the plugin is baked in.
    spdk_backend_lib = static_library('SPDK',
        spdk_sources,
        dependencies: spdk_plugin_deps,
        link_whole: spdk_env_lib,
        include_directories: spdk_inc_dirs,
        install: false,
        cpp_args: compile_defs + compile_flags,
        name_prefix: 'libplugin_')
    spdk_backend_interface = declare_dependency(
        link_whole: spdk_backend_lib,
        link_args: spdk_link_args)
else
    spdk_backend_lib = shared_library('SPDK',
        spdk_sources,
        dependencies: spdk_plugin_deps,
        link_args: spdk_link_args,
        link_whole: spdk_env_lib,
        include_directories: spdk_inc_dirs,
        install: true,
        cpp_args: compile_defs + ['-fPIC'],
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    spdk_backend_interface = declare_dependency(link_with: spdk_backend_lib)
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "SPDK=' + spdk_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true)
    endif
endif
