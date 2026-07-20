# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# AMD ROCm/HIP support for the libfabric plugin, selected with -Dwheel_variant=rocm
# (the same switch that controls nixl_rocm wheel naming). Defined here in the first
# libfabric subdir to be processed -- src/utils is entered before src/plugins (see
# src/meson.build) -- so both this file and src/plugins/libfabric/meson.build can
# consume rocm_dep without adding any ROCm wiring to the root meson.build. The
# plugin's CUDA sources are translated to HIP at build time via hipify-perl; rocm_dep
# supplies the HIP headers and runtime those translated sources compile and link
# against. __HIP_PLATFORM_AMD__ lets the HIP headers work under a non-hipcc host
# compiler. Kept separate from cuda_dep. The ROCm toolchain path comes from
# $ROCM_PATH, falling back to /opt/rocm.
#
# hipify injects <hip/hip_runtime.h> at the top of each translated source, ahead
# of the nixl/toml++ headers. HIP defines __noinline__ as a macro, which breaks
# toml++'s TOML_HAS_ATTR(__noinline__) preprocessor check; TOML_CUDA=1 routes
# toml++ around that line. HIP also maps some CUDA driver calls onto deprecated
# HIP APIs, so deprecation is downgraded from error (the tree builds with
# werror=true) to a warning.
if wheel_variant == 'rocm'
    rocm_path = run_command('bash', '-c', 'echo -n "$ROCM_PATH"', check: false).stdout().strip()
    if rocm_path == ''
        rocm_path = '/opt/rocm'
    endif
    rocm_dep = declare_dependency(
        compile_args : [
            '-I' + rocm_path + '/include',
            '-D__HIP_PLATFORM_AMD__',
            '-DTOML_CUDA=1',
            '-Wno-error=deprecated-declarations',
        ],
        link_args : ['-L' + rocm_path + '/lib', '-lamdhip64', '-lhiprtc'],
    )
    message('ROCm/HIP enabled; toolchain path: ' + rocm_path)
else
    rocm_dep = disabler()
endif

# On ROCm builds (-Dwheel_variant=rocm) the CUDA sources are translated to HIP
# with hipify-perl, which ships only with the ROCm/HIP toolchain, so it is
# required for that variant and unused (absent) on CUDA-only builds.
if wheel_variant == 'rocm'
    hipify_cmd = find_program('hipify-perl')
    hipify_src = generator(hipify_cmd,
                           output : ['@BASENAME@_hip.cpp'],
                           arguments : ['@INPUT@', '@EXTRA_ARGS@', '@OUTPUT@'])

    libfabric_topology_src = hipify_src.process('libfabric_topology.cpp', extra_args: ['-o'])
    libfabric_rail_manager_src = hipify_src.process('libfabric_rail_manager.cpp', extra_args: ['-o'])
else
    libfabric_topology_src = files('libfabric_topology.cpp')
    libfabric_rail_manager_src = files('libfabric_rail_manager.cpp')
endif

# Source files
libfabric_utils_sources = [
    'libfabric_rail.cpp',
    'libfabric_common.cpp',
    libfabric_topology_src,
    libfabric_rail_manager_src,
]

# Header files
libfabric_utils_headers = files(
    'libfabric_rail.h',
    'libfabric_rail_manager.h',
    'libfabric_common.h',
    'libfabric_topology.h',
)

# Find hwloc dependency
hwloc_dep = dependency('hwloc', required: true)

# Find numa dependency
numa_dep = dependency('numa', required: true)

# Set up dependencies and compile args
libfabric_utils_deps = [
    libfabric_dep,
    hwloc_dep,
    numa_dep,
    absl_log_dep,
]

libfabric_utils_cpp_args = []

# Detect optional libfabric features at compile time to enable/disable
# functionality based on the installed libfabric version
if libfabric_dep.found()
    # Check for FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV (libfabric >= 2.3)
    if cpp.has_header_symbol('rdma/fi_ext.h',
                             'FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV',
                             dependencies: libfabric_dep)
        libfabric_utils_cpp_args += ['-DHAVE_FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV']
        message('libfabric supports FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV')
    else
        message('libfabric does not support FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV')
    endif
endif

# Add GPU support: CUDA normally, or HIP/ROCm when wheel_variant=rocm.
if wheel_variant == 'rocm'
    libfabric_utils_deps += [rocm_dep]
    libfabric_utils_cpp_args += ['-DHAVE_CUDA']
elif cuda_dep.found()
    libfabric_utils_deps += [cuda_dep]
    libfabric_utils_cpp_args += ['-DHAVE_CUDA']
endif

# Create static library
libfabric_utils_lib = static_library(
    'nixl_libfabric_utils',
    libfabric_utils_sources,
    dependencies: libfabric_utils_deps,
    cpp_args: libfabric_utils_cpp_args,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
    ],
    install: false,
)

# Declare dependency for other parts of the build
libfabric_utils_dep = declare_dependency(
    link_with: libfabric_utils_lib,
    include_directories: include_directories('.'),
)
