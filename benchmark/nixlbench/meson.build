# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project('nixlbench', 'CPP', version: '1.5.0',
    default_options: ['buildtype=release',
                'werror=true',
                'cpp_std=c++20',
                'prefix=/usr/local/nixlbench'],
    meson_version: '>= 0.64.0'
)

# set up some global vars for compiler, platform, configuration, etc.
cpp = meson.get_compiler('cpp')
fs = import('fs')

# Allow overriding paths through environment variables
# CUDA
cuda_inc_path = get_option('cudapath_inc')
cuda_lib_path = get_option('cudapath_lib')
cuda_stub_path = get_option('cudapath_stub')
# device api
ucx_path = get_option('ucx_path')
ucx_device_cuda_arch_list = get_option('ucx_device_cuda_arch_list').strip()
# ETCD
etcd_inc_path = get_option('etcd_inc_path')
etcd_lib_path = get_option('etcd_lib_path')
# NIXL
nixl_path = get_option('nixl_path')
# NVSHMEM
nvshmem_inc_path = get_option('nvshmem_inc_path')
nvshmem_lib_path = get_option('nvshmem_lib_path')

# Find required libraries
# NIXL
host_cpu_family = host_machine.cpu_family()
host_system = host_machine.system().to_lower()

if host_system != 'linux' or host_cpu_family not in ['x86_64', 'aarch64']
    error('This build only supports Linux on x86_64 or aarch64 architectures.')
endif

nixl_lib_path = join_paths(nixl_path, get_option('libdir'))
nixl_lib = cpp.find_library('nixl', dirs: [nixl_lib_path])
nixl_build = cpp.find_library('nixl_build', dirs: [nixl_lib_path])
nixl_serdes = cpp.find_library('serdes', dirs: [nixl_lib_path])

if not nixl_lib.found() or not nixl_build.found() or not nixl_serdes.found()
    error('NIXL Libraries not found. Exiting.')
endif

# CUDA (always probed; independent of ROCm)
cuda_available = false
if cuda_lib_path == ''
    cuda_dep = dependency('cuda', required : false, modules : [ 'cudart', 'cuda' ])
    if not cuda_dep.found()
        # Meson is not detecting CUDA reliably on ARM, fallback to default
        cuda_home = run_command('bash', '-c', 'echo $CUDA_HOME', check: true).stdout().strip()
        if cuda_home == ''
            cuda_home = '/usr/local/cuda'
        endif
        cuda_lib = cuda_home + '/lib64'
        cuda_inc = cuda_home + '/include'
        cuda_stub = cuda_lib + '/stubs'
        if fs.exists(cuda_lib) and fs.exists(cuda_inc)
            cuda_dep = declare_dependency(
              link_args : ['-L' + cuda_lib, '-L' + cuda_stub, '-lcuda', '-lcudart'],
              include_directories : include_directories(cuda_inc))
            if cuda_dep.found()
                message('Found CUDA installation through fallback method:', cuda_home)
            endif
        endif
    endif
    if cuda_dep.found()
        cuda_available = true
    else
        warning('CUDA not found. VRAM support will be disabled.')
    endif
else
    message('cuda lib path ', cuda_lib_path)
    if cuda_stub_path == ''
        cuda_stub_path = cuda_lib_path + '/stubs'
    endif
    cuda_dep = declare_dependency(
        link_args : ['-L' + cuda_lib_path, '-L' + cuda_stub_path, '-lcuda', '-lcudart'],
        include_directories : include_directories(cuda_inc_path))
    cuda_available = true
endif

cuda_detected = cuda_available

# ROCm / HIP. As in the root project, ROCm is requested by setting
# -Drocm_path=<prefix>; an empty rocm_path means ROCm is not requested, so it is
# never enabled implicitly from the environment. Detection probes for
# libamdhip64 itself rather than stat'ing <prefix>, because ROCm >= 7.13 makes
# the install root an alternatives symlink; the prefix-scoped header check stops
# a system-wide libamdhip64 from satisfying a wrong rocm_path.
rocm_available = false
rocm_dep = declare_dependency()
rocm_path = get_option('rocm_path')
if rocm_path != ''
    rocm_lib_path = rocm_path + '/lib'
    rocm_inc = include_directories(rocm_path + '/include')
    amdhip64_lib = cpp.find_library('amdhip64', dirs: [rocm_lib_path], required: false)
    hiprtc_lib = cpp.find_library('hiprtc', dirs: [rocm_lib_path], required: false)
    if amdhip64_lib.found() and hiprtc_lib.found() and \
       cpp.has_header('hip/hip_runtime.h', include_directories: rocm_inc)
        rocm_dep = declare_dependency(
            dependencies : [amdhip64_lib, hiprtc_lib],
            link_args : ['-Wl,-rpath,' + rocm_lib_path],
            include_directories : rocm_inc)
        rocm_available = true
    endif
endif

rocm_detected = rocm_available

# One translation unit cannot include CUDA and HIP headers together, so a single
# build targets exactly one stack. nixlbench_gpu is the one control that picks
# it. use_rocm predates it and only asserted "ROCm must be present" without
# selecting it, so -Duse_rocm=true on a host with both SDKs configured a CUDA
# binary despite ROCm being demanded. It is kept as a compatibility alias that
# now forces the rocm flavor, and contradicting it is an error rather than a
# silent CUDA build.
gpu_flavor = get_option('nixlbench_gpu')
if get_option('use_rocm')
    if gpu_flavor == 'cuda'
        error('use_rocm=true contradicts nixlbench_gpu=cuda; pick one.')
    endif
    gpu_flavor = 'rocm'
endif

if gpu_flavor == 'cuda'
    cuda_available = cuda_detected
    rocm_bench = false
elif gpu_flavor == 'rocm'
    cuda_available = false
    rocm_bench = rocm_detected
else
    cuda_available = cuda_detected
    rocm_bench = rocm_detected
endif
nixlbench_rocm_enabled = rocm_bench and not cuda_available

if gpu_flavor == 'cuda' and not cuda_detected
    error('nixlbench_gpu=cuda was set but CUDA was not found.')
endif
if gpu_flavor == 'rocm' and not rocm_detected
    if rocm_path == ''
        error('A ROCm nixlbench build was requested but rocm_path is unset. ' +
              'Set -Drocm_path=<prefix> to the ROCm install root.')
    endif
    error('nixlbench_gpu=rocm was set but ROCm HIP was not found under ' + rocm_path + '.')
endif

if nixlbench_rocm_enabled
    add_project_arguments('-D__HIP_PLATFORM_AMD__', '-Wno-deprecated-declarations',
                          language: 'cpp')
endif

if cuda_detected and rocm_detected and gpu_flavor == 'auto'
    message(
        'CUDA and ROCm both found: default nixlbench build uses CUDA only. ' +
        'For a HIP binary as well, run a second Meson configure in another ' +
        'build directory with -Dnixlbench_gpu=rocm.',
    )
endif

cuda_fabric_available = false
if cuda_available
    if cpp.has_header_symbol('cuda.h', 'CU_MEM_HANDLE_TYPE_FABRIC', dependencies: cuda_dep)
        cuda_fabric_available = true
    endif
endif

# OpenMP
openmp_dep = dependency('openmp', required: true)

# Try CMake before pkg-config (see root meson.build: matches Debian etcd dev).
etcd_dep = dependency('etcd-cpp-api', required : false, method : 'cmake')
if not etcd_dep.found()
  etcd_dep = dependency('etcd-cpp-api', required : false, method : 'pkg-config')
endif

# GFlags
gflags_dep = dependency('gflags', required: true)

# CLI11 is used only by the opt-in `raw` command hierarchy. Legacy invocations
# continue to use gflags unchanged when CLI11 is unavailable or raw is disabled.
cli11_dep = disabler()
if get_option('build_raw_cli')
    cli11_dep = dependency('CLI11', required: false, fallback: ['cli11'])
    if not cli11_dep.found()
        warning('CLI11 not found. The raw command path will not be built.')
    endif
endif
raw_cli_available = get_option('build_raw_cli') and cli11_dep.found()

# ASIO (usually via libasio-dev package)
asio_dep = dependency('asio', required: true)

# Check for tomlplusplus (header only)
tomlplusplus_dep = dependency('tomlplusplus', required: true, default_options: ['build_lib=false', 'compile_library=false'])

# Ensure etcd is available
etcd_available = etcd_dep.found()
if etcd_available
    add_project_arguments('-DHAVE_ETCD', language: 'cpp')
else
    message('ETCD C++ client library not found. Disabling ETCD runtime.')
endif

etcd_inc = etcd_inc_path
nixl_inc = nixl_path + '/include'
nvshmem_inc = nvshmem_inc_path

inc_dir = include_directories('.', './src/', nixl_inc)

nvshmem_available = false
if nvshmem_lib_path != '' and cuda_available
    if cpp.has_header('nvshmem.h', include_directories: include_directories(nvshmem_inc))
        nvshmem_lib = cpp.find_library('nvshmem', dirs: [nvshmem_lib_path])
        nvshmem_host_lib = cpp.find_library('nvshmem_host', dirs: [nvshmem_lib_path])
        if nvshmem_lib.found()
            nvshmem_available = true
            inc_dir = include_directories('.', './src/', nixl_inc, nvshmem_inc)
            add_project_arguments('-DHAVE_NVSHMEM', language: 'cpp')
            add_project_arguments('-Wno-unused-variable', language: 'cpp')
        endif
    endif
endif

if cuda_available
    add_project_arguments('-DHAVE_CUDA', language: 'cpp')
endif

if nixlbench_rocm_enabled
    add_project_arguments('-DHAVE_ROCM', language: 'cpp')
endif

if cuda_fabric_available
    add_project_arguments('-DHAVE_CUDA_FABRIC', language: 'cpp')
endif

if not cuda_available and not nixlbench_rocm_enabled
    warning('Neither CUDA nor ROCm is enabled for this nixlbench build. VRAM support will be disabled.')
endif

ucx_device_cuda_args = []
ucx_device_override_options = []
ucx_gpu_device_api_available = false
nvcc_prog = find_program('nvcc', required: false)
doca_gpunetio_dep = disabler()

if ucx_path != ''
    ucx_lib_path = ucx_path + '/lib'
    ucx_inc_path = ucx_path + '/include'
    if ucx_inc_path.startswith('/')
        ucx_dep = declare_dependency(
            link_args: ['-L' + ucx_lib_path, '-lucp', '-lucs', '-luct'],
            compile_args: ['-I' + ucx_inc_path])
    else
        ucx_dep = declare_dependency(
            link_args: ['-L' + ucx_lib_path, '-lucp', '-lucs', '-luct'],
            include_directories: include_directories(ucx_inc_path))
    endif
else
    ucx_dep = dependency(
        'ucx',
        modules: ['ucx::ucs', 'ucx::ucp', 'ucx::uct'],
        required: false)
endif

if not ucx_dep.found()
    message('UCX GPU Device API is unavailable because UCX was not found.')
elif not cuda_available
    message('UCX GPU Device API is unavailable because CUDA is unavailable.')
elif not nvcc_prog.found()
    message('UCX GPU Device API is unavailable because nvcc was not found.')
elif nvshmem_available
    message('UCX GPU Device API is unavailable because the NVSHMEM build path is enabled.')
else
    add_languages('cuda', native: false)
    add_project_arguments('-std=c++20', language: 'cuda')
    cuda = meson.get_compiler('cuda')
    if not cuda.version().version_compare('>=12.8')
        message(
            'UCX GPU Device API is unavailable because CUDA ' +
            cuda.version() + ' is older than 12.8.')
    else
        if cuda.version().version_compare('>=12.9') and cuda.version().version_compare('<12.10')
            ucx_device_cuda_args += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
            message('Applying CUDA 12.9 atomic storage flag')
        endif
        doca_gpunetio_dep = dependency('doca-gpunetio', required: false)
        if ucx_device_cuda_arch_list == 'auto'
            ucx_device_cuda_arch_list = '80,86,89,90,100,103,120'
        endif
        selected_cuda_archs = []
        nvcc_ptx_arch_num = 0
        foreach cuda_arch : ucx_device_cuda_arch_list.split(',')
            cuda_arch = cuda_arch.strip()
            if cuda_arch == ''
                continue
            endif
            if cuda_arch.startswith('sm_')
                cuda_arch = cuda_arch.replace('sm_', '')
            endif
            # Validate format early so invalid option values fail fast.
            cuda_arch_num = cuda_arch.to_int()
            if cuda_arch_num > nvcc_ptx_arch_num
                nvcc_ptx_arch_num = cuda_arch_num
            endif
            ucx_device_cuda_args += [
                '-gencode=arch=compute_' + cuda_arch + ',code=sm_' + cuda_arch,
            ]
            selected_cuda_archs += ['sm_' + cuda_arch]
        endforeach
        if selected_cuda_archs.length() == 0
            error('No CUDA targets selected for UCX device kernels')
        endif
        nvcc_ptx_arch = 'compute_' + nvcc_ptx_arch_num.to_string()
        ucx_device_cuda_args += [
            '-gencode=arch=' + nvcc_ptx_arch + ',code=' + nvcc_ptx_arch,
        ]
        message(
            'UCX device kernel CUDA targets: ' +
            ', '.join(selected_cuda_archs) + ' + ' + nvcc_ptx_arch)

        have_gpu_side = cuda.compiles(
            '''
            #include <ucp/api/device/ucp_device_impl.h>
            int main() { return 0; }
            ''',
            dependencies: [ucx_dep, doca_gpunetio_dep],
            args: ucx_device_cuda_args,
            name: 'UCX GPU-side Device API')

        have_nixl_device_api = false
        if have_gpu_side
            have_nixl_device_api = cuda.compiles(
                '''
                #include <gpu/nixl_device.cuh>
                int main() { return 0; }
                ''',
                dependencies: [ucx_dep, doca_gpunetio_dep],
                args: ['-I' + nixl_inc] + ucx_device_cuda_args,
                name: 'NIXL Device API')
        endif

        if have_gpu_side and have_nixl_device_api
            message('UCX Device API is ready for nixlbench.')
            ucx_gpu_device_api_available = true
            add_project_arguments(
                '-DHAVE_UCX_GPU_DEVICE_API',
                language: ['cpp', 'cuda'])
        endif
    endif
endif

summary(
    {
        'UCX GPU Device API': ucx_gpu_device_api_available,
        'nvcc available': nvcc_prog.found(),
        'DOCA GPUNETIO found': doca_gpunetio_dep.found(),
    },
    section: 'Optional features',
    bool_yn: true)

# Subprojects
subdir('src/utils')
subdir('src/runtime')
subdir('src/worker')
subdir('src/kernels')

# Configure header file
configure_file(
    output: 'config.h',
    configuration: {
        'HAVE_ETCD': etcd_available ? '1' : '0',
        'HAVE_NVSHMEM': nvshmem_available ? '1' : '0',
        'HAVE_CUDA': cuda_available ? '1' : '0',
        'HAVE_ROCM': nixlbench_rocm_enabled ? '1' : '0',
        'HAVE_CUDA_FABRIC': cuda_fabric_available ? '1' : '0',
        'HAVE_RAW_CLI': raw_cli_available ? '1' : '0',
    },
    install: true,
    install_dir: get_option('includedir') / 'nixlbench'
)

deps = [nixl_lib, nixl_build, nixl_serdes, openmp_dep, gflags_dep, tomlplusplus_dep, asio_dep]
args = []
if raw_cli_available
    deps += [cli11_dep]
endif
if etcd_available
    deps += [etcd_dep]
endif
if cuda_available
    deps += [cuda_dep]
endif
if nixlbench_rocm_enabled
    deps += [rocm_dep]
endif
if ucx_gpu_device_api_available
    deps += [nixlbench_kernels_dep]
endif
if nvshmem_available
    deps += [nvshmem_lib]
    args += [
        '-Xcompiler', '-fopenmp',
        '-Xlinker', '-rpath=/usr/local/cuda/lib64',
        '-Xlinker', '--allow-shlib-undefined',
        '-lcudart',
        '-lcudadevrt',
    ]
endif

if nvshmem_available
    # Use nvcc directly for compilation and linking
    nvcc = find_program('nvcc')
    nvcc_args = []
    if etcd_available
        if etcd_inc != ''
            nvcc_args += ['-I' + etcd_inc]
        endif
    endif
    nvcc_args += ['-I' + nixl_inc]
    nvcc_args += ['-I' + nvshmem_inc]
    nvcc_args += ['-I.', '-I./src/', '-I../src/']
    nvcc_args += ['-I' + meson.current_build_dir()]
    if etcd_available
        if etcd_lib_path != ''
            nvcc_args += ['-L' + etcd_lib_path]
        endif
    endif
    nvcc_args += ['-L' + nixl_lib_path]
    nvcc_args += ['-L' + nvshmem_lib_path]
    nvcc_args += ['-L' + meson.current_build_dir() + '/src/utils']
    nvcc_args += ['-L' + meson.current_build_dir() + '/src/runtime']
    nvcc_args += ['-L' + meson.current_build_dir() + '/src/worker']
    nvcc_args += ['-lnixl', '-lnixl_build', '-lserdes', '-lcuda', '-lcudart', '-lnvshmem']
    nvcc_args += args
    nvcc_cmd_files = [
                 meson.current_build_dir() + '/src/utils/libutils.a.p/utils.cpp.o',
                 meson.current_build_dir() + '/src/runtime/libruntime.a.p/runtime.cpp.o',
                 meson.current_build_dir() + '/src/worker/libworker.a.p/worker.cpp.o',
                 meson.current_build_dir() + '/src/worker/nixl/libnixl_worker.a.p/nixl_worker.cpp.o',
                 meson.current_build_dir() + '/src/worker/nvshmem/libnvshmemWorker.a.p/nvshmem_worker.cpp.o'
                 ]

    if raw_cli_available
        nvcc_cmd_files += meson.current_build_dir() + '/src/utils/libutils.a.p/raw_cli.cpp.o'
    endif

    if etcd_available
        nvcc_args += ['-letcd-cpp-api-core', '-lcpprest']
        etcd_rt = meson.current_build_dir() + '/src/runtime/etcd/libetcd_rt.a.p/etcd_rt.cpp.o'
        nvcc_cmd_files += etcd_rt
    endif

    nvcc_command = [nvcc, nvcc_args, '-o', '@OUTPUT@', '@INPUT@']
    nvcc_command += nvcc_cmd_files

    nixlbench_target = custom_target('nixlbench',
        input: 'src/main.cpp',
        output: 'nixlbench',
        command: nvcc_command,
        build_by_default: true,
        install: true,
        install_dir: get_option('bindir'),
        depends: [nixlbench_runtimes, utils_lib, worker_libs])
else
    nixlbench_target = executable('nixlbench', 'src/main.cpp',
                include_directories: inc_dir,
                link_with: [nixlbench_runtimes, utils_lib, worker_libs],
                dependencies: deps,
                link_args: args,
                build_rpath: nixl_lib_path,
                install_rpath: nixl_lib_path,
                install: true,
                install_dir: get_option('bindir'))
endif

if get_option('build_tests') and get_option('buildtype') != 'release'
    if raw_cli_available
        subdir('tests')
    else
        message('Skipping raw CLI tests because the raw command path is not built.')
    endif
endif
