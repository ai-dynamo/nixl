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

# ROCm detection (mutually exclusive with CUDA; default off)
rocm_path = get_option('rocm_path')
if rocm_path != ''
    if rocm_path == 'default'
        rocm_path = '/opt/rocm'
    endif
    if not fs.exists(rocm_path)
        error('rocm_path set but ROCm install root not found at ' + rocm_path +
              '. Set -Drocm_path=<prefix> to the correct ROCm install root.')
    endif
    rocm_lib_path = rocm_path + '/lib'
    rocm_inc_path = rocm_path + '/include'
    rocm_dep = declare_dependency(
        link_args : ['-L' + rocm_lib_path, '-lamdhip64', '-lhiprtc'],
        include_directories : include_directories(rocm_inc_path))
else
    rocm_dep = disabler()
endif

ucx_test_dep = [ ucx_dep, nixl_common_deps, ucx_backend_interface]

if rocm_dep.found()
    ucx_test_dep += [ rocm_dep ]
    cpp_args = ['-DHAVE_GPU', '-DHAVE_ROCM', '-D__HIP_PLATFORM_AMD__', '-Wno-deprecated-declarations']
elif cuda_dep.found()
    ucx_test_dep += [ cuda_dep ]
    cpp_args = ['-DHAVE_GPU', '-DHAVE_CUDA']
else
    cpp_args = ['-UHAVE_CUDA']
endif

ucx_backend_test_dep = [nixl_dep, nixl_infra, thread_dep] + ucx_test_dep + nixl_test_utils_dep
ucx_test_include_directories = [nixl_inc_dirs, utils_inc_dirs]

ucx_backend_test = executable('ucx_backend_test',
        'ucx_backend_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories,
        cpp_args : cpp_args,
        install: true)

ucx_backend_multi = executable('ucx_backend_multi',
           'ucx_backend_multi.cpp',
           dependencies: ucx_backend_test_dep,
           include_directories: ucx_test_include_directories,
           cpp_args : cpp_args,
           install: true)

if get_option('buildtype') != 'release'

    ucx_worker_bin = executable('ucx_worker_test',
               'ucx_worker_test.cpp',
               dependencies: ucx_test_dep,
               include_directories: ucx_test_include_directories,
               install: true)

    if cuda_dep.found() or rocm_dep.found()
        ucx_worker_gpu_bin = executable('ucx_worker_test_gpu',
                       'ucx_worker_test.cpp',
                       dependencies: ucx_test_dep,
                       include_directories: ucx_test_include_directories,
                       cpp_args : ['-DUSE_VRAM'] + cpp_args,
                       install: true)
    endif

endif
