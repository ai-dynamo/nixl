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

nixl_worker_sources = [
    'nixl_worker.cpp',
    'nixl_worker.h',
]

nixl_worker_cuda_sources = [
    'gdaki_kernels.cu',
]

nixl_worker_deps = [openmp_dep, nixl_lib, nixl_build, nixl_serdes, cxxopts_dep, tomlplusplus_dep]
if cuda_available
    nixl_worker_deps += [cuda_dep]
endif

# Handle CUDA compilation if available
cuda_objects = []
if cuda_available and have_nixl_gdaki
    # Find nvcc compiler
    nvcc = find_program('nvcc', required: false)
    if nvcc.found()
      # Build NVCC command arguments
      nvcc_args = [
        nvcc,
        '-c', '@INPUT@',
        '-o', '@OUTPUT@',
        '-I' + nixl_inc,
        '-I' + meson.current_source_dir() + '/../../..',
        '-I' + meson.current_source_dir() + '/../..',
        '-I' + nixl_path + '/include',
      ]

      cxxopts_inc = cxxopts_dep.get_variable(pkgconfig: 'includedir',
                                             cmake: 'CMAKE_INSTALL_INCLUDEDIR',
                                             default_value: '')
      if cxxopts_inc != ''
        nvcc_args += ['-I' + cxxopts_inc]
      endif
      cxxopts_subproj_inc = join_paths(meson.project_source_root(),
                                       'subprojects', 'cxxopts-3.2.0', 'include')
      if fs.exists(cxxopts_subproj_inc)
        nvcc_args += ['-I' + cxxopts_subproj_inc]
      endif
      toml_inc = tomlplusplus_dep.get_variable(pkgconfig: 'includedir',
                                               cmake: 'CMAKE_INSTALL_INCLUDEDIR',
                                               default_value: '')
      if toml_inc != ''
        nvcc_args += ['-I' + toml_inc]
      endif
      toml_subproj_inc = join_paths(meson.project_source_root(),
                                    'subprojects', 'tomlplusplus-3.4.0', 'include')
      if fs.exists(toml_subproj_inc)
        nvcc_args += ['-I' + toml_subproj_inc]
      endif

      # Add UCX include path if ucxpath_inc option is provided
      ucxpath_inc = get_option('ucxpath_inc')
      if ucxpath_inc != ''
          nvcc_args += [
            '-I' + ucxpath_inc,
            '-I' + '/opt/mellanox/doca/include']
      endif

      nvcc_args += [
        '--std=c++17',
        '-arch=sm_80',
        '--ptxas-options=-v',
        '--maxrregcount=32',
        '--compiler-options', '-fPIC',
        '-DHAVE_CUDA'
      ]


      # Compile CUDA kernels
      nixl_cuda_obj = custom_target('nixl_gdaki_kernels_obj',
        input: nixl_worker_cuda_sources,
        output: 'gdaki_kernels.o',
        command: nvcc_args,
        install: false
      )
      cuda_objects = [nixl_cuda_obj]
    else
      warning('NVCC not found - CUDA kernels will not be available')
    endif
endif

nixl_worker_lib = static_library('nixl_worker',
    nixl_worker_sources,
    objects: cuda_objects,
    include_directories: inc_dir,
    dependencies: nixl_worker_deps,
    install: true,
)
