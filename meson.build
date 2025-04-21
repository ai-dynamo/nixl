# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project('nixl', 'CPP', version: '0.1.1',
    default_options: ['buildtype=debug',
                'werror=true',
                'cpp_std=c++17',
                'prefix=/opt/nvidia/nvda_nixl'],
    meson_version: '>= 0.64.0'
)

# set up some global vars for compiler, platform, configuration, etc.
cpp = meson.get_compiler('cpp')

# Forced to ignore this error due to:
# https://github.com/abseil/abseil-cpp/issues/1779
# This must be global since subprojects cannot be assigned with arguments.
# Adding this configuration only for release build to consider it while developing.
if get_option('buildtype') == 'release'
  abseil_flags = cpp.get_supported_arguments('-Wno-error=maybe-uninitialized', '-Wno-maybe-uninitialized')
  add_global_arguments(abseil_flags, language: 'cpp')
endif
abseil_proj = subproject('abseil-cpp')

cuda_inc_path = get_option('cudapath_inc')
cuda_lib_path = get_option('cudapath_lib')
cuda_stub_path = get_option('cudapath_stub')

if cuda_lib_path == ''
    cuda_dep = dependency('cuda', required : false, modules : [ 'cudart', 'cuda' ])
else
    message('cuda lib path ', cuda_lib_path)
    if cuda_stub_path == ''
        cuda_stub_path = cuda_lib_path + '/stubs'
    endif
    cuda_dep = declare_dependency(
    link_args : ['-L' + cuda_lib_path, '-L' + cuda_stub_path, '-lcuda', '-lcudart'],
    include_directories : include_directories(cuda_inc_path))
endif

prefix_path = get_option('prefix')
prefix_inc = prefix_path + '/include'

ucx_path = get_option('ucx_path')
if ucx_path != ''
  ucx_lib_path = ucx_path + '/lib'
  ucx_inc_path = ucx_path + '/include'
  ucx_dep = declare_dependency(
    link_args : ['-L' + ucx_lib_path, '-lucp', '-lucs', '-luct'],
    include_directories : include_directories(ucx_inc_path))
else
  ucx_dep = dependency('ucx', modules: ['ucx::ucs', 'ucx::ucp', 'ucx::uct'])
endif

if get_option('disable_gds_backend')
    add_project_arguments('-DDISABLE_GDS_BACKEND', language: 'cpp')
endif

static_plugins = []

# Check for static plugins, then set compiler flags to enable
if get_option('static_plugins') != ''
    static_plugins = get_option('static_plugins').split(',')
    foreach p : static_plugins
        flagname = '-DSTATIC_PLUGIN_' + p
        add_project_arguments(flagname, language: 'cpp')
    endforeach
endif

# Define a specific plugin directory
plugin_install_dir = join_paths(get_option('libdir'), 'plugins')
plugin_build_dir = meson.current_build_dir()

# Add to global args so plugin managers can find it
if get_option('buildtype') == 'debug'
    add_project_arguments('-DNIXL_USE_PLUGIN_FILE="' + plugin_build_dir + '/pluginlist"',  language: 'cpp')
endif

nixl_inc_dirs = include_directories('src/api/cpp', 'src/infra', 'src/core')
utils_inc_dirs = include_directories('src/utils')

subdir('src')

if get_option('buildtype') != 'release'
  subdir('test')
  subdir('examples')
endif

if get_option('install_headers')
  install_headers('src/api/cpp/nixl.h', install_dir: prefix_inc)
  install_headers('src/api/cpp/nixl_types.h', install_dir: prefix_inc)
  install_headers('src/api/cpp/nixl_params.h', install_dir: prefix_inc)
  install_headers('src/api/cpp/nixl_descriptors.h', install_dir: prefix_inc)
  install_headers('src/utils/serdes/serdes.h', install_dir: prefix_inc + '/utils/serdes')
  install_headers('src/utils/common/nixl_time.h', install_dir: prefix_inc + '/utils/common')
  install_headers('src/api/cpp/backend/backend_engine.h', install_dir: prefix_inc + '/backend')
  install_headers('src/api/cpp/backend/backend_aux.h', install_dir: prefix_inc + '/backend')
  install_headers('src/core/transfer_request.h', install_dir: prefix_inc)
  install_headers('src/core/agent_data.h', install_dir: prefix_inc)
  install_headers('src/infra/mem_section.h', install_dir: prefix_inc)
endif

# Doxygen documentation
if get_option('build_docs')
  doxygen = find_program('doxygen', required: false)
  if not doxygen.found()
    error('Doxygen not found, but documentation requested')
  endif

  docs_dir = join_paths(meson.current_build_dir(), 'docs')
  doxyfile = join_paths(meson.current_source_dir(), 'Doxyfile')
  doxygen_output_dir = join_paths(docs_dir, 'doxygen')

  # Create the output directory
  run_command('mkdir', '-p', doxygen_output_dir, check: true)

  # Configure Doxyfile with the correct output directory
  configure_file(
    input: doxyfile,
    output: 'Doxyfile.configured',
    configuration: {
      'DOXYGEN_OUTPUT_DIR': doxygen_output_dir,
    }
  )

  custom_target('docs',
    output: 'docs',
    command: [doxygen, join_paths(meson.current_build_dir(), 'Doxyfile.configured')],
    install: true,
    install_dir: join_paths(prefix_path),
    build_by_default: true
  )
endif
