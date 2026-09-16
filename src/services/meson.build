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

nixl_service_inc = include_directories('.')

nixl_service_deps = [nixl_dep, nixl_infra, serdes_interface, nixl_common_dep, cuda_dep]

subdir('marshal')

# Exported so consumers (e.g. the _service_bindings Python extension) can
# compile nixl_service.cpp directly into their own binary without depending
# on libnixl_service.so at runtime.  files() gives an absolute-path handle
# that meson accepts across subdir boundaries.
nixl_service_sources = files('nixl_service.cpp')

if not nvcomp_dep.found()
    error('nvCOMP is required to build nixl_service. Set -Dnvcomp_path_lib (and -Dnvcomp_path_inc) or install nvCOMP system-wide.')
endif

nixl_service_compile_args = []
nixl_service_extra_deps = [staging_dep]
nixl_service_extra_deps += [delta_dep]
nixl_service_extra_deps += [compression_dep]

nixl_service_lib = library('nixl_service',
                           nixl_service_sources,
                           cpp_args: nixl_service_compile_args,
                           include_directories: [nixl_inc_dirs, utils_inc_dirs, nixl_service_inc],
                           dependencies: nixl_service_deps + nixl_service_extra_deps,
                           link_with: nixl_lib,
                           install: true,
                           install_rpath: join_paths(get_option('prefix'), get_option('libdir')))

nixl_service_dep = declare_dependency(
    link_with: nixl_service_lib,
    include_directories: nixl_service_inc,
    compile_args: nixl_service_compile_args,
    dependencies: nixl_service_deps + nixl_service_extra_deps)

if get_option('install_headers')
    install_headers('nixl_service.h', 'nixl_service_types.h', subdir: 'nixl')
    # nixl_service.h pulls in the marshal backends, and those include each other by
    # relative path, so the installed tree has to mirror the source layout.
    marshal_headers = ['marshal/marshal_backend.h']
    if nvcomp_dep.found()
        marshal_headers += ['marshal/compression/compression_backend.h']
    endif
    install_headers(marshal_headers, subdir: 'nixl', preserve_path: true)
endif