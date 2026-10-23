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

# Check for AWS SDK dependencies for S3 utilities
aws_s3 = dependency('aws-cpp-sdk-s3', static: false, required: false)
aws_s3_crt = dependency('aws-cpp-sdk-s3-crt', static: false, required: false)
aws_core = dependency('aws-cpp-sdk-core', required: false, static: false)

if aws_s3.found() and aws_s3_crt.found() and aws_core.found()
    # By default aws-cpp-sdk sets a c++11 compile flag for the whole project, so
    # the SDK is consumed below with compile_args dropped. pkg-config carries the
    # header search paths in those same Cflags, so re-derive them here. Stock
    # aws-sdk-cpp .pc files have no prefix variable, and their includedir and
    # libdir are relative to the install prefix. The .pc file is somewhere below
    # that prefix, so walk up from pcfiledir to the first dir whose includedir
    # has the SDK headers. A relative libdir means the .pc file is in
    # <libdir>/pkgconfig. The relative -L resolves against the build dir, so add
    # the absolute one, and find the library by path so meson also puts its dir
    # on the build rpath. is_system also lets meson drop any path that is
    # already a compiler default.
    aws_sdk_include_dirs = []
    aws_sdk_seen_includes = []
    aws_sdk_libs = []
    foreach aws_dep : [aws_s3, aws_s3_crt, aws_core]
        if aws_dep.type_name() != 'pkgconfig'
            continue
        endif
        aws_inc = aws_dep.get_variable(pkgconfig: 'includedir', default_value: '')
        aws_lib = aws_dep.get_variable(pkgconfig: 'libdir', default_value: '')
        # Take the parent of the .pc file path, so a trailing slash in pcfiledir
        # does not change the result.
        aws_pc_dir = fs.parent(aws_dep.get_variable(pkgconfig: 'pcfiledir') / (aws_dep.name() + '.pc'))
        if aws_lib != '' and not fs.is_absolute(aws_lib) and fs.is_absolute(aws_pc_dir)
            aws_lib_dir = fs.parent(aws_pc_dir)
            aws_sdk_libs += declare_dependency(link_args: '-L' + aws_lib_dir)
            aws_sdk_libs += cpp.find_library(aws_dep.name(), dirs: aws_lib_dir, static: false, required: false)
        endif
        if aws_inc != '' and not fs.is_absolute(aws_inc)
            aws_dir = aws_pc_dir
            foreach _ : aws_pc_dir.split('/')
                aws_dir = fs.parent(aws_dir)
                if fs.is_file(aws_dir / aws_inc / 'aws/core/Aws.h')
                    aws_inc = aws_dir / aws_inc
                    break
                endif
            endforeach
        endif
        if aws_inc != '' and fs.is_absolute(aws_inc) and aws_inc not in aws_sdk_seen_includes and fs.is_dir(aws_inc)
            aws_sdk_seen_includes += aws_inc
            aws_sdk_include_dirs += include_directories(aws_inc, is_system: true)
        endif
    endforeach
    aws_sdk_paths = declare_dependency(include_directories: aws_sdk_include_dirs, dependencies: aws_sdk_libs)

    partial_aws_s3 = aws_s3.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)
    partial_aws_s3_crt = aws_s3_crt.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)
    partial_aws_core = aws_core.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)

    # Shared S3-over-RDMA infrastructure (NVIDIA cuObject / proposed aws-c-s3
    # protocol), split by responsibility: cuobj_client (buffer registration +
    # token minting), s3_control_plane_http (the signed control-plane GET/PUT),
    # and rdmaRetry (token-lifecycle wrappers). The cuObject-dependent units are
    # added to the build only when the cuObjClient library is found (below), so
    # they carry no in-source preprocessor guard; rdma_protocol.h is dependency-
    # free and always built, and rdma.h is the public umbrella. These live here
    # (not in the OBJ plugin) so the accel engines — and any future vendor
    # engine — can compose them.
    obj_utils_sources = ['s3/utils.cpp', 's3/utils.h',
                         'rdma/rdma_protocol.h']
    obj_utils_deps = [nixl_common_dep, aws_sdk_paths, partial_aws_s3, partial_aws_s3_crt, partial_aws_core]
    if cuobj_dep.found()
        obj_utils_sources += ['rdma/rdma.h',
                              'rdma/cuobj_client.h', 'rdma/cuobj_client.cpp',
                              'rdma/s3_control_plane_http.h', 'rdma/s3_control_plane_http.cpp',
                              'rdma/rdmaRetry.h', 'rdma/rdmaRetry.cpp']
        obj_utils_deps += [cuobj_dep]
    endif

    # Generic object utilities library containing S3 utilities and other object storage utilities
    obj_utils_lib = library('obj_utils',
               obj_utils_sources,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               dependencies: obj_utils_deps,
               install: true)

    # rdma.h is part of the public surface: it pulls in nixl_types.h (nixl_common_dep)
    # and, when enabled, <cuobjclient.h> (cuobj_dep). Forward the same public deps the
    # library was built with so downstream consumers compile against the header.
    obj_utils_interface = declare_dependency(
        link_with: obj_utils_lib,
        include_directories: [include_directories('.'), include_directories('s3')],
        dependencies: obj_utils_deps
    )
else
    warning('AWS SDK dependencies not found, skipping obj_utils library build')
endif

