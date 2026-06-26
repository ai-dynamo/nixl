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
    # By default aws-cpp-sdk sets c++11 compile flag for the whole project
    partial_aws_s3 = aws_s3.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)
    partial_aws_s3_crt = aws_s3_crt.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)
    partial_aws_core = aws_core.partial_dependency(compile_args: false, includes: true, link_args: true, links: true)

    # Shared S3-over-RDMA infrastructure (NVIDIA cuObject / proposed aws-c-s3
    # protocol). rdma.cpp is internally guarded by HAVE_CUOBJ_CLIENT and compiles
    # to nothing when the cuObjClient library is absent; rdma_protocol.h is
    # dependency-free. These live here (not in the OBJ plugin) so the accel
    # engines — and any future vendor engine — can compose them.
    obj_utils_sources = ['s3/utils.cpp', 's3/utils.h', 'rdma_protocol.h', 'rdma.cpp', 'rdma.h']
    obj_utils_deps = [nixl_common_dep, partial_aws_s3, partial_aws_s3_crt, partial_aws_core]
    if cuobj_dep.found()
        obj_utils_deps += [cuobj_dep]
    endif

    # Generic object utilities library containing S3 utilities and other object storage utilities
    obj_utils_lib = library('obj_utils',
               obj_utils_sources,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               dependencies: obj_utils_deps,
               install: true)

    # rdma.h is part of the public surface and pulls in nixl_types.h, so consumers
    # need nixl_common_dep too (alongside the AWS deps).
    obj_utils_interface = declare_dependency(
        link_with: obj_utils_lib,
        include_directories: [include_directories('.'), include_directories('s3')],
        dependencies: [nixl_common_dep, partial_aws_s3, partial_aws_s3_crt, partial_aws_core]
    )
else
    warning('AWS SDK dependencies not found, skipping obj_utils library build')
endif

