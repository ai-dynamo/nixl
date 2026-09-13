# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gpunetio_oob_endpoint_test = executable(
    'gpunetio_oob_endpoint_test',
    'gpunetio_oob_endpoint_test.cpp',
    include_directories: include_directories('../../../../src/plugins/gpunetio'),
    install: true,
)
test('gpunetio_oob_endpoint', gpunetio_oob_endpoint_test)

gpunetio_oob_socket_test = executable(
    'gpunetio_oob_socket_test',
    'gpunetio_oob_socket_test.cpp',
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        include_directories('../../../../src/plugins/gpunetio'),
    ],
    dependencies: [gpunetio_backend_interface, cuda_dep, absl_log_dep] + plugin_gpunetio_deps,
    install: true,
)
test('gpunetio_oob_socket', gpunetio_oob_socket_test)
