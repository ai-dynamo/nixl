# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gpunetio_test_inc = include_directories('../../../../src/plugins/gpunetio')
gpunetio_oob_endpoint_test = executable(
    'gpunetio_oob_endpoint_test',
    'gpunetio_oob_endpoint_test.cpp',
    include_directories: gpunetio_test_inc,
    dependencies: dependency('gtest', main: true),
    install: true,
)
test('gpunetio_oob_endpoint', gpunetio_oob_endpoint_test)

# Parser coverage must remain available without the optional plugin or DOCA SDK.
if not is_variable('gpunetio_backend_interface')
    subdir_done()
endif

gpunetio_oob_socket_test = executable(
    'gpunetio_oob_socket_test',
    'gpunetio_oob_socket_test.cpp',
    include_directories: [nixl_inc_dirs, utils_inc_dirs, gpunetio_test_inc],
    dependencies: [gpunetio_backend_interface, cuda_dep, absl_log_dep,
                   dependency('gtest', main: true)] + plugin_gpunetio_deps,
    install: true,
)
test('gpunetio_oob_socket', gpunetio_oob_socket_test)
