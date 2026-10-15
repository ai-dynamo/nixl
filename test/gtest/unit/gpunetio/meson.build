# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

netif_discovery_test = executable(
    'gpunetio_netif_discovery_test',
    'netif_discovery_test.cpp',
    include_directories: [nixl_inc_dirs, utils_inc_dirs,
                         include_directories('../../../../src/plugins/gpunetio')],
    dependencies: [gpunetio_backend_interface, cuda_dep, absl_log_dep,
                   dependency('gtest', main: true)] + plugin_gpunetio_deps,
    install: true,
)
test('gpunetio_netif_discovery', netif_discovery_test)
