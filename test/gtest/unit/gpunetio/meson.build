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

write_pairing_test = executable(
    'gpunetio_write_pairing_test',
    'write_pairing_test.cpp',
    include_directories: include_directories('../../../../src/plugins/gpunetio'),
    dependencies: dependency('gtest', main: true),
)
test('gpunetio_write_pairing', write_pairing_test)

if is_variable('gpunetio_backend_interface')
    peer_write2_test = executable(
        'gpunetio_peer_write2_test',
        'peer_write2_test.cpp',
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        dependencies: [nixl_dep, nixl_infra, cuda_dep, dependency('gtest', main: true)],
        link_with: [nixl_build_lib, serdes_lib],
    )
    test('gpunetio_peer_write2', peer_write2_test, timeout: 120)
endif
