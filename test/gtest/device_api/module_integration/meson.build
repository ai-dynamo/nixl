# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

module_integration_inc = include_directories('.')
module_integration_deps = [
    cuda_dep,
    gpu_device_api_deps,
    nixl_dep,
]
module_integration_link_with = [
    nixl_build_lib,
    nixl_gpu_proxy_lib,
]

early_dso_a = shared_library(
    'nixl_device_api_consumer_a',
    'early_dso_a.cu',
    dependencies: module_integration_deps,
    include_directories: [
        nixl_gpu_device_api_inc_dirs,
        nixl_inc_dirs,
        utils_inc_dirs,
        module_integration_inc,
    ],
    link_with: module_integration_link_with,
)

early_dso_b = shared_library(
    'nixl_device_api_consumer_b',
    'early_dso_b.cu',
    dependencies: module_integration_deps,
    include_directories: [
        nixl_gpu_device_api_inc_dirs,
        nixl_inc_dirs,
        utils_inc_dirs,
        module_integration_inc,
    ],
    link_with: module_integration_link_with,
)

late_dso = shared_library(
    'nixl_device_api_consumer_late',
    'late_dso.cu',
    dependencies: module_integration_deps,
    include_directories: [
        nixl_gpu_device_api_inc_dirs,
        nixl_inc_dirs,
        utils_inc_dirs,
        module_integration_inc,
    ],
    link_with: module_integration_link_with,
)

module_integration_test = executable(
    'device_module_integration_test',
    [
        'main.cpp',
        'module_integration_test.cpp',
        'executable_tu_a.cu',
        'executable_tu_b.cu',
    ],
    cpp_args: ['-DNIXL_TEST_LATE_DSO_PATH="' + late_dso.full_path() + '"'],
    dependencies: module_integration_deps + [gtest_dep],
    include_directories: [
        nixl_gpu_device_api_inc_dirs,
        nixl_inc_dirs,
        utils_inc_dirs,
        gtest_inc_dirs,
        module_integration_inc,
    ],
    link_with: module_integration_link_with + [early_dso_a, early_dso_b],
)

test(
    'device_module_integration',
    module_integration_test,
    depends: [late_dso],
    is_parallel: false,
    timeout: 120,
)
