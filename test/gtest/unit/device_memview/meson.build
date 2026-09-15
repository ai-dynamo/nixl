# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

device_memview_unit_test_dep = declare_dependency(
    sources: ['device_memview.cpp'],
    include_directories: [nixl_inc_dirs, nixl_device_inc_dirs, utils_inc_dirs, gtest_inc_dirs],
)

device_memview_test_exe = executable(
    'device-memview-test',
    'device_memview.cpp',
    include_directories: [nixl_inc_dirs, nixl_device_inc_dirs, utils_inc_dirs, gtest_inc_dirs],
    dependencies: [
        nixl_dep,
        device_allocator_interface,
        absl_log_dep,
        dependency('gtest', main: true),
    ],
)

test('device-memview-test', device_memview_test_exe)
