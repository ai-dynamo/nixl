# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cute_binding_test = executable('cute_device_binding',
    'device_binding.cpp',
    include_directories: [nixl_inc_dirs, include_directories('../../../../src/api/device')],
    dependencies: gtest_dep,
)
test('cute_device_binding', cute_binding_test)
