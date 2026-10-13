# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gpunetio_completion_status_test = executable(
    'gpunetio_completion_status_test',
    'gpunetio_completion_status_test.cpp',
    include_directories: [nixl_inc_dirs,
                         include_directories('../../../../src/plugins/gpunetio')],
    dependencies: dependency('gtest', main: true),
    install: true,
)
test('gpunetio_completion_status', gpunetio_completion_status_test)
