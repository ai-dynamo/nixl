# SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
wqs_unit_test = executable(
    'wqs_unit',
    sources: ['../../main.cpp', 'test_wqs.cpp'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../../src/plugins/wqs'],
    dependencies: [wqs_backend_interface, nixl_dep, gtest_dep],
)

test('wqs_unit', wqs_unit_test, env: sanitizer_env, suite: ['wqs', 'sanitizer'])
