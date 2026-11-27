# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

valkey_unit_test_dep = declare_dependency(
    sources: [
        'valkey.cpp',
    ],
    include_directories: [
        nixl_inc_dirs, utils_inc_dirs,
        '../../../../src/plugins/valkey',
    ],
    dependencies: [
        valkey_backend_interface,
    ],
)
