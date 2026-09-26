# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

daos_unit_test_dep = declare_dependency(
    sources: ['daos.cpp'],
    include_directories: [
        nixl_inc_dirs,
        '../../../../src/plugins/daos',
    ],
    link_with: daos_backend_lib,
)
