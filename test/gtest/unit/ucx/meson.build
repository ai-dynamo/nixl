# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ucx_unit_test_dep = declare_dependency(
    sources: ['ucx_ep_rkey_test.cpp'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, gtest_inc_dirs],
    dependencies: [ucx_backend_interface, ucx_dep, thread_dep],
)
