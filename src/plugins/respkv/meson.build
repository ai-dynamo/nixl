# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

respkv_engine_lib = static_library(
    'respkv_engine',
    sources: ['respkv_engine.cpp', 'respkv_engine.h'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, include_directories('.')],
    dependencies: [nixl_infra, nixl_common_dep, absl_strings_dep],
    install: false,
)

respkv_engine_dep = declare_dependency(
    link_with: respkv_engine_lib,
    include_directories: include_directories('.'),
    dependencies: [nixl_infra, nixl_common_dep],
)

if enabled_plugins.get('REDIS')
    subdir('redis')
endif

if enabled_plugins.get('VALKEY')
    subdir('valkey')
endif
