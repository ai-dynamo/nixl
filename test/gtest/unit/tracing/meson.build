# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

tracing_unit_test_sources = files('trace_context_test.cpp', 'tracing_test.cpp')
tracing_unit_test_inc_dirs = [nixl_inc_dirs, gtest_inc_dirs]
tracing_unit_test_deps = [gmock_dep, nixl_common_dep]

if nixl_trace_nvtx_enabled
    tracing_unit_test_sources += files(
        '../../../../src/plugins/tracing/nvtx/nvtx_events.cpp',
        'nvtx_test.cpp',
    )
    tracing_unit_test_inc_dirs += [include_directories('../../../../src/plugins/tracing/nvtx')]
    tracing_unit_test_deps += [nvtx_inc_dep, dl_dep]
endif

tracing_unit_test_dep = declare_dependency(
    sources: tracing_unit_test_sources,
    include_directories: tracing_unit_test_inc_dirs,
    dependencies: tracing_unit_test_deps,
)
