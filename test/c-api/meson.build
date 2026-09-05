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

# C consumer checks for the nixl C API: the header must be strict C11, and a
# plain C program must be able to drive the basic agent lifecycle.
if have_c
    capi_c_args = [
        '-std=c11',
        '-Wall',
        '-Wextra',
        '-Werror',
        '-Wstrict-prototypes',
    ]
    capi_test_inc = include_directories('../../src/api/c')

    static_library(
        'nixl_capi_c11_check',
        'capi_compile_check.c',
        c_args: capi_c_args,
        include_directories: [capi_test_inc],
        build_by_default: true,
    )

    capi_smoke = executable(
        'nixl_capi_smoke_test',
        'capi_smoke_test.c',
        c_args: capi_c_args,
        include_directories: [capi_test_inc],
        link_with: [nixl_capi_libs.get_shared_lib()],
    )
    # Point the smoke test at the in-tree UCX plugin so it can exercise the
    # metadata path; without it the test skips those checks.
    capi_test_env = environment()
    if is_variable('ucx_backend_lib')
        fs = import('fs')
        capi_test_env.set('NIXL_PLUGIN_DIR', fs.parent(ucx_backend_lib.full_path()))
    endif
    test('capi_smoke', capi_smoke, env: capi_test_env)
endif
