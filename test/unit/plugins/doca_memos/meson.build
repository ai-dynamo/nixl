# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DOCA_MEMOS backend unit tests with mocked DOCA (doca_kv) dependencies

# Check for GTest dependency
gtest_dep_local = dependency('gtest', version : '>=1.11.0', required : false)
if not gtest_dep_local.found()
    message('GTest not found, skipping DOCA_MEMOS backend tests')
    subdir_done()
endif

# Abseil: use deps from top-level meson.build (system or subproject)

doca_memos_test_sources = files(
    'doca_memos_backend_test.cpp',
    'doca_memos_mocks.cpp',
)

# Builds the backend translation units directly against the DOCA mocks
# headers in this directory rather than linking the plugin library.
# doca_memos_backend_sources is exported by src/plugins/doca_memos/meson.build.
doca_memos_backend_test = executable(
    'doca_memos_backend_test',
    sources: [
        doca_memos_test_sources,
        doca_memos_backend_sources,
    ],
    include_directories: [
        # Mock foundational headers (doca_compat/ctx/error/pe/types.h) and the
        # concrete struct definitions in doca_memos_mocks.h. MUST BE FIRST so the
        # mocks shadow the foundational DOCA headers; the real DOCA KV headers
        # (doca_kvdev*.h, doca_nvme_kernel_kvdev*.h) are picked up from the SDK
        # via the doca-kv/doca-common include paths below.
        include_directories('.'),
        include_directories('../../../../src/plugins/doca_memos'),  # For backend headers
        nixl_inc_dirs,  # Use existing NIXL include directories
        utils_inc_dirs,  # Use existing utils include directories
    ],
    dependencies: [
        gtest_dep_local,
        absl_strings_dep,
        absl_log_dep,
        nixl_infra,  # For nixl_time and other infra
        dependency('threads'),
        # Headers only: the DOCA functions are provided by doca_memos_mocks.cpp,
        # so pull in the real KV API declarations without linking libdoca_kv.
        doca_kv_dep.partial_dependency(includes: true, compile_args: true),
        doca_common_dep.partial_dependency(includes: true, compile_args: true),
    ],
    cpp_args: [
        '-DDOCA_ALLOW_EXPERIMENTAL_API',
        '-DNIXL_UNIT_TEST',  # Optional: can be used to ifdef test-specific code
    ],
    link_args: [],
    install: true,
)

# Register the test with meson
test('doca_memos_backend_unit_test', doca_memos_backend_test,
     timeout: 60,
     suite: ['unit', 'plugins', 'doca_memos'])
