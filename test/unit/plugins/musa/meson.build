# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
musa_test_includes = include_directories('../../../../src/api/cpp', '../../../../src/plugins')
musa_cpu_cases = {
    'memory_policy': files('memory_policy_test.cpp'),
    'registration': files('registration_test.cpp'),
    'musa_policy': files('musa_policy_test.cpp', '../../../../src/plugins/musa/musa_policy.cpp'),
    'musa_runtime_fake': files('musa_runtime_test.cpp', 'fake_sdk/runtime.cpp',
                              '../../../../src/plugins/musa/musa_runtime.cpp',
                              '../../../../src/plugins/musa/musa_policy.cpp'),
}
foreach name, sources : musa_cpu_cases
    extra_includes = name == 'musa_runtime_fake' ? [include_directories('fake_sdk')] : []
    target = executable('test_' + name, sources,
        include_directories: [musa_test_includes] + extra_includes)
    test(name, target, suite: 'musa-cpu', timeout: 30)
endforeach

musa_test_python = find_program('python3')
test('musa_build_contract', musa_test_python,
    args: [files('test_build_contract.py'), '-v'], suite: 'musa-configure', timeout: 180)

if ucx_dep.found() and (enabled_plugins.get('UCX') or
                       (enabled_plugins.get('MUSA_UCX') and musa_dep.found()))
    musa_factory_test = executable('test_ucx_policy_factory', 'ucx_policy_factory_test.cpp',
        dependencies: [ucx_impl_interface],
        include_directories: [musa_test_includes, nixl_inc_dirs, utils_inc_dirs])
    test('ucx_policy_factory', musa_factory_test, suite: 'musa-ucx', timeout: 60)
    test('ucx_policy_rejects_sgl', musa_factory_test,
        args: ['--reject-sgl'], env: ['NIXL_UCX_SGL_ENABLE=1'], suite: 'musa-ucx', timeout: 60)
    if host_machine.system() == 'linux'
        musa_failure_test = executable('test_ucx_registration_failure',
            'ucx_registration_failure_test.cpp',
            dependencies: [ucx_impl_interface],
            include_directories: [musa_test_includes, nixl_inc_dirs, utils_inc_dirs],
            link_args: ['-Wl,--wrap=ucp_mem_map', '-Wl,--wrap=ucp_mem_query',
                        '-Wl,--wrap=ucp_mem_unmap', '-Wl,--wrap=ucp_rkey_pack',
                        '-Wl,--wrap=ucp_ep_rkey_unpack', '-Wl,--wrap=ucp_rkey_destroy',
                        '-Wl,--wrap=ucp_config_modify'])
        test('ucx_registration_failure', musa_failure_test, suite: 'musa-ucx', timeout: 60)
    endif
endif

if enabled_plugins.get('MUSA_UCX') and musa_dep.found() and ucx_dep.found()
    musa_plugin_test = executable('test_musa_plugin', 'musa_plugin_test.cpp',
        dependencies: [dependency('dl'), nixl_common_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs])
    test('musa_plugin', musa_plugin_test, args: [musa_backend_lib.full_path()],
        depends: musa_backend_lib, suite: 'musa-plugin', timeout: 30)
endif

if get_option('musa_hardware_tests')
    if not enabled_plugins.get('MUSA_UCX') or not musa_dep.found() or not ucx_dep.found()
        error('musa_hardware_tests requires enabled MUSA_UCX with MUSA and UCX dependencies')
    endif
    musa_hardware_test = executable('musa_ucx_e2e',
        '../../../integration/musa/musa_ucx_e2e.cpp',
        dependencies: [nixl_dep, musa_dep])
    test('musa_ucx_e2e', musa_hardware_test,
        args: ['device', 'device', '65536', '0', '0', '0'],
        env: ['NIXL_PLUGIN_DIR=' + (meson.project_build_root() / 'src/plugins/musa')],
        suite: 'musa-hardware', timeout: 120)
endif
