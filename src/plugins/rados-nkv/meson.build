# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
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

# ---------------------------------------------------------------------------
# RADOS_NKV plugin: maps NIXL transfers onto the NVMe KV command set via the
# in-process SPDK KV host shim (kv_host_shim.{h,c}). WRITE -> KV Store,
# READ -> KV Retrieve.
# ---------------------------------------------------------------------------

cc = meson.get_compiler('c')

# Locations of the SPDK tree we link against. Overridable via meson options.
spdk_root = get_option('spdk_root')
spdk_shim_dir = get_option('spdk_kv_shim_dir')

spdk_build_lib = spdk_root / 'build' / 'lib'
spdk_build_inc = spdk_root / 'build' / 'include'
spdk_inc = spdk_root / 'include'
dpdk_lib = spdk_root / 'dpdk' / 'build' / 'lib'
vfu_inc = spdk_root / 'build' / 'libvfio-user' / 'usr' / 'local' / 'include'
vfu_lib = spdk_root / 'build' / 'libvfio-user' / 'usr' / 'local' / 'lib'

shim_c = spdk_shim_dir / 'kv_host_shim.c'

if not fs.exists(shim_c)
    warning('RADOS_NKV plugin disabled: SPDK KV shim not found at ' + shim_c)
    subdir_done()
endif
if not fs.exists(spdk_build_lib / 'libspdk_nvme.a')
    warning('RADOS_NKV plugin disabled: SPDK nvme lib not found under ' + spdk_build_lib)
    subdir_done()
endif

# Build the C shim into its own static lib so the C compile flags (and the
# C-only SPDK headers) stay isolated from the C++ plugin TU.
kv_shim_lib = static_library(
    'nixl_kv_host_shim',
    [shim_c],
    c_args: ['-D_GNU_SOURCE', '-w'],
    include_directories: include_directories(
        spdk_inc, spdk_build_inc, vfu_inc, spdk_shim_dir, is_system: true),
    install: false,
)

# SPDK whole-archive set (mirrors the kv_shim_test link line). The ordering and
# the whole-archive wrapping around the SPDK + DPDK static libs is required so
# the env/transport registration constructors are pulled in.
spdk_libs = [
    'spdk_sock_posix', 'spdk_nvme', 'spdk_keyring', 'spdk_sock', 'spdk_trace',
    'spdk_rpc', 'spdk_jsonrpc', 'spdk_json', 'spdk_dma', 'spdk_vfio_user',
    'spdk_vmd', 'spdk_util', 'spdk_log',
]

dpdk_libs = [
    'rte_argparse', 'rte_bus_pci', 'rte_cryptodev', 'rte_dmadev', 'rte_eal',
    'rte_ethdev', 'rte_hash', 'rte_kvargs', 'rte_log', 'rte_mbuf',
    'rte_mempool', 'rte_mempool_ring', 'rte_meter', 'rte_net', 'rte_pci',
    'rte_power', 'rte_power_acpi', 'rte_power_amd_pstate', 'rte_power_cppc',
    'rte_power_intel_pstate', 'rte_power_intel_uncore', 'rte_power_kvm_vm',
    'rte_rcu', 'rte_ring', 'rte_telemetry', 'rte_timer', 'rte_vhost',
]

spdk_link_args = ['-L' + spdk_build_lib, '-L' + vfu_lib, '-L' + dpdk_lib]
spdk_link_args += ['-Wl,--whole-archive', '-Wl,--no-as-needed']
foreach l : spdk_libs
    spdk_link_args += ['-l' + l]
endforeach
spdk_link_args += ['-Wl,--no-whole-archive']
spdk_link_args += [spdk_build_lib / 'libspdk_env_dpdk.a']
spdk_link_args += ['-Wl,--whole-archive']
foreach l : dpdk_libs
    spdk_link_args += [dpdk_lib / ('lib' + l + '.a')]
endforeach
spdk_link_args += ['-Wl,--no-whole-archive']

# isa-l static archives provide the crc/xor symbols referenced by spdk_util.
foreach isal : [spdk_root / 'isa-l' / '.libs' / 'libisal.a',
                spdk_root / 'isa-l-crypto' / '.libs' / 'libisal_crypto.a']
    if fs.exists(isal)
        spdk_link_args += [isal]
    endif
endforeach

# Tail system deps required by the SPDK/DPDK static archives.
foreach syslib : ['numa', 'dl', 'rt', 'uuid', 'ssl', 'crypto', 'm', 'lz4',
                  'keyutils', 'aio', 'rados', 'rbd', 'vfio-user', 'json-c']
    dep = cc.find_library(syslib, required: false)
    if dep.found()
        spdk_link_args += ['-l' + syslib]
    endif
endforeach

rados_nkv_sources = [
    'rados_nkv_backend.cpp',
    'rados_nkv_backend.h',
    'rados_nkv_key.cpp',
    'rados_nkv_key.h',
    'rados_nkv_plugin.cpp',
]

plugin_deps = [nixl_infra, nixl_common_dep]
# The C++ plugin TU (and the test) include kv_host_shim.h, which pulls in SPDK
# headers (spdk/stdinc.h, ...), so the SPDK include dirs must be on their path
# too -- not just the shim dir.
shim_inc = include_directories(
    spdk_shim_dir, spdk_inc, spdk_build_inc, vfu_inc, is_system: true)

if 'RADOS_NKV' in static_plugins
    rados_nkv_backend_lib = static_library(
        'RADOS_NKV',
        rados_nkv_sources,
        dependencies: plugin_deps,
        link_with: kv_shim_lib,
        link_args: spdk_link_args,
        include_directories: [nixl_inc_dirs, utils_inc_dirs, shim_inc],
        install: false,
        name_prefix: 'libplugin_')
else
    rados_nkv_backend_lib = shared_library(
        'RADOS_NKV',
        rados_nkv_sources,
        dependencies: plugin_deps,
        link_with: kv_shim_lib,
        link_args: spdk_link_args,
        cpp_args: ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs, shim_inc],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "RADOS_NKV=' + rados_nkv_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

if 'RADOS_NKV' in static_plugins
    rados_nkv_backend_interface = declare_dependency(link_whole: rados_nkv_backend_lib)
else
    rados_nkv_backend_interface = declare_dependency(link_with: rados_nkv_backend_lib)
endif

# Direct-engine round-trip test (links the engine TU + shim against a live
# SPDK KV target). Enabled by -Drados_nkv_build_test=true.
if get_option('rados_nkv_build_test')
    executable(
        'rados_nkv_roundtrip_test',
        ['rados_nkv_roundtrip_test.cpp', 'rados_nkv_backend.cpp', 'rados_nkv_backend.h',
         'rados_nkv_key.cpp', 'rados_nkv_key.h'],
        dependencies: plugin_deps,
        link_with: kv_shim_lib,
        link_args: spdk_link_args,
        include_directories: [nixl_inc_dirs, utils_inc_dirs, shim_inc],
        install: false,
    )
endif
