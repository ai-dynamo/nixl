# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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

if not cpp.links('''
        #include <cstddef>
        #include <cstdint>
        #include <rados/librados.h>
        #include <rados/librados.hpp>
        int main() {
            rados_t cluster = nullptr;
            rados_ioctx_t ioctx = nullptr;
            rados_completion_t c_comp = nullptr;
            char buf[8] = {};
            int rc = rados_create2(&cluster, nullptr, "client.fixture", uint64_t{0});
            rc += rados_conf_read_file(cluster, "fixture.conf");
            rc += rados_conf_set(cluster, "log_file", "");
            rc += rados_connect(cluster);
            rc += rados_ioctx_create(cluster, "pool", &ioctx);
            rados_ioctx_set_namespace(ioctx, "fixture-ns");
            rc += rados_aio_create_completion(nullptr, nullptr, nullptr, &c_comp);
            rc += rados_aio_read(ioctx, "oid", c_comp, buf, sizeof(buf), 0);
            rc += rados_aio_is_complete_and_cb(c_comp);
            rc += rados_aio_get_return_value(c_comp);
            rc += rados_aio_wait_for_complete_and_cb(c_comp);
            rados_aio_release(c_comp);
            librados::IoCtx view;
            librados::IoCtx::from_rados_ioctx_t(ioctx, view);
            librados::AioCompletion *cpp_comp = librados::Rados::aio_create_completion();
            librados::bufferlist bl = librados::bufferlist::static_from_mem(buf, sizeof(buf));
            rc += view.aio_write("oid", cpp_comp, bl, sizeof(buf), 0);
            librados::ObjectReadOperation read_op;
            uint64_t stat_size = 0;
            int stat_prval = 0;
            read_op.stat(&stat_size, nullptr, &stat_prval);
            rc += view.aio_operate("oid", cpp_comp, &read_op, &bl);
            cpp_comp->is_complete_and_cb();
            cpp_comp->get_return_value();
            cpp_comp->wait_for_complete_and_cb();
            bl.clear();
            cpp_comp->release();
            view.close();
            rados_ioctx_destroy(ioctx);
            rados_shutdown(cluster);
            return rc;
        }
    ''', dependencies: [rados_dep],
         name: 'librados C lifecycle/read and C++ IoCtx/bufferlist/AIO/async-query surface')
    if is_explicit_enable or 'RADOS' in static_plugins
        error('The RADOS plugin requires the librados C lifecycle and AIO-read surface plus the C++ ' +
              'IoCtx/bufferlist/AIO and async object-query surface: rados_create2, rados_conf_read_file, ' +
              'rados_conf_set, rados_connect, rados_ioctx_create, rados_ioctx_set_namespace, ' +
              'rados_ioctx_destroy, rados_shutdown, ' +
              'rados_aio_create_completion, rados_aio_read, rados_aio_is_complete_and_cb, ' +
              'rados_aio_get_return_value, rados_aio_wait_for_complete_and_cb, rados_aio_release, ' +
              'librados::IoCtx::from_rados_ioctx_t, librados::IoCtx::close, librados::IoCtx::aio_write, ' +
              'librados::IoCtx::aio_operate, librados::ObjectReadOperation::stat, ' +
              'librados::Rados::aio_create_completion, librados::bufferlist::static_from_mem, and the ' +
              'librados::AioCompletion is_complete_and_cb/get_return_value/wait_for_complete_and_cb/release ' +
              'operations.')
    endif
    warning('RADOS plugin disabled because the required librados API probe failed')
    subdir_done()
endif

rados_sources = ['rados_backend.cpp', 'rados_backend.h',
                        'rados_capi.h', 'rados_config.h',
                        'rados_plugin.cpp']
if 'RADOS' in static_plugins
  rados_backend_lib = static_library('RADOS', rados_sources,
    dependencies: [nixl_infra, nixl_common_dep, thread_dep, rados_dep],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    install: false, cpp_args: [], name_prefix: 'libplugin_')
else
  rados_backend_lib = shared_library('RADOS', rados_sources,
    dependencies: [nixl_infra, nixl_common_dep, thread_dep, rados_dep],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    install: true, cpp_args: ['-fPIC'], name_prefix: 'libplugin_',
    install_dir: plugin_install_dir, install_rpath: '$ORIGIN/..')
  if get_option('buildtype') == 'debug'
    run_command('sh', '-c', 'echo "RADOS=' + rados_backend_lib.full_path() +
                       '" >> ' + plugin_build_dir + '/pluginlist', check: true)
  endif
endif
rados_backend_interface = declare_dependency(link_with: rados_backend_lib)
