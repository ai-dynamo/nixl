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
        #include <string>
        #include <rados/librados.hpp>
        int main() {
            char buf[8] = {};
            librados::Rados cluster;
            int rc = cluster.init2("client.fixture", nullptr, uint64_t{0});
            rc += cluster.conf_read_file("fixture.conf");
            rc += cluster.conf_set("log_file", "");
            rc += cluster.connect();
            librados::IoCtx ioctx;
            rc += cluster.ioctx_create("pool", ioctx);
            ioctx.set_namespace(std::string("fixture-ns"));
            librados::AioCompletion *cpp_comp = librados::Rados::aio_create_completion();
            librados::bufferlist bl = librados::bufferlist::static_from_mem(buf, sizeof(buf));
            rc += ioctx.aio_write("oid", cpp_comp, bl, sizeof(buf), 0);
            rc += ioctx.aio_read("oid", cpp_comp, &bl, sizeof(buf), 0);
            bl.is_provided_buffer(buf);
            for (const auto &chunk : bl.buffers()) {
                rc += static_cast<int>(chunk.length());
                chunk.c_str();
            }
            librados::ObjectReadOperation read_op;
            uint64_t stat_size = 0;
            int stat_prval = 0;
            read_op.stat(&stat_size, nullptr, &stat_prval);
            rc += ioctx.aio_operate("oid", cpp_comp, &read_op, &bl);
            cpp_comp->is_complete_and_cb();
            cpp_comp->get_return_value();
            cpp_comp->wait_for_complete_and_cb();
            bl.clear();
            cpp_comp->release();
            return rc;
        }
     ''', dependencies: [rados_dep],
         name: 'librados C++ lifecycle and C++ IoCtx/bufferlist/AIO/async-query surface')
    if is_explicit_enable or 'RADOS' in static_plugins
        error('The RADOS plugin requires the librados C++ lifecycle surface plus the C++ ' +
              'IoCtx/bufferlist/AIO and async object-query surface: librados::Rados::init2, ' +
              'librados::Rados::conf_read_file, librados::Rados::conf_set, librados::Rados::connect, ' +
              'librados::Rados::ioctx_create, librados::IoCtx::set_namespace, ' +
              'librados::IoCtx::aio_write, ' +
              'librados::IoCtx::aio_read, librados::IoCtx::aio_operate, librados::ObjectReadOperation::stat, ' +
              'librados::Rados::aio_create_completion, librados::bufferlist::static_from_mem, ' +
              'librados::bufferlist::is_provided_buffer, ' +
              'librados::bufferlist::buffers, and the ' +
              'librados::AioCompletion is_complete_and_cb/get_return_value/wait_for_complete_and_cb/release ' +
              'operations.')
    endif
    warning('RADOS plugin disabled because the required librados API probe failed')
    subdir_done()
endif

rados_sources = ['rados_backend.cpp', 'rados_backend.h',
                        'rados_api.h', 'rados_config.h',
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
