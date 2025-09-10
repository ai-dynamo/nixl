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


# Enable libfabric utils layer
libfabric_plugin_deps = [nixl_infra, nixl_common_dep, libfabric_dep, serdes_interface, libfabric_utils_dep]

compile_flags = ['-DHAVE_LIBFABRIC']
if cuda_dep.found()
    libfabric_plugin_deps += [ cuda_dep ]
    compile_flags += [ '-DHAVE_CUDA' ]
endif

if 'LIBFABRIC' in static_plugins
    libfabric_backend_lib = static_library('LIBFABRIC',
               'libfabric_backend.cpp', 'libfabric_backend.h', 'libfabric_plugin.cpp',
               dependencies: libfabric_plugin_deps,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: false,
               cpp_args : compile_flags,
               name_prefix: 'libplugin_')
else
    libfabric_backend_lib = shared_library('LIBFABRIC',
               'libfabric_backend.cpp', 'libfabric_backend.h', 'libfabric_plugin.cpp',
               dependencies: libfabric_plugin_deps,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: true,
               cpp_args : compile_flags + ['-fPIC'],
               name_prefix: 'libplugin_',
               install_dir: plugin_install_dir)

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "LIBFABRIC=' + libfabric_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

libfabric_backend_interface = declare_dependency(link_with: libfabric_backend_lib)
