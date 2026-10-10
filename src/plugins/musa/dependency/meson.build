# SPDX-FileCopyrightText: Copyright (c) 2025-2026 MTHREADS CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Explicit opt-in: no environment or system-wide runtime fallback.
musa_dep = disabler()
musa_prefix = get_option('musa_path').strip()
foreach plugin : get_option('static_plugins').split(',')
    if enabled_plugins.get('MUSA_UCX') and plugin.strip() == 'MUSA_UCX'
        error('MUSA_UCX does not support static plugins; use the dynamic plugin')
    endif
endforeach

if musa_prefix != ''
    if not fs.is_file(musa_prefix / 'include/musa_runtime_api.h')
        error('MUSA runtime header not found under musa_path: ' + musa_prefix)
    endif
    musa_libdir = ''
    musa_static_libdir = ''
    foreach directory : ['lib', 'lib64']
        foreach filename : ['libmusart.so', 'libmusart.dylib']
            if fs.is_file(musa_prefix / directory / filename)
                musa_libdir = musa_prefix / directory
            endif
        endforeach
        if fs.is_file(musa_prefix / directory / 'libmusart.a')
            musa_static_libdir = musa_prefix / directory
        endif
    endforeach
    if musa_libdir == ''
        if musa_static_libdir != ''
            error('MUSA_UCX does not support a static MUSA runtime; provide libmusart.so or libmusart.dylib')
        endif
        error('MUSA runtime library musart not found under musa_path (the driver library is not sufficient)')
    endif
    musa_runtime_lib = cpp.find_library('musart', dirs: [musa_libdir], required: true)
    musa_dep = declare_dependency(
        dependencies: musa_runtime_lib,
        include_directories: include_directories(musa_prefix / 'include', is_system: true))
    if not cpp.links('''
        #include <musa_runtime_api.h>
        int main() {
            int count = 0;
            musaPointerAttributes attributes{};
            auto status = musaGetDeviceCount(&count);
            status = musaPointerGetAttributes(&attributes, nullptr);
            auto kind = attributes.type == musaMemoryTypeDevice;
            auto device = attributes.device;
            return (musaGetErrorString(status) != nullptr) + kind + device + count +
                   (status == musaErrorInvalidValue);
        }
        ''', dependencies: musa_dep, name: 'MUSA runtime adapter contract')
        error('MUSA runtime headers/library do not provide the required adapter fields and symbols')
    endif
endif
