# libblkio plugin

libblkio_dep = dependency('blkio', required: false)

if not libblkio_dep.found()
    libblkio_dep = dependency('libblkio', required: false)
endif

if libblkio_dep.found()
    blkio_sources = [
        'libblkio_backend.h',
        'libblkio_backend.cpp',
        'libblkio_plugin.cpp',
    ]
    compile_defs = []
    if 'LIBBLKIO' in static_plugins
        libblkio_backend_lib = static_library(
            'LIBBLKIO',
            blkio_sources,
            dependencies: [nixl_infra, absl_log_dep, libblkio_dep],
            include_directories: [nixl_inc_dirs, utils_inc_dirs],
            install: false,
            cpp_args: compile_defs + compile_flags,
            name_prefix: 'libplugin_',
        )
    else
        libblkio_backend_lib = shared_library(
            'LIBBLKIO',
            blkio_sources,
            dependencies: [nixl_infra, absl_log_dep, libblkio_dep],
            include_directories: [nixl_inc_dirs, utils_inc_dirs],
            install: true,
            cpp_args: compile_defs + ['-fPIC'],
            name_prefix: 'libplugin_',
            install_dir: plugin_install_dir,
            install_rpath: '$ORIGIN/..',
            version: meson.project_version(),
        )
        if get_option('buildtype') == 'debug'
            run_command(
                'sh', '-c',
                'echo "LIBBLKIO=' + libblkio_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                check: true,
            )
        endif
    endif
    message('Building libblkio plugin')
    libblkio_backend_interface = declare_dependency(link_with: libblkio_backend_lib)
else
    warning('libblkio not found - plugin disabled')
endif
