subdir('utils')
nixl_infra = library('nixl_build',
           'nixl_descriptors.cpp',
           'nixl_memory_section.cpp',
           'nixl_metadata_stream.cpp',
            include_directories: ['../include', '../include/internal', 'nixl_nw_backends',
                                  'nixl_storage_backends',
                                  './utils/serdes', './utils/data_structures',
                                   './utils/sys/'],
            dependencies: [serdes_interface],
            install: true)

subdir('nixl_nw_backends')
subdir('nixl_storage_backends')

nixl_lib   = library('nixl',
              'nixl_descriptors.cpp',
              'nixl_memory_section.cpp',
              'nixl_agent.cpp',
              'nixl_metadata_stream.cpp',
              include_directories: ['../include', '../include/internal', 'nixl_nw_backends',
                                    'nixl_storage_backends/gds/',
                                    './utils/serdes', './utils/data_structures',
                                    './utils/sys/', './utils/ucx/'],
              dependencies: [ucx_backend_interface, gds_backend_interface,
                             serdes_interface, cuda_dep],
              install: true)

subdir('adapter')
subdir('pybind')
