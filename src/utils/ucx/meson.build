ucx_utils_lib = library('ucx_utils',
           'ucx_utils.cpp', 'ucx_utils.h',
           dependencies: [ ucx_dep ],
           include_directories: inc_dir,
           install: true)

ucx_worker_bin = executable('ucx_worker_test',
           'ucx_worker_test.cpp',
           dependencies: [ ucx_dep ],
           include_directories: inc_dir,
           link_with: ucx_utils_lib,
           install: true)

ucx_am_bin = executable('ucx_am_test',
           'ucx_am_test.cpp',
           dependencies: [ ucx_dep ],
           include_directories: inc_dir,
           link_with: ucx_utils_lib,
           install: true)

if cuda_dep.found()
    ucx_worker_cuda_bin = executable('ucx_worker_test_cuda',
                   'ucx_worker_test.cpp',
                   dependencies: [ ucx_dep, cuda_dep ],
                   include_directories: inc_dir,
                   cpp_args : '-DUSE_VRAM',
                   link_with: ucx_utils_lib,
                   install: true)

    ucx_am_cuda_bin = executable('ucx_am_test_cuda',
                   'ucx_am_test.cpp',
                   dependencies: [ ucx_dep, cuda_dep ],
                   include_directories: inc_dir,
                   cpp_args : '-DUSE_VRAM',
                   link_with: ucx_utils_lib,
                   install: true)

endif


