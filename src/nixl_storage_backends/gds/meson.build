gds_path = get_option('gds_path')
if gds_path != ''
  gds_lib_path = gds_path + '/lib'
  gds_inc_path = gds_path + '/include'
  cufile_dep = declare_dependency(
    link_args : ['-L' + gds_lib_path, '-lcufile'],
    include_directories : include_directories(gds_inc_path))
endif

gds_backend_lib = library('gds_backend',
                  'gds_utils.cpp', 'gds_utils.h',
                  'gds_backend.cpp', 'gds_backend.h',
                  dependencies: [nixl_dep, cuda_dep, cufile_dep],
                  include_directories: inc_dir,
                  install: true)
gds_backend_interface = declare_dependency(link_with: gds_backend_lib)
