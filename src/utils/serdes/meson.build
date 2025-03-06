serdes_lib = library('serdes',
           'serdes.cpp', 'serdes.h',
           include_directories: inc_dir,
           install: true)

serdes_test_bin = executable('serdes_test',
           'serdes_test.cpp',
           include_directories: inc_dir,
           link_with: serdes_lib,
           install: true)

serdes_interface = declare_dependency(link_with: serdes_lib)
