#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of NVIDIA CORPORATION &
# AFFILIATES (the "Company") and all right, title, and interest in and to the
# software product, including all associated intellectual property rights, are
# and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

project('nixl', 'CPP', version: '0.1.0',
    default_options: ['buildtype=debug',
                'werror=true',
                'cpp_std=c++11',
                'prefix=/opt/nvidia/nvda_nixl'],
    meson_version: '>= 0.64.0'
)

# set up some global vars for compiler, platform, configuration, etc.
cpp = meson.get_compiler('cpp')

cuda_dep = dependency('cuda', required : false, modules : [ 'cudart', 'cuda' ])
prefix_path = get_option('prefix')
prefix_inc = prefix_path + '/include'

ucx_path = get_option('ucx_path')
if ucx_path != ''
  ucx_lib_path = ucx_path + '/lib'
  ucx_inc_path = ucx_path + '/include'
  ucx_dep = declare_dependency(
    link_args : ['-L' + ucx_lib_path, '-lucp', '-lucs', '-luct'],
    include_directories : include_directories(ucx_inc_path))
else
  ucx_dep = dependency('ucx')
endif

inc_dir = include_directories('include', 'include/internal', 'include/backend', 
                              'src', 'src/utils/serdes', 'src/utils/data_structures',
                              'src/utils/sys', 'src/utils/ucx')
subdir('include')
subdir('include/internal')
subdir('src')
subdir('test')

install_headers('include/nixl.h', install_dir: prefix_inc)
install_headers('include/nixl_types.h', install_dir: prefix_inc)
install_headers('include/nixl_params.h', install_dir: prefix_inc)
install_headers('include/nixl_descriptors.h', install_dir: prefix_inc)
install_headers('src/utils/serdes/serdes.h', install_dir: prefix_inc + '/utils/serdes')
install_headers('src/utils/sys/nixl_time.h', install_dir: prefix_inc + '/utils/sys')
install_headers('include/backend/backend_engine.h', install_dir: prefix_inc + '/backend')
install_headers('include/backend/backend_aux.h', install_dir: prefix_inc + '/backend')
install_headers('include/backend/backend_tester.h', install_dir: prefix_inc + '/backend')
install_headers('include/internal/transfer_request.h', install_dir: prefix_inc + '/internal')
install_headers('include/internal/agent_data.h', install_dir: prefix_inc + '/internal')
install_headers('include/internal/mem_section.h', install_dir: prefix_inc + '/internal')
