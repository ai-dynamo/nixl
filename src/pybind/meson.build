py = import('python').find_installation('python3', pure: false)

pybind_dep = dependency('pybind11')

nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: inc_dir)

py.extension_module('nixl_bindings',
           'nixl_bindings.cpp', 
           dependencies: [nixl_dep, serdes_interface, pybind_dep],
           include_directories: [inc_dir],
           install: true)

py.extension_module('nixl_utils',
           'nixl_utils.cpp', 
           dependencies: [pybind_dep],
           install: true)

