py = import('python').find_installation('python3', pure: false)

pybind_dep = dependency('pybind11')

nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: inc_dir)

py.extension_module('nixlAdapter',
		'nixl_adapter.cpp',
                dependencies: [nixl_dep, serdes_interface, pybind_dep],
		include_directories: [inc_dir],
		install: true)

