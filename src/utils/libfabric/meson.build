libfabric_utils_sources = files(
  'libfabric_rail.cpp',
  'libfabric_rail_manager.cpp',
  'libfabric_common.cpp',
  'libfabric_topology.cpp',
  # More implementation files will be added as we create them
)

libfabric_utils_headers = files(
  'libfabric_rail.h',
  'libfabric_rail_manager.h',
  'libfabric_common.h',
  'libfabric_topology.h',
)

# Find hwloc dependency
hwloc_dep = dependency('hwloc', required: true)

# Find NUMA library (optional)
numa_dep = dependency('numa', required: false)

# Set up dependencies and compile args
libfabric_utils_deps = [
  libfabric_dep,
  hwloc_dep,
  abseil_proj.get_variable('absl_log_dep'),
]

libfabric_utils_cpp_args = []

# Add NUMA support if available
if numa_dep.found()
  libfabric_utils_deps += [numa_dep]
  libfabric_utils_cpp_args += ['-DHAVE_NUMA']
endif

# Add CUDA support if available
if cuda_dep.found()
  libfabric_utils_deps += [cuda_dep]
  libfabric_utils_cpp_args += ['-DHAVE_CUDA']
endif

libfabric_utils_lib = static_library(
  'nixl_libfabric_utils',
  libfabric_utils_sources,
  dependencies: libfabric_utils_deps,
  cpp_args: libfabric_utils_cpp_args,
  include_directories: [
    nixl_inc_dirs,
    utils_inc_dirs,
  ],
  install: false,
)

libfabric_utils_dep = declare_dependency(
  link_with: libfabric_utils_lib,
  include_directories: include_directories('.'),
)
