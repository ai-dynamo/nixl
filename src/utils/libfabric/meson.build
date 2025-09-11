# LibFabric utilities meson build configuration

# Source files
libfabric_utils_sources = files(
    'libfabric_rail.cpp',
    'libfabric_rail_manager.cpp',
    'libfabric_common.cpp',
    'libfabric_topology.cpp',
    # More implementation files will be added as we create them
)

# Header files
libfabric_utils_headers = files(
    'libfabric_rail.h',
    'libfabric_rail_manager.h',
    'libfabric_common.h',
    'libfabric_topology.h',
)

# Find hwloc dependency
hwloc_dep = dependency('hwloc', required: true)

# Set up dependencies and compile args
libfabric_utils_deps = [
    libfabric_dep,
    hwloc_dep,
    abseil_proj.get_variable('absl_log_dep'),
]

libfabric_utils_cpp_args = []

# Add CUDA support if available
if cuda_dep.found()
    libfabric_utils_deps += [cuda_dep]
    libfabric_utils_cpp_args += ['-DHAVE_CUDA']
endif

# Create static library
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

# Declare dependency for other parts of the build
libfabric_utils_dep = declare_dependency(
    link_with: libfabric_utils_lib,
    include_directories: include_directories('.'),
)
