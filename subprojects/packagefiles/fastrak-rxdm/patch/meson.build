# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
project('fastrak-rxdm', ['cpp', 'cuda'],
  version: '1.0.0',
  license: 'BSD-3-Clause',
  default_options: [
    'buildtype=release',
    'warning_level=3',
    'cpp_std=c++20',
  ],
  meson_version: '>= 0.64.0')

cpp = meson.get_compiler('cpp')

# Abseil dependencies
absl_base_dep = dependency('absl_base', fallback: ['abseil-cpp', 'absl_base_dep'])
absl_container_dep = dependency('absl_container', fallback: ['abseil-cpp', 'absl_container_dep'])
absl_flags_dep = dependency('absl_flags', fallback: ['abseil-cpp', 'absl_flags_dep'])
absl_log_dep = dependency('absl_log', fallback: ['abseil-cpp', 'absl_log_dep'])
absl_numeric_dep = dependency('absl_numeric', fallback: ['abseil-cpp', 'absl_numeric_dep'])
absl_random_dep = dependency('absl_random', fallback: ['abseil-cpp', 'absl_random_dep'])
absl_status_dep = dependency('absl_status', fallback: ['abseil-cpp', 'absl_status_dep'])
absl_strings_dep = dependency('absl_strings', fallback: ['abseil-cpp', 'absl_strings_dep'])
absl_synchronization_dep = dependency('absl_synchronization', fallback: ['abseil-cpp', 'absl_synchronization_dep'])
absl_time_dep = dependency('absl_time', fallback: ['abseil-cpp', 'absl_time_dep'])
absl_types_dep = dependency('absl_types', fallback: ['abseil-cpp', 'absl_types_dep'])

# Build RxDM/DXS with bazel
rxdm_build_script = find_program('rxdm_build.sh', required: true)
rxdm_build_args = []
if get_option('skip_bazel_clean')
  rxdm_build_args += ['-s']
endif
run_command(rxdm_build_script, rxdm_build_args, check: true)

# Get the execution root from bazel
bazel_info = run_command('bazel', 'info', 'execution_root', check: true)
execution_root = bazel_info.stdout().strip()

# Get the bazel-bin path
# 'bazel info bazel-bin' may return the fastbuild dir instead of the opt dir.
# Resolving the local 'bazel-bin' symlink reliably points to the latest build configuration.
bazel_bin_info = run_command('bazel', 'info', 'bazel-bin', check: true)
bazel_bin_fallback = bazel_bin_info.stdout().strip()
bazel_bin_cmd = run_command('sh', '-c', 'if [ -d "bazel-bin" ]; then readlink -f bazel-bin; else echo "' + bazel_bin_fallback + '"; fi', check: true)
bazel_bin = bazel_bin_cmd.stdout().strip()

protoc = find_program('protoc', required : true)

# Note: Bazel often creates a symlink named 'bazel-bin' in the project root
bazel_gen_dir = include_directories('bazel-bin')
include_dir = include_directories('.')

fs = import('fs')
inc_dirs = [include_dir, bazel_gen_dir]

# Safely discover the googleapis directory to handle varying Bzlmod naming
googleapis_find = run_command('sh', '-c', 'ls -d ' + execution_root + '/external/googleapis* 2>/dev/null | head -n 1', check: false)
googleapis_path = googleapis_find.stdout().strip()

if googleapis_path != '' and fs.is_dir(googleapis_path)
  inc_dirs += include_directories(googleapis_path)
  googleapis_gen_path = bazel_bin / 'external' / fs.name(googleapis_path)
  if fs.is_dir(googleapis_gen_path)
    inc_dirs += include_directories(googleapis_gen_path)
  endif
endif

search_dirs = [
  bazel_bin / 'buffer_mgmt_daemon',
  bazel_bin / 'dxs',
  bazel_bin / 'dmabuf_bridge',
  bazel_bin / 'cuda_helpers',
  bazel_bin / 'webrtc'
]

bazel_libs = []
foreach d : search_dirs
  # Use -L to follow symlinks and -size +0c to ignore empty 0-byte umbrella archives
  find_cmd = run_command('sh', '-c', 'if [ -d "' + d + '" ]; then find -L "' + d + '" -type f -name "*.a" ! -name "*test*" -size +0c 2>/dev/null; fi', check: false)
  foreach lib : find_cmd.stdout().strip().split('\n')
    if lib != ''
      bazel_libs += lib
    endif
  endforeach
endforeach

if bazel_libs.length() == 0
  error('Could not find any static libraries from Bazel build in ' + bazel_bin)
endif

bazel_dep = declare_dependency(
  link_args: ['-Wl,--start-group'] + bazel_libs + ['-Wl,--end-group'],
)

# Dependencies for DXS and RxDM
tcpxo_rxdm_dxs_dep = declare_dependency(
  include_directories: inc_dirs,
  dependencies: [
    bazel_dep,
    absl_base_dep,
    absl_container_dep,
    absl_flags_dep,
    absl_log_dep,
    absl_numeric_dep,
    absl_random_dep,
    absl_status_dep,
    absl_strings_dep,
    absl_synchronization_dep,
    absl_time_dep,
    absl_types_dep
  ]
)

