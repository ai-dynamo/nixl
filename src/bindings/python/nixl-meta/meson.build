# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

fs = import('fs')

build_dir = meson.current_build_dir()

packages_list = get_option('build_nixl_ep') ? '"nixl", "nixl_ep"' : '"nixl"'

# Give each build_nixl_ep configuration its own setuptools build-base so that
# reconfiguring in place never reuses a stale build/lib that still holds
# nixl_ep. This replaces the deprecated `clean` command that setuptools>=80.9.0
# no longer accepts via --global-option.
build_base = join_paths(build_dir,
  get_option('build_nixl_ep') ? 'setuptools-build-ep' : 'setuptools-build-noep')

pyproject_toml = configure_file(
  input: 'pyproject.toml.in',
  output: 'pyproject.toml',
  configuration: {
    'VERSION': meson.project_version(),
    'PACKAGES': packages_list,
    'WHEEL_DEPS': get_option('release_wheel') \
      ? '"nixl-cu12==' + meson.project_version() + '", "nixl-cu13==' + meson.project_version() + '"' \
      : '"' + cuda_wheel_dir.replace('_', '-') + '==' + meson.project_version() + '"',
  }
)

# setup.cfg carries the per-configuration build_base (stable setuptools option,
# unlike the deprecated --global-option clean or the experimental
# [tool.distutils] table in pyproject.toml).
setup_cfg = configure_file(
  input: 'setup.cfg.in',
  output: 'setup.cfg',
  configuration: {
    'BUILD_BASE': build_base,
  }
)

readme_md = fs.copyfile('README.md')
nixl_meta_utils_copy = fs.copyfile('nixl_meta_utils.py')

source_root = meson.project_source_root()
root_license_path = join_paths(source_root, 'LICENSE')
license_path = fs.copyfile(root_license_path)

subdir('nixl')

if get_option('build_nixl_ep')
  subdir('nixl_ep')
endif

uv = find_program('uv', required: false)

if uv.found()
  meta_inputs = [
    pyproject_toml,
    setup_cfg,
    readme_md,
    license_path,
    nixl_meta_utils_copy,
    nixl_init_copy,
    nixl_api_copy,
    nixl_logging_copy,
  ]
  if get_option('build_nixl_ep')
    meta_inputs += [nixl_ep_init_copy]
  endif

  wheel_name = 'nixl-@0@-py3-none-any.whl'.format(meson.project_version())
  # Inputs point at build-dir copies (return values of fs.copyfile in subdirs)
  # so ninja runs copies before `uv build` and picks up source edits.
  # setup.cfg's per-configuration build_base (see build_base above) keeps
  # build/lib from retaining nixl_ep after build_nixl_ep is disabled in an
  # existing Meson build directory.
  meta_wheel = custom_target(
    'build_nixl_meta',
    input: meta_inputs,
    output: [wheel_name],
    command: [
      uv, 'build', '--wheel',
      '--out-dir', build_dir, build_dir,
    ],
    install: false,
    build_by_default: true
  )
else
  warning('uv not found, skipping meta package build')
endif
