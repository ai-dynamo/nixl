# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

pyproject_toml = configure_file(
  input: 'pyproject.toml.in',
  output: 'pyproject.toml',
  configuration: { 'VERSION': meson.project_version() }
)

readme_md = fs.copyfile('README.md')

source_root = meson.project_source_root()
root_license_path = join_paths(source_root, 'LICENSE')
license_path = fs.copyfile(root_license_path)

nixl_sources = files('nixl/__init__.py')

subdir('nixl')

uv = find_program('uv', required: false)
if uv.found()
  wheel_name = 'nixl-@0@-py3-none-any.whl'.format(meson.project_version())
  meta_wheel = custom_target(
    'build_nixl_meta',
    input: [pyproject_toml, readme_md, license_path] + nixl_sources,
    output: [wheel_name],
    command: [uv, 'build', '--wheel', '--out-dir', build_dir, build_dir],
    install: false,
    build_by_default: true
  )
else
  warning('uv not found, skipping meta package build')
endif
