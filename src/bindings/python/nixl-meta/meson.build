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

source_dir = meson.current_source_dir()
build_dir = meson.current_build_dir()

configure_file(
  input: 'pyproject.toml.in',
  output: 'pyproject.toml',
  configuration: { 'VERSION': meson.project_version() }
)

uv = find_program('uv', required: false)
if uv.found()
  wheel_name = 'nixl-@0@-py3-none-any.whl'.format(meson.project_version())
  meta_wheel = custom_target(
    'build_nixl_meta',
    output: [wheel_name],
    command: [uv, 'build', '--wheel', '--out-dir', build_dir, build_dir],
    install: false,
    build_by_default: true
  )
else
  warning('uv not found, skipping meta package build')
endif
