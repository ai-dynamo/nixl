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
fs.copyfile('__init__.py')
fs.copyfile('_api.py')
fs.copyfile('logging.py')

# nixl_ep meta-dispatcher: copy into build dir so the meta wheel picks it up
nixl_ep_meta_dir = join_paths(meson.current_build_dir(), '..', 'nixl_ep')
nixl_ep_meta_src = join_paths(meson.current_source_dir(), '..', 'nixl_ep', '__init__.py')
custom_target('nixl_ep_meta_copy',
    output: 'nixl_ep_meta.stamp',
    command: ['bash', '-c',
        'mkdir -p @0@ && cp @1@ @0@/__init__.py && touch @2@'.format(
            nixl_ep_meta_dir,
            nixl_ep_meta_src,
            join_paths(meson.current_build_dir(), 'nixl_ep_meta.stamp'),
        )
    ],
    build_by_default: true,
)
