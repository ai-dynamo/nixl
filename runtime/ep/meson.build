# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install the API-only shell without native code,
# toolkit discovery, JIT headers, or an extension module.
ep_install_dir = join_paths(cuda_wheel_dir, 'runtime', 'ep')

py.install_sources(
    'python/__init__.py',
    subdir: ep_install_dir,
    pure: false,
)
py.install_sources(
    'python/buffers/__init__.py',
    'python/buffers/elastic.py',
    subdir: join_paths(ep_install_dir, 'buffers'),
    pure: false,
)
py.install_sources(
    'python/utils/__init__.py',
    'python/utils/event.py',
    subdir: join_paths(ep_install_dir, 'utils'),
    pure: false,
)
