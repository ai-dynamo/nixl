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

import os

import nixl

__all__ = ["nixl"]

if "NIXL_PLUGIN_DIR" not in os.environ:
    # name for local installation
    plugin_dir = nixl.__file__[:-16] + ".nixl.mesonpy.libs/plugins/"

    # name for pypi installation
    if not os.path.isdir(plugin_dir):
        plugin_dir = nixl.__file__[:-16] + ".nixl_pybind.mesonpy.libs/plugins/"

    if os.path.isdir(plugin_dir):
        os.environ["NIXL_PLUGIN_DIR"] = plugin_dir
