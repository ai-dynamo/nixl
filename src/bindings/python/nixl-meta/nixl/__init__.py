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

import importlib
import sys

# Try packages in order
candidates = ["nixl_cu13", "nixl_cu12"]
_pkg = None
for pkg in candidates:
    try:
        _pkg = importlib.import_module(pkg)
        break
    except ModuleNotFoundError:
        continue

if _pkg is None:
    raise ImportError(
        "Could not find CUDA-specific NIXL package. Please install NIXL with `pip install nixl[cu12]` or `pip install nixl[cu13]`"
    )

submodules = ["_api", "_bindings", "_utils", "logging"]
for sub_name in submodules:
    # Import submodule from actual wheel
    module = importlib.import_module(f"{pkg}.{sub_name}")
    # Make it accessible as nixl._api, nixl._utils, nixl.logging
    sys.modules[f"nixl.{sub_name}"] = module
    # Also add the submodule itself to the nixl namespace
    setattr(sys.modules[__name__], sub_name, module)

    # Expose all symbols from the submodule under the nixl namespace
    for attr in dir(module):
        if not attr.startswith("_"):
            setattr(sys.modules[__name__], attr, getattr(module, attr))
