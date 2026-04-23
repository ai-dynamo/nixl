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

import importlib
import sys
from typing import TYPE_CHECKING


def _get_torch_cuda_major() -> int | None:
    """Return the CUDA major version that torch was built for, or None."""
    from torch.version import cuda as _torch_cuda_ver

    return int(_torch_cuda_ver.split(".")[0]) if _torch_cuda_ver else None


def _load_cuda_backend() -> str:
    cuda_major = _get_torch_cuda_major()
    if cuda_major is not None:
        pip_name = f"nixl-cu{cuda_major}"
        mod_name = f"nixl_cu{cuda_major}"
        try:
            return importlib.import_module(mod_name).__name__
        except ModuleNotFoundError as e:
            raise ImportError(
                f"torch reports CUDA {cuda_major} but {pip_name} is not installed"
            ) from e
    # CPU-only torch — try whatever backend is installed
    for mod_name in ("nixl_cu13", "nixl_cu12"):
        try:
            return importlib.import_module(mod_name).__name__
        except ModuleNotFoundError:
            continue
    raise ImportError("No NIXL CUDA backend found")


_pkg = sys.modules[_load_cuda_backend()]

submodules = ["_api", "_bindings", "_utils", "logging"]
for sub_name in submodules:
    # Import submodule from actual wheel
    module = importlib.import_module(f"{_pkg.__name__}.{sub_name}")
    # Make it accessible as nixl._api, nixl._utils, nixl.logging
    sys.modules[f"nixl.{sub_name}"] = module
    # Also add the submodule itself to the nixl namespace
    setattr(sys.modules[__name__], sub_name, module)

    # Expose all symbols from the submodule under the nixl namespace
    for attr in dir(module):
        if not attr.startswith("_"):
            setattr(sys.modules[__name__], attr, getattr(module, attr))

if TYPE_CHECKING:
    from nixl import logging  # noqa: F401
    from nixl._api import (  # type: ignore[attr-defined]  # noqa: F401
        nixl_agent,
        nixl_agent_config,
    )
