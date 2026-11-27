# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
import warnings
from typing import TYPE_CHECKING

from nixl_meta_utils import (
    detect_cuda_major,
    detect_rocm,
    detect_torch_cuda,
    find_cuda_backends,
    find_rocm,
    find_rocm_backends,
    load_rocm,
)


def _cuda_torch_with_rocm_backends_error() -> ImportError | None:
    """The error for a CUDA PyTorch next to ROCm-only NIXL backends, if that applies."""
    rocm_backends = find_rocm_backends()
    if not rocm_backends or find_cuda_backends():
        return None
    torch_cuda = detect_torch_cuda()
    if torch_cuda is None:
        return None
    return ImportError(
        f"found a CUDA build of PyTorch (CUDA {torch_cuda}) but only ROCm NIXL "
        f"backends are installed ({', '.join(rocm_backends)}). NIXL for ROCm "
        "needs AMD's ROCm build of PyTorch, which is not on PyPI: install it "
        "in place of the CUDA build"
    )


def _load_cuda_backend() -> str:
    cuda_major = detect_cuda_major()
    if cuda_major is not None:
        pip_name = f"nixl-cu{cuda_major}"
        mod_name = f"nixl_cu{cuda_major}"
        try:
            return importlib.import_module(mod_name).__name__
        except ModuleNotFoundError as e:
            if e.name != mod_name:
                raise
            wrong_torch = _cuda_torch_with_rocm_backends_error()
            if wrong_torch is not None:
                raise wrong_torch from e
            raise ImportError(
                f"detected CUDA {cuda_major} but {pip_name} is not installed"
            ) from e
    # No CUDA stack detected: pick the backend matching the ROCm version, unless
    # a CUDA backend is installed, which keeps the selection below unchanged.
    cuda_backends = find_cuda_backends()
    rocm = None if cuda_backends else find_rocm()
    rocm_backends = [] if cuda_backends else find_rocm_backends()
    if rocm is not None:
        mod_name = "nixl_rocm{}_{}".format(*rocm.version)
        if mod_name in rocm_backends:
            # The backend's plugins find the HIP runtime by library name only:
            # load it from the install the backend was selected for.
            load_rocm(rocm)
        try:
            return importlib.import_module(mod_name).__name__
        except ModuleNotFoundError as e:
            if e.name != mod_name:
                raise
            if rocm_backends:
                raise ImportError(
                    "detected ROCm {0}.{1} from {2} but no NIXL backend for it "
                    "is installed (backends installed: {3})".format(
                        *rocm.version, rocm.source, ", ".join(rocm_backends)
                    )
                ) from e
            # No nixl_rocm<X>_<Y> backend is installed, so none can mismatch:
            # fall through to the legacy nixl_rocm package, whose name carries
            # no ROCm version, or to the error below.
    elif rocm_backends:
        warnings.warn(
            "could not tell which ROCm install to select the NIXL backend for: "
            "found none, or several versions side by side (set ROCM_PATH to "
            "the one to use). Using the backend for the newest ROCm version, "
            f"{rocm_backends[0]}",
            RuntimeWarning,
        )
    # ROCm version unknown, or only the legacy backend: use whatever backend is
    # installed.
    backends = ("nixl_cu13", *rocm_backends, "nixl_rocm")
    if rocm is not None or detect_rocm():
        backends = (*rocm_backends, "nixl_rocm", "nixl_cu13")
    for mod_name in backends:
        try:
            return importlib.import_module(mod_name).__name__
        except ModuleNotFoundError as e:
            if e.name != mod_name:
                # Re-raise if the error is not about the module we're trying to import
                raise
            continue
    raise ImportError("No NIXL backend found (tried nixl_cu13, nixl_rocm)")


_pkg = sys.modules[_load_cuda_backend()]

submodules = ["_bindings", "_utils", "logging", "_api"]
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
        DEFAULT_COMM_PORT,
        nixl_agent,
        nixl_agent_config,
        nixl_backend_handle,
        nixl_completion_monitor,
        nixl_prepped_dlist_handle,
        nixl_thread_sync_t,
        nixl_xfer_handle,
    )
