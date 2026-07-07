# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared helpers for the NIXL meta packages (``nixl`` and ``nixl_ep``)."""

import importlib.util
import pathlib
import sys


def detect_cuda_major() -> int | None:
    """CUDA major version used to select the nixl_cuXX backend wheel.

    Read from whichever CUDA stack the process has already imported, so that
    importing a NIXL meta package does not itself pull in torch (a ~1s cost).
    If nothing CUDA is loaded yet, read torch's build version off disk (still
    torch-free); only if that fails do we import torch and use its official API
    as a last resort.
    """

    def major(version: str | None) -> int | None:
        return int(version.split(".")[0]) if version else None

    # torch already imported: use its official API.
    torch = sys.modules.get("torch")
    if torch is not None:
        found = major(getattr(getattr(torch, "version", None), "cuda", None))
        if found is not None:
            return found

    # cuda-python already imported: use it.
    cuda_bindings = sys.modules.get("cuda.bindings")
    if cuda_bindings is not None:
        found = major(getattr(cuda_bindings, "__version__", None))
        if found is not None:
            return found

    # cupy already imported: use it.
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            return cupy.cuda.runtime.runtimeGetVersion() // 1000
        except Exception:
            pass

    # torch installed but not imported: read build version off disk.
    found = major(_torch_cuda_version_from_disk())
    if found is not None:
        return found

    # Last resort: import torch and use its official API. Slow.
    try:
        from torch.version import cuda as torch_cuda

        return major(torch_cuda)
    except ImportError:
        return None


def _torch_cuda_version_from_disk() -> str | None:
    """Return torch's build CUDA version (e.g. "12.6") without full torch import.

    ``find_spec`` locates the package without running its ``__init__``, and the
    standalone module name keeps importlib from importing the torch package; we
    then exec only the tiny ``version.py`` to read its ``cuda`` attribute.
    """
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None

    version_py = pathlib.Path(spec.origin).parent / "version.py"
    try:
        version_spec = importlib.util.spec_from_file_location(
            "_nixl_torch_version", version_py
        )
        if version_spec is None or version_spec.loader is None:
            return None
        module = importlib.util.module_from_spec(version_spec)
        version_spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "cuda", None)
