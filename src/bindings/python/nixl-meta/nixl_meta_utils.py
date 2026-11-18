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
import os
import pathlib
import pkgutil
import re
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
        version = major(getattr(getattr(torch, "version", None), "cuda", None))
        if version is not None:
            return version

    # cuda-python already imported: use it.
    cuda_bindings = sys.modules.get("cuda.bindings")
    if cuda_bindings is not None:
        version = major(getattr(cuda_bindings, "__version__", None))
        if version is not None:
            return version

    # cupy already imported: use it.
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            return cupy.cuda.runtime.runtimeGetVersion() // 1000
        except Exception:
            pass

    # torch installed but not imported: read build version off disk.
    version = major(_torch_cuda_version_from_disk())
    if version is not None:
        return version

    # Last resort: import torch and use its official API. Slow.
    try:
        from torch.version import cuda as torch_cuda

        return major(torch_cuda)
    except ImportError:
        return None


def detect_rocm() -> bool:
    """Whether the process uses a ROCm stack, to select the nixl_rocm backend.

    Called after detect_cuda_major() returns None, by which point torch is
    already imported if it is installed.
    """
    # torch already imported: use its official API.
    torch = sys.modules.get("torch")
    if torch is not None:
        if getattr(getattr(torch, "version", None), "hip", None) is not None:
            return True

    # hip-python already imported: use it.
    return "rocm.bindings" in sys.modules or "hip" in sys.modules


def parse_rocm_version(text: str | None) -> tuple[int, int] | None:
    """ROCm (major, minor) from "X.Y[.Z][-build]", or from a "+rocmX.Y[.Z]" tag."""
    if not text:
        return None
    if "+" in text:
        match = re.search(r"\+rocm(\d+)\.(\d+)", text)
    else:
        match = re.match(r"\s*(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def detect_rocm_version() -> tuple[int, int] | None:
    """ROCm (major, minor) used to select the nixl_rocm<X>_<Y> backend.

    Never imports torch: called after detect_cuda_major() returns None, by which
    point torch is already imported if it is installed.
    """
    # torch: its local version tag (e.g. "2.14.0+rocm10.1.0").
    torch = sys.modules.get("torch")
    if torch is not None:
        version = parse_rocm_version(_rocm_tag(getattr(torch, "__version__", None)))
    else:
        version = parse_rocm_version(_rocm_tag(_distribution_version("torch")))
    if version is not None:
        return version

    # ROCm install: its version file.
    rocm_path = pathlib.Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    for version_file in (
        rocm_path / ".info" / "version",
        rocm_path / "core" / ".info" / "version",
    ):
        try:
            version = parse_rocm_version(version_file.read_text())
        except (OSError, UnicodeDecodeError):
            continue
        if version is not None:
            return version

    # ROCm installed from pip wheels.
    return parse_rocm_version(_distribution_version("rocm"))


def find_rocm_backends() -> list[str]:
    """Installed nixl_rocm<X>_<Y> backend packages, newest ROCm first."""
    found = {}
    for module in pkgutil.iter_modules():
        match = re.fullmatch(r"nixl_rocm(\d+)_(\d+)", module.name)
        if match:
            found[module.name] = (int(match.group(1)), int(match.group(2)))
    return sorted(found, key=lambda name: found[name], reverse=True)


def find_cuda_backends() -> list[str]:
    """Installed nixl_cu<NN> backend packages."""
    return sorted(
        module.name
        for module in pkgutil.iter_modules()
        if re.fullmatch(r"nixl_cu\d+", module.name)
    )


def detect_torch_cuda() -> str | None:
    """CUDA version an installed PyTorch was built for, None if it is not a CUDA build.

    Never imports torch.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        return getattr(getattr(torch, "version", None), "cuda", None)
    return _torch_cuda_version_from_disk()


def _rocm_tag(version: str | None) -> str | None:
    """The version if it carries a "+rocm" local tag, else None."""
    return version if version and "+rocm" in version else None


def _distribution_version(name: str) -> str | None:
    """Version of an installed distribution, from its metadata."""
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
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
