# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared helpers for the NIXL meta packages (``nixl`` and ``nixl_ep``)."""

import importlib.util
import os
import pathlib
import pkgutil
import re
import sys
import types
from collections.abc import Iterator
from typing import NamedTuple

_DEFAULT_ROCM_ROOT = "/opt/rocm"
_PROC_MAPS = "/proc/self/maps"
# libamdhip64 loads the libraries it depends on from its own directory.
_HIP_RUNTIME_LIBS = ("libamdhip64.so", "libhipfile.so")


def detect_cuda_major() -> int | None:
    """CUDA major version used to select the nixl_cuXX backend wheel.

    Read from whichever CUDA stack the process has already imported, so that
    importing a NIXL meta package does not itself pull in torch (a ~1s cost).
    If nothing CUDA is loaded yet, read torch's build version off disk (still
    torch-free); only if that fails do we import torch and use its official API
    as a last resort. A ROCm build of torch has no CUDA version to give, so it
    is not imported for one.
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
    if _torch_rocm_version_from_disk() is not None:
        return None

    # Last resort: import torch and use its official API. Slow.
    try:
        from torch.version import cuda as torch_cuda

        return major(torch_cuda)
    except ImportError:
        return None


def detect_rocm() -> bool:
    """Whether the process uses a ROCm stack, to select the nixl_rocm backend.

    Never imports torch: a ROCm build that is installed but not imported
    counts as well.
    """
    # torch already imported: use its official API.
    torch = sys.modules.get("torch")
    if torch is not None:
        if getattr(getattr(torch, "version", None), "hip", None) is not None:
            return True
    elif _torch_rocm_version_from_disk() is not None:
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


class RocmInstall(NamedTuple):
    """A ROCm version, and the prefix its HIP runtime is to be loaded from."""

    root: pathlib.Path | None
    version: tuple[int, int]
    # Where the version was read from, for error messages.
    source: str = ""


def find_rocm() -> RocmInstall | None:
    """The ROCm install the nixl_rocm<X>_<Y> backend is selected for.

    Every ROCm version ships its HIP runtime under the same library names, so
    the dynamic loader accepts any backend against any install: the backend has
    to be the one built for the ROCm the process uses. In order:

    1. the install whose HIP runtime the process has already loaded;
    2. the "+rocm" version tag of an imported PyTorch, for a build that bundles
       its ROCm without a version file (root is None: there is nothing to load);
    3. $ROCM_PATH;
    4. ROCm installed from AMD's pip packages (rocm, rocm-sdk-core);
    5. /opt/rocm, if ROCM_PATH is not set.

    A ROCm build of PyTorch that is installed but not imported is imported only
    if none of these finds an install: such a build bundles its ROCm, and
    importing it is what loads that HIP runtime (1 and 2 then apply).
    """
    install = _find_rocm()
    if (
        install is None
        and "torch" not in sys.modules
        and _torch_rocm_version_from_disk() is not None
    ):
        try:
            importlib.import_module("torch")
        except ImportError:
            return None
        install = _find_rocm()
    return install


def _find_rocm() -> RocmInstall | None:
    """find_rocm() for the process as it is, without importing anything."""
    loaded = _loaded_hip_runtime()
    if loaded is not None:
        install = _rocm_install(loaded.parent.parent)
        if install is not None:
            return install._replace(source=f"the loaded HIP runtime {loaded}")

    torch = sys.modules.get("torch")
    torch_version = _rocm_tag(getattr(torch, "__version__", None))
    version = parse_rocm_version(torch_version)
    if version is not None:
        return RocmInstall(None, version, f"PyTorch {torch_version}")

    for root in _rocm_roots():
        candidates = [root, root / "core"]
        # The component packages populate only <root>/core-<X.Y>. Several
        # versions side by side are ambiguous.
        versioned = {path.resolve() for path in root.glob("core-*")}
        if len(versioned) == 1:
            candidates.extend(versioned)
        for candidate in candidates:
            install = _rocm_install(candidate)
            if install is not None:
                # A HIP runtime from elsewhere is already loaded: keep the
                # version as the best guess, but do not load a second runtime.
                return install if loaded is None else install._replace(root=None)
    return None


def load_rocm(install: RocmInstall) -> None:
    """Load the HIP runtime of a ROCm install, ahead of the NIXL backend.

    A library loaded by path satisfies later lookups of its name, so the NIXL
    plugins use this install whatever the loader path holds. It also makes a
    ROCm installed from pip wheels usable, which is on no loader path.
    """
    if install.root is None:
        return
    import ctypes

    for name in _HIP_RUNTIME_LIBS:
        if _loaded_library(name) is not None:
            # Later lookups of the name get the one that is loaded: do not map
            # a second copy.
            continue
        # The unversioned name, or the versioned one it links to, sorts first.
        paths = sorted((install.root / "lib").glob(name + "*"))
        if not paths:
            continue
        try:
            # Not RTLD_GLOBAL: the loader matches a loaded library by name in
            # any scope, and symbols in the global scope come ahead of a NIXL
            # plugin's own. libhipfile exports another Taskflow version under
            # the names the AIS plugin uses, which crashes the plugin's workers.
            ctypes.CDLL(str(paths[0]), mode=ctypes.RTLD_LOCAL)
        except OSError:
            # Left to the dynamic loader, as it is without an install.
            pass


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


def _loaded_hip_runtime() -> pathlib.Path | None:
    """The HIP runtime mapped into this process, if one is."""
    return _loaded_library(_HIP_RUNTIME_LIBS[0])


def _loaded_library(name: str) -> pathlib.Path | None:
    """The library of that name mapped into this process, if one is."""
    try:
        with open(_PROC_MAPS, encoding="utf-8", errors="replace") as maps:
            for line in maps:
                # address, permissions, offset, device, inode, path
                fields = line.split(None, 5)
                if len(fields) == 6:
                    path = pathlib.Path(fields[5].rstrip("\n"))
                    if path.name.startswith(name):
                        return path
    except OSError:
        pass
    return None


def _rocm_roots() -> Iterator[pathlib.Path]:
    """The prefixes find_rocm() looks for a ROCm install under, in order."""
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        yield pathlib.Path(rocm_path)
    pip_root = _pip_rocm_root()
    if pip_root is not None:
        yield pip_root
    if not rocm_path:
        yield pathlib.Path(_DEFAULT_ROCM_ROOT)


def _pip_rocm_root() -> pathlib.Path | None:
    """The prefix of a ROCm installed from AMD's pip packages, if one is."""
    try:
        import rocm_sdk

        # <site-packages>/_rocm_sdk_core/lib/libamdhip64.so.<N>
        return pathlib.Path(rocm_sdk.find_libraries("amdhip64")[0]).parent.parent
    except Exception:
        # Not installed, without its core package, or not AMD's package.
        return None


def _rocm_version_from_header(root: pathlib.Path) -> tuple[int, int] | None:
    """ROCm version from rocm-core/rocm_version.h, matching Meson's third fallback."""
    try:
        text = (root / "include" / "rocm-core" / "rocm_version.h").read_text()
    except (OSError, UnicodeDecodeError):
        return None
    major = re.search(r"#define\s+ROCM_VERSION_MAJOR\s+(\d+)", text)
    minor = re.search(r"#define\s+ROCM_VERSION_MINOR\s+(\d+)", text)
    if major and minor:
        return (int(major.group(1)), int(minor.group(1)))
    return None


def _rocm_install(root: pathlib.Path) -> RocmInstall | None:
    """The ROCm install at a prefix, read from its version file or header.

    A prefix without the HIP runtime is not one: the process cannot be using
    it.
    """
    version = None
    for version_file in (
        root / ".info" / "version",
        root / "core" / ".info" / "version",
    ):
        try:
            version = parse_rocm_version(version_file.read_text())
        except (OSError, UnicodeDecodeError):
            continue
        if version is not None:
            break
    if version is None:
        version = _rocm_version_from_header(root)
    if version is None or not any((root / "lib").glob(_HIP_RUNTIME_LIBS[0] + "*")):
        return None
    return RocmInstall(root, version, str(root))


def _torch_cuda_version_from_disk() -> str | None:
    """Return torch's build CUDA version (e.g. "12.6") without full torch import."""
    return getattr(_torch_version_module_from_disk(), "cuda", None)


def _torch_rocm_version_from_disk() -> str | None:
    """Return torch's version if it is a ROCm build (e.g. "2.15.0+rocm10.2.0").

    Without full torch import, as _torch_cuda_version_from_disk().
    """
    module = _torch_version_module_from_disk()
    version = _rocm_tag(getattr(module, "__version__", None))
    if version is None and getattr(module, "hip", None) is not None:
        # A ROCm build without the tag: report what version there is.
        version = str(getattr(module, "__version__", None) or "unknown")
    return version


def _torch_version_module_from_disk() -> types.ModuleType | None:
    """Return torch's ``version`` module without full torch import.

    ``find_spec`` locates the package without running its ``__init__``, and the
    standalone module name keeps importlib from importing the torch package; we
    then exec only the tiny ``version.py`` to read its attributes.
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
    return module
