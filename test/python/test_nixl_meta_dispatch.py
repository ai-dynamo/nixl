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

"""Backend selection of the ``nixl`` meta package.

Runs against the dispatcher sources with fake backend packages, so it needs
neither a built NIXL nor a GPU.
"""

import importlib
import pathlib
import pkgutil
import re
import sys
import types

import pytest

META_DIR = pathlib.Path(__file__).resolve().parents[2] / "src/bindings/python/nixl-meta"
BACKEND_SUBMODULES = ("_bindings", "_utils", "logging", "_api")


class _HideInstalledBackends:
    """Meta path finder hiding NIXL backends that are really installed."""

    def __init__(self, backends):
        self.backends = backends

    def find_spec(self, name, path=None, target=None):
        if (
            re.fullmatch(r"nixl_(cu|rocm)\w*", name)
            and not (self.backends / name).is_dir()
        ):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated import state: dispatcher sources plus a dir for fake backends."""
    if not META_DIR.is_dir():
        pytest.skip("NIXL source tree not available")
    saved = dict(sys.modules)
    for name in list(sys.modules):
        if name.split(".")[0].startswith("nixl") or name.split(".")[0] == "torch":
            del sys.modules[name]
    backends = tmp_path / "site"
    backends.mkdir()
    rocm = tmp_path / "rocm"
    rocm.mkdir()
    monkeypatch.syspath_prepend(str(backends))
    monkeypatch.syspath_prepend(str(META_DIR))
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    monkeypatch.setattr(
        sys, "meta_path", [_HideInstalledBackends(backends), *sys.meta_path]
    )
    importlib.invalidate_caches()

    utils = importlib.import_module("nixl_meta_utils")
    real_detect_cuda_major = utils.detect_cuda_major
    monkeypatch.setattr(utils, "detect_cuda_major", lambda: None)
    monkeypatch.setattr(utils, "_torch_cuda_version_from_disk", lambda: None)
    monkeypatch.setattr(utils, "_distribution_version", lambda name: None)
    iter_modules = pkgutil.iter_modules
    monkeypatch.setattr(
        utils.pkgutil, "iter_modules", lambda: iter_modules([str(backends)])
    )

    yield types.SimpleNamespace(
        utils=utils,
        backends=backends,
        rocm=rocm,
        real_detect_cuda_major=real_detect_cuda_major,
    )

    sys.modules.clear()
    sys.modules.update(saved)


def add_backend(env, name):
    pkg = env.backends / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for sub in BACKEND_SUBMODULES:
        (pkg / f"{sub}.py").write_text(f"marker = {name!r}\n")
    importlib.invalidate_caches()


def selected_backend():
    return importlib.import_module("nixl").marker


def set_rocm_install(env, version, subdir=".info"):
    info = env.rocm / subdir
    info.mkdir(parents=True)
    (info / "version").write_text(version + "\n")


def fake_torch(monkeypatch, version, hip=None, cuda=None):
    torch = types.ModuleType("torch")
    torch.__version__ = version
    torch.version = types.SimpleNamespace(cuda=cuda, hip=hip)
    monkeypatch.setitem(sys.modules, "torch", torch)


def use_cuda_torch(env, monkeypatch):
    """A CUDA 13 PyTorch, found by the real CUDA detection."""
    fake_torch(monkeypatch, "2.14.0+cu130", cuda="13.0")
    monkeypatch.setattr(env.utils, "detect_cuda_major", env.real_detect_cuda_major)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10.1.0", (10, 1)),
        ("7.14.1\n", (7, 14)),
        ("7.14.0-123", (7, 14)),
        ("10.1", (10, 1)),
        ("2.14.0+rocm10.1.0", (10, 1)),
        ("2.12.1+rocm7.14", (7, 14)),
        ("2.14.0+cu130", None),
        ("2.14.0+cpu", None),
        ("garbage", None),
        ("10", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_rocm_version(env, text, expected):
    assert env.utils.parse_rocm_version(text) == expected


def test_version_from_imported_torch(env, monkeypatch):
    fake_torch(monkeypatch, "2.14.0+rocm10.1.0")
    set_rocm_install(env, "7.14.1")
    assert env.utils.detect_rocm_version() == (10, 1)


def test_version_from_torch_metadata_without_importing_torch(env, monkeypatch):
    versions = {"torch": "2.13.0+rocm7.14.0", "rocm": "10.1.0"}
    monkeypatch.setattr(env.utils, "_distribution_version", versions.get)
    assert env.utils.detect_rocm_version() == (7, 14)
    assert "torch" not in sys.modules


def test_non_rocm_torch_falls_back_to_rocm_install(env, monkeypatch):
    fake_torch(monkeypatch, "2.14.0+cpu")
    set_rocm_install(env, "10.1.0")
    assert env.utils.detect_rocm_version() == (10, 1)


@pytest.mark.parametrize("subdir", [".info", "core/.info"])
def test_version_from_rocm_install(env, subdir):
    set_rocm_install(env, "7.14.1-42", subdir)
    assert env.utils.detect_rocm_version() == (7, 14)


def test_version_from_pip_rocm(env, monkeypatch):
    monkeypatch.setattr(env.utils, "_distribution_version", {"rocm": "10.1.0"}.get)
    assert env.utils.detect_rocm_version() == (10, 1)


def test_no_version(env):
    assert env.utils.detect_rocm_version() is None


def test_find_rocm_backends_newest_first(env):
    for name in ("nixl_rocm7_14", "nixl_rocm10_1", "nixl_rocm", "nixl_cu12"):
        add_backend(env, name)
    assert env.utils.find_rocm_backends() == ["nixl_rocm10_1", "nixl_rocm7_14"]


def test_selects_backend_matching_rocm_version(env):
    for name in ("nixl_rocm7_14", "nixl_rocm10_1", "nixl_rocm"):
        add_backend(env, name)
    set_rocm_install(env, "7.14.1")
    assert selected_backend() == "nixl_rocm7_14"
    assert "torch" not in sys.modules


@pytest.mark.parametrize(
    "version, expected",
    [("7.14.1", "nixl_rocm7_14"), ("10.1.0", "nixl_rocm10_1")],
)
def test_selects_among_all_supported_rocm_backends(env, version, expected):
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    set_rocm_install(env, version)
    assert selected_backend() == expected


@pytest.mark.parametrize(
    "version, expected",
    [("2.13.0+rocm7.14.0", "nixl_rocm7_14"), ("2.14.0+rocm10.1.0", "nixl_rocm10_1")],
)
def test_selects_among_all_supported_rocm_backends_by_torch(
    env, monkeypatch, version, expected
):
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    monkeypatch.setattr(env.utils, "_distribution_version", {"torch": version}.get)
    assert selected_backend() == expected
    assert "torch" not in sys.modules


def test_mismatched_rocm_backend_is_an_error(env):
    add_backend(env, "nixl_rocm7_14")
    set_rocm_install(env, "10.1.0")
    with pytest.raises(ImportError, match=r"ROCm 10\.1 .*nixl_rocm7_14"):
        selected_backend()


def test_unsupported_rocm_version_names_installed_backends(env):
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    set_rocm_install(env, "6.4.2")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    message = str(excinfo.value)
    assert "ROCm 6.4" in message
    assert "this rocm-nixl has no backend for it" in message
    assert "nixl-rocm6.4" not in message
    assert "nixl_rocm10_1, nixl_rocm7_14" in message


def test_unknown_version_uses_installed_rocm_backend(env):
    add_backend(env, "nixl_rocm10_1")
    assert selected_backend() == "nixl_rocm10_1"


def test_unknown_version_prefers_versioned_over_legacy(env, monkeypatch):
    add_backend(env, "nixl_rocm")
    add_backend(env, "nixl_rocm7_14")
    fake_torch(monkeypatch, "2.14.0", hip="7.14")
    assert selected_backend() == "nixl_rocm7_14"


def test_legacy_backend_with_known_version(env):
    add_backend(env, "nixl_rocm")
    set_rocm_install(env, "10.1.0")
    assert selected_backend() == "nixl_rocm"


@pytest.mark.parametrize("rocm_backend", ["nixl_rocm", "nixl_rocm10_1"])
def test_cuda_backend_with_rocm_install_keeps_upstream_selection(env, rocm_backend):
    add_backend(env, rocm_backend)
    add_backend(env, "nixl_cu12")
    set_rocm_install(env, "10.1.0")
    assert selected_backend() == "nixl_cu12"


def test_cuda_backend_with_unsupported_rocm_keeps_upstream_selection(env):
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_cu12")
    set_rocm_install(env, "7.14.1")
    assert selected_backend() == "nixl_cu12"


def test_cuda_backend_with_rocm_torch_keeps_upstream_selection(env, monkeypatch):
    for name in ("nixl_rocm", "nixl_rocm10_1", "nixl_cu13"):
        add_backend(env, name)
    fake_torch(monkeypatch, "2.14.0+rocm10.1.0", hip="10.1")
    assert selected_backend() == "nixl_rocm"


def test_cuda_backend_with_rocm_torch_ignores_versioned_backend(env, monkeypatch):
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_cu13")
    fake_torch(monkeypatch, "2.14.0+rocm10.1.0", hip="10.1")
    assert selected_backend() == "nixl_cu13"


def test_cuda_fallback_order_without_rocm(env):
    for name in ("nixl_cu12", "nixl_cu13", "nixl_rocm10_1", "nixl_rocm"):
        add_backend(env, name)
    assert selected_backend() == "nixl_cu13"


def test_detected_cuda_is_unaffected_by_rocm(env, monkeypatch):
    monkeypatch.setattr(env.utils, "detect_cuda_major", lambda: 12)
    add_backend(env, "nixl_cu12")
    add_backend(env, "nixl_rocm10_1")
    set_rocm_install(env, "10.1.0")
    assert selected_backend() == "nixl_cu12"


def test_cuda_torch_with_only_rocm_backends_is_an_error(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    set_rocm_install(env, "10.1.0")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    message = str(excinfo.value)
    assert "CUDA build of PyTorch (CUDA 13.0)" in message
    assert "only ROCm NIXL backends are installed" in message
    assert "nixl_rocm10_1, nixl_rocm7_14" in message
    assert '"rocm-nixl[rocm<X>.<Y>]" --extra-index-url' in message
    assert "extras available: rocm10.1, rocm7.14" in message
    assert "http" not in message


def test_cuda_torch_on_disk_with_only_rocm_backends_is_an_error(env, monkeypatch):
    monkeypatch.setattr(env.utils, "detect_cuda_major", lambda: 12)
    monkeypatch.setattr(env.utils, "_torch_cuda_version_from_disk", lambda: "12.6")
    add_backend(env, "nixl_rocm10_1")
    with pytest.raises(ImportError, match=r"CUDA build of PyTorch \(CUDA 12\.6\)"):
        selected_backend()
    assert "torch" not in sys.modules


def test_cuda_torch_with_matching_cuda_backend(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    add_backend(env, "nixl_cu13")
    assert selected_backend() == "nixl_cu13"


def test_cuda_torch_with_other_cuda_backend_keeps_upstream_error(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    add_backend(env, "nixl_cu12")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == "detected CUDA 13 but nixl-cu13 is not installed"


def test_cuda_torch_with_both_kinds_selects_cuda_backend(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    add_backend(env, "nixl_cu13")
    add_backend(env, "nixl_rocm10_1")
    set_rocm_install(env, "10.1.0")
    assert selected_backend() == "nixl_cu13"


def test_cuda_torch_with_both_kinds_keeps_upstream_error(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    add_backend(env, "nixl_cu12")
    add_backend(env, "nixl_rocm10_1")
    set_rocm_install(env, "10.1.0")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == "detected CUDA 13 but nixl-cu13 is not installed"


def test_cuda_torch_without_backends_keeps_upstream_error(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == "detected CUDA 13 but nixl-cu13 is not installed"


def test_cuda_without_cuda_torch_keeps_upstream_error(env, monkeypatch):
    # CUDA detected through cuda-python or cupy, PyTorch absent or not a CUDA build.
    monkeypatch.setattr(env.utils, "detect_cuda_major", lambda: 12)
    add_backend(env, "nixl_rocm10_1")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == "detected CUDA 12 but nixl-cu12 is not installed"


def test_no_torch_with_rocm_backend(env):
    add_backend(env, "nixl_rocm10_1")
    set_rocm_install(env, "10.1.0")
    assert selected_backend() == "nixl_rocm10_1"
    assert "torch" not in sys.modules


NO_BACKEND_MESSAGE = "No NIXL backend found (tried nixl_cu13, nixl_cu12, nixl_rocm)"


def test_no_backend(env):
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == NO_BACKEND_MESSAGE


def test_no_backend_with_rocm_detected(env):
    set_rocm_install(env, "10.1.0")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == NO_BACKEND_MESSAGE
