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

"""Backend selection of the ``nixl`` meta package.

Runs against the dispatcher sources with fake backend packages, so it needs
neither a built NIXL nor a GPU.
"""

import ctypes
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


def _drop_nixl_and_torch_modules():
    for name in list(sys.modules):
        top = name.split(".")[0]
        if top.startswith("nixl") or top == "torch":
            del sys.modules[name]


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated import state: dispatcher sources plus a dir for fake backends."""
    if not META_DIR.is_dir():
        pytest.skip("NIXL source tree not available")
    saved = dict(sys.modules)
    _drop_nixl_and_torch_modules()
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
    real_torch_rocm_version_from_disk = utils._torch_rocm_version_from_disk
    monkeypatch.setattr(utils, "_torch_rocm_version_from_disk", lambda: None)
    # No ROCm of the machine running the tests: not loaded, not from pip, and
    # not under the default root.
    real_loaded_library = utils._loaded_library
    real_pip_rocm_root = utils._pip_rocm_root
    monkeypatch.setattr(utils, "_loaded_library", lambda name: None)
    monkeypatch.setattr(utils, "_pip_rocm_root", lambda: None)
    monkeypatch.setattr(utils, "_DEFAULT_ROCM_ROOT", str(tmp_path / "opt-rocm"))
    # Libraries the dispatcher loads, and whether it had the backend by then.
    loaded = []

    def fake_cdll(path, mode=ctypes.DEFAULT_MODE):
        # Never the global scope: test_load_rocm_stays_out_of_the_global_scope.
        assert mode == ctypes.RTLD_LOCAL
        backend_imported = any(
            re.fullmatch(r"nixl_(cu|rocm)\w*", name) for name in sys.modules
        )
        loaded.append((path, backend_imported))

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    iter_modules = pkgutil.iter_modules
    monkeypatch.setattr(
        utils.pkgutil, "iter_modules", lambda: iter_modules([str(backends)])
    )

    yield types.SimpleNamespace(
        utils=utils,
        backends=backends,
        rocm=rocm,
        loaded=loaded,
        real_detect_cuda_major=real_detect_cuda_major,
        real_torch_rocm_version_from_disk=real_torch_rocm_version_from_disk,
        real_loaded_library=real_loaded_library,
        real_pip_rocm_root=real_pip_rocm_root,
    )

    _drop_nixl_and_torch_modules()
    sys.modules.update({k: v for k, v in saved.items() if k not in sys.modules})


def add_backend(env, name):
    pkg = env.backends / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for sub in BACKEND_SUBMODULES:
        (pkg / f"{sub}.py").write_text(f"marker = {name!r}\n")
    importlib.invalidate_caches()


def selected_backend():
    return importlib.import_module("nixl").marker


def make_rocm(prefix, version, libs=("libamdhip64.so.7", "libhipfile.so.0")):
    """A ROCm install: its version file and, empty, its HIP runtime."""
    (prefix / ".info").mkdir(parents=True)
    (prefix / ".info" / "version").write_text(version + "\n")
    (prefix / "lib").mkdir()
    for lib in libs:
        (prefix / "lib" / lib).write_text("")
    return prefix


def set_rocm_install(env, version, subdir="."):
    """A ROCm install under $ROCM_PATH."""
    return make_rocm(env.rocm / subdir, version)


def set_pip_rocm(env, monkeypatch, version):
    """A ROCm installed from AMD's pip packages."""
    prefix = make_rocm(env.backends / "_rocm_sdk_core", version)
    monkeypatch.setattr(env.utils, "_pip_rocm_root", lambda: prefix)
    return prefix


def set_loaded_hip_runtime(env, monkeypatch, prefix):
    """A process that has the HIP runtime under a prefix loaded already."""
    set_loaded_libraries(env, monkeypatch, prefix / "lib" / "libamdhip64.so.7")


def set_loaded_libraries(env, monkeypatch, *paths):
    """A process that has those libraries loaded already."""

    def loaded_library(name):
        return next((path for path in paths if path.name.startswith(name)), None)

    monkeypatch.setattr(env.utils, "_loaded_library", loaded_library)


def hip_runtime(prefix):
    """What the dispatcher loads from a make_rocm() install, in order."""
    return [
        (str(prefix / "lib" / "libamdhip64.so.7"), False),
        (str(prefix / "lib" / "libhipfile.so.0"), False),
    ]


def fake_torch(version, hip=None, cuda=None):
    torch = types.ModuleType("torch")
    torch.__version__ = version
    torch.version = types.SimpleNamespace(cuda=cuda, hip=hip)
    # The env fixture restores the real torch; monkeypatch would delete it again.
    sys.modules["torch"] = torch


def install_torch(env, monkeypatch, version, hip=None, importable=True):
    """A PyTorch that is installed and not imported, found by the real detection.

    Importing it records that in its "imported" attribute, or fails the test.
    """
    package = env.backends / "torch"
    package.mkdir()
    (package / "version.py").write_text(
        f"__version__ = {version!r}\ncuda = None\nhip = {hip!r}\n"
    )
    body = "from . import version\n__version__ = version.__version__\nimported = True\n"
    if not importable:
        body = "raise AssertionError('torch was imported')\n"
    (package / "__init__.py").write_text(body)
    importlib.invalidate_caches()
    monkeypatch.setattr(env.utils, "detect_cuda_major", env.real_detect_cuda_major)
    monkeypatch.setattr(
        env.utils,
        "_torch_rocm_version_from_disk",
        env.real_torch_rocm_version_from_disk,
    )
    monkeypatch.setattr(
        env.utils,
        "_torch_cuda_version_from_disk",
        lambda: getattr(env.utils._torch_version_module_from_disk(), "cuda", None),
    )


def use_cuda_torch(env, monkeypatch):
    """A CUDA 13 PyTorch, found by the real CUDA detection."""
    fake_torch("2.14.0+cu130", cuda="13.0")
    monkeypatch.setattr(env.utils, "detect_cuda_major", env.real_detect_cuda_major)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10.2.0", (10, 2)),
        ("10.1.0", (10, 1)),
        ("7.14.1\n", (7, 14)),
        ("7.14.0-123", (7, 14)),
        ("10.1", (10, 1)),
        ("2.15.0+rocm10.2.0", (10, 2)),
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


def on_disk(prefix, version):
    """What find_rocm() returns for a make_rocm() install it will load."""
    return (prefix, version, str(prefix))


def test_rocm_from_rocm_path(env):
    prefix = set_rocm_install(env, "7.14.1-42")
    assert env.utils.find_rocm() == on_disk(prefix, (7, 14))


def test_rocm_from_core_under_rocm_path(env):
    prefix = set_rocm_install(env, "10.2.0", "core")
    assert env.utils.find_rocm() == on_disk(prefix, (10, 2))


def test_rocm_version_from_core_info_with_hip_at_root(env):
    # Layout: version at <root>/core/.info/version, HIP at <root>/lib.
    # Meson reads the nested version file; the runtime loader must match it.
    root = env.rocm
    (root / "core" / ".info").mkdir(parents=True)
    (root / "core" / ".info" / "version").write_text("10.1.0\n")
    (root / "lib").mkdir()
    (root / "lib" / "libamdhip64.so.7").write_text("")
    assert env.utils.find_rocm() == on_disk(root, (10, 1))


def test_rocm_version_from_header_fallback(env):
    # Layout: no version file, only rocm-core/rocm_version.h with defines.
    # Matches Meson's third fallback so the runtime selects the same backend.
    root = env.rocm
    (root / "include" / "rocm-core").mkdir(parents=True)
    header = (
        "#define ROCM_VERSION_MAJOR 10\n"
        "#define ROCM_VERSION_MINOR 2\n"
        "#define ROCM_VERSION_PATCH 0\n"
    )
    (root / "include" / "rocm-core" / "rocm_version.h").write_text(header)
    (root / "lib").mkdir()
    (root / "lib" / "libamdhip64.so.7").write_text("")
    assert env.utils.find_rocm() == on_disk(root, (10, 2))


def test_rocm_from_the_only_versioned_core(env):
    # The component packages: <root>/core-<X.Y>, with links to it next to it.
    prefix = set_rocm_install(env, "10.1.0", "core-10.1")
    (env.rocm / "core-10").symlink_to(prefix)
    assert env.utils.find_rocm() == on_disk(prefix.resolve(), (10, 1))

    set_rocm_install(env, "10.2.0", "core-10.2")
    assert env.utils.find_rocm() is None


def test_rocm_from_pip(env, monkeypatch):
    prefix = set_pip_rocm(env, monkeypatch, "10.2.0")
    assert env.utils.find_rocm() == on_disk(prefix, (10, 2))


def test_a_prefix_without_the_hip_runtime_is_not_an_install(env, monkeypatch):
    make_rocm(env.rocm, "7.2.4", libs=("libhipfile.so.0",))
    assert env.utils.find_rocm() is None
    pip = set_pip_rocm(env, monkeypatch, "10.2.0")
    assert env.utils.find_rocm() == on_disk(pip, (10, 2))


def test_rocm_from_the_default_root(env, monkeypatch):
    prefix = make_rocm(pathlib.Path(env.utils._DEFAULT_ROCM_ROOT), "10.2.0")
    # ROCM_PATH is set, to a directory without ROCm: the default is not used.
    assert env.utils.find_rocm() is None
    monkeypatch.delenv("ROCM_PATH")
    assert env.utils.find_rocm() == on_disk(prefix, (10, 2))


def test_rocm_path_beats_pip_beats_the_default_root(env, monkeypatch):
    make_rocm(pathlib.Path(env.utils._DEFAULT_ROCM_ROOT), "7.14.1")
    pip = set_pip_rocm(env, monkeypatch, "10.1.0")
    assert env.utils.find_rocm() == on_disk(pip, (10, 1))
    monkeypatch.delenv("ROCM_PATH")
    assert env.utils.find_rocm() == on_disk(pip, (10, 1))
    monkeypatch.setenv("ROCM_PATH", str(env.rocm))
    rocm_path = set_rocm_install(env, "10.2.0")
    assert env.utils.find_rocm() == on_disk(rocm_path, (10, 2))


def test_loaded_hip_runtime_beats_everything(env, monkeypatch):
    set_rocm_install(env, "10.2.0")
    pip = set_pip_rocm(env, monkeypatch, "10.1.0")
    set_loaded_hip_runtime(env, monkeypatch, pip)
    fake_torch("2.13.0+rocm7.14.0")
    runtime = pip / "lib" / "libamdhip64.so.7"
    assert env.utils.find_rocm() == (
        pip,
        (10, 1),
        f"the loaded HIP runtime {runtime}",
    )


def test_imported_torch_tag_beats_the_installs(env, monkeypatch):
    # A PyTorch that bundles ROCm: its HIP runtime has no version file.
    set_rocm_install(env, "7.14.1")
    set_loaded_hip_runtime(env, monkeypatch, env.backends / "torch")
    fake_torch("2.14.0+rocm10.1.0")
    assert env.utils.find_rocm() == (None, (10, 1), "PyTorch 2.14.0+rocm10.1.0")


def test_non_rocm_torch_falls_back_to_rocm_install(env):
    fake_torch("2.14.0+cpu")
    prefix = set_rocm_install(env, "10.2.0")
    assert env.utils.find_rocm() == on_disk(prefix, (10, 2))


def test_install_is_not_loaded_next_to_another_hip_runtime(env, monkeypatch):
    prefix = set_rocm_install(env, "10.2.0")
    set_loaded_hip_runtime(env, monkeypatch, env.backends / "elsewhere")
    assert env.utils.find_rocm() == (None, (10, 2), str(prefix))


def test_no_rocm(env):
    assert env.utils.find_rocm() is None


def test_loaded_libraries_from_process_maps(env, monkeypatch, tmp_path):
    maps = tmp_path / "maps"
    monkeypatch.setattr(env.utils, "_PROC_MAPS", str(maps))
    monkeypatch.setattr(env.utils, "_loaded_library", env.real_loaded_library)
    line = "7f00-7f01 r-xp 00000000 08:01 42    {}\n"
    maps.write_text(
        "7ffc-7ffd rw-p 00000000 00:00 0\n"
        + line.format("/usr/lib/libc.so.6")
        + line.format("/opt/my rocm/lib/libamdhip64.so.7.16.26385")
        + line.format("/other/lib/libamdhip64.so.7")
    )
    assert env.utils._loaded_hip_runtime() == pathlib.Path(
        "/opt/my rocm/lib/libamdhip64.so.7.16.26385"
    )
    assert env.utils._loaded_library("libhipfile.so") is None
    maps.write_text(
        line.format("/usr/lib/libc.so.6")
        + line.format("/opt/rocm/lib/libhipfile.so.0.5.0")
    )
    assert env.utils._loaded_hip_runtime() is None
    assert env.utils._loaded_library("libhipfile.so") == pathlib.Path(
        "/opt/rocm/lib/libhipfile.so.0.5.0"
    )
    maps.unlink()
    assert env.utils._loaded_hip_runtime() is None
    assert env.utils._loaded_library("libhipfile.so") is None


def test_pip_rocm_root(env, monkeypatch):
    library = env.backends / "_rocm_sdk_core" / "lib" / "libamdhip64.so.7"
    rocm_sdk = types.ModuleType("rocm_sdk")
    rocm_sdk.find_libraries = lambda *names: [library]
    monkeypatch.setitem(sys.modules, "rocm_sdk", rocm_sdk)
    assert env.real_pip_rocm_root() == env.backends / "_rocm_sdk_core"

    # Without its core package, find_libraries() raises.
    def missing(*names):
        raise ModuleNotFoundError("Missing required rocm libraries")

    rocm_sdk.find_libraries = missing
    assert env.real_pip_rocm_root() is None
    # Another package of that name.
    monkeypatch.setitem(sys.modules, "rocm_sdk", types.ModuleType("rocm_sdk"))
    assert env.real_pip_rocm_root() is None
    # Not installed.
    monkeypatch.setitem(sys.modules, "rocm_sdk", None)
    assert env.real_pip_rocm_root() is None


def test_load_rocm(env, tmp_path):
    prefix = make_rocm(
        tmp_path / "prefix",
        "10.2.0",
        libs=(
            "libamdhip64.so",
            "libamdhip64.so.7",
            "libhipfile.so.0",
            "libhipfile.so.0.5.0",
            "libamd_comgr.so.3",
        ),
    )
    env.utils.load_rocm(env.utils.RocmInstall(prefix, (10, 2)))
    assert env.loaded == [
        (str(prefix / "lib" / "libamdhip64.so"), False),
        (str(prefix / "lib" / "libhipfile.so.0"), False),
    ]


def test_load_rocm_stays_out_of_the_global_scope(env, monkeypatch, tmp_path):
    # In the global scope the symbols of libhipfile come ahead of the AIS
    # plugin's own, and the plugin then runs the Taskflow of libhipfile.
    prefix = make_rocm(tmp_path / "prefix", "10.2.0")
    modes = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode: modes.append(mode))
    env.utils.load_rocm(env.utils.RocmInstall(prefix, (10, 2)))
    assert modes == [ctypes.RTLD_LOCAL, ctypes.RTLD_LOCAL]


def test_load_rocm_skips_what_is_loaded_already(env, monkeypatch, tmp_path):
    prefix = make_rocm(tmp_path / "prefix", "10.2.0")
    install = env.utils.RocmInstall(prefix, (10, 2))
    # As after AMD's PyTorch, or the application, loaded the HIP runtime.
    set_loaded_hip_runtime(env, monkeypatch, prefix)
    env.utils.load_rocm(install)
    assert env.loaded == hip_runtime(prefix)[1:]
    # As on a second call, and whichever file has the name.
    del env.loaded[:]
    set_loaded_libraries(
        env,
        monkeypatch,
        prefix / "lib" / "libamdhip64.so.7",
        tmp_path / "elsewhere" / "libhipfile.so.0.5.0",
    )
    env.utils.load_rocm(install)
    assert env.loaded == []


def test_load_rocm_without_a_prefix_or_libraries(env, tmp_path):
    env.utils.load_rocm(env.utils.RocmInstall(None, (10, 2)))
    env.utils.load_rocm(env.utils.RocmInstall(tmp_path / "missing", (10, 2)))
    prefix = make_rocm(tmp_path / "prefix", "10.2.0", libs=("libhipfile.so.0",))
    env.utils.load_rocm(env.utils.RocmInstall(prefix, (10, 2)))
    assert env.loaded == [(str(prefix / "lib" / "libhipfile.so.0"), False)]


def test_load_rocm_leaves_a_failure_to_the_dynamic_loader(env, monkeypatch, tmp_path):
    prefix = make_rocm(tmp_path / "prefix", "10.2.0")
    attempts = []

    def failing_cdll(path, mode=0):
        attempts.append(path)
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", failing_cdll)
    env.utils.load_rocm(env.utils.RocmInstall(prefix, (10, 2)))
    assert attempts == [path for path, _ in hip_runtime(prefix)]


def test_find_rocm_backends_newest_first(env):
    for name in (
        "nixl_rocm7_14",
        "nixl_rocm10_2",
        "nixl_rocm10_1",
        "nixl_rocm",
        "nixl_cu13",
    ):
        add_backend(env, name)
    assert env.utils.find_rocm_backends() == [
        "nixl_rocm10_2",
        "nixl_rocm10_1",
        "nixl_rocm7_14",
    ]


def test_selects_backend_matching_rocm_version(env):
    for name in ("nixl_rocm7_14", "nixl_rocm10_1", "nixl_rocm"):
        add_backend(env, name)
    prefix = set_rocm_install(env, "7.14.1")
    assert selected_backend() == "nixl_rocm7_14"
    assert "torch" not in sys.modules
    # The HIP runtime of that install is loaded, ahead of the backend.
    assert env.loaded == hip_runtime(prefix)


@pytest.mark.parametrize(
    "version, expected",
    [
        ("7.14.1", "nixl_rocm7_14"),
        ("10.1.0", "nixl_rocm10_1"),
        ("10.2.0", "nixl_rocm10_2"),
    ],
)
def test_selects_among_all_supported_rocm_backends(env, version, expected):
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, version)
    assert selected_backend() == expected


@pytest.mark.parametrize(
    "version, expected",
    [
        ("7.14.1", "nixl_rocm7_14"),
        ("10.1.0", "nixl_rocm10_1"),
        ("10.2.0", "nixl_rocm10_2"),
    ],
)
def test_selects_among_all_supported_rocm_backends_by_pip_rocm(
    env, monkeypatch, version, expected
):
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    prefix = set_pip_rocm(env, monkeypatch, version)
    assert selected_backend() == expected
    assert "torch" not in sys.modules
    assert env.loaded == hip_runtime(prefix)


def test_selects_backend_for_the_rocm_torch_has_loaded(env, monkeypatch):
    # AMD's PyTorch loads the HIP runtime of the pip ROCm it depends on.
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, "10.2.0")
    prefix = set_pip_rocm(env, monkeypatch, "10.1.0")
    set_loaded_hip_runtime(env, monkeypatch, prefix)
    fake_torch("2.14.0+rocm10.1.0", hip="7.16.26385")
    assert selected_backend() == "nixl_rocm10_1"
    # Only what PyTorch has not loaded, from the same install.
    assert env.loaded == hip_runtime(prefix)[1:]


@pytest.mark.parametrize(
    "torch_version, on_disk, expected",
    [
        ("2.15.0+rocm10.2.0", "10.1.0", "nixl_rocm10_2"),
        ("2.14.0+rocm10.1.0", "7.14.1", "nixl_rocm10_1"),
        ("2.13.0+rocm7.14.0", "10.2.0", "nixl_rocm7_14"),
    ],
)
def test_selects_backend_by_the_tag_of_a_torch_bundling_rocm(
    env, torch_version, on_disk, expected
):
    # The tag wins over an install of another version on disk.
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, on_disk)
    fake_torch(torch_version, hip="7.16.26385")
    assert selected_backend() == expected
    assert env.loaded == []


def test_mismatched_rocm_backend_is_an_error(env):
    add_backend(env, "nixl_rocm7_14")
    set_rocm_install(env, "10.1.0")
    with pytest.raises(ImportError, match=r"ROCm 10\.1 .*nixl_rocm7_14"):
        selected_backend()
    assert env.loaded == []


def test_unsupported_rocm_version_names_installed_backends(env):
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    set_rocm_install(env, "6.4.2")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    message = str(excinfo.value)
    assert f"detected ROCm 6.4 from {env.rocm} but" in message
    assert "no NIXL backend for it is installed" in message
    assert "nixl-rocm6.4" not in message
    assert "nixl_rocm10_1, nixl_rocm7_14" in message


def test_unsupported_rocm_version_names_the_torch_it_came_from(env):
    # ROCM_PATH cannot change this one: the message has to say so.
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, "10.2.0")
    fake_torch("2.14.0+rocm10.0.0", hip="10.0")
    with pytest.raises(
        ImportError,
        match=r"detected ROCm 10\.0 from PyTorch 2\.14\.0\+rocm10\.0\.0 but",
    ):
        selected_backend()


NO_ROCM_WARNING = (
    r"could not tell which ROCm install .*found none, or several versions.*"
    r"set ROCM_PATH.*newest ROCm version, nixl_rocm{}$"
)


def test_unknown_version_uses_newest_rocm_backend_with_a_warning(env):
    add_backend(env, "nixl_rocm7_14")
    add_backend(env, "nixl_rocm10_1")
    with pytest.warns(RuntimeWarning, match=NO_ROCM_WARNING.format("10_1")):
        assert selected_backend() == "nixl_rocm10_1"
    assert env.loaded == []


def test_several_versioned_cores_use_newest_rocm_backend_with_a_warning(env):
    # Neither core-<X.Y> is the install: the backend may match none of them.
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, "7.14.1", "core-7.14")
    set_rocm_install(env, "10.1.0", "core-10.1")
    with pytest.warns(RuntimeWarning, match=NO_ROCM_WARNING.format("10_2")):
        assert selected_backend() == "nixl_rocm10_2"
    assert env.loaded == []


def test_unknown_version_prefers_versioned_over_legacy(env):
    add_backend(env, "nixl_rocm")
    add_backend(env, "nixl_rocm7_14")
    fake_torch("2.14.0", hip="7.14")
    with pytest.warns(RuntimeWarning, match=NO_ROCM_WARNING.format("7_14")):
        assert selected_backend() == "nixl_rocm7_14"


def test_legacy_backend_with_known_version(env):
    add_backend(env, "nixl_rocm")
    set_rocm_install(env, "10.2.0")
    assert selected_backend() == "nixl_rocm"
    assert env.loaded == []


@pytest.mark.parametrize("rocm_backend", ["nixl_rocm", "nixl_rocm10_2"])
def test_cuda_backend_with_rocm_install_keeps_upstream_selection(env, rocm_backend):
    add_backend(env, rocm_backend)
    add_backend(env, "nixl_cu13")
    set_rocm_install(env, "10.2.0")
    assert selected_backend() == "nixl_cu13"


def test_cuda_backend_with_unsupported_rocm_keeps_upstream_selection(env):
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_cu13")
    set_rocm_install(env, "7.14.1")
    assert selected_backend() == "nixl_cu13"


def test_cuda_backend_with_rocm_torch_keeps_upstream_selection(env):
    for name in ("nixl_rocm", "nixl_rocm10_2", "nixl_cu13"):
        add_backend(env, name)
    fake_torch("2.15.0+rocm10.2.0", hip="10.2")
    assert selected_backend() == "nixl_rocm"


def test_cuda_backend_with_rocm_torch_ignores_versioned_backend(env):
    add_backend(env, "nixl_rocm10_2")
    add_backend(env, "nixl_cu13")
    fake_torch("2.15.0+rocm10.2.0", hip="10.2")
    assert selected_backend() == "nixl_cu13"


def test_cuda_fallback_order_without_rocm(env):
    for name in ("nixl_cu13", "nixl_rocm10_2", "nixl_rocm"):
        add_backend(env, name)
    assert selected_backend() == "nixl_cu13"


def test_detected_cuda_is_unaffected_by_rocm(env, monkeypatch):
    monkeypatch.setattr(env.utils, "detect_cuda_major", lambda: 13)
    add_backend(env, "nixl_cu13")
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, "10.2.0")
    assert selected_backend() == "nixl_cu13"


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
    assert message.endswith("install it in place of the CUDA build")


def test_cuda_torch_on_disk_with_only_rocm_backends_is_an_error(env, monkeypatch):
    monkeypatch.setattr(env.utils, "detect_cuda_major", lambda: 13)
    monkeypatch.setattr(env.utils, "_torch_cuda_version_from_disk", lambda: "13.0")
    add_backend(env, "nixl_rocm10_2")
    with pytest.raises(ImportError, match=r"CUDA build of PyTorch \(CUDA 13\.0\)"):
        selected_backend()
    assert "torch" not in sys.modules


def test_cuda_torch_with_matching_cuda_backend(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    add_backend(env, "nixl_cu13")
    assert selected_backend() == "nixl_cu13"


def test_cuda_torch_with_both_kinds_selects_cuda_backend(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    add_backend(env, "nixl_cu13")
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, "10.2.0")
    assert selected_backend() == "nixl_cu13"


def test_cuda_torch_without_backends_keeps_upstream_error(env, monkeypatch):
    use_cuda_torch(env, monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == "detected CUDA 13 but nixl-cu13 is not installed"


def test_cuda_without_cuda_torch_keeps_upstream_error(env, monkeypatch):
    # CUDA detected through cuda-python or cupy, PyTorch absent or not a CUDA build.
    monkeypatch.setattr(env.utils, "detect_cuda_major", lambda: 13)
    add_backend(env, "nixl_rocm10_2")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == "detected CUDA 13 but nixl-cu13 is not installed"


@pytest.mark.parametrize(
    "version, hip", [("2.15.0+rocm10.2.0", "10.2"), ("2.14.0", "7.14")]
)
def test_installed_rocm_torch_is_not_imported(env, monkeypatch, version, hip):
    # AMD's PyTorch is what the rocm<X>.<Y> extra installs: selecting the
    # backend from the ROCm install must not cost its import.
    install_torch(env, monkeypatch, version, hip=hip, importable=False)
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, "10.2.0")
    assert selected_backend() == "nixl_rocm10_2"
    assert "torch" not in sys.modules
    assert env.utils.detect_rocm()


def test_rocm_path_beats_an_installed_rocm_torch(env, monkeypatch):
    # $ROCM_PATH is an explicit choice: a ROCm there wins over the one a
    # PyTorch that has not been imported would load.
    install_torch(env, monkeypatch, "2.15.0+rocm10.2.0", hip="10.2", importable=False)
    set_pip_rocm(env, monkeypatch, "10.2.0")
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    install = set_rocm_install(env, "10.1.0")
    assert selected_backend() == "nixl_rocm10_1"
    assert "torch" not in sys.modules
    assert all(str(install) in path for path, _ in env.loaded)


def test_installed_rocm_torch_without_rocm_install_is_imported(env, monkeypatch):
    # No ROCm install to select by: this PyTorch bundles its ROCm, and only
    # importing it loads that HIP runtime. Its tag then selects the backend.
    install_torch(env, monkeypatch, "2.14.0+rocm10.1.0", hip="10.1")
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    assert selected_backend() == "nixl_rocm10_1"
    assert sys.modules["torch"].imported


def test_installed_rocm_torch_that_does_not_import_is_ignored(env, monkeypatch):
    install_torch(env, monkeypatch, "2.14.0+rocm10.1.0", hip="10.1")
    (env.backends / "torch" / "__init__.py").write_text("import _no_such_module\n")
    add_backend(env, "nixl_rocm10_1")
    add_backend(env, "nixl_rocm10_2")
    with pytest.warns(RuntimeWarning, match="could not tell which ROCm install"):
        assert selected_backend() == "nixl_rocm10_2"


def test_installed_rocm_torch_with_cuda_backend_is_not_imported(env, monkeypatch):
    install_torch(env, monkeypatch, "2.15.0+rocm10.2.0", hip="10.2", importable=False)
    add_backend(env, "nixl_cu13")
    add_backend(env, "nixl_rocm")
    assert selected_backend() == "nixl_rocm"
    assert "torch" not in sys.modules


def test_no_torch_with_rocm_backend(env):
    add_backend(env, "nixl_rocm10_2")
    set_rocm_install(env, "10.2.0")
    assert selected_backend() == "nixl_rocm10_2"
    assert "torch" not in sys.modules


NO_BACKEND_MESSAGE = "No NIXL backend found (tried nixl_cu13, nixl_rocm)"


def test_no_backend(env):
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == NO_BACKEND_MESSAGE


def test_no_backend_with_rocm_detected(env):
    set_rocm_install(env, "10.2.0")
    with pytest.raises(ImportError) as excinfo:
        selected_backend()
    assert str(excinfo.value) == NO_BACKEND_MESSAGE
