# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check CuTe packaging without requiring CUDA or a CuTe installation."""

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def package_tree(tmp_path):
    """Use the real dispatcher/wrapper with two stub native backends."""
    package = tmp_path / "nixl"
    package.mkdir()
    shutil.copyfile(
        ROOT / "src/bindings/python/nixl-meta/nixl/__init__.py",
        package / "__init__.py",
    )
    shutil.copyfile(
        ROOT / "src/bindings/python/nixl-meta/nixl/cute.py", package / "cute.py"
    )
    shutil.copyfile(
        ROOT / "src/bindings/python/nixl-meta/nixl_meta_utils.py",
        tmp_path / "nixl_meta_utils.py",
    )
    for major in (12, 13):
        backend = tmp_path / f"nixl_cu{major}"
        (backend / "lib").mkdir(parents=True)
        for sm in ("sm_90", "sm_100"):
            (backend / "lib" / f"libnixl_device_{sm}.bc").touch()
        # Distinct values catch hard-coded constants in the Python wrapper.
        (backend / "_bindings.py").write_text(
            "class nixl_status_t:\n    NIXL_SUCCESS = 0\n    NIXL_IN_PROG = 1\n"
            "class nixl_gpu_level_t:\n    THREAD = 11\n    WARP = 22\n    BLOCK = 33\n"
            "NIXL_GPU_DEFER = 8\nNIXL_GPU_REQUEST_BYTES = 128\n"
        )
        for name in ("__init__", "_api", "_utils", "logging"):
            (backend / f"{name}.py").touch()
    cutlass = tmp_path / "cutlass"
    cutlass.mkdir()
    (cutlass / "__init__.py").write_text(
        "class Scalar(int):\n    def bitcast(self, dtype): return dtype(self)\n"
        "Uint64 = Uint32 = Int32 = Scalar\n"
    )
    (cutlass / "cute.py").write_text(
        "calls = []\nBitCode = str\n"
        "def extern(fn, *, name, source):\n"
        "    def emit(*args):\n        calls.append((name, source))\n        return 0\n"
        "    return emit\n"
    )
    (cutlass / "cutlass_dsl.py").write_text(
        "class CuTeDSL:\n"
        "    _get_dsl = classmethod(lambda cls: cls)\n"
        "    get_arch_enum = classmethod(lambda cls: cls.target)\n"
    )
    return tmp_path


def run_import(package_tree, code, major=13):
    setup = (
        f"import nixl_meta_utils\nnixl_meta_utils.detect_cuda_major = lambda: {major}\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", setup + dedent(code)],
        cwd=package_tree,
        env={
            **os.environ,
            "PYTHONPATH": str(package_tree),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("major", [12, 13])
def test_selected_backend_bitcode(package_tree, major):
    run_import(
        package_tree,
        "import sys, nixl\n"
        "assert 'nixl.cute' not in sys.modules and 'cutlass' not in sys.modules\n"
        "from nixl import cute\n"
        "from nixl import _bindings\n"
        "assert cute.THREAD == int(_bindings.nixl_gpu_level_t.THREAD)\n"
        "assert cute.WARP == int(_bindings.nixl_gpu_level_t.WARP)\n"
        "assert cute.BLOCK == int(_bindings.nixl_gpu_level_t.BLOCK)\n"
        "assert cute.DEFER == _bindings.NIXL_GPU_DEFER\n"
        "assert cute.REQUEST_BYTES == _bindings.NIXL_GPU_REQUEST_BYTES\n"
        "import nixl.cute\n"
        "assert nixl.cute is cute\n",
        major,
    )


@pytest.mark.parametrize("major", [12, 13])
def test_target_selected_per_compilation(package_tree, major):
    run_import(
        package_tree,
        """
        from pathlib import Path
        from types import SimpleNamespace
        from cutlass.cute import calls
        from cutlass.cutlass_dsl import CuTeDSL
        from nixl import _bindings, cute

        for major, minor, suffix in [(9, 0, "a"), (10, 0, "f"), (9, 0, "")]:
            CuTeDSL.target = SimpleNamespace(major=major, minor=minor, suffix=suffix)
            cute.put(1, 2, 3)
            cute.atomic_add(1, 2)
            cute.flush(2, channel=3, level=cute.WARP)
            cute.get_xfer_status(1)
            path = Path(_bindings.__file__).parent / "lib" / f"libnixl_device_sm_{major}{minor}.bc"
            assert calls == [(name, str(path)) for name in
                             ("cute_nixl_put", "cute_nixl_atomic_add", "cute_nixl_flush",
                              "cute_nixl_get_xfer_status")]
            calls.clear()
        """,
        major,
    )


def test_missing_target(package_tree):
    (package_tree / "nixl_cu13/lib/libnixl_device_sm_100.bc").unlink()
    run_import(
        package_tree,
        """
        from types import SimpleNamespace
        from cutlass.cutlass_dsl import CuTeDSL
        from nixl import cute

        CuTeDSL.target = SimpleNamespace(major=10, minor=0)
        try:
            cute.get_xfer_status(1)
        except ValueError as exc:
            assert "nixl_cu13" in str(exc) and "Available targets: sm_90." in str(exc)
        else:
            raise AssertionError("Expected a missing-target diagnostic")
        """,
    )


def test_missing_optional_dependency(package_tree):
    shutil.rmtree(package_tree / "cutlass")
    run_import(
        package_tree,
        "import nixl\n"
        "try:\n    import nixl.cute\n"
        "except ImportError as exc:\n    assert 'nixl[cute]' in str(exc)\n"
        "else:\n    raise AssertionError('CuTe should be unavailable')\n",
    )


def test_missing_bitcode_does_not_use_other_backend(package_tree):
    shutil.rmtree(package_tree / "nixl_cu13/lib")
    run_import(
        package_tree,
        "import nixl\n"
        "try:\n    import nixl.cute\n"
        "except ImportError as exc:\n"
        "    assert 'nixl_cu13' in str(exc) and '-Dbuild_cute=true' in str(exc)\n"
        "else:\n    raise AssertionError('Missing selected-backend bitcode')\n",
    )


def test_libnvvm_lifetime_markers():
    clean_ir = runpy.run_path(str(ROOT / "src/bindings/cute/build.py"))["clean_ir"]
    new = "  call void @llvm.lifetime.start.p0(ptr nonnull %buf)\n"
    declaration = "declare void @llvm.lifetime.end.p0(ptr captures(none))\n"
    old = "  call void @llvm.lifetime.start.p0(i64 8, ptr %buf)\n"
    store = "  store i64 1, ptr %buf\n"
    assert clean_ir(new + store + declaration + old) == store + old


def test_libnvvm_keeps_pointer_alignment():
    clean_ir = runpy.run_path(str(ROOT / "src/bindings/cute/build.py"))["clean_ir"]
    signature = (
        "define align 8 ptr @f(ptr byval(i32) align 16 %p) local_unnamed_addr #0"
    )
    body = "  %v = load i32, ptr %p, align 16\n  ret ptr %p\n}\n"
    assert clean_ir(signature + " align 2 {\n" + body) == signature + " {\n" + body
