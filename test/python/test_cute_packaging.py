# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise package imports without requiring CUDA or a CuTe installation."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

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
    (tmp_path / "nixl_meta_utils.py").write_text(
        "import os\n"
        "def detect_cuda_major(): return int(os.environ['TEST_CUDA_MAJOR'])\n"
        "def detect_rocm(): return False\n"
    )
    for major in (12, 13):
        backend = tmp_path / f"nixl_cu{major}"
        (backend / "lib").mkdir(parents=True)
        (backend / "lib/libnixl_device.bc").write_bytes(b"test bitcode")
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
    (cutlass / "__init__.py").write_text("Uint64 = Uint32 = Int32 = int\n")
    (cutlass / "cute.py").write_text(
        "BitCode = str\ndef extern(**kwargs): return lambda function: function\n"
    )
    return tmp_path


def run_import(package_tree, code, major=13):
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=package_tree,
        env={
            **os.environ,
            "PYTHONPATH": str(package_tree),
            "TEST_CUDA_MAJOR": str(major),
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
        "from pathlib import Path\n"
        "assert cute._bitcode == str(Path(_bindings.__file__).parent / 'lib/libnixl_device.bc')\n"
        "assert cute.THREAD == int(_bindings.nixl_gpu_level_t.THREAD)\n"
        "assert cute.WARP == int(_bindings.nixl_gpu_level_t.WARP)\n"
        "assert cute.BLOCK == int(_bindings.nixl_gpu_level_t.BLOCK)\n"
        "assert cute.DEFER == _bindings.NIXL_GPU_DEFER\n"
        "assert cute.REQUEST_BYTES == _bindings.NIXL_GPU_REQUEST_BYTES\n"
        "import nixl.cute\n"
        "assert nixl.cute is cute\n",
        major,
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
    (package_tree / "nixl_cu13/lib/libnixl_device.bc").unlink()
    run_import(
        package_tree,
        "import nixl\n"
        "try:\n    import nixl.cute\n"
        "except ImportError as exc:\n"
        "    assert 'nixl_cu13' in str(exc) and '-Dbuild_cute=true' in str(exc)\n"
        "else:\n    raise AssertionError('Missing selected-backend bitcode')\n",
    )
