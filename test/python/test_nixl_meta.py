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

import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NIXL_META_INIT = (
    Path(__file__).parents[2] / "src/bindings/python/nixl-meta/nixl/__init__.py"
)


def test_missing_backends_chain_first_import_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "nixl_meta_utils",
        SimpleNamespace(detect_cuda_major=lambda: None, detect_rocm=lambda: False),
    )
    errors = []

    def missing_backend(name):
        error = ModuleNotFoundError(f"No module named '{name}'", name=name)
        errors.append(error)
        raise error

    monkeypatch.setattr(importlib, "import_module", missing_backend)

    candidates = ["nixl_cu13", "nixl_cu12", "nixl_rocm"]
    with pytest.raises(ImportError, match=f"tried {', '.join(candidates)}") as exc_info:
        runpy.run_path(str(NIXL_META_INIT))

    assert [error.name for error in errors] == candidates
    assert exc_info.value.__cause__ is errors[0]
