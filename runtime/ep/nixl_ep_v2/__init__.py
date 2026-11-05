# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import importlib
import os
import subprocess
import sys
from contextlib import contextmanager

import torch


def _deepbind_enabled() -> bool:
    value = os.getenv("NIXL_UCX_DEEPBIND", "0").strip().lower()
    return value not in {"0", "false", "no", "off", "disable", "disabled"}


@contextmanager
def _rtld_deepbind_import():
    deepbind = getattr(os, "RTLD_DEEPBIND", 0)
    if not deepbind or not _deepbind_enabled() or not hasattr(sys, "getdlopenflags"):
        yield
        return

    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | deepbind)
    try:
        yield
    finally:
        sys.setdlopenflags(old_flags)


# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs

    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass


with _rtld_deepbind_import():
    _torch_mm = "".join(torch.__version__.split(".")[:2])
    _C = importlib.import_module(f".nixl_ep_v2_cpp_torch{_torch_mm}", __package__)
sys.modules[f"{__package__}._C"] = _C

from .utils.find_pkgs import find_nixl_root, find_ucx_root  # noqa: E402

if requested_topk_idx_bits := os.getenv("NIXL_EP_NUM_TOPK_IDX_BITS"):
    compiled_topk_idx_bits = torch.iinfo(_C.topk_idx_t).bits
    if int(requested_topk_idx_bits) != compiled_topk_idx_bits:
        raise RuntimeError(
            f"NIXL_EP_NUM_TOPK_IDX_BITS={requested_topk_idx_bits}, but the extension uses {compiled_topk_idx_bits}"
        )


# Initialize
@functools.lru_cache()
def find_cuda_home() -> str:
    """
    Find the CUDA installation directory, cached.

    Returns:
        cuda_home: the CUDA installation path.
    """
    # TODO: reuse PyTorch API later
    # For some PyTorch versions, the original `_find_cuda_home` will initialize CUDA, which is incompatible with process forks
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # noinspection PyBroadException
        try:
            with open(os.devnull, "w") as devnull:
                nvcc = (
                    subprocess.check_output(["which", "nvcc"], stderr=devnull)
                    .decode()
                    .rstrip("\r\n")
                )
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


def init_jit():
    """
    Initialize the JIT compilation runtime. Sets up the CUDA, NIXL and UCX root paths for the JIT compiler.
    """
    library_root_path = os.path.dirname(os.path.abspath(__file__))
    _C.init_jit(
        library_root_path,  # Library root directory path
        find_cuda_home(),  # CUDA home
        find_nixl_root(),  # NIXL root
        find_ucx_root(),  # UCX root
    )


# Run initialization
init_jit()


# noinspection PyUnresolvedReferences
from ._C import topk_idx_t  # noqa: F401,E402

# Import APIs after initialization
from .buffers.elastic import ElasticBuffer, EPHandle  # noqa: F401,E402
from .utils.envs import (  # noqa: F401,E402
    get_logical_domain_size,
    get_physical_domain_size,
)

# noinspection PyUnresolvedReferences
from .utils.event import EventHandle, EventOverlap  # noqa: F401,E402

__version__ = "2.1.0"
