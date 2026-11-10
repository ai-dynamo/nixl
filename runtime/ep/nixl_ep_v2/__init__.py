# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import functools
import importlib
import os
import shutil
import sys
from contextlib import contextmanager

import torch


@contextmanager
def _rtld_deepbind_import():
    value = os.getenv("NIXL_UCX_DEEPBIND", "0").strip().lower()
    deepbind = getattr(os, "RTLD_DEEPBIND", 0)
    if (
        not deepbind
        or value in {"0", "false", "no", "off", "disable", "disabled"}
        or not hasattr(sys, "getdlopenflags")
    ):
        yield
        return
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | deepbind)
    try:
        yield
    finally:
        sys.setdlopenflags(old_flags)


with _rtld_deepbind_import():
    _torch_mm = "".join(torch.__version__.split(".")[:2])
    _C = importlib.import_module(f".nixl_ep_v2_cpp_torch{_torch_mm}", __package__)
sys.modules[f"{__package__}._C"] = _C

from .utils.find_pkgs import find_nixl_root, find_ucx_root  # noqa: E402


@functools.lru_cache()
def find_cuda_home() -> str:
    """Find CUDA without initializing its runtime, preserving process-fork safety."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        nvcc = shutil.which("nvcc")
        if nvcc is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        elif os.path.exists("/usr/local/cuda"):
            cuda_home = "/usr/local/cuda"
    assert cuda_home is not None
    return cuda_home


def init_jit():
    # noinspection PyUnresolvedReferences
    library_root_path = os.path.dirname(os.path.abspath(__file__))
    _C.init_jit(
        library_root_path,
        find_cuda_home(),
        find_nixl_root(),
        find_ucx_root(),
    )


init_jit()


# noinspection PyUnresolvedReferences
from ._C import topk_idx_t as topk_idx_t  # noqa: E402,F401

# Import APIs after initialization
from .buffers.elastic import ElasticBuffer as ElasticBuffer  # noqa: E402,F401
from .buffers.elastic import EPHandle as EPHandle  # noqa: E402,F401
from .utils.envs import (  # noqa: E402,F401
    get_logical_domain_size as get_logical_domain_size,
)

# noinspection PyUnresolvedReferences
from .utils.event import EventHandle as EventHandle  # noqa: E402,F401
from .utils.event import EventOverlap as EventOverlap  # noqa: E402,F401

__all__ = [
    "ElasticBuffer",
    "EPHandle",
    "EventHandle",
    "EventOverlap",
    "topk_idx_t",
    "get_logical_domain_size",
]
