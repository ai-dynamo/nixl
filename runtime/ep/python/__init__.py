# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""NIXL EP API shell: meta placeholders and local state, with no native runtime."""
import torch

from .buffers.elastic import ElasticBuffer, EPHandle
from .utils.event import EventHandle, EventOverlap

topk_idx_t = torch.int64


def init_jit():
    """No-op retained for API compatibility; this shell has no compiler or kernels."""
    pass


__all__ = [
    "ElasticBuffer",
    "EPHandle",
    "EventHandle",
    "EventOverlap",
    "topk_idx_t",
]
