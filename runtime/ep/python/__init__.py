# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
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
