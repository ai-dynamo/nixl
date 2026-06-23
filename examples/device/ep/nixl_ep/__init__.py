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

import importlib
import os
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


with _rtld_deepbind_import():
    _nixl_ep_cpp = importlib.import_module(f"{__name__}.nixl_ep_cpp")
nixl_ep_cpp = _nixl_ep_cpp

from .buffer import Buffer  # noqa: E402
from .utils import EventOverlap  # noqa: E402

topk_idx_t = getattr(_nixl_ep_cpp, "topk_idx_t", torch.int64)
Config = _nixl_ep_cpp.Config

__all__ = ["Buffer", "EventOverlap", "Config"]
