# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import functools
import os


def _find_header_root(env_name, header):
    root = os.environ.get(
        env_name, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if not os.path.isfile(os.path.join(root, "include", header)):
        raise ModuleNotFoundError(
            f"{env_name}={root} has no include/{header}: install the matching NIXL EP wheel or set {env_name}"
        )
    return root


@functools.lru_cache()
def find_nixl_root():
    """Use packaged NIXL device headers unless explicitly overridden."""
    return _find_header_root("NIXL_EP_NIXL_ROOT_DIR", "gpu/nixl_device.cuh")


@functools.lru_cache()
def find_ucx_root():
    """Use packaged UCX device headers unless explicitly overridden."""
    return _find_header_root("NIXL_EP_UCX_ROOT_DIR", "ucp/api/device/ucp_device_impl.h")
