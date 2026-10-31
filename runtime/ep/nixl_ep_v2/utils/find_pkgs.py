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
import os
import warnings


def _package_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warn_if_rdma_headers_missing():
    include_paths = [
        path
        for env_name in ("CPATH", "CPLUS_INCLUDE_PATH")
        for path in os.environ.get(env_name, "").split(os.pathsep)
        if path
    ]
    include_paths += ["/usr/local/include", "/usr/include"]
    if not any(
        os.path.isfile(os.path.join(path, "infiniband", "mlx5dv.h"))
        for path in include_paths
    ):
        warnings.warn(
            "rdma-core header infiniband/mlx5dv.h was not found; inter-node NIXL EP puts will be unavailable",
            RuntimeWarning,
            stacklevel=2,
        )


@functools.lru_cache()
def find_nixl_root():
    """
    Find the packaged NIXL device headers (`NIXL_EP_NIXL_ROOT_DIR` overrides the package root), cached.
    """
    root = os.environ.get("NIXL_EP_NIXL_ROOT_DIR", _package_root())
    header = os.path.join("gpu", "nixl_device.cuh")
    assert os.path.isfile(
        os.path.join(root, "include", header)
    ), f"NIXL_EP_NIXL_ROOT_DIR={root} has no include/{header}"
    return root


@functools.lru_cache()
def find_ucx_root():
    """
    Find the packaged UCX device headers (`NIXL_EP_UCX_ROOT_DIR` overrides the package root), cached.
    """

    def has_device_api(root: str) -> bool:
        return os.path.isfile(
            os.path.join(root, "include", "ucp", "api", "device", "ucp_device_impl.h")
        )

    root = os.environ.get("NIXL_EP_UCX_ROOT_DIR", _package_root())
    assert has_device_api(
        root
    ), f"NIXL_EP_UCX_ROOT_DIR={root} has no UCX device API headers"
    _warn_if_rdma_headers_missing()
    return root


def find_nixl_lib_dir(nixl_root: str) -> str:
    """
    The directory holding `libnixl.so` inside a NIXL prefix (`lib` or a multiarch `lib/<triple>`).
    """
    import glob

    candidates = (
        glob.glob(os.path.join(nixl_root, "lib", "libnixl.so"))
        + glob.glob(os.path.join(nixl_root, "lib", "*-linux-gnu", "libnixl.so"))
        + glob.glob(os.path.join(nixl_root, "lib64", "libnixl.so"))
    )
    assert candidates, f"No libnixl.so under {nixl_root}"
    return os.path.dirname(candidates[0])
