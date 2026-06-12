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

"""
NIXL Python Examples - Shared Utilities

This package provides utilities for NIXL Python examples:
- tcp_server: Simple TCP-based metadata exchange server
- memory_utils: Memory read/write helpers
- metadata_utils: Metadata exchange helpers (TCP and NIXL built-in)
"""

from .memory_utils import read_uint64, write_uint64
from .metadata_utils import (
    publish_agent_metadata,
    publish_descriptors,
    retrieve_agent_metadata,
    retrieve_descriptors,
)
from .tcp_server import clear_metadata, get_metadata, set_metadata, start_server

__all__ = [
    # Memory utilities
    "read_uint64",
    "write_uint64",
    # Metadata utilities
    "publish_agent_metadata",
    "retrieve_agent_metadata",
    "publish_descriptors",
    "retrieve_descriptors",
    # TCP server
    "start_server",
    "set_metadata",
    "get_metadata",
    "clear_metadata",
]
