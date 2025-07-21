# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum
from typing import Any, List, Optional


class ReduceOp(Enum):
    """Reduction operations for distributed computing"""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class _RTUtils:
    """Base class for distributed runtime utilities"""

    def __init__(self):
        pass

    def get_rank(self) -> int:
        """Get the rank of the current process"""
        raise NotImplementedError("Subclasses must implement get_rank")

    def get_world_size(self) -> int:
        """Get the total number of processes"""
        raise NotImplementedError("Subclasses must implement get_world_size")

    def barrier(self, ranks: Optional[List[int]] = None, timeout_sec: int = 600):
        """Synchronization barrier for processes"""
        raise NotImplementedError("Subclasses must implement barrier")

    def allgather_obj(self, obj: Any) -> List[Any]:
        """All-gather operation for objects"""
        raise NotImplementedError("Subclasses must implement allgather_obj")

    def alltoall_obj(self, send_objs: List[Any]) -> List[Any]:
        """All-to-all operation for objects"""
        raise NotImplementedError("Subclasses must implement alltoall_obj")

    def all_reduce(
        self, vals: List[float | int], op: ReduceOp, root: int = 0
    ) -> List[float | int]:
        """All-reduce operation with specified reduction operation"""
        raise NotImplementedError("Subclasses must implement all_reduce")

    def destroy_dist(self):
        """Clean up distributed resources"""
        raise NotImplementedError("Subclasses must implement destroy_dist")

    def _get_group_id(self, ranks: List[int]) -> int:
        """Get the id for a group of ranks"""
        key = tuple(sorted(ranks))
        return hash(key)
