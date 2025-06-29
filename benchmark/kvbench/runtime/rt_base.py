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

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from nixl._api import nixl_agent

log = logging.getLogger(__name__)


class ReduceOp(Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class _RTUtils(ABC):
    """Allow for different distributed backends"""

    def __init__(self):
        # Key is tuple of the sorted ranks, value is backend group
        self.groups: Dict[Tuple[int, ...], Any] = {}

    def init_dist(self):
        pass

    def destroy_dist(self):
        """Cleanup distributed process group"""
        pass

    @abstractmethod
    def barrier(self, ranks: Optional[List[int]] = None):
        """Barrier for a group of ranks

        Args:
            ranks: List of ranks to barrier, if None, barrier for all ranks
        """
        pass

    @abstractmethod
    def allgather_obj(self, obj: Any) -> List[Any]:
        """Allgather arbitrary object on world

        Args:
            obj: Object to gather from all ranks

        Returns:
            List[Any]: List of gathered objects, one from each rank
        """
        pass

    @abstractmethod
    def alltoall_obj(self, send_objs: List[Any]) -> List[Any]:
        """All-to-all communication of arbitrary objects on world

        Args:
            send_objs: List of objects to send, length must equal world_size

        Returns:
            List[Any]: List of received objects

        Raises:
            AssertionError: If length of send_objs doesn't match world_size
        """
        pass

    @abstractmethod
    def all_reduce(self, vals: List[float | int], op: ReduceOp) -> List[float | int]:
        pass

    @abstractmethod
    def get_rank(self) -> int:
        pass

    @abstractmethod
    def get_world_size(self) -> int:
        pass

    def init_group(self, ranks: List[int]):
        pass

    def share_world_metadata(self, nixl_agent: "nixl_agent") -> None:
        my_rank = self.get_rank()

        log.debug(f"[Rank {my_rank}] Sharing agent metadata with other ranks")
        md = nixl_agent.get_agent_metadata()
        world_mds = self.allgather_obj(md)
        for other_rank, metadata in enumerate(world_mds):
            if other_rank == my_rank:
                continue
            nixl_agent.add_remote_agent(metadata)
            log.debug(f"[Rank {my_rank}] Added remote agent {other_rank}'s metadata")

    def init_groups(self, groups_ranks: List[List[int]]):
        """Initialize groups of ranks

        Args:
            groups_ranks: List of lists of ranks, each list is a group
        """
        for ranks in groups_ranks:
            self.init_group(ranks)
