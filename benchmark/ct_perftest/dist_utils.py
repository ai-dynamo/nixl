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
import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, final, Dict, Optional
from enum import Enum

from nixl._api import nixl_agent

try:
    import torch
    import torch.distributed as dist

    has_torch = True
except ImportError:
    has_torch = False

log = logging.getLogger(__name__)


class ReduceOp(Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class _DistUtils(ABC):
    """Allow for different distributed backends"""

    @final
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


class _TorchDistUtils(_DistUtils):
    def get_rank(self) -> int:
        return dist.get_rank()

    def get_world_size(self) -> int:
        return dist.get_world_size()

    def init_group(self, ranks: List[int]):
        """Initialize a group of ranks

        Args:
            ranks: List of ranks to initialize a group for
        """
        if not ranks:
            return None

        key = tuple(sorted(ranks))
        if key not in self.groups:
            self.groups[key] = dist.new_group(ranks)
        return key

    def init_dist(self) -> Tuple[int, int]:
        """Init torch distributed module

        Returns:
            Tuple[int, int]: Tuple of (rank, world_size)

        Raises:
            ValueError: If rank and world size cannot be determined
            RuntimeError: If CUDA is not available
        """
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()

        log.debug("Initializing torch distributed module")
        if os.environ.get("SLURM_PROCID"):
            rank_str = os.environ.get("SLURM_PROCID", "")
            world_size_str = os.environ.get("SLURM_NTASKS", "")
        elif os.environ.get("RANK"):
            rank_str = os.environ.get("RANK", "")
            world_size_str = os.environ.get("WORLD_SIZE", "")
        else:
            raise ValueError("Could not parse rank and world size")

        if not rank_str.isdigit() or not world_size_str.isdigit():
            raise ValueError("Could not parse rank and world size")

        rank: int = int(rank_str)
        world_size: int = int(world_size_str)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

        rank = dist.get_rank()

        if torch.cuda.device_count() == 0:
            print(
                "No CUDA device have been detected, maybe you forgot to add --gpus-per-node option in srun?"
            )
            return rank, world_size

        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        log.debug(f"[Rank {rank}] Using CUDA device {device}")

        return rank, world_size

    def barrier(self, ranks: Optional[List[int]] = None):
        """Barrier for a group of ranks

        Args:
            ranks: List of ranks to barrier, if None, barrier for all ranks
        """
        if ranks is None:
            dist.barrier()
            return

        if self.get_rank() not in ranks:
            return

        key = tuple(sorted(ranks))
        group = self.groups.get(key)
        if group is None:
            raise ValueError(
                f"[Rank {self.get_rank()}] Group with ranks {ranks} was not created"
            )

        dist.barrier(group=group)

    def destroy_dist(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def all_reduce(self, vals: List[float | int], op: ReduceOp) -> List[float | int]:
        val_tensor = torch.tensor(vals, device=torch.device("cuda"))
        if op == ReduceOp.SUM:
            op = dist.ReduceOp.SUM
        elif op == ReduceOp.AVG:
            op = dist.ReduceOp.AVG
        elif op == ReduceOp.MIN:
            op = dist.ReduceOp.MIN
        elif op == ReduceOp.MAX:
            op = dist.ReduceOp.MAX
        dist.all_reduce(val_tensor, op=op)
        return val_tensor.tolist()

    def allgather_obj(self, obj: Any) -> List[Any]:
        """Allgather arbitrary object on world

        Args:
            obj: Object to gather from all ranks

        Returns:
            List[Any]: List of gathered objects, one from each rank
        """
        to = [None for _ in range(self.get_world_size())]
        dist.all_gather_object(to, obj)
        return to

    def alltoall_obj(self, send_objs: List[Any]) -> List[Any]:
        """All-to-all communication of arbitrary objects on world

        Args:
            send_objs: List of objects to send, length must equal world_size

        Returns:
            List[Any]: List of received objects

        Raises:
            AssertionError: If length of send_objs doesn't match world_size
        """
        world_size = self.get_world_size()

        assert (
            len(send_objs) == world_size
        ), f"Invalid number of objects {len(send_objs)}, expected {world_size}"

        recv_objs = [None for _ in range(len(send_objs))]

        for other_rank in range(world_size):
            output = [None]
            dist.scatter_object_list(
                scatter_object_output_list=output,
                scatter_object_input_list=send_objs,
                src=other_rank,
            )
            recv_objs[other_rank] = output[0]

        return recv_objs


dist_utils = _TorchDistUtils()
dist_utils.init_dist()
