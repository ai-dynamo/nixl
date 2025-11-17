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
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional

import numpy as np
import torch


@dataclass
class TrafficPattern:
    """Represents a communication pattern between distributed processes.

    Attributes:
        matrix: Communication matrix as numpy array
        mem_type: Type of memory to use
        xfer_op: Transfer operation type
        shards: Number of shards for distributed processing
        dtype: PyTorch data type for the buffers
        sleep_before_launch_sec: Number of seconds to sleep before launch
        sleep_after_launch_sec: Number of seconds to sleep after launch
        id: Unique identifier for this traffic pattern
    """

    matrix: np.ndarray
    mem_type: Literal["cuda", "vram", "cpu", "dram"]
    xfer_op: Literal["WRITE", "READ"] = "WRITE"
    shards: int = 1
    dtype: torch.dtype = torch.int8
    sleep_before_launch_sec: Optional[int] = None
    sleep_after_launch_sec: Optional[int] = None

    id: int = field(default_factory=lambda: TrafficPattern._get_next_id())
    _id_counter: ClassVar[int] = 0

    @classmethod
    def _get_next_id(cls) -> int:
        """Get the next available ID and increment the counter"""
        current_id = cls._id_counter
        cls._id_counter += 1
        return current_id

    def senders_ranks(self, world_size: Optional[int] = None):
        """Return the ranks (process indices) that send messages.

        If world_size is provided, only indices < world_size are returned
        (storage endpoints are ignored).
        """
        senders_ranks = []
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    senders_ranks.append(i)
                    break
        ranks = list(set(senders_ranks))
        if world_size is not None:
            ranks = [r for r in ranks if r < world_size]
        return ranks

    def receivers_ranks(
        self, from_ranks: Optional[list[int]] = None, world_size: Optional[int] = None
    ):
        """Return the ranks (process indices) that receive messages.

        If world_size is provided, only indices < world_size are returned
        (storage endpoints are ignored).
        """
        if from_ranks is None:
            from_ranks = list(range(self.matrix.shape[0]))
        receivers_ranks = []
        for i in from_ranks:
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    receivers_ranks.append(j)
                    break
        ranks = list(set(receivers_ranks))
        if world_size is not None:
            ranks = [r for r in ranks if r < world_size]
        return ranks

    def ranks(self):
        """Return all ranks that are involved in the traffic pattern"""
        return list(set(self.senders_ranks() + self.receivers_ranks()))

    def buf_size(self, src, dst):
        return self.matrix[src, dst]

    def total_src_size(self, rank):
        """Return the total size sent by <rank> across all destinations."""
        total_src_size = 0
        # iterate over columns (destinations)
        for dst in range(self.matrix.shape[1]):
            total_src_size += self.matrix[rank][dst]
        return total_src_size

    def total_src_size_to_ranks(self, rank: int, world_size: int) -> int:
        """Return the total size sent by <rank> to rank destinations only.

        Only columns < world_size are considered (storage columns are ignored).
        """
        total_size = 0
        for dst in range(world_size):
            total_size += int(self.matrix[rank][dst])
        return total_size

    def total_dst_size(self, rank):
        """Return the total size received by <rank> across all sources."""
        total_dst_size = 0
        # iterate over rows (sources)
        for src in range(self.matrix.shape[0]):
            total_dst_size += self.matrix[src][rank]
        return total_dst_size

    def total_dst_size_from_ranks(self, rank: int, world_size: int) -> int:
        """Return total size received by <rank> from rank sources only."""
        total_size = 0
        for src in range(world_size):
            total_size += int(self.matrix[src][rank])
        return total_size
