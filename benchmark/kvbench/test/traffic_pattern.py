# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import ClassVar, Dict, Literal, Optional

import numpy as np
import torch


@dataclass
class StorageOp:
    """Storage operation configuration for a single rank."""

    file_path: str
    file_size: int
    read_offset: int
    read_size: int
    write_offset: int
    write_size: int


@dataclass
class TrafficPattern:
    """Represents a communication pattern between distributed processes.

    Attributes:
        matrix: Communication matrix as numpy array (optional, None for storage-only)
        mem_type: Type of memory to use
        xfer_op: Transfer operation type
        shards: Number of shards for distributed processing
        dtype: PyTorch data type for the buffers
        sleep_before_launch_sec: Number of seconds to sleep before launch
        sleep_after_launch_sec: Number of seconds to sleep after RDMA
        storage_ops: Per-rank storage operations (loaded from external config)
        id: Unique identifier for this traffic pattern
    """

    mem_type: Literal["cuda", "vram", "cpu", "dram"]
    matrix: Optional[np.ndarray] = None  # None for storage-only patterns
    xfer_op: Literal["WRITE", "READ"] = "WRITE"
    shards: int = 1
    dtype: torch.dtype = torch.int8
    sleep_before_launch_sec: Optional[float] = None
    sleep_after_launch_sec: Optional[float] = None
    storage_ops: Optional[Dict[int, StorageOp]] = None  # rank -> StorageOp

    id: int = field(default_factory=lambda: TrafficPattern._get_next_id())
    _id_counter: ClassVar[int] = 0

    # Cached rank lists (computed once in __post_init__, matrix is immutable)
    _senders: list = field(default_factory=list, init=False, repr=False)
    _receivers: list = field(default_factory=list, init=False, repr=False)
    _all_participating: list = field(default_factory=list, init=False, repr=False)

    @classmethod
    def _get_next_id(cls) -> int:
        """Get the next available ID and increment the counter"""
        current_id = cls._id_counter
        cls._id_counter += 1
        return current_id

    def __post_init__(self):
        """Pre-compute and cache rank lists from the immutable matrix."""
        if self.matrix is not None:
            senders = set()
            receivers = set()
            for i in range(self.matrix.shape[0]):
                for j in range(self.matrix.shape[1]):
                    if self.matrix[i, j] > 0:
                        senders.add(i)
                        receivers.add(j)
            self._senders = sorted(senders)
            self._receivers = sorted(receivers)
        else:
            self._senders = []
            self._receivers = []

        # All participating ranks: senders + receivers + storage ranks
        all_ranks = set(self._senders + self._receivers)
        if self.storage_ops:
            all_ranks.update(self.storage_ops.keys())
        self._all_participating = sorted(all_ranks)

    def has_rdma(self) -> bool:
        """Check if this traffic pattern has any RDMA traffic."""
        return len(self._senders) > 0

    def senders_ranks(self):
        """Return the ranks (process indices) that send messages."""
        return self._senders

    def receivers_ranks(self, from_ranks: Optional[list[int]] = None):
        """Return the ranks (process indices) that receive messages."""
        if from_ranks is None:
            return self._receivers
        # Filtered case: only receivers that receive from specified senders
        if self.matrix is None:
            return []
        result = set()
        for i in from_ranks:
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    result.add(j)
        return sorted(result)

    def ranks(self):
        """Return all ranks that are involved in RDMA traffic"""
        return sorted(set(self._senders + self._receivers))

    def all_participating_ranks(self):
        """Return all ranks that actively participate in this TP.

        Includes:
        - RDMA senders (they initiate transfers)
        - RDMA receivers (they wait for notifications and participate in barriers)
        - Storage ranks (they do read/write ops)
        """
        return self._all_participating

    def buf_size(self, src, dst):
        if self.matrix is None:
            return 0
        return self.matrix[src, dst]

    def total_src_size(self, rank):
        """Return the total size sent by <rank> across all destinations."""
        if self.matrix is None:
            return 0
        total_src_size = 0
        # iterate over columns (destinations)
        for dst in range(self.matrix.shape[1]):
            total_src_size += self.matrix[rank][dst]
        return total_src_size

    def total_dst_size(self, rank):
        """Return the total size received by <rank> across all sources."""
        if self.matrix is None:
            return 0
        total_dst_size = 0
        # iterate over rows (sources)
        for src in range(self.matrix.shape[0]):
            total_dst_size += self.matrix[src][rank]
        return total_dst_size
