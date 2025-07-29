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
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


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

    def senders_ranks(self):
        """Return the ranks that send messages"""
        senders_ranks = []
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    senders_ranks.append(i)
                    break
        return list(set(senders_ranks))

    def receivers_ranks(self, from_ranks: Optional[list[int]] = None):
        """Return the ranks that receive messages"""
        if from_ranks is None:
            from_ranks = list(range(self.matrix.shape[0]))
        receivers_ranks = []
        for i in from_ranks:
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    receivers_ranks.append(j)
                    break
        return list(set(receivers_ranks))

    def ranks(self):
        """Return all ranks that are involved in the traffic pattern"""
        return list(set(self.senders_ranks() + self.receivers_ranks()))

    def buf_size(self, src, dst):
        return self.matrix[src, dst]

    def total_src_size(self, rank):
        """Return the sum of the sizes received by <rank>"""
        total_src_size = 0
        for other_rank in range(self.matrix.shape[0]):
            total_src_size += self.matrix[rank][other_rank]
        return total_src_size

    def total_dst_size(self, rank):
        """Return the sum of the sizes received by <rank>"""
        total_dst_size = 0
        for other_rank in range(self.matrix.shape[0]):
            total_dst_size += self.matrix[other_rank][rank]
        return total_dst_size
