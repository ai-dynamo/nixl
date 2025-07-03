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
import uuid
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from dist_utils import dist_utils
from collections import defaultdict

from nixl._api import nixl_agent


log = logging.getLogger(__name__)

class NixlHandle:
    def __init__(self, remote_rank, handle, traffic_pattern):
        self.remote_rank = remote_rank
        self.handle = handle
        self.tp = traffic_pattern

    def __str__(self):
        return f"to:{self.remote_rank}"


class NixlBuffer:
    """Can be sharded"""

    def __init__(
        self,
        size: int,
        mem_type: str,
        nixl_agent: nixl_agent,
        shards=1,
        fill_value=0,
        dtype: torch.dtype = torch.int8,
    ):
        self.size = size
        self.nixl_agent = nixl_agent
        if mem_type in ("cuda", "vram"):
            device = torch.device('cuda')
        elif mem_type in ("cpu", "dram"):
            device = torch.device('cpu')
        else:
            raise ValueError(f"Unsupported memory type: {mem_type}")
        
        if shards > 1:
            raise ValueError("Sharding is not supported yet")

        log.debug(
            f"[Rank {dist_utils.get_rank()}] Initializing NixlBuffer with size {size}, device {device}, shards {shards}, fill_value {fill_value}"
        )
        self.buf = torch.full((size,), fill_value, dtype=dtype, device=device)
        
        log.debug(f"[Rank {dist_utils.get_rank()}] Registering memory for buffer {self.buf}")
        self.reg_descs = nixl_agent.get_reg_descs(self.buf)
        assert (
            nixl_agent.register_memory(self.reg_descs) is not None
        ), "Failed to register memory"
    
    def get_chunk(self, size, offset):
        if offset + size > self.size:
            raise ValueError(f"Offset {offset} + size {size} is greater than buffer size {self.size}")
        return self.buf[offset:offset+size]

    def destroy(self):
        self.nixl_agent.deregister_memory(self.reg_descs)
        if hasattr(self.buf, "is_cuda") and self.buf.is_cuda:
            del self.buf
            torch.cuda.empty_cache()


@dataclass
class TrafficPattern:
    """Represents a communication pattern between distributed processes.

    Attributes:
        matrix_file: Path to the file containing the communication matrix
        shards: Number of shards for distributed processing
        mem_type: Type of memory to use
        xfer_op: Transfer operation type
        dtype: PyTorch data type for the buffers
        sleep_sec: Number of seconds to sleep after finish
        id: Unique identifier for this traffic pattern
    """

    matrix: np.ndarray
    mem_type: Literal["cuda", "vram", "cpu", "dram"]
    xfer_op: Literal["WRITE", "READ"] = "WRITE"
    shards: int = 1
    dtype: torch.dtype = torch.int8
    sleep_before_launch_sec: int = 0
    sleep_after_launch_sec: int = 0

    id: str = str(uuid.uuid4())

    def senders_ranks(self):
        """Return the ranks that send messages"""
        senders_ranks = []
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    senders_ranks.append(i)
                    break
        return list(set(senders_ranks))
    
    def receivers_ranks(self):
        """Return the ranks that receive messages"""
        receivers_ranks = []
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    receivers_ranks.append(j)
                    break
        return list(set(receivers_ranks))
    
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
