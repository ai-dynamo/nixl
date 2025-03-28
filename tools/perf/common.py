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

import torch
from dist_utils import dist_utils

from nixl._api import nixl_agent

log = logging.getLogger(__name__)


class NixlBuffer:
    """Can be sharded"""

    def __init__(
        self,
        size: int,
        mem_type: str,
        nixl_agent: nixl_agent,
        shards=2,
        fill_value=0,
        dtype: torch.dtype = torch.int8,
    ):
        self.nixl_agent = nixl_agent
        if mem_type in ("cuda", "vram"):
            device = "cuda"
        elif mem_type in ("cpu", "dram"):
            device = "cpu"
        else:
            raise ValueError(f"Unsupported memory type: {mem_type}")

        log.debug(
            f"[Rank {dist_utils.get_rank()}] Initializing NixlBuffer with size {size}, device {device}, shards {shards}, fill_value {fill_value}"
        )
        self.bufs = []
        chunk_size = size // shards
        self.bufs = [
            torch.full((chunk_size,), fill_value=fill_value, dtype=dtype, device=device)
            for _ in range(shards)
        ]
        if size % chunk_size != 0:
            self.bufs.append(
                torch.full(
                    (size % chunk_size,),
                    fill_value=fill_value,
                    dtype=dtype,
                    device=device,
                )
            )

        self.reg_descs = nixl_agent.get_reg_descs(self.bufs, mem_type=mem_type)
        self.xfer_descs = nixl_agent.get_xfer_descs(self.bufs, mem_type=mem_type)

        log.debug(
            f"[Rank {dist_utils.get_rank()}] Registering memory for bufs {self.bufs}"
        )
        assert (
            nixl_agent.register_memory(self.reg_descs) is not None
        ), "Failed to register memory"

    def deregister(self):
        self.nixl_agent.deregister_memory(self.reg_descs)
