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

log = logging.getLogger(__name__)


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
        return senders_ranks
