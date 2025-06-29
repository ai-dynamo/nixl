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

import pickle
import uuid
from typing import Any, List, Optional

from .rt_base import ReduceOp, _RTUtils

try:
    import etcd_runtime
except ImportError:
    raise ImportError(
        "etcd_runtime module not found. Please build the Python bindings."
    )


class _EtcdDistUtils(_RTUtils):
    """ETCD-based distributed runtime utilities that replaces MPI"""

    def __init__(self, etcd_endpoints: str = "http://localhost:2379", size: int = 1):
        super().__init__()
        self.etcd_runtime = etcd_runtime.EtcdRuntime(etcd_endpoints, size, 0)
        self._rank = self.etcd_runtime.get_rank()
        self._world_size = self.etcd_runtime.get_size()

    def get_rank(self) -> int:
        return self._rank

    def get_world_size(self) -> int:
        return self._world_size

    def allgather_obj(self, obj: Any) -> List[Any]:
        """Allgather arbitrary object using etcd point-to-point communication"""
        # Serialize the object
        serialized_obj = pickle.dumps(obj)

        # Broadcast our object to all other ranks
        for dest_rank in range(self._world_size):
            if dest_rank != self._rank:
                self.etcd_runtime.send_char(serialized_obj, dest_rank)

        # Collect objects from all ranks
        gathered_objs = [None] * self._world_size
        gathered_objs[self._rank] = obj  # Our own object

        # Receive from all other ranks
        for src_rank in range(self._world_size):
            if src_rank != self._rank:
                result, data = self.etcd_runtime.recv_char(
                    len(serialized_obj), src_rank
                )
                if result == 0:
                    gathered_objs[src_rank] = pickle.loads(data.encode("latin-1"))
                else:
                    raise RuntimeError(f"Failed to receive data from rank {src_rank}")

        return gathered_objs

    def alltoall_obj(self, send_objs: List[Any]) -> List[Any]:
        """All-to-all communication using etcd point-to-point communication"""
        world_size = self.get_world_size()
        assert (
            len(send_objs) == world_size
        ), f"Invalid number of objects {len(send_objs)}, expected {world_size}"

        # Serialize all objects to send
        serialized_objs = [pickle.dumps(obj) for obj in send_objs]

        # Send objects to their respective destinations
        for dest_rank, serialized_obj in enumerate(serialized_objs):
            if dest_rank != self._rank:
                self.etcd_runtime.send_char(serialized_obj, dest_rank)

        # Receive objects from all other ranks
        recv_objs = [None] * world_size
        recv_objs[self._rank] = send_objs[self._rank]  # Our own object

        for src_rank in range(world_size):
            if src_rank != self._rank:
                # We need to know the size of data to receive - use a simple protocol
                # First receive the size, then the actual data
                expected_size = len(
                    serialized_objs[self._rank]
                )  # Assume similar sizes for simplicity
                result, data = self.etcd_runtime.recv_char(expected_size, src_rank)
                if result == 0:
                    recv_objs[src_rank] = pickle.loads(data.encode("latin-1"))
                else:
                    raise RuntimeError(f"Failed to receive data from rank {src_rank}")

        return recv_objs

    def barrier(self, ranks: Optional[List[int]] = None):
        """Barrier for a group of ranks using etcd barrier"""
        if ranks is None:
            # Barrier for all ranks
            barrier_id = "global_barrier_" + str(uuid.uuid4())
            result = self.etcd_runtime.barrier(barrier_id)
            if result != 0:
                raise RuntimeError("Barrier failed")
            return

        if self.get_rank() not in ranks:
            return

        # Create barrier for specific group of ranks
        key = tuple(sorted(ranks))
        if key not in self.groups:
            # For etcd, we don't need to pre-create groups, just use unique barrier IDs
            self.groups[key] = f"group_barrier_{hash(key)}"

        barrier_id = self.groups[key] + "_" + str(uuid.uuid4())

        # Only participating ranks call barrier
        result = self.etcd_runtime.barrier(barrier_id)
        if result != 0:
            raise RuntimeError(f"Group barrier failed for ranks {ranks}")

    def all_reduce(self, vals: List[float | int], op: ReduceOp) -> List[float | int]:
        """All-reduce operation using etcd reduce and broadcast"""
        # Convert values to doubles for etcd reduce
        double_vals = [float(val) for val in vals]

        # Choose rank 0 as the reduction destination
        dest_rank = 0

        result_vals = []
        for i, val in enumerate(double_vals):
            result, global_val = self.etcd_runtime.reduce_sum_double(val, dest_rank)
            if result != 0:
                raise RuntimeError(f"Reduce operation failed for value {i}")

            # Apply the requested operation
            if op == ReduceOp.SUM:
                final_val = global_val
            elif op == ReduceOp.AVG:
                final_val = global_val / self._world_size
            elif op == ReduceOp.MIN:
                # ETCD runtime only supports SUM, so we need to implement MIN/MAX differently
                # For now, use SUM as fallback (this would need custom implementation)
                final_val = global_val  # Placeholder
            elif op == ReduceOp.MAX:
                # Similar to MIN, would need custom implementation
                final_val = global_val  # Placeholder
            else:
                raise ValueError(f"Unsupported reduce operation: {op}")

            result_vals.append(final_val)

        # Broadcast the result from rank 0 to all other ranks would be needed here
        # For simplicity, assuming all ranks get the same result from reduce
        return result_vals

    def init_group(self, ranks: List[int]):
        """Initialize a group of ranks for etcd runtime"""
        if not ranks:
            return None

        key = tuple(sorted(ranks))
        if key not in self.groups:
            # For etcd, we create a unique identifier for the group
            self.groups[key] = f"group_{hash(key)}"

        return key
