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
import pickle
import re
import time
from collections import defaultdict
from typing import Any, List, Optional

import etcd3

from .rt_base import ReduceOp, _RTUtils

log = logging.getLogger(__name__)


def int_to_bytes(val: int) -> bytes:
    return int.to_bytes(val, length=4)


class _EtcdDistUtils(_RTUtils):
    """ETCD-based MPI utilities - NOT PERFORMANCE OPTIMIZED (for control path only)"""

    def __init__(
        self,
        etcd_endpoints: str = "http://localhost:2379",
        prefix: str = "/nixl/kvbench",
    ):
        super().__init__()
        self.prefix = prefix
        self.ops_counter: dict[str, dict[Any, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Initialize rank & world size
        if os.environ.get("SLURM_PROCID"):
            self.rank = int(os.environ["SLURM_PROCID"])
            self.world_size = int(os.environ["SLURM_NTASKS"])
        elif os.environ.get("OMPI_COMM_WORLD_RANK"):
            self.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            self.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        elif os.environ.get("RANK"):
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            raise ValueError(
                "Rank and world size not found in environment variables SLURM_PROCID/SLURM_NTASKS or RANK/WORLD_SIZE"
            )

        # Parse endpoint host & port
        url_pattern = r"^(https?://)?([^:]+)(?::(\d+))?$"
        match = re.match(url_pattern, etcd_endpoints)

        if match:
            protocol = match.group(1) or "http://"
            protocol = protocol.rstrip("://")
            host = match.group(2)
            port = int(match.group(3)) if match.group(3) else 2379
        else:
            raise ValueError(
                f"Invalid etcd endpoint format: {etcd_endpoints}, expected format is [http://]host[:port]"
            )

        log.info(f"ETCD client initialized with host {host} & port {port}")

        try:
            self.client = etcd3.client(host=host, port=port)
        except Exception as e:
            raise ValueError(f"Failed to initialize ETCD client: {e}")

        if self.rank == 0:
            log.info(f"Wiping ETCD prefix {self.prefix}")
            self.client.delete_prefix(self.prefix)

    def destroy_dist(self):
        self.client.delete_prefix(self.prefix)

    def _get_int_val(self, key: str) -> int | None:
        val = self.client.get(key)[0]
        if val is None:
            return None
        return int.from_bytes(val)

    def barrier(self, ranks: Optional[List[int]] = None, timeout_sec=600):
        """Barrier for a group of ranks using etcd barrier"""
        if ranks is None:
            ranks = list(range(self.world_size))

        if self.rank not in ranks:
            return

        ranks = sorted(ranks)
        root = ranks[0]
        # Create barrier for specific group of ranks
        group_id = self._get_group_id(ranks)
        group_size = len(ranks)
        barrier_ix = self.ops_counter["barrier"][group_id]
        self.ops_counter["barrier"][group_id] += 1

        key = f"{self.prefix}/barrier/{group_id}/{barrier_ix}"
        start_time = time.time()

        if self.rank == root:
            # Fan in - count from 1 to len(ranks)
            self.client.put(key, int_to_bytes(1))

            # Fan out - put it back to 0 to signal that all ranks have entered the barrier
            while not self.client.replace(
                key, int_to_bytes(len(ranks)), int_to_bytes(0)
            ):
                if timeout_sec and time.time() - start_time > timeout_sec:
                    raise TimeoutError(
                        f"[Rank {self.rank}] ROOT - Barrier {key} timed out after {timeout_sec} seconds, current value: {self.client.get(key)}, waiting for val={len(ranks)} (i.e all the ranks have entered the barrier), (ranks: {ranks})"
                    )
        else:
            my_index = ranks.index(self.rank)
            # Fan in - count from 1 to len(ranks)
            while not self.client.replace(
                key, int_to_bytes(my_index), int_to_bytes(my_index + 1)
            ):
                if timeout_sec and time.time() - start_time > timeout_sec:
                    raise TimeoutError(
                        f"[Rank {self.rank}] Barrier {key} timed out after {timeout_sec} seconds, current value: {self.client.get(key)}, waiting for val={my_index} (i.e rank {ranks[my_index - 1]} entered barrier), (ranks: {ranks})"
                    )
            # Fan out - wait for root to set 0 again
            while not self._get_int_val(key) == 0:
                if timeout_sec and time.time() - start_time > timeout_sec:
                    raise TimeoutError(
                        f"[Rank {self.rank}] Barrier {key} timed out after {timeout_sec} seconds, current value: {self.client.get(key)}, waiting for val={group_size} (i.e all the ranks have entered the barrier), (ranks: {ranks})"
                    )

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        return self.world_size

    def allgather_obj(self, obj: Any) -> List[Any]:
        allgather_ix = self.ops_counter["allgather"]["world"]
        self.ops_counter["allgather"]["world"] += 1

        result = [None for _ in range(self.world_size)]
        # Serialize the object
        serialized_obj = pickle.dumps(obj)

        self.client.put(
            f"{self.prefix}/allgather/{allgather_ix}/{self.rank}", serialized_obj
        )

        for dest_rank in range(self.world_size):
            val = None
            while val is None:
                val = self.client.get(
                    f"{self.prefix}/allgather/{allgather_ix}/{dest_rank}"
                )[0]

            result[dest_rank] = pickle.loads(val)

        return result

    def alltoall_obj(self, send_objs: List[Any]) -> List[Any]:
        result = [None for _ in range(self.world_size)]
        serialized_objs = [pickle.dumps(obj) for obj in send_objs]

        self.barrier()
        for dest_rank in range(self.world_size):
            self.client.put(
                f"{self.prefix}/alltoall/{self.rank}_to_{dest_rank}",
                serialized_objs[dest_rank],
            )

        self.barrier()
        for src_rank in range(self.world_size):
            val = self.client.get(f"{self.prefix}/alltoall/{src_rank}_to_{self.rank}")[
                0
            ]
            result[src_rank] = pickle.loads(val)

        return result

    def all_reduce(
        self, vals: List[float | int], op: ReduceOp, root: int = 0
    ) -> List[float | int]:
        self.barrier()
        self.client.put(f"{self.prefix}/all_reduce/{self.rank}", pickle.dumps(vals))
        if self.rank == root:
            self.client.delete(f"{self.prefix}/all_reduce/result")
        self.barrier()

        if self.rank == root:
            vals = []
            for dest_rank in range(self.world_size):
                val = self.client.get(f"{self.prefix}/all_reduce/{dest_rank}")[0]
                vals.append(pickle.loads(val))

            print(vals)
            if op == ReduceOp.SUM:
                final_val = [sum(col) for col in zip(*vals)]
            elif op == ReduceOp.AVG:
                final_val = [sum(col) / self.world_size for col in zip(*vals)]
            elif op == ReduceOp.MIN:
                final_val = [min(col) for col in zip(*vals)]
            elif op == ReduceOp.MAX:
                final_val = [max(col) for col in zip(*vals)]
            else:
                raise ValueError(f"Unsupported reduce operation: {op}")

            self.client.put(f"{self.prefix}/all_reduce/result", pickle.dumps(final_val))
        else:
            while self.client.get(f"{self.prefix}/all_reduce/result")[0] is None:
                pass
            val = self.client.get(f"{self.prefix}/all_reduce/result")[0]
            final_val = pickle.loads(val)

        return final_val

    def _get_group_id(self, ranks: List[int]) -> int:
        """Get the id for a group of ranks"""
        key = tuple(sorted(ranks))
        return hash(key)


if not os.environ.get("NIXL_ETCD_NAMESPACE"):
    log.warning(
        "Environment variable NIXL_ETCD_NAMESPACE is not set, using default prefix /nixl/kvbench. "
        "Note that it can lead to conflicts if multiple instances of KVBench are running. "
        "To avoid this, set NIXL_ETCD_NAMESPACE to a unique value for each instance of KVBench. "
        'For example, export NIXL_ETCD_NAMESPACE="/nixl/kvbench/$(uuidgen)"'
    )


etcd_dist_utils = _EtcdDistUtils(
    etcd_endpoints=os.environ.get("NIXL_ETCD_ENDPOINTS", "http://localhost:2379"),
    prefix=os.environ.get("NIXL_ETCD_NAMESPACE", "/nixl/kvbench"),
)
