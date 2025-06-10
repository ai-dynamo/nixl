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
import time
from itertools import chain
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from common import NixlBuffer, NixlHandle, TrafficPattern
from dist_utils import dist_utils
from tabulate import tabulate

from nixl._api import nixl_agent

log = logging.getLogger(__name__)


class CTPerftest:

    def __init__(
        self, traffic_pattern: TrafficPattern, iters: int = 1, warmup_iters: int = 0
    ):
        """
        Args:
            traffic_pattern: The communication pattern to test
            iters: Number of test iterations
            warmup_iters: Number of warmup iterations before timing
        """
        self.my_rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self._check_tp_config(traffic_pattern)
        self.traffic_pattern = traffic_pattern
        self.iters = iters
        self.warmup_iters = warmup_iters

        self.nixl_agent = nixl_agent(f"{self.my_rank}")

        # One big buffer is used for all the transfers 
        self.send_buf = None
        self.recv_buf = None

        assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"


    def _share_md(self) -> None:
        """Share agent metadata between all ranks. (Need to be run after registering buffers)"""
        log.debug(f"[Rank {self.my_rank}] Sharing MD")
        md = self.nixl_agent.get_agent_metadata()
        mds = dist_utils.allgather_obj(md)
        for other_rank, metadata in enumerate(mds):
            if other_rank == self.my_rank:
                continue
            self.nixl_agent.add_remote_agent(metadata)

    def _share_recv_buf_descs(self, my_recv_bufs: list[Optional[NixlBuffer]]) -> list:
        """Share receive buffer descriptors between all ranks, in alltoall style.

        Args:
            my_recv_bufs: List of receive buffers for current rank

        Returns:
            List of buffer descriptors from all ranks
        """
        xfer_descs = [
            self.nixl_agent.get_xfer_descs(buf) if buf is not None else None for buf in my_recv_bufs
        ]

        my_recv_bufs_serdes = [
            self.nixl_agent.get_serialized_descs(xfer_descs) for xfer_descs in xfer_descs
        ]

        dst_bufs_serdes = dist_utils.alltoall_obj(my_recv_bufs_serdes)
        dst_bufs_descs = [
            self.nixl_agent.deserialize_descs(serdes) for serdes in dst_bufs_serdes
        ]
        return dst_bufs_descs

    def _init_buffers(self) -> tuple[list[Optional[NixlBuffer]], list[Optional[NixlBuffer]]]:
        """Initialize the buffers, one big send and recv buffer is used for all the transfers
        it has to be chunked inside each transfer to get buffers per ranks
        the buffer is big enough to handle any of the transfers
        For now, support only CUDA/CPU buffers
        """
        
        self.send_buf = NixlBuffer(self.traffic_pattern.total_src_size(self.my_rank), mem_type=tp.mem_type, nixl_agent=self.nixl_agent, dtype=tp.dtype)
        self.recv_buf = NixlBuffer(self.traffic_pattern.total_dst_size(self.my_rank), mem_type=tp.mem_type, nixl_agent=self.nixl_agent, dtype=tp.dtype)

    def _destroy(self):
        del self.send_buf
        del self.recv_buf
    
    def _init_pgs(self):
        senders_ranks = self.traffic_pattern.senders_ranks()
        dist_utils.init_group(senders_ranks)
    
    def _get_bufs(self, tp: TrafficPattern):
        senders_ranks = tp.senders_ranks()

        send_bufs = [None for _ in range(self.world_size)]
        recv_bufs = [None for _ in range(self.world_size)]
        send_offset = recv_offset = 0

        for other_rank in range(self.world_size):
            send_size = tp.matrix[self.my_rank][other_rank]
            recv_size = tp.matrix[other_rank][self.my_rank]
            send_buf = recv_buf = None
            if send_size > 0:
                send_buf = self.send_buf.get_chunk(send_size, send_offset)
                send_offset += send_size
            if recv_size > 0:
                recv_buf = self.recv_buf.get_chunk(recv_size, recv_offset)
                recv_offset += recv_size

            send_bufs[other_rank] = send_buf
            recv_bufs[other_rank] = recv_buf
        
        return send_bufs, recv_bufs

    def _prepare_tp(
        self, tp: TrafficPattern
    ) -> Tuple[
        list[NixlHandle], list[Optional[NixlBuffer]], list[Optional[NixlBuffer]]
    ]:

        send_bufs, recv_bufs = self._get_bufs(tp)

        log.debug(f"[Rank {self.my_rank}] Sharing recv buf descs")
        dst_bufs_descs = self._share_recv_buf_descs(recv_bufs)
        handles: list[NixlHandle] = []
        for other, buf in enumerate(send_bufs):
            if buf is None:
                continue

            log.debug(f"[Rank {self.my_rank}] Initializing xfer handle - Getting xfer desc for rank {other}")
            xfer_desc = self.nixl_agent.get_xfer_descs(buf)

            log.debug(f"[Rank {self.my_rank}] Initializing xfer handle - init xfer with rank {other}")
            handle = self.nixl_agent.initialize_xfer(
                "WRITE",
                xfer_desc,
                dst_bufs_descs[other],
                f"{other}",
                f"{tp.id}_{self.my_rank}_{other}",
            )
            handles.append(NixlHandle(other, handle, tp))

        return handles, send_bufs, recv_bufs

    def _warmup(
        self,
        iters=15,
        fill_value: int = 100000,
        mem_type: Literal["cuda", "vram", "cpu", "dram"] = "cuda",
    ):
        full_matrix = np.full((self.world_size, self.world_size), fill_value=fill_value)
        tp = TrafficPattern(matrix=full_matrix, mem_type=mem_type)
        handles, send_bufs, recv_bufs = self._prepare_tp(tp)
        for _ in range(iters):
            self._run_tp(handles)
            self._wait(handles)

    def _run_tp(self, handles: list[NixlHandle], blocking=False) -> list[NixlHandle]:
        pending = []
        for handle in handles:
            status = self.nixl_agent.transfer(handle.handle)
            assert status != "ERR", "Transfer failed"
            if status != "DONE":
                pending.append(handle)

        if not blocking:
            return pending
        else:
            self._wait(pending)

    def _wait(self, handles: list[NixlHandle]):
        # Wait for transfers to complete
        while True:
            pending = []
            for handle in handles:
                state = self.nixl_agent.check_xfer_state(handle.handle)
                assert state != "ERR", "Transfer got to Error state."
                if state != "DONE":
                    pending.append(handle)

            if not pending:
                break
            handles = pending

    def _destroy( self, handles: list[NixlHandle]):
        for handle in handles:
            self.nixl_agent.release_xfer_handle(handle.handle)

        for other_rank in range(self.world_size):
            if other_rank == self.my_rank:
                continue
            self.nixl_agent.remove_remote_agent(f"{other_rank}")
    
    def _destroy_buffers(self):
        self.send_buf.destroy()
        self.recv_buf.destroy()

    def _check_tp_config(self, tp: TrafficPattern):
        # Matrix size should be world * world
        assert tp.matrix.shape == (
            self.world_size,
            self.world_size,
        ), f"Matrix size is not the same as world size, got {tp.matrix.shape}, world_size={self.world_size}"

    def _verify_tp(
        self,
        tp: TrafficPattern,
        recv_bufs: list[Optional[NixlBuffer]],
        print_recv_buffers: bool = False,
    ):
        for r, recv_buf in enumerate(recv_bufs):
            if recv_buf is None:
                if tp.matrix[r][self.my_rank] > 0:
                    log.error(
                        f"Rank {self.my_rank} expected {tp.matrix[r][self.my_rank]} bytes from rank {r}, but got 0"
                    )
                    raise RuntimeError("Buffer verification failed")
                continue

            if print_recv_buffers:
                s = ""
                for b in recv_buf.bufs:
                    s += f"{b}\n"
                log.info(f"Recv buffer {r}:\n{s}")

            # recv_buf has to be filled with the rank of the sender
            # and its size has to be the same as matrix[r][my_rank]
            full_recv_buf = torch.cat([b for b in recv_buf.bufs])
            expected = torch.full_like(full_recv_buf, r)
            assert torch.all(
                full_recv_buf == expected
            ), f"Vector not equal to {r}, got {full_recv_buf}"
            assert (
                full_recv_buf.size(0) == tp.matrix[r][self.my_rank]
            ), f"Size of vector {r} is not the same as matrix[r][{self.my_rank}], got {full_recv_buf.size(0)}"

    def _get_tp_total_size(self, tp: TrafficPattern) -> int:
        """Return total size of matrix in bytes"""
        return np.sum(tp.matrix, axis=(0, 1)) * tp.dtype.itemsize

    def run(
        self, verify_buffers: bool = False, print_recv_buffers: bool = False
    ) -> float:
        """Execute the performance test.

        Args:
            verify_buffers: Whether to verify buffer contents after transfer
            print_recv_buffers: Whether to print receive buffer contents

        Returns:
            Total execution time in seconds
        """
        log.debug(f"[Rank {self.my_rank}] Running CT perftest")
        self._init_buffers()
        self._init_pgs()
        self._share_md()

        handles, send_bufs, recv_bufs = self._prepare_tp(self.traffic_pattern)

        for _ in range(self.warmup_iters):
            pending_handles = self._run_tp(handles)
            self._wait(pending_handles)

        start = time.time()
        for i in range(self.iters):
            pending_handles = self._run_tp(handles)
            self._wait(pending_handles)
        end = time.time()

        # Metrics report
        start_times = dist_utils.allgather_obj(start)
        end_times = dist_utils.allgather_obj(end)

        global_time = max(end_times) - min(start_times)
        avg_time_per_iter_sec = global_time / self.iters

        total_size_gb = self._get_tp_total_size(self.traffic_pattern) / 1e9

        # Print metrics as a table
        if self.my_rank == 0:
            headers = [
                "Iters",
                "Total time (s)",
                "Avg latency/iter (s)",
                "Total size (GB)",
            ]
            data = [
                [
                    self.iters,
                    global_time,
                    avg_time_per_iter_sec,
                    total_size_gb,
                ]
            ]
            print(tabulate(data, headers=headers, floatfmt=".6f"))

        if verify_buffers:
            self._verify_tp(self.traffic_pattern, recv_bufs, print_recv_buffers)
        
        # Destroy
        self._destroy(handles)
        self._destroy_buffers()

        return end - start
