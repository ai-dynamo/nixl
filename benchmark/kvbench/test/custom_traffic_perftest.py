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

import os
import time
from test.traffic_pattern import TrafficPattern
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from runtime.etcd_rt import etcd_dist_utils as dist_rt
from tabulate import tabulate

from nixl._api import nixl_agent
from nixl.logging import get_logger

logger = get_logger(__name__)


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
            device = torch.device("cuda")
        elif mem_type in ("cpu", "dram"):
            device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported memory type: {mem_type}")

        if shards > 1:
            raise ValueError("Sharding is not supported yet")

        logger.debug(
            "[Rank %d] Initializing NixlBuffer with size %d, device %s, shards %d, fill_value %d",
            dist_rt.get_rank(),
            size,
            device,
            shards,
            fill_value,
        )
        self.buf = torch.full((size,), fill_value, dtype=dtype, device=device)

        logger.debug(
            "[Rank %d] Registering memory for buffer %s",
            dist_rt.get_rank(),
            self.buf,
        )
        self.reg_descs = nixl_agent.get_reg_descs(self.buf)
        assert (
            nixl_agent.register_memory(self.reg_descs) is not None
        ), "Failed to register memory"

    def get_chunk(self, size, offset):
        if offset + size > self.size:
            raise ValueError(
                f"Offset {offset} + size {size} is greater than buffer size {self.size}"
            )
        return self.buf[offset : offset + size]

    def destroy(self):
        self.nixl_agent.deregister_memory(self.reg_descs)
        if hasattr(self.buf, "is_cuda") and self.buf.is_cuda:
            del self.buf
            torch.cuda.empty_cache()


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
        self.my_rank = dist_rt.get_rank()
        self.world_size = dist_rt.get_world_size()
        self.traffic_pattern = traffic_pattern
        self.iters = iters
        self.warmup_iters = warmup_iters

        self.nixl_agent = nixl_agent(f"{self.my_rank}")

        if (
            not os.environ.get("CUDA_VISIBLE_DEVICES")
            and self.traffic_pattern.mem_type == "cuda"
        ):
            logger.warning(
                "Cuda buffers detected, but the env var CUDA_VISIBLE_DEVICES is not set, this will cause every process in the same host to use the same GPU device."
            )

        """Initialize the buffers, one big send and recv buffer is used for all the transfers
        it has to be chunked inside each transfer to get buffers per ranks
        the buffer is big enough to handle any of the transfers
        For now, support only CUDA/CPU buffers
        """
        self.send_buf: NixlBuffer = NixlBuffer(
            self.traffic_pattern.total_src_size(self.my_rank),
            mem_type=self.traffic_pattern.mem_type,
            nixl_agent=self.nixl_agent,
            dtype=self.traffic_pattern.dtype,
        )
        self.recv_buf: NixlBuffer = NixlBuffer(
            self.traffic_pattern.total_dst_size(self.my_rank),
            mem_type=self.traffic_pattern.mem_type,
            nixl_agent=self.nixl_agent,
            dtype=self.traffic_pattern.dtype,
        )

        self._check_tp_config(traffic_pattern)
        assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"

    def _barrier_tp(self, tp: TrafficPattern, senders_only=True):
        """Barrier for a traffic pattern"""
        if senders_only:
            dist_rt.barrier(tp.senders_ranks())
        else:
            dist_rt.barrier(tp.ranks())

    def _share_md(self) -> None:
        """Share agent metadata between all ranks. (Need to be run after registering buffers)"""
        logger.debug("[Rank %d] Sharing MD", self.my_rank)
        md = self.nixl_agent.get_agent_metadata()
        mds = dist_rt.allgather_obj(md)
        for other_rank, metadata in enumerate(mds):
            if other_rank == self.my_rank:
                continue
            self.nixl_agent.add_remote_agent(metadata)

        dist_rt.barrier()

    def _share_recv_buf_descs(self, my_recv_bufs: list[Optional[NixlBuffer]]) -> list:
        """Share receive buffer descriptors between all ranks, in alltoall style.
        Args:
            my_recv_bufs: List of receive buffers for current rank
        Returns:
            List of buffer descriptors from all ranks
        """
        xfer_descs = [
            self.nixl_agent.get_xfer_descs(buf) if buf is not None else None
            for buf in my_recv_bufs
        ]

        my_recv_bufs_serdes = [
            self.nixl_agent.get_serialized_descs(xfer_descs)
            for xfer_descs in xfer_descs
        ]

        dst_bufs_serdes = dist_rt.alltoall_obj(my_recv_bufs_serdes)

        dst_bufs_descs = [
            self.nixl_agent.deserialize_descs(serdes) for serdes in dst_bufs_serdes
        ]
        return dst_bufs_descs

    def _get_bufs(self, tp: TrafficPattern):
        """Returns lists of buffers where bufs[i] is the send/recv buffer for rank i"""
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
        """Timing everything in this function because it takes a lot of time"""

        send_bufs, recv_bufs = self._get_bufs(tp)

        logger.debug("[Rank %d] Sharing recv buf descs", self.my_rank)
        dst_bufs_descs = self._share_recv_buf_descs(recv_bufs)

        handles: list[NixlHandle] = []
        for other, buf in enumerate(send_bufs):
            if buf is None:
                continue

            xfer_desc = self.nixl_agent.get_xfer_descs(buf)

            logger.debug(
                "[Rank %d] Initializing xfer for %d - xfer desc: %s, dst buf desc: %s",
                self.my_rank,
                other,
                xfer_desc,
                dst_bufs_descs[other],
            )
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
            return []

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

    def _destroy(self, handles: list[NixlHandle]):
        logger.debug("[Rank %d] Releasing XFER handles", self.my_rank)
        for handle in handles:
            self.nixl_agent.release_xfer_handle(handle.handle)

        logger.debug("[Rank %d] Removing remote agents", self.my_rank)
        for other_rank in range(self.world_size):
            if other_rank == self.my_rank:
                continue
            self.nixl_agent.remove_remote_agent(f"{other_rank}")

        self._destroy_buffers()

    def _destroy_buffers(self):
        logger.debug("[Rank %d] Destroying buffers", self.my_rank)
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
                    logger.error(
                        f"Rank {self.my_rank} expected {tp.matrix[r][self.my_rank]} bytes from rank {r}, but got 0"
                    )
                    raise RuntimeError("Buffer verification failed")
                continue

            if print_recv_buffers:
                logger.info("Recv buffer %d:\n%s", r, recv_buf.buf)

            # recv_buf has to be filled with the rank of the sender
            # and its size has to be the same as matrix[r][my_rank]
            full_recv_buf = recv_buf.buf
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
        logger.debug("[Rank %d] Running CT perftest", self.my_rank)
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
        start_times = dist_rt.allgather_obj(start)
        end_times = dist_rt.allgather_obj(end)

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
            logger.info(
                "Performance metrics:\n%s",
                tabulate(data, headers=headers, floatfmt=".6f"),
            )

        if verify_buffers:
            self._verify_tp(self.traffic_pattern, recv_bufs, print_recv_buffers)

        # Destroy
        self._destroy(handles)

        return end - start
