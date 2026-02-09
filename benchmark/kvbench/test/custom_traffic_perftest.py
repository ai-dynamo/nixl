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

import logging
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

BUFFER_ALIGNMENT = 4096  # 4K alignment for O_DIRECT


def allocate_aligned_buffer(
    size: int,
    device: torch.device,
    alignment: int = BUFFER_ALIGNMENT,
    fill_value: int = 0,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate a buffer with guaranteed alignment.

    Args:
        size: Desired buffer size in bytes
        device: Torch device (cpu or cuda)
        alignment: Required alignment in bytes (default 4K for O_DIRECT)
        fill_value: Value to fill buffer with
        dtype: Torch dtype for buffer

    Returns:
        Tuple of (raw_buffer, aligned_view) where aligned_view is a slice
        of raw_buffer starting at an aligned address.
    """
    # Over-allocate by alignment-1 bytes to guarantee we can find an aligned start
    alloc_size = size + alignment - 1
    raw_buf = torch.full((alloc_size,), fill_value, dtype=dtype, device=device)

    # Find aligned offset within the buffer
    raw_ptr = raw_buf.data_ptr()
    align_offset = (alignment - (raw_ptr % alignment)) % alignment

    # Create aligned view
    aligned_buf = raw_buf[align_offset : align_offset + size]

    # Verify alignment
    aligned_ptr = aligned_buf.data_ptr()
    assert (
        aligned_ptr % alignment == 0
    ), f"Buffer not aligned: ptr={aligned_ptr:#x}, alignment={alignment}"

    logger.debug(
        "Buffer aligned: raw_ptr=%#x, aligned_ptr=%#x, offset=%d",
        raw_ptr,
        aligned_ptr,
        align_offset,
    )

    return raw_buf, aligned_buf


class NixlHandle:
    """Base class for NIXL transfer handles."""

    def __init__(self, handle):
        self.handle = handle

    def __str__(self):
        return "NixlHandle"


class RDMAHandle(NixlHandle):
    """Handle for RDMA transfers to a remote rank."""

    def __init__(self, handle, remote_rank, traffic_pattern):
        super().__init__(handle)
        self.remote_rank = remote_rank
        self.tp = traffic_pattern

    def __str__(self):
        return f"RDMA→rank:{self.remote_rank}"


class StorageXferHandle(NixlHandle):
    """Handle for storage I/O transfer operations."""

    def __init__(self, handle, file_path: str, operation: str):
        super().__init__(handle)
        self.file_path = file_path
        self.operation = operation  # "read" or "write"

    def __str__(self):
        return f"Storage:{self.operation}:{self.file_path}"


class NixlBuffer:
    """Can be sharded. Allocates 4K-aligned buffers for O_DIRECT compatibility.

    When using get_chunk(), offsets should be pre-aligned using align_offset().
    Use aligned_total_size() to calculate buffer size when chunks need alignment.
    """

    ALIGNMENT = BUFFER_ALIGNMENT

    @staticmethod
    def align_up(value: int, alignment: int = BUFFER_ALIGNMENT) -> int:
        """Round up value to next alignment boundary."""
        return ((value + alignment - 1) // alignment) * alignment

    @staticmethod
    def aligned_total_size(
        chunk_sizes: list[int], alignment: int = BUFFER_ALIGNMENT
    ) -> int:
        """Calculate total buffer size needed for aligned chunks.

        Each chunk starts at an aligned offset, so we need padding between chunks.
        Example: chunks [5000, 3000] with 4K alignment needs:
          - Chunk 0: offset=0, size=5000
          - Chunk 1: offset=8192 (next 4K boundary after 5000), size=3000
          - Total: 8192 + 3000 = 11192, rounded up to 12288
        """
        if not chunk_sizes:
            return 0
        offset = 0
        for size in chunk_sizes:
            if size > 0:
                # Align current offset before placing chunk
                offset = NixlBuffer.align_up(offset, alignment)
                offset += size
        # Final size aligned
        return NixlBuffer.align_up(offset, alignment)

    def __init__(
        self,
        size: int,
        mem_type: str,
        nixl_agent: nixl_agent,
        shards=1,
        fill_value=0,
        dtype: torch.dtype = torch.int8,
        backends: (
            list[str] | None
        ) = None,  # Additional backends to register with (e.g., ["POSIX"])
    ):
        self.nixl_agent = nixl_agent
        self._backends = backends
        if mem_type in ("cuda", "vram"):
            device = torch.device("cuda")
        elif mem_type in ("cpu", "dram"):
            device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported memory type: {mem_type}")

        if shards > 1:
            raise ValueError("Sharding is not supported yet")

        # 4K align the buffer size
        size = self.align_up(size, self.ALIGNMENT)
        self.size = size

        logger.debug(
            "[Rank %d] Initializing NixlBuffer with size %d, device %s, shards %d, fill_value %d",
            dist_rt.get_rank(),
            size,
            device,
            shards,
            fill_value,
        )

        # Allocate aligned buffer
        self._raw_buf, self.buf = allocate_aligned_buffer(
            size, device, self.ALIGNMENT, fill_value, dtype
        )

        logger.debug(
            "[Rank %d] Registering memory for buffer %s",
            dist_rt.get_rank(),
            self.buf,
        )
        self.reg_descs = nixl_agent.get_reg_descs(self.buf)
        # First register with default backend (UCX for RDMA)
        assert (
            nixl_agent.register_memory(self.reg_descs) is not None
        ), "Failed to register memory"
        # Also register with additional backends if specified (e.g., POSIX for storage)
        if self._backends:
            assert (
                nixl_agent.register_memory(self.reg_descs, backends=self._backends)
                is not None
            ), f"Failed to register memory with backends {self._backends}"

    def get_chunk(
        self, size: int, offset: int, check_alignment: bool = True
    ) -> torch.Tensor:
        """Get a chunk of the buffer at the specified offset.

        Args:
            size: Size of the chunk in bytes
            offset: Offset into the buffer (should be aligned for O_DIRECT)
            check_alignment: If True, warn if offset is not aligned

        Returns:
            Tensor slice of the buffer

        Raises:
            ValueError: If chunk would exceed buffer bounds
        """
        if offset + size > self.size:
            raise ValueError(
                f"Chunk out of bounds: offset={offset} + size={size} = {offset + size} > buffer_size={self.size}"
            )
        if check_alignment and offset % self.ALIGNMENT != 0:
            logger.warning(
                "Chunk offset %d is not %d-byte aligned. This may cause performance issues with O_DIRECT.",
                offset,
                self.ALIGNMENT,
            )
        return self.buf[offset : offset + size]

    def destroy(self):
        # Deregister from additional backends first
        if self._backends:
            self.nixl_agent.deregister_memory(self.reg_descs, backends=self._backends)
        # Deregister from default backend
        self.nixl_agent.deregister_memory(self.reg_descs)
        # Delete the raw buffer (buf is just a view into it)
        if hasattr(self._raw_buf, "is_cuda") and self._raw_buf.is_cuda:
            del self.buf
            del self._raw_buf
            torch.cuda.empty_cache()
        else:
            del self.buf
            del self._raw_buf


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

        # Initialize the buffers with aligned size calculation.
        # One big send and recv buffer is used for all transfers, chunked per destination.
        # Buffer must be large enough for aligned chunks (padding between chunks).
        tp = self.traffic_pattern
        if tp.matrix is not None:
            # Get individual chunk sizes for aligned total calculation
            send_sizes = [
                int(tp.matrix[self.my_rank][dst]) for dst in range(tp.matrix.shape[1])
            ]
            recv_sizes = [
                int(tp.matrix[src][self.my_rank]) for src in range(tp.matrix.shape[0])
            ]
            send_total = NixlBuffer.aligned_total_size(send_sizes)
            recv_total = NixlBuffer.aligned_total_size(recv_sizes)
        else:
            send_total = recv_total = 0

        self.send_buf: NixlBuffer = NixlBuffer(
            send_total,
            mem_type=tp.mem_type,
            nixl_agent=self.nixl_agent,
            dtype=tp.dtype,
        )
        self.recv_buf: NixlBuffer = NixlBuffer(
            recv_total,
            mem_type=tp.mem_type,
            nixl_agent=self.nixl_agent,
            dtype=tp.dtype,
        )

        self._check_tp_config(traffic_pattern)
        assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable string."""
        if size_bytes >= 1e9:
            return f"{size_bytes / 1e9:.2f} GB"
        elif size_bytes >= 1e6:
            return f"{size_bytes / 1e6:.2f} MB"
        elif size_bytes >= 1e3:
            return f"{size_bytes / 1e3:.2f} KB"
        return f"{size_bytes} B"

    def _barrier_tp(self, tp: TrafficPattern, senders_only=True, include_storage=True):
        """Barrier for a traffic pattern (participating ranks only).

        Args:
            tp: The traffic pattern
            senders_only: If True, only RDMA senders. If False, all RDMA participants.
            include_storage: If True, also include ranks with storage ops.
        """
        if senders_only:
            ranks = set(tp.senders_ranks())
        else:
            ranks = set(tp.senders_ranks() + tp.receivers_ranks())

        # Include storage ranks if requested
        if include_storage and tp.storage_ops:
            ranks.update(tp.storage_ops.keys())

        if ranks:
            dist_rt.barrier(list(ranks))

    def _share_md(self) -> None:
        """Share agent metadata between all ranks. (Need to be run after registering buffers)"""
        # Skip remote metadata when running single-rank or when no remote-capable backend is available
        if self.world_size == 1:
            logger.debug(
                "[Rank %d] Single-rank run, skipping metadata exchange", self.my_rank
            )
            return
        logger.debug("[Rank %d] Getting local agent metadata...", self.my_rank)
        try:
            md = self.nixl_agent.get_agent_metadata()
        except Exception as e:
            logger.warning(
                "[Rank %d] Skipping metadata exchange due to agent error: %s",
                self.my_rank,
                e,
            )
            return

        logger.debug("[Rank %d] Exchanging metadata with all ranks...", self.my_rank)
        mds = dist_rt.allgather_obj(md)

        agent_names = []
        for other_rank, metadata in enumerate(mds):
            if other_rank == self.my_rank:
                agent_names.append(f"{other_rank}(local)")
                continue
            logger.debug(
                "[Rank %d] Adding remote agent: rank %d", self.my_rank, other_rank
            )
            self.nixl_agent.add_remote_agent(metadata)
            agent_names.append(f"{other_rank}")

        logger.info(
            "[Rank %d] Metadata exchange complete. Agents: [%s]",
            self.my_rank,
            ", ".join(agent_names),
        )
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
        """Returns lists of buffers where bufs[i] is the send/recv buffer for rank i.

        Chunks are placed at aligned offsets for O_DIRECT compatibility.
        """
        send_bufs = [None for _ in range(self.world_size)]
        recv_bufs = [None for _ in range(self.world_size)]

        # No RDMA buffers needed if no matrix (storage-only pattern)
        if tp.matrix is None:
            return send_bufs, recv_bufs

        send_offset = recv_offset = 0

        for other_rank in range(self.world_size):
            send_size = tp.matrix[self.my_rank][other_rank]
            recv_size = tp.matrix[other_rank][self.my_rank]
            send_buf = recv_buf = None
            if send_size > 0:
                # Align offset before getting chunk
                send_offset = NixlBuffer.align_up(send_offset)
                send_buf = self.send_buf.get_chunk(send_size, send_offset)
                send_offset += send_size
            if recv_size > 0:
                # Align offset before getting chunk
                recv_offset = NixlBuffer.align_up(recv_offset)
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
            handles.append(RDMAHandle(handle, other, tp))

        return handles, send_bufs, recv_bufs

    def _warmup(
        self,
        iters=15,
        fill_value: int = 100000,
        mem_type: Literal["cuda", "vram", "cpu", "dram"] = "cpu",
    ):
        full_matrix = np.full((self.world_size, self.world_size), fill_value=fill_value)
        tp = TrafficPattern(matrix=full_matrix, mem_type=mem_type)
        handles, send_bufs, recv_bufs = self._prepare_tp(tp)
        for _ in range(iters):
            self._run_tp(handles)
            self._wait(handles)

    def _run_handle(self, handle: NixlHandle) -> float:
        """Run a single transfer handle and return latency in seconds."""
        t = time.time()
        status = self.nixl_agent.transfer(handle.handle)
        assert status != "ERR", "Transfer failed"
        if status != "DONE":
            self._wait([handle])
        return time.time() - t

    def _run_tp(self, handles: list[NixlHandle], blocking=False) -> list:
        pending = []
        for h in handles:
            logger.debug("[Rank %d] Transfer: %s", self.my_rank, h)
            status = self.nixl_agent.transfer(h.handle)
            assert status != "ERR", "Transfer failed"
            if status != "DONE":
                pending.append(h)

        logger.debug(
            "[Rank %d] Transfer: %d handles initiated, %d pending, blocking=%s",
            self.my_rank,
            len(handles),
            len(pending),
            blocking,
        )

        if not blocking:
            return pending
        else:
            self._wait(pending)
            return []

    def _run_isolated_tp(
        self, handles: list[NixlHandle], n_iters: int = 1
    ) -> list[float]:
        """Run each handle in isolation and return per-handle latencies.

        Each handle is run n_iters times separately, measuring time for each.
        Returns a list of average latencies, one per handle (in same order as input).

        This provides a true "speed of light" measurement where each transfer
        runs without interference from other transfers in the same TP.

        Args:
            handles: List of transfer handles to run
            n_iters: Number of iterations per handle for averaging

        Returns:
            List of average latencies (seconds), one per handle
        """
        if not handles:
            return []

        latencies = []
        for h in handles:
            handle_total = 0.0
            for _ in range(n_iters):
                t = time.time()
                status = self.nixl_agent.transfer(h.handle)
                assert status != "ERR", "Transfer failed"
                if status != "DONE":
                    self._wait([h])
                handle_total += time.time() - t
            latencies.append(handle_total / n_iters)

        return latencies

    def _wait(self, handles: list[NixlHandle]):
        """Wait for transfers to complete using in-place swap-remove."""
        if not handles:
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[Rank %d] Waiting for %d handles to complete...",
                self.my_rank,
                len(handles),
            )
        # Make a mutable copy to avoid modifying caller's list
        remaining = list(handles)
        poll_count = 0
        while remaining:
            i = 0
            while i < len(remaining):
                state = self.nixl_agent.check_xfer_state(remaining[i].handle)
                if state == "ERR":
                    raise RuntimeError(
                        f"[Rank {self.my_rank}] Transfer {remaining[i]} got to Error state."
                    )
                if state == "DONE":
                    # Swap-remove: O(1) removal without shifting
                    remaining[i] = remaining[-1]
                    remaining.pop()
                else:
                    i += 1
            poll_count += 1
            if remaining:
                time.sleep(0)  # yield CPU timeslice

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[Rank %d] All handles completed (polled %d times)",
                self.my_rank,
                poll_count,
            )

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
        # Matrix size should be world * world (if matrix exists)
        if tp.matrix is not None:
            assert tp.matrix.shape == (
                self.world_size,
                self.world_size,
            ), f"Matrix size is not the same as world size, got {tp.matrix.shape}, world_size={self.world_size}"
        elif not tp.storage_ops:
            raise ValueError(
                "Traffic pattern must have either RDMA matrix or storage ops"
            )

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
        """Return total size of matrix in bytes (0 if no matrix)"""
        if tp.matrix is None:
            return 0
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
