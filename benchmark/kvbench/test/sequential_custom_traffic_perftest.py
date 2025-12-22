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

"""Sequential is different from multi in that every rank processes only one TP at a time, but they can process different ones.

Supports optional storage operations: Per-TP flow READ → COMPUTE → WRITE → RDMA.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
from pathlib import Path
from test.custom_traffic_perftest import CTPerftest, NixlBuffer
from test.storage_backend import FilesystemBackend, StorageBackend, StorageHandle
from test.traffic_pattern import TrafficPattern
from typing import Any, Dict, List, Optional, Tuple

from runtime.etcd_rt import etcd_dist_utils as dist_rt
from tabulate import tabulate

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


class StorageOpType(Enum):
    """Type of storage operation."""

    READ = auto()
    WRITE = auto()


@dataclass
class StorageOp:
    """Storage operation configuration for a single rank."""

    file_path: str
    file_size: int
    read_offset: int
    read_size: int
    write_offset: int
    write_size: int


class SequentialCTPerftest(CTPerftest):
    """Extends CTPerftest to handle multiple traffic patterns sequentially.

    The patterns are executed in sequence, and the results are aggregated.
    Allows testing multiple communication patterns sequentially between distributed processes.
    Supports optional storage operations: READ → COMPUTE → WRITE → RDMA.
    """

    def __init__(
        self,
        traffic_patterns: list[TrafficPattern],
        storage_path: Optional[Path] = None,
        storage_nixl_backend: Optional[str] = None,
        storage_direct_io: bool = False,
        n_iters: int = 3,
        n_isolation_iters: int = 30,
        warmup_iters: int = 30,
    ) -> None:
        """Initialize sequential traffic performance test.

        Args:
            traffic_patterns: List of traffic patterns to test sequentially
            storage_path: Optional base path for storage operations
            storage_nixl_backend: Storage backend type (POSIX, GDS, GDS_MT)
            storage_direct_io: Whether to use O_DIRECT for storage I/O
            n_iters: Number of benchmark iterations
            n_isolation_iters: Number of isolated latency measurement iterations
            warmup_iters: Number of warmup iterations
        """
        self.my_rank = dist_rt.get_rank()
        self.world_size = dist_rt.get_world_size()
        self.traffic_patterns = traffic_patterns
        self.n_iters = n_iters
        self.n_isolation_iters = n_isolation_iters
        self.warmup_iters = warmup_iters

        logger.debug("[Rank %d] Initializing Nixl agent", self.my_rank)
        # Check if storage is enabled - need to control backend creation order
        has_storage = any(tp.storage_ops for tp in traffic_patterns) and storage_path
        if has_storage:
            # Don't auto-create UCX - we'll create GDS first, then UCX
            # This avoids UCX/GDS conflicts on systems without UMR QP support
            config = nixl_agent_config(backends=[])
            self.nixl_agent = nixl_agent(f"{self.my_rank}", config)
        else:
            # No storage, default UCX auto-creation is fine
            self.nixl_agent = nixl_agent(f"{self.my_rank}")

        for tp in self.traffic_patterns:
            self._check_tp_config(tp)
        if not os.environ.get("CUDA_VISIBLE_DEVICES") and any(
            tp.mem_type == "cuda" for tp in self.traffic_patterns
        ):
            logger.warning(
                "Cuda buffers detected, but the env var CUDA_VISIBLE_DEVICES is not set, "
                "this will cause every process in the same host to use the same GPU device."
            )

        # NixlBuffer caches buffers and reuse them if they are big enough,
        # let's initialize them once, with the largest needed size
        self.send_buf_by_mem_type: dict[str, NixlBuffer] = {}
        self.recv_buf_by_mem_type: dict[str, NixlBuffer] = {}
        self._has_storage = any(tp.storage_ops for tp in traffic_patterns)
        self._storage_backend: Optional[StorageBackend] = None
        self._storage_handles: Dict[str, StorageHandle] = {}
        self._storage_nixl_backend: Optional[str] = (
            None  # Backend for buffer registration
        )

        if self._has_storage and storage_path:
            nixl_backend = storage_nixl_backend or "POSIX"
            self._storage_nixl_backend = nixl_backend
            use_direct_io = storage_direct_io or nixl_backend in ("GDS", "GDS_MT")
            logger.info(
                "[Rank %d] Storage: %s, backend=%s, O_DIRECT=%s",
                self.my_rank,
                storage_path,
                nixl_backend,
                use_direct_io,
            )
            self._storage_backend = FilesystemBackend(
                agent=self.nixl_agent,
                base_path=storage_path,
                nixl_backend=nixl_backend,
                use_direct_io=use_direct_io,
            )
            # Now create UCX backend (after GDS to avoid UMR QP conflicts)
            self.nixl_agent.create_backend("UCX")
            logger.debug("[Rank %d] Created UCX backend after storage", self.my_rank)
        elif self._has_storage:
            logger.warning(
                "[Rank %d] Storage ops in TPs but no storage_path", self.my_rank
            )
            self._has_storage = False

        logger.debug(
            "[Rank %d] Init: world=%d, iters=%d, warmup=%d, tps=%d, storage=%s",
            self.my_rank,
            self.world_size,
            n_iters,
            warmup_iters,
            len(traffic_patterns),
            self._has_storage,
        )

    def _get_storage_key(self, tp_idx: int) -> str:
        """Get storage handle key for a traffic pattern index."""
        return f"{tp_idx}:{self.my_rank}"

    def _prepare_storage(self):
        """Prepare all storage handles."""
        if not self._has_storage or not self._storage_backend:
            return
        for tp_idx, tp in enumerate(self.traffic_patterns):
            my_ops = tp.storage_ops.get(self.my_rank) if tp.storage_ops else None
            if not my_ops:
                continue
            key = self._get_storage_key(tp_idx)
            if key not in self._storage_handles:
                self._storage_handles[key] = self._storage_backend.prepare(
                    tp_idx=tp_idx,
                    rank=self.my_rank,
                    read_size=my_ops.read_size,
                    write_size=my_ops.write_size,
                )
        logger.info(
            "[Rank %d] Prepared %d storage handles",
            self.my_rank,
            len(self._storage_handles),
        )

    def _init_buffers(self):
        """Initialize send/recv buffers sized for max RDMA or storage."""
        logger.debug("[Rank %d] Initializing buffers", self.my_rank)
        max_src_by_mem_type = defaultdict(int)
        max_dst_by_mem_type = defaultdict(int)

        for tp in self.traffic_patterns:
            my_ops = tp.storage_ops.get(self.my_rank) if tp.storage_ops else None
            storage_size = (my_ops.read_size + my_ops.write_size) if my_ops else 0
            max_src_by_mem_type[tp.mem_type] = max(
                max_src_by_mem_type[tp.mem_type],
                tp.total_src_size(self.my_rank),
                storage_size,
            )
            max_dst_by_mem_type[tp.mem_type] = max(
                max_dst_by_mem_type[tp.mem_type], tp.total_dst_size(self.my_rank)
            )

        # If storage is enabled, also register buffers with storage backend
        storage_backends = (
            [self._storage_nixl_backend] if self._storage_nixl_backend else None
        )

        for mem_type, size in max_src_by_mem_type.items():
            if not size:
                continue
            self.send_buf_by_mem_type[mem_type] = NixlBuffer(
                size,
                mem_type=mem_type,
                nixl_agent=self.nixl_agent,
                backends=storage_backends,
            )

        for mem_type, size in max_dst_by_mem_type.items():
            if not size:
                continue
            self.recv_buf_by_mem_type[mem_type] = NixlBuffer(
                size,
                mem_type=mem_type,
                nixl_agent=self.nixl_agent,
                backends=storage_backends,
            )

    def _destroy_buffers(self):
        logger.debug("[Rank %d] Destroying buffers", self.my_rank)
        for buf in chain(
            self.send_buf_by_mem_type.values(), self.recv_buf_by_mem_type.values()
        ):
            buf.destroy()

    def _get_bufs(self, tp: TrafficPattern):
        logger.debug("[Rank %d] Getting buffers for TP %s", self.my_rank, tp.id)

        send_bufs = [None for _ in range(self.world_size)]
        recv_bufs = [None for _ in range(self.world_size)]

        # If no matrix, return empty buffers (storage-only pattern)
        if tp.matrix is None:
            return send_bufs, recv_bufs

        send_offset_by_memtype: dict[str, int] = defaultdict(int)
        recv_offset_by_memtype: dict[str, int] = defaultdict(int)

        for other_rank in range(self.world_size):
            send_size = tp.matrix[self.my_rank][other_rank]
            recv_size = tp.matrix[other_rank][self.my_rank]
            send_buf = recv_buf = None

            if send_size > 0:
                send_buf = self.send_buf_by_mem_type[tp.mem_type].get_chunk(
                    send_size, send_offset_by_memtype[tp.mem_type]
                )
                send_offset_by_memtype[tp.mem_type] += send_size
            if recv_size > 0:
                recv_buf = self.recv_buf_by_mem_type[tp.mem_type].get_chunk(
                    recv_size, recv_offset_by_memtype[tp.mem_type]
                )
                recv_offset_by_memtype[tp.mem_type] += recv_size

            send_bufs[other_rank] = send_buf
            recv_bufs[other_rank] = recv_buf

        return send_bufs, recv_bufs

    def _prepare_storage_xfer(self, tp_idx: int, operation: StorageOpType) -> List[Any]:
        """Prepare a NIXL transfer handle for storage read or write.

        Creates a transfer handle that can be passed to _run_tp() for execution.
        The handle is reusable - can be transferred multiple times before release.

        File layout per rank:
            [0, read_size) = READ region (pre-filled with rank pattern)
            [read_size, read_size + write_size) = WRITE region

        Args:
            tp_idx: Traffic pattern index
            operation: StorageOpType.READ or StorageOpType.WRITE

        Returns:
            List containing single transfer handle, or empty list if:
            - No storage backend configured
            - No storage handle for this TP/rank
            - Requested operation has size 0
            - No buffer available for this memory type
        """
        if not self._storage_backend:
            return []

        handle = self._storage_handles.get(self._get_storage_key(tp_idx))
        if not handle:
            return []

        # Determine size, buffer offset, and backend method based on operation
        if operation == StorageOpType.READ:
            if handle.read_size == 0:
                return []
            size = handle.read_size
            buf_offset = 0  # Read uses start of buffer
            get_handle_fn = self._storage_backend.get_read_handle
        else:  # StorageOpType.WRITE
            if handle.write_size == 0:
                return []
            size = handle.write_size
            buf_offset = handle.read_size  # Write uses buffer after read region
            get_handle_fn = self._storage_backend.get_write_handle

        # Get buffer chunk for the transfer (must match registered memory type)
        buf = self.send_buf_by_mem_type.get(self.traffic_patterns[tp_idx].mem_type)
        if not buf:
            return []

        # Create transfer handle: buffer_chunk -> file (write) or file -> buffer_chunk (read)
        xfer = get_handle_fn(handle, buf.get_chunk(size, offset=buf_offset))
        return [xfer] if xfer else []

    def _prepare_storage_read(self, tp_idx: int) -> List[Any]:
        """Get storage read transfer handle. See _prepare_storage_xfer for details."""
        return self._prepare_storage_xfer(tp_idx, StorageOpType.READ)

    def _prepare_storage_write(self, tp_idx: int) -> List[Any]:
        """Get storage write transfer handle. See _prepare_storage_xfer for details."""
        return self._prepare_storage_xfer(tp_idx, StorageOpType.WRITE)

    # ═══════════════════════════════════════════════════════════════════════════
    # ISOLATED LATENCY MEASUREMENT HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _measure_isolated_storage_latency(
        self,
        handles_by_tp: List[List[Any]],
        operation_name: str,
    ) -> List[float]:
        """Measure isolated latency for storage operations (READ or WRITE).

        Iterates through TPs sequentially, executing storage transfers and measuring
        average time. Uses global barrier before each TP to ensure TPs don't overlap
        across ranks (even ranks not participating must wait).

        Args:
            handles_by_tp: List of transfer handles per TP index
            operation_name: "READ" or "WRITE" for logging

        Returns:
            List of average latencies (seconds) per TP, 0.0 for non-participating TPs
        """
        n_tps = len(self.traffic_patterns)
        latencies: List[float] = [0.0] * n_tps

        for tp_idx in range(n_tps):
            tp = self.traffic_patterns[tp_idx]
            handle = handles_by_tp[tp_idx]
            has_storage = tp.storage_ops and self.my_rank in tp.storage_ops

            # Global barrier ensures TPs don't overlap across ranks
            dist_rt.barrier()

            if not handle or not has_storage:
                continue

            for _ in range(self.n_isolation_iters):
                start = time.time()
                self._run_tp(handle, blocking=True)
                latencies[tp_idx] += time.time() - start

            latencies[tp_idx] /= self.n_isolation_iters
            logger.debug(
                "[Rank %d] Isolated %s tp %d/%d: %.3f ms",
                self.my_rank,
                operation_name,
                tp_idx,
                n_tps,
                latencies[tp_idx] * 1e3,
            )
        return latencies

    def _measure_isolated_rdma_latency(
        self,
        rdma_handles_by_tp: List[List[Any]],
    ) -> List[float]:
        """Measure isolated RDMA latency (baseline without storage overhead).

        Both senders and receivers participate and measure time (matching main loop).
        Uses global barrier before each TP, then TP-specific barrier for RDMA
        participants only (excludes storage-only ranks).

        Args:
            rdma_handles_by_tp: List of RDMA transfer handles per TP index

        Returns:
            List of average latencies (seconds) per TP, 0.0 for non-participants
        """
        n_tps = len(self.traffic_patterns)
        latencies: List[float] = [0.0] * n_tps

        for tp_idx, rdma_h in enumerate(rdma_handles_by_tp):
            tp = self.traffic_patterns[tp_idx]

            # Global barrier ensures TPs don't overlap
            dist_rt.barrier()

            is_participant = self.my_rank in tp.senders_ranks()
            if not is_participant:
                continue

            # TP-specific barrier (exclude storage ranks - this is RDMA-only measurement)
            self._barrier_tp(tp, senders_only=False, include_storage=False)

            for _ in range(self.n_isolation_iters):
                start = time.time()
                # All participants execute transfer (sender sends, receiver receives)
                self._run_tp(rdma_h, blocking=True)
                latencies[tp_idx] += time.time() - start
                # Barrier after each iteration ensures all transfers complete before next
                self._barrier_tp(tp, senders_only=False, include_storage=False)

            latencies[tp_idx] /= self.n_isolation_iters
            logger.debug(
                "[Rank %d] Isolated RDMA tp %d/%d: %.3f ms",
                self.my_rank,
                tp_idx,
                n_tps,
                latencies[tp_idx] * 1e3,
            )
        return latencies

    def _aggregate_isolated_latencies(
        self,
        my_read: List[float],
        my_write: List[float],
        my_rdma: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        """Aggregate isolated latencies across ranks and compute max per TP.

        Gathers latency measurements from all ranks using allgather, then computes
        max latency per TP (the slowest rank determines overall TP performance).
        Filters out 0.0 values from non-participating ranks.

        Args:
            my_read: This rank's read latencies (seconds) per TP
            my_write: This rank's write latencies (seconds) per TP
            my_rdma: This rank's RDMA latencies (seconds) per TP

        Returns:
            Tuple of (read_ms, write_ms, rdma_ms) lists with max latency per TP in ms
        """
        n_tps = len(self.traffic_patterns)
        read_all = dist_rt.allgather_obj(my_read)
        write_all = dist_rt.allgather_obj(my_write)
        rdma_all = dist_rt.allgather_obj(my_rdma)

        def max_per_tp(all_ranks: List[List[float]]) -> List[float]:
            """Compute max latency across ranks for each TP, filtering 0.0 values."""
            return [
                max([r[i] for r in all_ranks if r[i] > 0], default=0) * 1e3
                for i in range(n_tps)
            ]

        return max_per_tp(read_all), max_per_tp(write_all), max_per_tp(rdma_all)

    def _print_summary(
        self,
        results: Dict[str, Any],
        isolated_read_ms: List[float],
        isolated_write_ms: List[float],
        isolated_rdma_ms: List[float],
    ) -> None:
        """Print benchmark summary table (rank 0 only).

        Uses cross-rank aggregated data from iter_results for accurate averages.

        Args:
            results: Full results dict with iterations_results
            isolated_read_ms: Isolated read latencies (ms) per TP
            isolated_write_ms: Isolated write latencies (ms) per TP
            isolated_rdma_ms: Isolated RDMA latencies (ms) per TP
        """
        if self.my_rank != 0:
            return

        n_tps = len(self.traffic_patterns)
        n_iters = len(results["iterations_results"])

        logger.info("═" * 80)
        logger.info("BENCHMARK SUMMARY (%d iterations)", n_iters)
        headers = [
            "TP",
            "Size(GB)",
            "Read(ms)",
            "Write(ms)",
            "RDMA(ms)",
            "Iso.Rd(ms)",
            "Iso.Wr(ms)",
            "Iso.RDMA(ms)",
            "Rd(GB/s)",
            "Wr(GB/s)",
            "RDMA(GB/s)",
        ]
        data = []

        for tp_idx in range(n_tps):
            # Average across iterations using cross-rank aggregated data
            avg_read = (
                sum(
                    results["iterations_results"][i][tp_idx]["storage_read_avg_ms"]
                    for i in range(n_iters)
                )
                / n_iters
            )
            avg_write = (
                sum(
                    results["iterations_results"][i][tp_idx]["storage_write_avg_ms"]
                    for i in range(n_iters)
                )
                / n_iters
            )
            avg_rdma = (
                sum(
                    results["iterations_results"][i][tp_idx]["latency"] or 0
                    for i in range(n_iters)
                )
                / n_iters
            )
            avg_rdma_bw = (
                sum(
                    results["iterations_results"][i][tp_idx]["mean_bw"]
                    for i in range(n_iters)
                )
                / n_iters
            )
            size_gb = results["iterations_results"][0][tp_idx]["size"]

            # Calculate BW for each type (GB/s = size_gb / time_sec)
            read_bw = size_gb / (avg_read / 1e3) if avg_read > 0 else 0
            write_bw = size_gb / (avg_write / 1e3) if avg_write > 0 else 0

            data.append(
                [
                    tp_idx,
                    f"{size_gb:.3f}",
                    f"{avg_read:.1f}" if avg_read > 0 else "-",
                    f"{avg_write:.1f}" if avg_write > 0 else "-",
                    f"{avg_rdma:.1f}" if avg_rdma > 0 else "-",
                    (
                        f"{isolated_read_ms[tp_idx]:.1f}"
                        if isolated_read_ms[tp_idx] > 0
                        else "-"
                    ),
                    (
                        f"{isolated_write_ms[tp_idx]:.1f}"
                        if isolated_write_ms[tp_idx] > 0
                        else "-"
                    ),
                    (
                        f"{isolated_rdma_ms[tp_idx]:.1f}"
                        if isolated_rdma_ms[tp_idx] > 0
                        else "-"
                    ),
                    f"{read_bw:.1f}" if read_bw > 0 else "-",
                    f"{write_bw:.1f}" if write_bw > 0 else "-",
                    f"{avg_rdma_bw:.1f}" if avg_rdma_bw > 0 else "-",
                ]
            )
        logger.info("\n%s", tabulate(data, headers=headers))
        logger.info("═" * 80)

    def _destroy(self, handles: List[Any]):
        """Release all NIXL resources: transfer handles, remote agents, buffers, and storage.

        Handles can be in different formats depending on source:
        - dict with "handle" key (from some internal structures)
        - object with .handle attribute (wrapper objects)
        - raw handle (from storage backend's get_read_handle/get_write_handle)
        """
        for h in handles:
            # Extract raw handle from dict/object wrapper, or use as-is if already raw
            handle = (
                h.get("handle", h) if isinstance(h, dict) else getattr(h, "handle", h)
            )
            try:
                self.nixl_agent.release_xfer_handle(handle)
            except Exception:
                pass  # Ignore errors during cleanup

        # Remove connections to remote NIXL agents
        for other in range(self.world_size):
            if other != self.my_rank:
                try:
                    self.nixl_agent.remove_remote_agent(f"{other}")
                except Exception:
                    pass

        # Deregister memory buffers from NIXL
        self._destroy_buffers()

        # Close storage backend (closes file descriptors, deregisters file memory)
        if self._storage_backend:
            self._storage_backend.close()

    def run(
        self,
        verify_buffers: bool = False,
        print_recv_buffers: bool = False,
        json_output_path: Optional[str] = None,
    ):
        """Run sequential perftest with optional storage: READ → COMPUTE → WRITE → RDMA.

        Args:
            verify_buffers: Whether to verify buffer contents after transfer
            print_recv_buffers: Whether to print receive buffer contents
            json_output_path: Path to save results in JSON format
        """
        logger.debug("[Rank %d] Running sequential CT perftest", self.my_rank)
        self._init_buffers()
        self._prepare_storage()
        self._share_md()

        results: Dict[str, Any] = {
            "iterations_results": [],
            "metadata": {
                "ts": time.time(),
                "iters": [{} for _ in range(self.n_iters)],
                "storage_enabled": self._has_storage,
            },
        }

        # Prepare RDMA handles
        rdma_handles_by_tp, tp_bufs = [], []
        prepare_start = time.time()
        for tp in self.traffic_patterns:
            rdma_handles, send_bufs, recv_bufs = self._prepare_tp(tp)
            tp_bufs.append((send_bufs, recv_bufs))
            rdma_handles_by_tp.append(rdma_handles)
        results["metadata"]["prepare_tp_time"] = time.time() - prepare_start
        logger.info(
            "[Rank %d] Prepared %d TPs in %.3fs",
            self.my_rank,
            len(self.traffic_patterns),
            results["metadata"]["prepare_tp_time"],
        )

        # Prepare storage handles
        storage_read_handles_by_tp = [
            self._prepare_storage_read(i) if self._has_storage else []
            for i in range(len(self.traffic_patterns))
        ]
        storage_write_handles_by_tp = [
            self._prepare_storage_write(i) if self._has_storage else []
            for i in range(len(self.traffic_patterns))
        ]

        # Combined handles per TP: (rdma, read, write) - each can be empty []
        tp_handles = list(
            zip(
                rdma_handles_by_tp,
                storage_read_handles_by_tp,
                storage_write_handles_by_tp,
            )
        )

        # WARMUP - Storage then RDMA
        logger.info("[Rank %d] Warming up (%d iters)", self.my_rank, self.warmup_iters)
        for rdma_h, read_h, write_h in tp_handles:
            for _ in range(self.warmup_iters):
                if read_h:
                    self._run_tp(read_h, blocking=True)
                if write_h:
                    self._run_tp(write_h, blocking=True)

        # RDMA warmup: only warmup connections to destinations we haven't warmed yet.
        # This optimization skips redundant warmup when multiple TPs share destinations.
        warmed: set[int] = set()
        for tp_idx, (rdma_h, _, _) in enumerate(tp_handles):
            if not rdma_h:
                continue
            tp = self.traffic_patterns[tp_idx]
            # Get destinations this rank sends to in this TP
            dests = set(tp.receivers_ranks(from_ranks=[self.my_rank]))
            # Skip if all destinations already warmed by previous TPs
            if dests.issubset(warmed):
                continue
            for _ in range(self.warmup_iters):
                self._run_tp(rdma_h, blocking=True)
            warmed.update(dests)
        dist_rt.barrier()
        logger.info("[Rank %d] ✓ Warmup done", self.my_rank)

        # ISOLATED LATENCY - measure baseline performance without noise
        logger.info(
            "[Rank %d] Running isolated benchmark (to measure perf without noise)",
            self.my_rank,
        )
        results["metadata"]["sol_calculation_ts"] = time.time()
        n_tps = len(self.traffic_patterns)

        # Measure isolated latencies using helper methods
        my_isolated_read = self._measure_isolated_storage_latency(
            storage_read_handles_by_tp, "READ"
        )
        my_isolated_write = self._measure_isolated_storage_latency(
            storage_write_handles_by_tp, "WRITE"
        )
        my_isolated_rdma = self._measure_isolated_rdma_latency(rdma_handles_by_tp)

        # Aggregate across ranks (max latency per TP determines overall performance)
        (
            isolated_read_latencies_ms,
            isolated_write_latencies_ms,
            isolated_rdma_latencies_ms,
        ) = self._aggregate_isolated_latencies(
            my_isolated_read, my_isolated_write, my_isolated_rdma
        )

        # WORKLOAD BENCHMARK - execute full flow per iteration
        logger.info(
            "[Rank %d] Running benchmark (%d iters, %d TPs)",
            self.my_rank,
            self.n_iters,
            n_tps,
        )
        total_storage_read_time = [0.0] * n_tps
        total_storage_write_time = [0.0] * n_tps
        total_rdma_time = [0.0] * n_tps
        tp_sizes_gb = [
            self._get_tp_total_size(tp) / 1e9 for tp in self.traffic_patterns
        ]

        for iter_idx in range(self.n_iters):
            logger.debug(
                "[Rank %d] Running iteration %d/%d",
                self.my_rank,
                iter_idx + 1,
                self.n_iters,
            )
            iter_metadata = results["metadata"]["iters"][iter_idx]
            # Global barrier before iteration (matches old code behavior)
            dist_rt.barrier(timeout_sec=None)
            iter_metadata["start_ts"] = time.time()

            # Per-TP timing for this iteration
            iter_storage_read = [0.0] * n_tps
            iter_storage_write = [0.0] * n_tps
            iter_rdma = [0.0] * n_tps
            rdma_start_ts: List[Optional[float]] = [None] * n_tps
            rdma_end_ts: List[Optional[float]] = [None] * n_tps

            # Execute each TP sequentially: READ → COMPUTE → WRITE → RDMA
            # Each rank participates if it has any role: sender or storage.
            # Sequential execution ensures TPs don't overlap and timing is accurate.
            for tp_idx, (rdma_h, read_h, write_h) in enumerate(tp_handles):
                tp = self.traffic_patterns[tp_idx]
                is_sender = self.my_rank in tp.senders_ranks()
                has_storage = tp.storage_ops and self.my_rank in tp.storage_ops

                # Skip if this rank has no role in this TP
                # For RDMA-only (no storage), only senders participate
                if tp.storage_ops:
                    if not is_sender and not has_storage:
                        continue
                else:
                    if not is_sender:
                        continue

                # Barrier ensures all participating ranks start together
                # Includes senders + storage participants (not receivers)
                self._barrier_tp(tp, senders_only=True)

                logger.debug(
                    "[Rank %d] Running TP %d/%d",
                    self.my_rank,
                    tp_idx,
                    len(tp_handles),
                )

                # Step 1: Storage READ
                if read_h and has_storage:
                    start = time.time()
                    self._run_tp(read_h, blocking=True)
                    # Extra barrier ensures all storage reads complete before sleep
                    if tp.storage_ops:
                        self._barrier_tp(tp, senders_only=False, include_storage=True)

                # Step 2: Compute sleep (simulates GPU work between storage and RDMA)
                if tp.compute_time_sec:
                    time.sleep(tp.compute_time_sec)

                # Step 3: Storage WRITE
                if write_h and has_storage:
                    start = time.time()
                    self._run_tp(write_h, blocking=True)
                    iter_storage_write[tp_idx] = time.time() - start
                    # No need to barrier here because storage writes are independent of RDMA

                # Step 4: RDMA transfer
                if is_sender:
                    start = time.time()
                    self._run_tp(rdma_h, blocking=True)
                    end = time.time()
                    iter_rdma[tp_idx] = end - start
                    rdma_start_ts[tp_idx] = start
                    rdma_end_ts[tp_idx] = end

                if tp.sleep_after_launch_sec:
                    time.sleep(tp.sleep_after_launch_sec)

            # Accumulate totals across iterations
            for i in range(n_tps):
                total_storage_read_time[i] += iter_storage_read[i]
                total_storage_write_time[i] += iter_storage_write[i]
                total_rdma_time[i] += iter_rdma[i]

            # Store iteration metadata
            iter_metadata["tps_start_ts"] = rdma_start_ts[:]
            iter_metadata["tps_end_ts"] = rdma_end_ts[:]
            iter_metadata["storage_read_times"] = iter_storage_read[:]
            iter_metadata["storage_write_times"] = iter_storage_write[:]
            iter_metadata["rdma_times"] = iter_rdma[:]

            # Gather timing from all ranks for cross-rank analysis
            rdma_starts_all = dist_rt.allgather_obj(rdma_start_ts)
            rdma_ends_all = dist_rt.allgather_obj(rdma_end_ts)
            storage_read_all = dist_rt.allgather_obj(iter_storage_read)
            storage_write_all = dist_rt.allgather_obj(iter_storage_write)

            # Calculate cross-rank RDMA latencies and bandwidth.
            # Cross-rank latency = wall-clock time from when first rank started to when last finished.
            # This captures the full TP completion time including any stragglers.
            rdma_latencies_ms: List[Optional[float]] = []
            mean_bw = 0.0
            for tp_idx, tp in enumerate(self.traffic_patterns):
                # Collect non-None timestamps from all ranks for this TP
                starts = [
                    rdma_starts_all[r][tp_idx]
                    for r in range(len(rdma_starts_all))
                    if rdma_starts_all[r][tp_idx]
                ]
                ends = [
                    rdma_ends_all[r][tp_idx]
                    for r in range(len(rdma_ends_all))
                    if rdma_ends_all[r][tp_idx]
                ]

                if not ends or not starts:
                    rdma_latencies_ms.append(None)
                else:
                    # Cross-rank latency: earliest start → latest end (captures full TP duration)
                    rdma_latencies_ms.append((max(ends) - min(starts)) * 1e3)
                    # Per-sender bandwidth: each sender's data size / their individual transfer time
                    sender_bws = [
                        tp.total_src_size(r)
                        * 1e-9
                        / (rdma_ends_all[r][tp_idx] - rdma_starts_all[r][tp_idx])
                        for r in tp.senders_ranks()
                        if rdma_starts_all[r][tp_idx] and rdma_ends_all[r][tp_idx]
                    ]
                    mean_bw = sum(sender_bws) / max(1, len(sender_bws))

            if verify_buffers:
                for tp_idx, tp in enumerate(self.traffic_patterns):
                    self._verify_tp(tp, tp_bufs[tp_idx][1], print_recv_buffers)

            # Build per-TP results with per-rank metrics
            iter_results = []
            for tp_idx, tp in enumerate(self.traffic_patterns):
                # Per-rank storage times (ms) for min/max/avg
                read_times_ms = [
                    storage_read_all[r][tp_idx] * 1e3
                    for r in range(len(storage_read_all))
                    if storage_read_all[r][tp_idx] > 0
                ]
                write_times_ms = [
                    storage_write_all[r][tp_idx] * 1e3
                    for r in range(len(storage_write_all))
                    if storage_write_all[r][tp_idx] > 0
                ]

                # Per-rank RDMA bandwidth
                bw_per_rank = []
                for rank in tp.senders_ranks():
                    if rdma_starts_all[rank][tp_idx] and rdma_ends_all[rank][tp_idx]:
                        duration = (
                            rdma_ends_all[rank][tp_idx] - rdma_starts_all[rank][tp_idx]
                        )
                        bw_per_rank.append(tp.total_src_size(rank) * 1e-9 / duration)

                iter_results.append(
                    {
                        # Original keys (backward compatible with plot.py)
                        "size": tp_sizes_gb[tp_idx],
                        "latency": rdma_latencies_ms[tp_idx],
                        "isolated_latency": isolated_rdma_latencies_ms[tp_idx],
                        "mean_bw": mean_bw,
                        # Extended metadata
                        "tp_idx": tp_idx,
                        "has_storage": bool(tp.storage_ops),
                        "num_senders": len(tp.senders_ranks()),
                        "bw_min_gbps": min(bw_per_rank, default=0),
                        "bw_max_gbps": max(bw_per_rank, default=0),
                        "min_start_ts": min(
                            (
                                rdma_starts_all[r][tp_idx]
                                for r in range(len(rdma_starts_all))
                                if rdma_starts_all[r][tp_idx]
                            ),
                            default=None,
                        ),
                        "max_end_ts": max(
                            (
                                rdma_ends_all[r][tp_idx]
                                for r in range(len(rdma_ends_all))
                                if rdma_ends_all[r][tp_idx]
                            ),
                            default=None,
                        ),
                        # Storage metrics (cross-rank aggregated)
                        # Use MAX for workload latency (matches isolated measurement)
                        # Pipeline is limited by slowest rank
                        "storage_read_min_ms": min(read_times_ms, default=0),
                        "storage_read_max_ms": max(read_times_ms, default=0),
                        "storage_read_avg_ms": max(
                            read_times_ms, default=0
                        ),  # MAX for consistency with isolated
                        "storage_write_min_ms": min(write_times_ms, default=0),
                        "storage_write_max_ms": max(write_times_ms, default=0),
                        "storage_write_avg_ms": max(
                            write_times_ms, default=0
                        ),  # MAX for consistency with isolated
                        "isolated_read_ms": isolated_read_latencies_ms[tp_idx],
                        "isolated_write_ms": isolated_write_latencies_ms[tp_idx],
                    }
                )
            results["iterations_results"].append(iter_results)

            # Per-iteration logging with all metrics
            if self.my_rank == 0:
                headers = [
                    "Size(GB)",
                    "Read(ms)",
                    "Write(ms)",
                    "RDMA(ms)",
                    "Iso.Rd(ms)",
                    "Iso.Wr(ms)",
                    "Iso.RDMA(ms)",
                    "Rd(GB/s)",
                    "Wr(GB/s)",
                    "RDMA(GB/s)",
                ]
                data = []
                for i in range(len(self.traffic_patterns)):
                    # Get cross-rank max for this iteration (from iter_results)
                    ir = iter_results[i]
                    size = tp_sizes_gb[i]
                    read_ms = ir["storage_read_avg_ms"]
                    write_ms = ir["storage_write_avg_ms"]
                    # Calculate BW (GB/s = size_gb / time_sec)
                    read_bw = size / (read_ms / 1e3) if read_ms > 0 else 0
                    write_bw = size / (write_ms / 1e3) if write_ms > 0 else 0
                    data.append(
                        [
                            size,
                            read_ms if read_ms > 0 else None,
                            write_ms if write_ms > 0 else None,
                            rdma_latencies_ms[i],
                            (
                                ir["isolated_read_ms"]
                                if ir["isolated_read_ms"] > 0
                                else None
                            ),
                            (
                                ir["isolated_write_ms"]
                                if ir["isolated_write_ms"] > 0
                                else None
                            ),
                            isolated_rdma_latencies_ms[i],
                            read_bw if read_bw > 0 else None,
                            write_bw if write_bw > 0 else None,
                            ir["mean_bw"],
                        ]
                    )
                logger.info(
                    "Iteration %d/%d\n%s",
                    iter_idx + 1,
                    self.n_iters,
                    tabulate(data, headers=headers, floatfmt=".3f", missingval="-"),
                )

        # SUMMARY (rank 0 only) - print benchmark results table
        self._print_summary(
            results,
            isolated_read_latencies_ms,
            isolated_write_latencies_ms,
            isolated_rdma_latencies_ms,
        )

        # CLEANUP
        results["metadata"]["finished_ts"] = time.time()
        if json_output_path and self.my_rank == 0:
            logger.info("Saving results to %s", json_output_path)
            with open(json_output_path, "w") as f:
                json.dump(results, f)

        logger.info("[Rank %d] Finished run, destroying objects", self.my_rank)
        # Collect all transfer handles for cleanup (flatten list of lists)
        all_handles = [
            h
            for hs in rdma_handles_by_tp
            + storage_read_handles_by_tp
            + storage_write_handles_by_tp
            for h in hs
        ]
        self._destroy(all_handles)
