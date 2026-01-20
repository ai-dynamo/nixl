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

"""Sequential is different from multi in that every rank processes only one TP at a time, but they can process different ones"""

import json
import os
import time
from collections import defaultdict
from enum import Enum, auto
from itertools import chain
from pathlib import Path
from test.custom_traffic_perftest import CTPerftest, NixlBuffer, StorageXferHandle
from test.storage_backend import FilesystemBackend, StorageBackend, StorageHandle
from test.traffic_pattern import TrafficPattern
from typing import Any, Dict, List, Optional

import yaml
from runtime.etcd_rt import etcd_dist_utils as dist_rt
from tabulate import tabulate

from nixl._api import nixl_agent
from nixl._api import nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


class StorageOpType(Enum):
    """Type of storage operation."""
    READ = auto()
    WRITE = auto()


class SequentialCTPerftest(CTPerftest):
    """Extends CTPerftest to handle multiple traffic patterns sequentially.
    The patterns are executed in sequence, and the results are aggregated.

    Allows testing multiple communication patterns sequentially between distributed processes.
    """

    def __init__(
        self,
        traffic_patterns: list[TrafficPattern],
        n_iters: int = 3,
        n_isolation_iters=30,
        warmup_iters=30,
        # Storage options (optional)
        storage_path: Optional[Path] = None,
        storage_nixl_backend: Optional[str] = None,
        storage_direct_io: bool = False,
    ) -> None:
        """Initialize multi-pattern performance test.

        Args:
            traffic_patterns: List of traffic patterns to test simultaneously
            storage_path: Optional base path for storage operations
            storage_nixl_backend: Storage backend type (POSIX, GDS, GDS_MT)
            storage_direct_io: Whether to use O_DIRECT for storage I/O
        """
        self.my_rank = dist_rt.get_rank()
        self.world_size = dist_rt.get_world_size()
        self.traffic_patterns = traffic_patterns
        self.n_iters = n_iters
        self.n_isolation_iters = n_isolation_iters
        self.warmup_iters = warmup_iters

        # Storage setup
        self._has_storage = any(tp.storage_ops for tp in traffic_patterns) and storage_path
        self._storage_backend: Optional[StorageBackend] = None
        self._storage_handles: Dict[str, StorageHandle] = {}
        self._storage_nixl_backend: Optional[str] = None
        
        # Check if any TP has RDMA (matrix is not None)
        self._has_rdma = any(tp.matrix is not None for tp in traffic_patterns)

        logger.debug("[Rank %d] Initializing Nixl agent", self.my_rank)
        if self._has_storage:
            # Create agent without auto UCX - we'll create GDS first, then UCX
            config = nixl_agent_config(backends=[])
            self.nixl_agent = nixl_agent(f"{self.my_rank}", config)
        else:
            self.nixl_agent = nixl_agent(f"{self.my_rank}")

        for tp in self.traffic_patterns:
            self._check_tp_config(tp)
        if not os.environ.get("CUDA_VISIBLE_DEVICES") and any(
            tp.mem_type == "cuda" for tp in self.traffic_patterns
        ):
            logger.warning(
                "Cuda buffers detected, but the env var CUDA_VISIBLE_DEVICES is not set, this will cause every process in the same host to use the same GPU device."
            )
        # UCX is required only if we have RDMA traffic patterns
        if self._has_rdma:
            assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"

        # NixlBuffer caches buffers and reuse them if they are big enough, let's initialize them once, with the largest needed size
        self.send_buf_by_mem_type: dict[str, NixlBuffer] = {}
        self.recv_buf_by_mem_type: dict[str, NixlBuffer] = {}

        # Initialize storage backend if needed
        if self._has_storage:
            nixl_backend = storage_nixl_backend or "POSIX"
            self._storage_nixl_backend = nixl_backend
            use_direct_io = storage_direct_io or nixl_backend in ("GDS", "GDS_MT")
            logger.info(
                "[Rank %d] Storage: %s, backend=%s, O_DIRECT=%s",
                self.my_rank, storage_path, nixl_backend, use_direct_io,
            )
            self._storage_backend = FilesystemBackend(
                agent=self.nixl_agent,
                base_path=storage_path,
                nixl_backend=nixl_backend,
                use_direct_io=use_direct_io,
            )
            # Only create UCX if we have RDMA traffic patterns
            if self._has_rdma:
                self.nixl_agent.create_backend("UCX")

    # =========================================================================
    # STORAGE METHODS
    # =========================================================================

    def _run_isolated_storage_benchmark(
        self,
        storage_handles: List[List[Any]],
        op_type: str,  # "read" or "write"
    ) -> List[Dict[str, float]]:
        """Run isolated storage benchmark for all TPs.
        
        Returns list of stat dicts per TP for this rank.
        Each dict has: avg, p50, p90, p99, min, max (all in seconds)
        """
        empty_stats = {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
        my_stats = [empty_stats.copy() for _ in self.traffic_patterns]
        
        for tp_ix in range(len(self.traffic_patterns)):
            handles = storage_handles[tp_ix]
            tp = self.traffic_patterns[tp_ix]
            
            # Get ranks that have storage ops for this TP
            storage_ranks = set()
            if tp.storage_ops:
                for rank, ops in tp.storage_ops.items():
                    size = ops.read_size if op_type == "read" else ops.write_size
                    if size > 0:
                        storage_ranks.add(rank)
            
            dist_rt.barrier()  # Global barrier before each TP
            
            if self.my_rank not in storage_ranks:
                continue
            
            dist_rt.barrier(storage_ranks)  # Barrier among participating ranks
            
            iter_latencies = []
            for iter_idx in range(self.n_isolation_iters):
                dist_rt.barrier(storage_ranks)  # Sync before each iteration
                t = time.time()
                self._run_tp(handles, blocking=True)
                iter_latency = time.time() - t
                iter_latencies.append(iter_latency)
                if iter_idx < 3 or iter_idx == self.n_isolation_iters - 1:
                    logger.info(
                        "[Rank %d] Isolated %s TP %d iter %d: %.3f ms",
                        self.my_rank, op_type, tp_ix, iter_idx, iter_latency * 1e3
                    )
            
            # Calculate fio-style percentile statistics
            sorted_lats = sorted(iter_latencies)
            n = len(sorted_lats)
            avg_lat = sum(iter_latencies) / n
            p50 = sorted_lats[n // 2]
            p90 = sorted_lats[int(n * 0.9)]
            p99 = sorted_lats[int(n * 0.99)] if n >= 100 else sorted_lats[-1]
            
            my_stats[tp_ix] = {
                "avg": avg_lat,
                "p50": p50,
                "p90": p90,
                "p99": p99,
                "min": sorted_lats[0],
                "max": sorted_lats[-1],
            }
            logger.info(
                "[Rank %d] Isolated %s TP %d: avg=%.3f p50=%.3f p90=%.3f p99=%.3f ms (min=%.3f max=%.3f)",
                self.my_rank, op_type, tp_ix, avg_lat * 1e3, p50 * 1e3, p90 * 1e3, p99 * 1e3,
                sorted_lats[0] * 1e3, sorted_lats[-1] * 1e3
            )
            
            # Barrier after TP completes - ensures all participating ranks finish
            # before next TP starts (prevents OST contention spillover)
            dist_rt.barrier(storage_ranks)
        
        return my_stats

    def _get_storage_key(self, tp_idx: int) -> str:
        """Get storage handle key for a traffic pattern index."""
        return f"{tp_idx}:{self.my_rank}"

    def _prepare_storage(self):
        """Prepare all storage handles for traffic patterns with storage ops."""
        if not self._has_storage or not self._storage_backend:
            return
        
        for tp_idx in range(len(self.traffic_patterns)):
            tp = self.traffic_patterns[tp_idx]
            if not tp.storage_ops:
                continue
            my_ops = tp.storage_ops.get(self.my_rank)
            if my_ops:
                self._storage_handles[self._get_storage_key(tp_idx)] = self._storage_backend.prepare(
                    tp_idx=tp_idx,
                    rank=self.my_rank,
                    read_size=my_ops.read_size,
                    write_size=my_ops.write_size,
                )
        logger.info("[Rank %d] Prepared %d storage handles", self.my_rank, len(self._storage_handles))

    def _prepare_storage_xfer(self, tp_idx: int, operation: StorageOpType) -> List[StorageXferHandle]:
        """Prepare a NIXL transfer handle for storage read or write."""
        if not self._storage_backend:
            return []
        storage_handle = self._storage_handles.get(self._get_storage_key(tp_idx))
        if not storage_handle:
            return []
        op_name = "read" if operation == StorageOpType.READ else "write"
        if operation == StorageOpType.READ:
            if storage_handle.read_size == 0:
                return []
            size = storage_handle.read_size
            buf_offset = 0
            get_handle_fn = self._storage_backend.get_read_handle
        else:
            if storage_handle.write_size == 0:
                return []
            size = storage_handle.write_size
            buf_offset = storage_handle.read_size
            get_handle_fn = self._storage_backend.get_write_handle
        buf = self.send_buf_by_mem_type.get(self.traffic_patterns[tp_idx].mem_type)
        if not buf:
            return []
        raw_xfer = get_handle_fn(storage_handle, buf.get_chunk(size, offset=buf_offset))
        if not raw_xfer:
            return []
        # Wrap raw handle in StorageXferHandle for consistent interface
        file_path = str(storage_handle.backend_data.get("file_path", f"tp_{tp_idx}_rank_{self.my_rank}"))
        return [StorageXferHandle(raw_xfer, file_path, op_name)]

    def _prepare_storage_read(self, tp_idx: int) -> List[Any]:
        """Get storage read transfer handle."""
        return self._prepare_storage_xfer(tp_idx, StorageOpType.READ)

    def _prepare_storage_write(self, tp_idx: int) -> List[Any]:
        """Get storage write transfer handle."""
        return self._prepare_storage_xfer(tp_idx, StorageOpType.WRITE)

    # =========================================================================
    # BUFFER METHODS (with storage support)
    # =========================================================================

    def _init_buffers(self):
        """Initialize buffers with aligned size calculation.
        
        Buffer size accounts for alignment padding between chunks.
        Uses max across all TPs since each TP reuses the same buffer.
        """
        from test.custom_traffic_perftest import NixlBuffer
        
        logger.debug("[Rank %d] Initializing buffers", self.my_rank)
        max_src_by_mem_type = defaultdict(int)
        max_dst_by_mem_type = defaultdict(int)

        for tp in self.traffic_patterns:
            # Calculate aligned RDMA buffer sizes
            if tp.matrix is not None:
                send_sizes = [int(tp.matrix[self.my_rank][dst]) for dst in range(tp.matrix.shape[1])]
                recv_sizes = [int(tp.matrix[src][self.my_rank]) for src in range(tp.matrix.shape[0])]
                rdma_send = NixlBuffer.aligned_total_size(send_sizes)
                rdma_recv = NixlBuffer.aligned_total_size(recv_sizes)
            else:
                rdma_send = rdma_recv = 0
            
            # Include storage sizes (already 4K aligned in main.py)
            my_ops = tp.storage_ops.get(self.my_rank) if tp.storage_ops else None
            storage_size = (my_ops.read_size + my_ops.write_size) if my_ops else 0
            
            max_src_by_mem_type[tp.mem_type] = max(
                max_src_by_mem_type[tp.mem_type], rdma_send, storage_size
            )
            max_dst_by_mem_type[tp.mem_type] = max(
                max_dst_by_mem_type[tp.mem_type], rdma_recv
            )

        # If storage is enabled, also register buffers with storage backend
        storage_backends = [self._storage_nixl_backend] if self._storage_nixl_backend else None

        for mem_type, size in max_src_by_mem_type.items():
            if not size:
                continue
            self.send_buf_by_mem_type[mem_type] = NixlBuffer(
                size, mem_type=mem_type, nixl_agent=self.nixl_agent, backends=storage_backends
            )

        for mem_type, size in max_dst_by_mem_type.items():
            if not size:
                continue
            self.recv_buf_by_mem_type[mem_type] = NixlBuffer(
                size, mem_type=mem_type, nixl_agent=self.nixl_agent, backends=storage_backends
            )

    def _destroy_buffers(self):
        logger.debug("[Rank %d] Destroying buffers", self.my_rank)
        for buf in chain(
            self.send_buf_by_mem_type.values(), self.recv_buf_by_mem_type.values()
        ):
            buf.destroy()

    def _get_bufs(self, tp: TrafficPattern):
        """Get send/recv buffers for a traffic pattern with aligned offsets."""
        from test.custom_traffic_perftest import NixlBuffer
        
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
                # Align offset before getting chunk
                send_offset_by_memtype[tp.mem_type] = NixlBuffer.align_up(
                    send_offset_by_memtype[tp.mem_type]
                )
                send_buf = self.send_buf_by_mem_type[tp.mem_type].get_chunk(
                    send_size, send_offset_by_memtype[tp.mem_type]
                )
                send_offset_by_memtype[tp.mem_type] += send_size
            if recv_size > 0:
                # Align offset before getting chunk
                recv_offset_by_memtype[tp.mem_type] = NixlBuffer.align_up(
                    recv_offset_by_memtype[tp.mem_type]
                )
                recv_buf = self.recv_buf_by_mem_type[tp.mem_type].get_chunk(
                    recv_size, recv_offset_by_memtype[tp.mem_type]
                )
                recv_offset_by_memtype[tp.mem_type] += recv_size

            send_bufs[other_rank] = send_buf
            recv_bufs[other_rank] = recv_buf

        return send_bufs, recv_bufs

    # =========================================================================
    # MAIN RUN METHOD
    # =========================================================================

    def run(
        self,
        verify_buffers: bool = False,
        print_recv_buffers: bool = False,
        json_output_path: Optional[str] = None,
    ):
        """
        Args:
            verify_buffers: Whether to verify buffer contents after transfer
            print_recv_buffers: Whether to print receive buffer contents
            json_output_path: Path to save results in YAML format

        Returns:
            Total execution time in seconds

        This method initializes and executes multiple traffic patterns simultaneously,
        measures their performance, and optionally verifies the results.
        """
        logger.debug("[Rank %d] Running sequential CT perftest", self.my_rank)
        self._init_buffers()
        # Only exchange metadata if we have RDMA traffic patterns
        if self._has_rdma:
            self._share_md()
        self._prepare_storage()  # <-- ADDED: prepare storage handles

        results: Dict[str, Any] = {
            "iterations_results": [],
            "metadata": {
                "ts": time.time(),
                "iters": [{} for _ in range(self.n_iters)],
                "storage_enabled": self._has_storage,  # <-- ADDED
            },
        }

        tp_handles: list[list] = []
        tp_bufs = []

        s = time.time()
        logger.info("[Rank %d] Preparing TPs", self.my_rank)
        for i, tp in enumerate(self.traffic_patterns):
            handles, send_bufs, recv_bufs = self._prepare_tp(tp)
            # Consider:Only prepare RDMA handles for patterns with a matrix
            #if tp.matrix is not None:
            #    handles, send_bufs, recv_bufs = self._prepare_tp(tp)
            #else:
                # Storage-only pattern - no RDMA handles needed
            #    handles, send_bufs, recv_bufs = [], [None] * self.world_size, [None] * self.world_size
            tp_bufs.append((send_bufs, recv_bufs))
            tp_handles.append(handles)

        results["metadata"]["prepare_tp_time"] = time.time() - s

        # ADDED: Prepare storage handles
        storage_read_handles = [self._prepare_storage_read(i) for i in range(len(self.traffic_patterns))]
        storage_write_handles = [self._prepare_storage_write(i) for i in range(len(self.traffic_patterns))]

        # RDMA Warmup
        warm_dsts: set[int] = set()
        for tp_ix, handles in enumerate(tp_handles):
            if not handles:  # Skip storage-only patterns (no RDMA handles)
                continue
            tp = self.traffic_patterns[tp_ix]
            dsts = set(tp.receivers_ranks(from_ranks=[self.my_rank]))
            if dsts.issubset(warm_dsts):
                # All the dsts have been warmed up
                continue
            for _ in range(self.warmup_iters):
                self._run_tp(handles, blocking=True)
            warm_dsts.update(dsts)

        # Storage warmup
        if self._has_storage:
            warmup_start = time.time()
            for tp_idx in range(len(self.traffic_patterns)):
                read_h = storage_read_handles[tp_idx]
                write_h = storage_write_handles[tp_idx]
                # Log start of warmup for this TP
                if read_h or write_h:
                    logger.info("[Rank %d] Starting warmup TP %d/%d, read=%s, write=%s",
                                self.my_rank, tp_idx + 1, len(self.traffic_patterns), bool(read_h), bool(write_h))
                for _ in range(self.warmup_iters):
                    if read_h:
                        self._run_tp(read_h, blocking=True)
                    if write_h:
                        self._run_tp(write_h, blocking=True)
                # Log progress every TP
                if read_h or write_h:
                    logger.info("[Rank %d] Warmup TP %d/%d done, elapsed=%.1fs",
                                self.my_rank, tp_idx + 1, len(self.traffic_patterns), time.time() - warmup_start)            

        dist_rt.barrier()
        if self.my_rank == 0:
            logger.info("[Rank 0] All ranks finished warmup, starting isolated benchmark")

        # Isolated mode -  Measure SOL for every matrix
        logger.info(
            "[Rank %d] Running isolated benchmark (to measure perf without noise)",
            self.my_rank,
        )
        empty_stats = {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
        my_isolated_rdma_stats: List[Dict] = [empty_stats.copy() for _ in tp_handles]
        my_isolated_read_stats: List[Dict] = [empty_stats.copy() for _ in tp_handles]
        my_isolated_write_stats: List[Dict] = [empty_stats.copy() for _ in tp_handles]

        results["metadata"]["sol_calculation_ts"] = time.time()

        # Isolated RDMA measurement
        for tp_ix, handles in enumerate(tp_handles):
            tp = self.traffic_patterns[tp_ix]
            dist_rt.barrier()
            if self.my_rank not in tp.senders_ranks():
                continue

            self._barrier_tp(tp)

            iter_latencies = []
            for iter_idx in range(self.n_isolation_iters):
                t = time.time()
                self._run_tp(handles, blocking=True)
                iter_latency = time.time() - t
                iter_latencies.append(iter_latency)
                self._barrier_tp(tp)
                if iter_idx < 3 or iter_idx == self.n_isolation_iters - 1:
                    logger.info(
                        "[Rank %d] Isolated RDMA TP %d iter %d: %.3f ms",
                        self.my_rank, tp_ix, iter_idx, iter_latency * 1e3
                    )

            # Calculate fio-style percentile statistics
            sorted_lats = sorted(iter_latencies)
            n = len(sorted_lats)
            avg_lat = sum(iter_latencies) / n
            p50 = sorted_lats[n // 2]
            p90 = sorted_lats[int(n * 0.9)]
            p99 = sorted_lats[int(n * 0.99)] if n >= 100 else sorted_lats[-1]
            
            my_isolated_rdma_stats[tp_ix] = {
                "avg": avg_lat,
                "p50": p50,
                "p90": p90,
                "p99": p99,
                "min": sorted_lats[0],
                "max": sorted_lats[-1],
            }
            logger.info(
                "[Rank %d] Isolated RDMA TP %d: avg=%.3f p50=%.3f p90=%.3f p99=%.3f ms (min=%.3f max=%.3f)",
                self.my_rank, tp_ix, avg_lat * 1e3, p50 * 1e3, p90 * 1e3, p99 * 1e3,
                sorted_lats[0] * 1e3, sorted_lats[-1] * 1e3
            )

        # Isolated storage read/write measurements
        if self._has_storage:
            logger.info("[Rank %d] Running isolated storage read benchmark", self.my_rank)
            my_isolated_read_stats = self._run_isolated_storage_benchmark(
                storage_read_handles, "read"
            )
            
            logger.info("[Rank %d] Running isolated storage write benchmark", self.my_rank)
            my_isolated_write_stats = self._run_isolated_storage_benchmark(
                storage_write_handles, "write"
            )

        # Store isolated results
        isolated_rdma_stats_by_ranks = dist_rt.allgather_obj(my_isolated_rdma_stats)
        isolated_read_stats_by_ranks = dist_rt.allgather_obj(my_isolated_read_stats)
        isolated_write_stats_by_ranks = dist_rt.allgather_obj(my_isolated_write_stats)

        # Process isolated stats per TP - aggregate across ranks
        # For each stat, take max across ranks (bottleneck determines performance)
        def aggregate_stats(stats_by_ranks, tp_idx):
            """Aggregate stats across ranks for a TP. Returns dict with p50, p90, p99, min, max in ms."""
            empty = {"p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
            p50s = [r[tp_idx]["p50"] for r in stats_by_ranks if r[tp_idx]["p50"] > 0]
            if not p50s:
                return empty
            p90s = [r[tp_idx]["p90"] for r in stats_by_ranks if r[tp_idx]["p90"] > 0]
            p99s = [r[tp_idx]["p99"] for r in stats_by_ranks if r[tp_idx]["p99"] > 0]
            mins = [r[tp_idx]["min"] for r in stats_by_ranks if r[tp_idx]["min"] > 0]
            maxs = [r[tp_idx]["max"] for r in stats_by_ranks if r[tp_idx]["max"] > 0]
            return {
                "p50": max(p50s) * 1e3,
                "p90": max(p90s) * 1e3 if p90s else 0.0,
                "p99": max(p99s) * 1e3 if p99s else 0.0,
                "min": min(mins) * 1e3 if mins else 0.0,
                "max": max(maxs) * 1e3 if maxs else 0.0,
            }

        isolated_rdma_stats_ms = []
        isolated_read_stats_ms = []
        isolated_write_stats_ms = []
        
        for i in range(len(self.traffic_patterns)):
            isolated_rdma_stats_ms.append(aggregate_stats(isolated_rdma_stats_by_ranks, i))
            isolated_read_stats_ms.append(aggregate_stats(isolated_read_stats_by_ranks, i))
            isolated_write_stats_ms.append(aggregate_stats(isolated_write_stats_by_ranks, i))

        logger.info("[Rank %d] Running workload benchmark", self.my_rank)

        # Workload mode - Measure perf of the matrices while running the full workload
        for iter_ix in range(self.n_iters):
            logger.debug(
                "[Rank %d] Running iteration %d/%d",
                self.my_rank,
                iter_ix + 1,
                self.n_iters,
            )
            iter_metadata = results["metadata"]["iters"][iter_ix]

            tp_starts: list[float | None] = [None] * len(tp_handles)
            tp_ends: list[float | None] = [None] * len(tp_handles)
            # Storage timing per TP (start timestamp and duration)
            storage_read_times: list[float] = [0.0] * len(tp_handles)
            storage_write_times: list[float] = [0.0] * len(tp_handles)
            storage_read_starts: list[float | None] = [None] * len(tp_handles)
            storage_read_ends: list[float | None] = [None] * len(tp_handles)
            logger.debug("[Rank %d] Warmup done.", self.my_rank)
            dist_rt.barrier(timeout_sec=None)

            iter_metadata["start_ts"] = time.time()
            for tp_ix, handles in enumerate(tp_handles):
                tp = self.traffic_patterns[tp_ix]

                # Check if this rank participates (RDMA or storage)
                if self.my_rank not in tp.all_participating_ranks():
                    continue

                self._barrier_tp(tp)
                if tp.sleep_before_launch_sec is not None:
                    time.sleep(tp.sleep_before_launch_sec)

                # Determine what operations this TP has (for barrier logic)
                read_h = storage_read_handles[tp_ix]
                write_h = storage_write_handles[tp_ix]
                has_rdma = len(tp.senders_ranks()) > 0
                has_read = tp.storage_ops and any(ops.read_size > 0 for ops in tp.storage_ops.values())
                has_write = tp.storage_ops and any(ops.write_size > 0 for ops in tp.storage_ops.values())

                # Storage READ (before RDMA)
                if read_h:
                    read_start = time.time()
                    self._run_tp(read_h, blocking=True)
                    read_end = time.time()
                    storage_read_times[tp_ix] = read_end - read_start
                    storage_read_starts[tp_ix] = read_start
                    storage_read_ends[tp_ix] = read_end

                # Barrier between storage read and RDMA (if both exist)
                if has_read and has_rdma:
                    self._barrier_tp(tp, include_storage=True)

                # Run RDMA transfer (only if this rank is a sender)
                if self.my_rank in tp.senders_ranks():
                    logger.debug(
                        "[Rank %d] Running TP %d/%d",
                        self.my_rank,
                        tp_ix,
                        len(tp_handles),
                    )

                    tp_start_ts = time.time()
                    self._run_tp(handles, blocking=True)
                    tp_end_ts = time.time()

                    logger.debug(
                        "[Rank %d] TP %d took %.3f seconds",
                        self.my_rank,
                        tp_ix,
                        tp_end_ts - tp_start_ts,
                    )

                    tp_starts[tp_ix] = tp_start_ts
                    tp_ends[tp_ix] = tp_end_ts

                # Barrier between RDMA and storage write (if both exist)
                if has_rdma and has_write:
                    self._barrier_tp(tp, include_storage=True)

                # Storage WRITE (after RDMA)
                if write_h:
                    write_start = time.time()
                    self._run_tp(write_h, blocking=True)
                    storage_write_times[tp_ix] = time.time() - write_start

                if tp.sleep_after_launch_sec is not None:
                    time.sleep(tp.sleep_after_launch_sec)

            iter_metadata["tps_start_ts"] = tp_starts.copy()
            iter_metadata["tps_end_ts"] = tp_ends.copy()

            tp_starts_by_ranks = dist_rt.allgather_obj(tp_starts)
            tp_ends_by_ranks = dist_rt.allgather_obj(tp_ends)
            # Gather storage times from all ranks
            storage_read_by_ranks = dist_rt.allgather_obj(storage_read_times)
            storage_write_by_ranks = dist_rt.allgather_obj(storage_write_times)
            # Gather storage start/end timestamps for overlap analysis
            storage_read_starts_by_ranks = dist_rt.allgather_obj(storage_read_starts)
            storage_read_ends_by_ranks = dist_rt.allgather_obj(storage_read_ends)
            
            # Analyze TP overlap (only rank 0 prints)
            if self.my_rank == 0 and self._has_storage:
                logger.info("=== TP TIMING OVERLAP ANALYSIS (iter %d) ===", iter_ix)
                for tp_idx in range(len(self.traffic_patterns)):
                    # Get all ranks' start/end for this TP
                    starts = [storage_read_starts_by_ranks[r][tp_idx] for r in range(self.world_size) 
                              if storage_read_starts_by_ranks[r][tp_idx] is not None]
                    ends = [storage_read_ends_by_ranks[r][tp_idx] for r in range(self.world_size)
                            if storage_read_ends_by_ranks[r][tp_idx] is not None]
                    if starts and ends:
                        tp_start = min(starts)
                        tp_end = max(ends)
                        node = tp_idx // 2 % 12  # Approximate node from TP index
                        logger.info("  TP %2d (node ~%d): %.3f - %.3f (%.1f ms)", 
                                    tp_idx, node, tp_start, tp_end, (tp_end - tp_start) * 1000)
                
                # Check for overlap between consecutive TPs
                overlap_count = 0
                for tp_idx in range(len(self.traffic_patterns) - 1):
                    ends_curr = [storage_read_ends_by_ranks[r][tp_idx] for r in range(self.world_size)
                                 if storage_read_ends_by_ranks[r][tp_idx] is not None]
                    starts_next = [storage_read_starts_by_ranks[r][tp_idx + 1] for r in range(self.world_size)
                                   if storage_read_starts_by_ranks[r][tp_idx + 1] is not None]
                    if ends_curr and starts_next:
                        if max(ends_curr) > min(starts_next):
                            overlap_count += 1
                logger.info("=== OVERLAP: %d TP pairs overlap in time ===", overlap_count)

            tp_latencies_ms: list[float | None] = []
            storage_read_max_ms: list[float] = []
            storage_write_max_ms: list[float] = []

            tp_sizes_gb = [
                self._get_tp_total_size(tp) / 1e9 for tp in self.traffic_patterns
            ]

            # Calculate total storage sizes per TP (sum across all ranks)
            storage_read_sizes_gb: list[float] = []
            storage_write_sizes_gb: list[float] = []
            for tp in self.traffic_patterns:
                read_total = 0
                write_total = 0
                if tp.storage_ops:
                    for ops in tp.storage_ops.values():
                        read_total += ops.read_size
                        write_total += ops.write_size
                storage_read_sizes_gb.append(read_total / 1e9)
                storage_write_sizes_gb.append(write_total / 1e9)

            for i, tp in enumerate(self.traffic_patterns):
                starts = [
                    tp_starts_by_ranks[rank][i]
                    for rank in range(len(tp_starts_by_ranks))
                ]
                ends = [
                    tp_ends_by_ranks[rank][i] for rank in range(len(tp_ends_by_ranks))
                ]
                starts = [x for x in starts if x is not None]
                ends = [x for x in ends if x is not None]

                # Max storage times across ranks (slowest determines pipeline)
                read_times = [storage_read_by_ranks[r][i] for r in range(len(storage_read_by_ranks)) if storage_read_by_ranks[r][i] > 0]
                write_times = [storage_write_by_ranks[r][i] for r in range(len(storage_write_by_ranks)) if storage_write_by_ranks[r][i] > 0]
                storage_read_max_ms.append(max(read_times) * 1e3 if read_times else 0.0)
                storage_write_max_ms.append(max(write_times) * 1e3 if write_times else 0.0)

                mean_bw = 0.0  # Default for storage-only patterns
                if not ends or not starts:
                    tp_latencies_ms.append(None)
                else:
                    tp_latencies_ms.append((max(ends) - min(starts)) * 1e3)

                    senders = tp.senders_ranks()
                    for rank in senders:
                        rank_start = tp_starts_by_ranks[rank][i]
                        rank_end = tp_ends_by_ranks[rank][i]
                        if not rank_start or not rank_end:
                            raise ValueError(
                                f"Rank {rank} has no start or end time, but participated in TP, this is not normal."
                            )
                        mean_bw += (
                            tp.total_src_size(rank) * 1e-9 / (rank_end - rank_start)
                        )

                    if senders:
                        mean_bw /= len(senders)

            if self.my_rank == 0:
                headers = [
                    "RDMA (GB)",
                    "RDMA (ms)",
                    "Iso p50",
                    "Iso p90",
                    "RDMA BW",
                    "Iso BW",
                    "Read (GB)",
                    "Read (ms)",
                    "Rd p50",
                    "Rd p90",
                    "Read BW",
                    "Iso Rd BW",
                    "Write (GB)",
                    "Write (ms)",
                    "Wr p50",
                    "Wr p90",
                    "Write BW",
                    "Iso Wr BW",
                ]
                data = []
                for i, tp in enumerate(self.traffic_patterns):
                    read_ms = storage_read_max_ms[i]
                    write_ms = storage_write_max_ms[i]
                    iso_rdma_stats = isolated_rdma_stats_ms[i]
                    iso_read_stats = isolated_read_stats_ms[i]
                    iso_write_stats = isolated_write_stats_ms[i]
                    read_size = storage_read_sizes_gb[i]
                    write_size = storage_write_sizes_gb[i]

                    # Calculate aggregate BWs from isolated p50 latencies (GB/s)
                    iso_read_p50 = iso_read_stats["p50"]
                    iso_write_p50 = iso_write_stats["p50"]
                    read_bw = (read_size / (iso_read_p50 / 1e3)) if iso_read_p50 > 0 else None
                    write_bw = (write_size / (iso_write_p50 / 1e3)) if iso_write_p50 > 0 else None

                    # Calculate per-rank isolated BWs using each rank's p50
                    # Then take min (bottleneck BW) across ranks
                    
                    # RDMA: per-rank BW = rank's size / rank's p50 latency
                    rdma_bws = []
                    for rank in tp.senders_ranks():
                        rank_stats = isolated_rdma_stats_by_ranks[rank][i]
                        if rank_stats["p50"] > 0:
                            rank_size_gb = tp.total_src_size(rank) * 1e-9
                            rdma_bws.append(rank_size_gb / rank_stats["p50"])
                    iso_rdma_bw = min(rdma_bws) if rdma_bws else None
                    
                    # Storage read: per-rank BW = rank's read_size / rank's p50
                    read_bws = []
                    if tp.storage_ops:
                        for rank, ops in tp.storage_ops.items():
                            if ops.read_size > 0:
                                rank_stats = isolated_read_stats_by_ranks[rank][i]
                                if rank_stats["p50"] > 0:
                                    read_bws.append((ops.read_size * 1e-9) / rank_stats["p50"])
                    iso_read_bw = min(read_bws) if read_bws else None
                    
                    # Storage write: per-rank BW = rank's write_size / rank's p50
                    write_bws = []
                    if tp.storage_ops:
                        for rank, ops in tp.storage_ops.items():
                            if ops.write_size > 0:
                                rank_stats = isolated_write_stats_by_ranks[rank][i]
                                if rank_stats["p50"] > 0:
                                    write_bws.append((ops.write_size * 1e-9) / rank_stats["p50"])
                    iso_write_bw = min(write_bws) if write_bws else None

                    data.append([
                        tp_sizes_gb[i],
                        tp_latencies_ms[i],
                        iso_rdma_stats["p50"] if iso_rdma_stats["p50"] > 0 else None,
                        iso_rdma_stats["p90"] if iso_rdma_stats["p90"] > 0 else None,
                        mean_bw,
                        iso_rdma_bw,
                        read_size if read_size > 0 else None,
                            read_ms if read_ms > 0 else None,
                        iso_read_p50 if iso_read_p50 > 0 else None,
                        iso_read_stats["p90"] if iso_read_stats["p90"] > 0 else None,
                        read_bw,
                        iso_read_bw,
                        write_size if write_size > 0 else None,
                            write_ms if write_ms > 0 else None,
                        iso_write_p50 if iso_write_p50 > 0 else None,
                        iso_write_stats["p90"] if iso_write_stats["p90"] > 0 else None,
                        write_bw,
                        iso_write_bw,
                    ])
                logger.info(
                    f"Iteration {iter_ix + 1}/{self.n_iters}\n{tabulate(data, headers=headers, floatfmt='.3f', missingval='-')}"
                )

                # Per-rank detailed report (only on last iteration to avoid spam)
                if iter_ix == self.n_iters - 1:
                    logger.info("Per-rank performance breakdown:")
                    for tp_idx, tp in enumerate(self.traffic_patterns):
                        rank_headers = ["Rank", "RDMA (GB)", "RDMA BW", "Iso BW", "Read (GB)", "Read BW", "Write (GB)", "Write BW"]
                        rank_data = []
                        for rank in range(self.world_size):
                            rdma_start = tp_starts_by_ranks[rank][tp_idx]
                            rdma_end = tp_ends_by_ranks[rank][tp_idx]
                            rdma_sec = (rdma_end - rdma_start) if rdma_start and rdma_end else 0
                            iso_rdma_sec = isolated_rdma_stats_by_ranks[rank][tp_idx]["p50"]
                            read_sec = storage_read_by_ranks[rank][tp_idx]
                            iso_read_sec = isolated_read_stats_by_ranks[rank][tp_idx]["p50"]
                            write_sec = storage_write_by_ranks[rank][tp_idx]
                            iso_write_sec = isolated_write_stats_by_ranks[rank][tp_idx]["p50"]
                            
                            # Calculate per-rank sizes and BWs (GB/s)
                            rdma_size_gb = tp.total_src_size(rank) * 1e-9 if rank in tp.senders_ranks() else 0
                            rdma_bw = (rdma_size_gb / rdma_sec) if rdma_sec > 0 else None
                            iso_rdma_bw = (rdma_size_gb / iso_rdma_sec) if iso_rdma_sec > 0 else None
                            
                            read_size_gb = (tp.storage_ops[rank].read_size * 1e-9) if tp.storage_ops and rank in tp.storage_ops else 0
                            read_bw = (read_size_gb / read_sec) if read_sec > 0 else None
                            
                            write_size_gb = (tp.storage_ops[rank].write_size * 1e-9) if tp.storage_ops and rank in tp.storage_ops else 0
                            write_bw = (write_size_gb / write_sec) if write_sec > 0 else None
                            
                            # Only add rows for ranks that participated
                            if any([rdma_bw, iso_rdma_bw, read_bw, write_bw]):
                                rank_data.append([rank, 
                                    rdma_size_gb if rdma_size_gb > 0 else None, rdma_bw, iso_rdma_bw,
                                    read_size_gb if read_size_gb > 0 else None, read_bw,
                                    write_size_gb if write_size_gb > 0 else None, write_bw])
                        
                        if rank_data:
                            logger.info(f"TP {tp_idx}:\n{tabulate(rank_data, headers=rank_headers, floatfmt='.3f', missingval='-')}")

            if verify_buffers:
                for i, tp in enumerate(self.traffic_patterns):
                    send_bufs, recv_bufs = tp_bufs[i]
                    self._verify_tp(tp, recv_bufs, print_recv_buffers)

            iter_results = []
            for i, tp in enumerate(self.traffic_patterns):
                starts = list(filter(None, (tp_starts_by_ranks[rank][i] for rank in range(len(tp_starts_by_ranks)))))
                ends = list(filter(None, (tp_ends_by_ranks[rank][i] for rank in range(len(tp_ends_by_ranks)))))
                # Storage timing for overlap analysis
                stor_starts = [storage_read_starts_by_ranks[r][i] for r in range(self.world_size) 
                               if storage_read_starts_by_ranks[r][i] is not None]
                stor_ends = [storage_read_ends_by_ranks[r][i] for r in range(self.world_size)
                             if storage_read_ends_by_ranks[r][i] is not None]
                iter_results.append({
                    "size": tp_sizes_gb[i],
                    "latency": tp_latencies_ms[i],
                    # RDMA isolated stats (fio-style percentiles)
                    "isolated_rdma_p50_ms": isolated_rdma_stats_ms[i]["p50"],
                    "isolated_rdma_p90_ms": isolated_rdma_stats_ms[i]["p90"],
                    "isolated_rdma_p99_ms": isolated_rdma_stats_ms[i]["p99"],
                    "isolated_rdma_min_ms": isolated_rdma_stats_ms[i]["min"],
                    "isolated_rdma_max_ms": isolated_rdma_stats_ms[i]["max"],
                    "num_senders": len(tp.senders_ranks()),
                    "mean_bw": mean_bw,
                    "min_start_ts": min(starts) if starts else None,
                    "max_end_ts": max(ends) if ends else None,
                    # Storage workload timing (with timestamps for overlap analysis)
                    "storage_read_avg_ms": storage_read_max_ms[i],
                    "storage_write_avg_ms": storage_write_max_ms[i],
                    "storage_read_start_ts": min(stor_starts) if stor_starts else None,
                    "storage_read_end_ts": max(stor_ends) if stor_ends else None,
                    # Storage isolated stats (fio-style percentiles)
                    "isolated_read_p50_ms": isolated_read_stats_ms[i]["p50"],
                    "isolated_read_p90_ms": isolated_read_stats_ms[i]["p90"],
                    "isolated_read_p99_ms": isolated_read_stats_ms[i]["p99"],
                    "isolated_read_min_ms": isolated_read_stats_ms[i]["min"],
                    "isolated_read_max_ms": isolated_read_stats_ms[i]["max"],
                    "isolated_write_p50_ms": isolated_write_stats_ms[i]["p50"],
                    "isolated_write_p90_ms": isolated_write_stats_ms[i]["p90"],
                    "isolated_write_p99_ms": isolated_write_stats_ms[i]["p99"],
                    "isolated_write_min_ms": isolated_write_stats_ms[i]["min"],
                    "isolated_write_max_ms": isolated_write_stats_ms[i]["max"],
                    "storage_read_size_gb": storage_read_sizes_gb[i],
                    "storage_write_size_gb": storage_write_sizes_gb[i],
                })
            results["iterations_results"].append(iter_results)

        results["metadata"]["finished_ts"] = time.time()
        if json_output_path and self.my_rank == 0:
            logger.info("Saving results to %s", json_output_path)
            with open(json_output_path, "w") as f:
                # Use default=str to handle Path objects
                json.dump(results, f, default=str)

        # Destroy
        logger.info("[Rank %d] Finished run, destroying objects", self.my_rank)
        all_handles = [h for hs in tp_handles + storage_read_handles + storage_write_handles for h in hs]
        self._destroy(all_handles)

        # ADDED: Close storage backend
        if self._storage_backend:
            self._storage_backend.close()

    def _write_yaml_results(
        self,
        output_path: str,
        headers: List[str],
        data: List[List],
        traffic_patterns: List[TrafficPattern],
    ) -> None:
        """Write performance test results to a YAML file.

        Args:
            output_path: Path to save the YAML file
            headers: Column headers for the results
            data: Performance data rows
            traffic_patterns: List of traffic patterns tested
        """
        results: Dict[str, Any] = {
            "performance_results": {
                "timestamp": time.time(),
                "world_size": self.world_size,
                "traffic_patterns": [],
            }
        }

        for i in range(len(traffic_patterns)):
            tp_data = {}
            for j, header in enumerate(headers):
                # Convert header to a valid YAML key
                key = header.lower().replace(" ", "_").replace("(", "").replace(")", "")
                # Format floating point values to 2 decimal places for readability
                if isinstance(data[i][j], float):
                    tp_data[key] = round(data[i][j], 2)
                else:
                    tp_data[key] = data[i][j]

            # Add traffic pattern name or index for reference
            tp_data["pattern_index"] = i

            # You can add more pattern-specific information here if needed
            # For example:
            # tp_data["sender_ranks"] = list(tp.senders_ranks())

            results["performance_results"]["traffic_patterns"].append(tp_data)

        try:
            with open(output_path, "w") as f:
                yaml.dump(results, f, default_flow_style=False, sort_keys=False)
            logger.info("Results saved to YAML file: %s", output_path)
        except Exception as e:
            logger.error("Failed to write YAML results to %s: %s", output_path, e)
