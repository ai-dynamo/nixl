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
        assert "UCX" in self.nixl_agent.get_plugin_list() or self._has_storage, "UCX plugin is not loaded"

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
            self.nixl_agent.create_backend("UCX")

    # =========================================================================
    # STORAGE METHODS
    # =========================================================================

    def _get_storage_key(self, tp_idx: int) -> str:
        """Get storage handle key for a traffic pattern index."""
        return f"{tp_idx}:{self.my_rank}"

    def _prepare_storage(self):
        """Prepare all storage handles for traffic patterns with storage ops."""
        if not self._has_storage or not self._storage_backend:
            return
        for tp_idx, tp in enumerate(self.traffic_patterns):
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
        logger.debug("[Rank %d] Initializing buffers", self.my_rank)
        max_src_by_mem_type = defaultdict(int)
        max_dst_by_mem_type = defaultdict(int)

        for tp in self.traffic_patterns:
            # Include storage sizes in buffer calculation
            my_ops = tp.storage_ops.get(self.my_rank) if tp.storage_ops else None
            storage_size = (my_ops.read_size + my_ops.write_size) if my_ops else 0
            max_src_by_mem_type[tp.mem_type] = max(
                max_src_by_mem_type[tp.mem_type], tp.total_src_size(self.my_rank), storage_size
            )
            max_dst_by_mem_type[tp.mem_type] = max(
                max_dst_by_mem_type[tp.mem_type], tp.total_dst_size(self.my_rank)
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
            yaml_output_path: Path to save results in YAML format

        Returns:
            Total execution time in seconds

        This method initializes and executes multiple traffic patterns simultaneously,
        measures their performance, and optionally verifies the results.
        """
        logger.debug("[Rank %d] Running sequential CT perftest", self.my_rank)
        self._init_buffers()
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
            tp_bufs.append((send_bufs, recv_bufs))
            tp_handles.append(handles)

        results["metadata"]["prepare_tp_time"] = time.time() - s

        # ADDED: Prepare storage handles
        storage_read_handles = [self._prepare_storage_read(i) for i in range(len(self.traffic_patterns))]
        storage_write_handles = [self._prepare_storage_write(i) for i in range(len(self.traffic_patterns))]

        # Warmup
        warm_dsts: set[int] = set()
        for tp_ix, handles in enumerate(tp_handles):
            dsts = set(tp.receivers_ranks(from_ranks=[self.my_rank]))
            if dsts.issubset(warm_dsts):
                # All the dsts have been warmed up
                continue
            for _ in range(self.warmup_iters):
                self._run_tp(handles, blocking=True)
            warm_dsts.update(dsts)

        # ADDED: Storage warmup
        if self._has_storage:
            for read_h, write_h in zip(storage_read_handles, storage_write_handles):
                for _ in range(self.warmup_iters):
                    if read_h:
                        self._run_tp(read_h, blocking=True)
                    if write_h:
                        self._run_tp(write_h, blocking=True)

        dist_rt.barrier()

        # Isolated mode -  Measure SOL for every matrix
        logger.info(
            "[Rank %d] Running isolated benchmark (to measure perf without noise)",
            self.my_rank,
        )
        my_isolated_tp_latencies: list[float] = [0 for _ in tp_handles]
        my_isolated_read_latencies: list[float] = [0 for _ in tp_handles]
        my_isolated_write_latencies: list[float] = [0 for _ in tp_handles]

        results["metadata"]["sol_calculation_ts"] = time.time()

        # Isolated RDMA measurement
        for tp_ix, handles in enumerate(tp_handles):
            tp = self.traffic_patterns[tp_ix]
            dist_rt.barrier()
            if self.my_rank not in tp.senders_ranks():
                continue

            self._barrier_tp(tp)

            for _ in range(self.n_isolation_iters):
                t = time.time()
                self._run_tp(handles, blocking=True)
                e = time.time()
                my_isolated_tp_latencies[tp_ix] += e - t
                self._barrier_tp(tp)

            logger.debug(
                "[Rank %d] Ran %d isolated iters for tp %d/%d, took %.3f secs",
                self.my_rank,
                self.n_isolation_iters,
                tp_ix,
                len(tp_handles),
                e - t,
            )

            my_isolated_tp_latencies[tp_ix] /= self.n_isolation_iters

        # Isolated storage read measurement
        if self._has_storage:
            logger.info("[Rank %d] Running isolated storage read benchmark", self.my_rank)
            for tp_ix, read_h in enumerate(storage_read_handles):
                dist_rt.barrier()
                if not read_h:
                    continue
                for _ in range(self.n_isolation_iters):
                    t = time.time()
                    self._run_tp(read_h, blocking=True)
                    my_isolated_read_latencies[tp_ix] += time.time() - t
                my_isolated_read_latencies[tp_ix] /= self.n_isolation_iters

        # Isolated storage write measurement
        if self._has_storage:
            logger.info("[Rank %d] Running isolated storage write benchmark", self.my_rank)
            for tp_ix, write_h in enumerate(storage_write_handles):
                dist_rt.barrier()
                if not write_h:
                    continue
                for _ in range(self.n_isolation_iters):
                    t = time.time()
                    self._run_tp(write_h, blocking=True)
                    my_isolated_write_latencies[tp_ix] += time.time() - t
                my_isolated_write_latencies[tp_ix] /= self.n_isolation_iters

        # Store isolated results
        isolated_tp_latencies_by_ranks = dist_rt.allgather_obj(my_isolated_tp_latencies)
        isolated_read_latencies_by_ranks = dist_rt.allgather_obj(my_isolated_read_latencies)
        isolated_write_latencies_by_ranks = dist_rt.allgather_obj(my_isolated_write_latencies)

        isolated_tp_latencies_ms = []
        isolated_read_latencies_ms = []
        isolated_write_latencies_ms = []
        for i in range(len(self.traffic_patterns)):
            # RDMA
            tp_lats = [
                rank_lats[i]
                for rank_lats in isolated_tp_latencies_by_ranks
                if rank_lats[i] > 0
            ]
            if tp_lats:
                isolated_tp_latencies_ms.append(max(tp_lats) * 1e3)
            else:
                isolated_tp_latencies_ms.append(0.0)

            # Storage read
            read_lats = [
                rank_lats[i]
                for rank_lats in isolated_read_latencies_by_ranks
                if rank_lats[i] > 0
            ]
            isolated_read_latencies_ms.append(max(read_lats) * 1e3 if read_lats else 0.0)

            # Storage write
            write_lats = [
                rank_lats[i]
                for rank_lats in isolated_write_latencies_by_ranks
                if rank_lats[i] > 0
            ]
            isolated_write_latencies_ms.append(max(write_lats) * 1e3 if write_lats else 0.0)

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
            # Storage timing per TP
            storage_read_times: list[float] = [0.0] * len(tp_handles)
            storage_write_times: list[float] = [0.0] * len(tp_handles)
            logger.debug("[Rank %d] Warmup done.", self.my_rank)
            dist_rt.barrier(timeout_sec=None)

            iter_metadata["start_ts"] = time.time()
            for tp_ix, handles in enumerate(tp_handles):
                tp = self.traffic_patterns[tp_ix]

                if self.my_rank not in tp.senders_ranks():
                    continue

                self._barrier_tp(tp)
                if tp.sleep_before_launch_sec is not None:
                    time.sleep(tp.sleep_before_launch_sec)

                # Storage READ (before RDMA)
                read_h = storage_read_handles[tp_ix]
                if read_h:
                    read_start = time.time()
                    self._run_tp(read_h, blocking=True)
                    storage_read_times[tp_ix] = time.time() - read_start

                # Run RDMA transfer
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

                # Storage WRITE (after RDMA)
                write_h = storage_write_handles[tp_ix]
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

                if not ends or not starts:
                    tp_latencies_ms.append(None)
                else:
                    tp_latencies_ms.append((max(ends) - min(starts)) * 1e3)

                    mean_bw = 0.0
                    for rank in tp.senders_ranks():
                        rank_start = tp_starts_by_ranks[rank][i]
                        rank_end = tp_ends_by_ranks[rank][i]
                        if not rank_start or not rank_end:
                            raise ValueError(
                                f"Rank {rank} has no start or end time, but participated in TP, this is not normal."
                            )
                        mean_bw += (
                            tp.total_src_size(rank) * 1e-9 / (rank_end - rank_start)
                        )

                    mean_bw /= len(tp.senders_ranks())

            if self.my_rank == 0:
                headers = [
                    "RDMA (GB)",
                    "RDMA (ms)",
                    "Iso RDMA (ms)",
                    "RDMA BW",
                    "Read (GB)",
                    "Read (ms)",
                    "Iso Read (ms)",
                    "Read BW",
                    "Write (GB)",
                    "Write (ms)",
                    "Iso Write (ms)",
                    "Write BW",
                ]
                data = []
                for i, tp in enumerate(self.traffic_patterns):
                    read_ms = storage_read_max_ms[i]
                    write_ms = storage_write_max_ms[i]
                    iso_read = isolated_read_latencies_ms[i]
                    iso_write = isolated_write_latencies_ms[i]
                    read_size = storage_read_sizes_gb[i]
                    write_size = storage_write_sizes_gb[i]

                    # Calculate BWs from isolated latencies (GB/s)
                    read_bw = (read_size / (iso_read / 1e3)) if iso_read > 0 else None
                    write_bw = (write_size / (iso_write / 1e3)) if iso_write > 0 else None

                    data.append([
                        tp_sizes_gb[i],
                        tp_latencies_ms[i],
                        isolated_tp_latencies_ms[i] if isolated_tp_latencies_ms[i] > 0 else None,
                        mean_bw,
                        read_size if read_size > 0 else None,
                        read_ms if read_ms > 0 else None,
                        iso_read if iso_read > 0 else None,
                        read_bw,
                        write_size if write_size > 0 else None,
                        write_ms if write_ms > 0 else None,
                        iso_write if iso_write > 0 else None,
                        write_bw,
                    ])
                logger.info(
                    f"Iteration {iter_ix + 1}/{self.n_iters}\n{tabulate(data, headers=headers, floatfmt='.3f', missingval='-')}"
                )

            if verify_buffers:
                for i, tp in enumerate(self.traffic_patterns):
                    send_bufs, recv_bufs = tp_bufs[i]
                    self._verify_tp(tp, recv_bufs, print_recv_buffers)

            iter_results = [
                {
                    "size": tp_sizes_gb[i],
                    "latency": tp_latencies_ms[i],
                    "isolated_latency": isolated_tp_latencies_ms[i],
                    "num_senders": len(tp.senders_ranks()),
                    "mean_bw": mean_bw,
                    "min_start_ts": min(
                        filter(
                            None,
                            (
                                tp_starts_by_ranks[rank][i]
                                for rank in range(len(tp_starts_by_ranks))
                            ),
                        )
                    ),
                    "max_end_ts": max(
                        filter(
                            None,
                            (
                                tp_ends_by_ranks[rank][i]
                                for rank in range(len(tp_ends_by_ranks))
                            ),
                        )
                    ),
                }
                for i, tp in enumerate(self.traffic_patterns)
            ]
            results["iterations_results"].append(iter_results)

        results["metadata"]["finished_ts"] = time.time()
        if json_output_path and self.my_rank == 0:
            logger.info("Saving results to %s", json_output_path)
            with open(json_output_path, "w") as f:
                # Use default=str to handle Path objects
                json.dump(results, f, default=str)

        # Destroy
        logger.info("[Rank %d] Finished run, destroying objects", self.my_rank)
        all_handles = [h for hs in tp_handles + storage_read_handles + storage_write_handles for h in hs]  # <-- MODIFIED
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
