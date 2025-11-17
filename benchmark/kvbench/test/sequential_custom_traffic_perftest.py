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
import shutil
import time
from collections import defaultdict
from itertools import chain
from test.custom_traffic_perftest import CTPerftest, NixlBuffer
from test.traffic_pattern import TrafficPattern
from typing import Any, Dict, List, Optional, Tuple

import yaml
from runtime.etcd_rt import etcd_dist_utils as dist_rt
from tabulate import tabulate

from nixl._api import nixl_agent
from nixl.logging import get_logger

logger = get_logger(__name__)


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
        base_storage_path: Optional[str] = None,
    ) -> None:
        """Initialize multi-pattern performance test.

        Args:
            traffic_patterns: List of traffic patterns to test simultaneously
        """
        self.my_rank = dist_rt.get_rank()
        self.world_size = dist_rt.get_world_size()
        self.traffic_patterns = traffic_patterns
        self.n_iters = n_iters
        self.n_isolation_iters = n_isolation_iters
        self.warmup_iters = warmup_iters

        logger.debug("[Rank %d] Initializing Nixl agent", self.my_rank)
        self.nixl_agent = nixl_agent(f"{self.my_rank}")

        for tp in self.traffic_patterns:
            self._check_tp_config(tp)
        if not os.environ.get("CUDA_VISIBLE_DEVICES") and any(
            tp.mem_type == "cuda" for tp in self.traffic_patterns
        ):
            logger.warning(
                "Cuda buffers detected, but the env var CUDA_VISIBLE_DEVICES is not set, this will cause every process in the same host to use the same GPU device."
            )
        # Require UCX only if any TP contains rank<->rank flows
        need_ucx = any(
            tp.matrix.shape[0] >= self.world_size
            and tp.matrix.shape[1] >= self.world_size
            and (tp.matrix[: self.world_size, : self.world_size].sum() > 0)
            for tp in self.traffic_patterns
        )
        if need_ucx:
            assert (
                "UCX" in self.nixl_agent.get_plugin_list()
            ), "UCX plugin is not loaded"

        # If any storage endpoints exist in any TP, require a storage backend (GDS or POSIX)
        has_storage = any(
            (tp.matrix.shape[0] > self.world_size)
            or (tp.matrix.shape[1] > self.world_size)
            for tp in self.traffic_patterns
        )
        if has_storage:
            plugin_list = self.nixl_agent.get_plugin_list()
            assert ("GDS" in plugin_list) or (
                "POSIX" in plugin_list
            ), "No storage backend available (need GDS or POSIX)"
            self._storage_backend_name = "GDS" if "GDS" in plugin_list else "POSIX"
            assert (
                base_storage_path is not None
            ), "base_storage_path is required. Provide --storage-path in CLI."
            self.base_storage_path: str = base_storage_path
            # Initialize chosen storage backend
            try:
                self.nixl_agent.create_backend(self._storage_backend_name)
            except Exception:
                pass

        # NixlBuffer caches buffers and reuse them if they are big enough, let's initialize them once, with the largest needed size
        self.send_buf_by_mem_type: dict[str, NixlBuffer] = {}
        self.recv_buf_by_mem_type: dict[str, NixlBuffer] = {}

        # Per-TP storage state
        self._tp_fds: Dict[int, Dict[int, int]] = {}
        self._tp_file_reg_descs: Dict[int, Any] = {}
        # Per-TP write base offsets for storage endpoints (so writes start after READ region)
        self._tp_write_base: Dict[int, Dict[int, int]] = {}

    def _check_tp_config(self, tp: TrafficPattern):
        # Enforce square matrix, size >= world_size. Extra rows/cols are storage endpoints
        rows, cols = tp.matrix.shape
        assert rows == cols, f"Matrix must be square, got {tp.matrix.shape}"
        assert (
            rows >= self.world_size
        ), f"Matrix must be at least world_size x world_size, got {tp.matrix.shape}, world_size={self.world_size}"

    def _init_buffers(self):
        logger.debug("[Rank %d] Initializing buffers", self.my_rank)
        max_src_by_mem_type = defaultdict(int)
        max_dst_by_mem_type = defaultdict(int)

        for tp in self.traffic_patterns:
            max_src_by_mem_type[tp.mem_type] = max(
                max_src_by_mem_type[tp.mem_type], tp.total_src_size(self.my_rank)
            )
            max_dst_by_mem_type[tp.mem_type] = max(
                max_dst_by_mem_type[tp.mem_type], tp.total_dst_size(self.my_rank)
            )

        for mem_type, size in max_src_by_mem_type.items():
            if not size:
                continue
            self.send_buf_by_mem_type[mem_type] = NixlBuffer(
                size, mem_type=mem_type, nixl_agent=self.nixl_agent
            )

        for mem_type, size in max_dst_by_mem_type.items():
            if not size:
                continue
            self.recv_buf_by_mem_type[mem_type] = NixlBuffer(
                size, mem_type=mem_type, nixl_agent=self.nixl_agent
            )

    def _open_storage_for_tp(self, tp: TrafficPattern) -> None:
        # No storage for this TP if matrix has no extra endpoints
        num_storage = max(0, tp.matrix.shape[0] - self.world_size)
        if num_storage == 0:
            return

        tp_id = tp.id
        if tp_id in self._tp_fds:
            return  # already opened

        # Choose storage path: base path + tp_<id> if provided; else YAML path; else default
        storage_path = os.path.join(self.base_storage_path, f"tp_{tp.id}")
        file_prefix = "obj_"

        # Prepare files (compute sizes, delete dir + prefill on rank 0) and sync once
        (
            read_size_by_storage_idx,
            write_size_by_storage_idx,
        ) = self._prepare_storage_files_for_tp(
            tp, storage_path, file_prefix, num_storage
        )
        # Store write bases: writes start after the READ region
        self._tp_write_base[tp_id] = {
            idx: read_size_by_storage_idx.get(idx, 0) for idx in range(num_storage)
        }

        # All ranks open their own FDs (always request O_DIRECT; fallback if not supported)
        fds = self._open_storage_fds(storage_path, file_prefix, num_storage)

        self._tp_fds[tp_id] = fds
        # Optionally register file memory with GDS for this TP
        try:
            reg_list = []
            for storage_idx, fd in fds.items():
                total_size = read_size_by_storage_idx.get(
                    storage_idx, 0
                ) + write_size_by_storage_idx.get(storage_idx, 0)
                if total_size <= 0:
                    # Register a minimal length to satisfy API
                    total_size = 1
                reg_list.append((0, int(total_size), fd, str(storage_idx)))
            if reg_list:
                file_reg_descs = self.nixl_agent.register_memory(
                    reg_list, "FILE", backends=[self._storage_backend_name]
                )
                self._tp_file_reg_descs[tp_id] = file_reg_descs
        except Exception:
            # If registration fails, we will still attempt xfers using FILE xfer descs
            pass

    def _has_storage_io_for_rank(self, tp: TrafficPattern) -> bool:
        """Return True if current rank has any storage reads or writes in this TP."""
        num_storage = max(0, tp.matrix.shape[0] - self.world_size)
        if num_storage == 0:
            return False
        # Writes to storage (row = my_rank, dst >= world_size)
        for col in range(self.world_size, tp.matrix.shape[1]):
            if int(tp.matrix[self.my_rank][col]) > 0:
                return True
        # Reads from storage (col = my_rank, src >= world_size)
        for row in range(self.world_size, tp.matrix.shape[0]):
            if int(tp.matrix[row][self.my_rank]) > 0:
                return True
        return False

    def _compute_storage_sizes_for_tp(
        self, tp: TrafficPattern, num_storage: int, is_read: bool
    ) -> Dict[int, int]:
        """Compute per-storage-endpoint total bytes.

        - is_read=True: bytes read from storage rows to rank columns
        - is_read=False: bytes written from rank rows to storage columns
        """
        size_by_storage_idx: Dict[int, int] = defaultdict(int)
        for storage_idx in range(num_storage):
            size = 0
            if is_read:
                row_idx = self.world_size + storage_idx
                size = tp.total_src_size_to_ranks(row_idx, self.world_size)
            else:
                col_idx = self.world_size + storage_idx
                size = tp.total_dst_size_from_ranks(col_idx, self.world_size)
            size_by_storage_idx[storage_idx] = size
        return size_by_storage_idx

    def _prepare_storage_files_for_tp(
        self,
        tp: TrafficPattern,
        storage_path: str,
        file_prefix: str,
        num_storage: int,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Compute sizes, clean TP dir, prefill per-endpoint files for this TP, then barrier.

        Returns:
            (read_size_by_storage_idx, write_size_by_storage_idx)
        """
        read_size_by_storage_idx = self._compute_storage_sizes_for_tp(
            tp, num_storage, True
        )
        write_size_by_storage_idx = self._compute_storage_sizes_for_tp(
            tp, num_storage, False
        )

        if dist_rt.get_rank() == 0:
            # Clean TP dir fully (remove if exists), then recreate empty
            if os.path.isdir(storage_path):
                shutil.rmtree(storage_path, ignore_errors=True)
            os.makedirs(storage_path, exist_ok=True)
            total_size_by_idx = {
                idx: read_size_by_storage_idx.get(idx, 0)
                + write_size_by_storage_idx.get(idx, 0)
                for idx in range(num_storage)
            }
            self._truncate_and_prefill_storage_files(
                storage_path,
                file_prefix,
                total_size_by_idx,
                prefill_size_by_storage_idx=read_size_by_storage_idx,
            )
        dist_rt.barrier()

        return read_size_by_storage_idx, write_size_by_storage_idx

    def _truncate_and_prefill_storage_files(
        self,
        storage_path: str,
        file_prefix: str,
        total_size_by_storage_idx: Dict[int, int],
        prefill_size_by_storage_idx: Optional[Dict[int, int]] = None,
    ) -> None:
        for storage_idx, total_bytes in total_size_by_storage_idx.items():
            filename = os.path.join(storage_path, f"{file_prefix}{storage_idx}.bin")
            # Create new file (remove if exists)
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
            fd = os.open(filename, os.O_CREAT | os.O_RDWR)
            try:
                prefill_bytes = (
                    total_bytes
                    if prefill_size_by_storage_idx is None
                    else prefill_size_by_storage_idx.get(storage_idx, 0)
                )
                if prefill_bytes > 0:
                    os.lseek(fd, 0, os.SEEK_SET)
                    chunk_size = min(8 * 1024 * 1024, prefill_bytes)
                    pattern_byte = bytes([storage_idx % 256])
                    chunk = pattern_byte * chunk_size
                    remaining = prefill_bytes
                    pos = 0
                    while remaining > 0:
                        to_write = min(chunk_size, remaining)
                        os.pwrite(fd, chunk[:to_write], pos)
                        pos += to_write
                        remaining -= to_write
                # Ensure file size is exactly total_bytes without truncation: extend if needed
                if total_bytes > 0:
                    os.pwrite(fd, b"\x00", max(0, total_bytes - 1))
            finally:
                os.close(fd)

    def _open_storage_fds(
        self, storage_path: str, file_prefix: str, num_storage: int
    ) -> Dict[int, int]:
        fds: Dict[int, int] = {}
        for storage_idx in range(num_storage):
            filename = os.path.join(storage_path, f"{file_prefix}{storage_idx}.bin")
            flags = os.O_CREAT | os.O_RDWR
            fd = os.open(filename, flags)
            fds[storage_idx] = fd
        return fds

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

    def _compute_tp_chunks(
        self, tp: TrafficPattern
    ) -> Tuple[List[Tuple[int, Any, int]], List[Tuple[int, Any, int]],]:
        """Compute storage write/read chunks for current rank."""
        storage_writes: List[Tuple[int, Any, int]] = []
        storage_reads: List[Tuple[int, Any, int]] = []

        # Base offsets so storage slices don't overlap rank<->rank slices
        send_offset = tp.total_src_size_to_ranks(self.my_rank, self.world_size)
        recv_offset = tp.total_dst_size_from_ranks(self.my_rank, self.world_size)

        # Writes to storage columns (dst >= world_size)
        for dst in range(self.world_size, tp.matrix.shape[1]):
            size = int(tp.matrix[self.my_rank][dst])
            if size <= 0:
                continue
            buf = self.send_buf_by_mem_type[tp.mem_type].get_chunk(size, send_offset)
            send_offset += size
            storage_writes.append((dst - self.world_size, buf, size))

        # Reads from storage rows (src >= world_size)
        for src in range(self.world_size, tp.matrix.shape[0]):
            size = int(tp.matrix[src][self.my_rank])
            if size <= 0:
                continue
            buf = self.recv_buf_by_mem_type[tp.mem_type].get_chunk(size, recv_offset)
            recv_offset += size
            storage_reads.append((src - self.world_size, buf, size))

        return storage_writes, storage_reads

    def _prepare_tp(
        self, tp: TrafficPattern
    ) -> Tuple[List[Any], List[Optional[Any]], List[Optional[Any]]]:
        # Build rank<->rank handles using base implementation
        rank_handles, rank_send_bufs, rank_recv_bufs = super()._prepare_tp(tp)
        handles: list = list(rank_handles)

        # Prepare storage ops (open, compute chunks, initialize xfers)
        storage_handles = self._initialize_storage_xfers(tp)
        handles.extend(storage_handles)

        return handles, rank_send_bufs, rank_recv_bufs

    # _barrier_tp, _run_tp and _wait are inherited from CTPerftest

    # Storage transfers are pre-initialized into handles via _initialize_storage_xfers

    def _initialize_storage_xfers(self, tp: TrafficPattern) -> List[Any]:
        """Open storage, compute storage chunks and create NIXL handles with proper offsets."""
        # Ensure storage is open/registered
        self._open_storage_for_tp(tp)

        # Compute storage chunks locally (avoid overlap with rank<->rank buffers)
        storage_writes: List[Tuple[int, Any, int]] = []
        storage_reads: List[Tuple[int, Any, int]] = []
        send_offset = tp.total_src_size_to_ranks(self.my_rank, self.world_size)
        recv_offset = tp.total_dst_size_from_ranks(self.my_rank, self.world_size)

        # Writes to storage columns (dst >= world_size)
        for dst in range(self.world_size, tp.matrix.shape[1]):
            size = int(tp.matrix[self.my_rank][dst])
            if size <= 0:
                continue
            buf = self.send_buf_by_mem_type[tp.mem_type].get_chunk(size, send_offset)
            send_offset += size
            storage_writes.append((dst - self.world_size, buf, size))

        # Reads from storage rows (src >= world_size)
        for src in range(self.world_size, tp.matrix.shape[0]):
            size = int(tp.matrix[src][self.my_rank])
            if size <= 0:
                continue
            buf = self.recv_buf_by_mem_type[tp.mem_type].get_chunk(size, recv_offset)
            recv_offset += size
            storage_reads.append((src - self.world_size, buf, size))

        # Initialize xfers with correct file offsets
        handles: list[Any] = []
        tp_id = tp.id
        write_base_by_storage_idx = self._tp_write_base.get(tp_id, {})
        write_offset_by_storage: Dict[int, int] = defaultdict(int)
        for k, v in write_base_by_storage_idx.items():
            write_offset_by_storage[k] = int(v)
        read_offset_by_storage: Dict[int, int] = defaultdict(int)

        # Writes: start at write base (after READ region)
        for storage_idx, tensor, size in storage_writes:
            offset = write_offset_by_storage[storage_idx]
            write_offset_by_storage[storage_idx] += size
            fd = self._tp_fds[tp_id][storage_idx]
            local_descs = self.nixl_agent.get_xfer_descs(tensor)
            file_descs = self.nixl_agent.get_xfer_descs(
                [(int(offset), int(size), int(fd))], "FILE"
            )
            h = self.nixl_agent.initialize_xfer(
                "WRITE",
                local_descs,
                file_descs,
                self.nixl_agent.name,
                backends=[self._storage_backend_name],
            )
            handles.append(h)

        # Reads: start at 0 and advance
        for storage_idx, tensor, size in storage_reads:
            offset = read_offset_by_storage[storage_idx]
            read_offset_by_storage[storage_idx] += size
            fd = self._tp_fds[tp_id][storage_idx]
            file_descs = self.nixl_agent.get_xfer_descs(
                [(int(offset), int(size), int(fd))], "FILE"
            )
            local_descs = self.nixl_agent.get_xfer_descs(tensor)
            h = self.nixl_agent.initialize_xfer(
                "READ",
                local_descs,
                file_descs,
                self.nixl_agent.name,
                backends=[self._storage_backend_name],
            )
            handles.append(h)

        return handles

    def _destroy(self, handles: List[Any]):
        logger.debug("[Rank %d] Releasing XFER handles", self.my_rank)
        for h in handles:
            # Support dict, wrapper with .handle, or bare handle
            if isinstance(h, dict):
                handle = h.get("handle", h)
            else:
                handle = getattr(h, "handle", h)
            try:
                self.nixl_agent.release_xfer_handle(handle)
            except Exception:
                pass

        logger.debug("[Rank %d] Removing remote agents", self.my_rank)
        for other_rank in range(self.world_size):
            if other_rank == self.my_rank:
                continue
            self.nixl_agent.remove_remote_agent(f"{other_rank}")

        self._destroy_buffers()

        # Close storage FDs (all TPs)
        for tp_id, fds in self._tp_fds.items():
            for fd in fds.values():
                try:
                    os.close(fd)
                except Exception:
                    pass
        self._tp_fds.clear()
        # Deregister file memory
        for tp_id, reg_descs in self._tp_file_reg_descs.items():
            try:
                self.nixl_agent.deregister_memory(
                    reg_descs, backends=[self._storage_backend_name]
                )
            except Exception:
                pass
        self._tp_file_reg_descs.clear()

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

        results: Dict[str, Any] = {
            "iterations_results": [],
            "metadata": {
                "ts": time.time(),
                "iters": [{} for _ in range(self.n_iters)],
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

        # Warmup
        warm_dsts: set[int] = set()
        for tp_ix, handles in enumerate(tp_handles):
            tp = self.traffic_patterns[tp_ix]
            dsts = set(
                tp.receivers_ranks(
                    from_ranks=[self.my_rank], world_size=self.world_size
                )
            )
            if dsts.issubset(warm_dsts):
                continue
            for _ in range(self.warmup_iters):
                self._run_tp(handles, blocking=True)
            warm_dsts.update(dsts)

        dist_rt.barrier()

        # Isolated mode -  Measure SOL for every matrix
        logger.info(
            "[Rank %d] Running isolated benchmark (to measure perf without noise)",
            self.my_rank,
        )
        my_isolated_tp_latencies: list[float] = [0 for _ in tp_handles]

        results["metadata"]["sol_calculation_ts"] = time.time()
        for tp_ix, handles in enumerate(tp_handles):
            tp = self.traffic_patterns[tp_ix]
            dist_rt.barrier()
            if self.my_rank not in tp.senders_ranks(world_size=self.world_size):
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

        # Store isolated results
        isolated_tp_latencies_by_ranks = dist_rt.allgather_obj(my_isolated_tp_latencies)
        isolated_tp_latencies_ms = []
        for i in range(len(self.traffic_patterns)):
            tp_lats = [
                rank_lats[i]
                for rank_lats in isolated_tp_latencies_by_ranks
                if rank_lats[i] > 0
            ]

            if tp_lats:
                isolated_tp_latencies_ms.append(max(tp_lats) * 1e3)

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

                # Run TP
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

                if tp.sleep_after_launch_sec is not None:
                    time.sleep(tp.sleep_after_launch_sec)

            iter_metadata["tps_start_ts"] = tp_starts.copy()
            iter_metadata["tps_end_ts"] = tp_ends.copy()

            tp_starts_by_ranks = dist_rt.allgather_obj(tp_starts)
            tp_ends_by_ranks = dist_rt.allgather_obj(tp_ends)

            tp_latencies_ms: list[float | None] = []

            tp_sizes_gb = [
                self._get_tp_total_size(tp) / 1e9 for tp in self.traffic_patterns
            ]

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

                if not ends or not starts:
                    tp_latencies_ms.append(None)
                else:
                    tp_latencies_ms.append((max(ends) - min(starts)) * 1e3)

                    mean_bw = 0.0
                    for rank in tp.senders_ranks(world_size=self.world_size):
                        rank_start = tp_starts_by_ranks[rank][i]
                        rank_end = tp_ends_by_ranks[rank][i]
                        if not rank_start or not rank_end:
                            raise ValueError(
                                f"Rank {rank} has no start or end time, but participated in TP, this is not normal."
                            )
                        mean_bw += (
                            tp.total_src_size(rank) * 1e-9 / (rank_end - rank_start)
                        )

                    num_senders = max(
                        1, len(tp.senders_ranks(world_size=self.world_size))
                    )
                    mean_bw /= num_senders

            if self.my_rank == 0:
                headers = [
                    "Transfer size (GB)",
                    "Latency (ms)",
                    "Isolated Latency (ms)",
                    "Num Senders",
                    "Mean BW (GB/s)",  # Bandwidth
                ]
                data = [
                    [
                        tp_sizes_gb[i],
                        tp_latencies_ms[i],
                        isolated_tp_latencies_ms[i],
                        len(tp.senders_ranks(world_size=self.world_size)),
                        mean_bw,
                    ]
                    for i, tp in enumerate(self.traffic_patterns)
                ]
                logger.info(
                    f"Iteration {iter_ix + 1}/{self.n_iters}\n{tabulate(data, headers=headers, floatfmt='.3f')}"
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
                    "num_senders": len(tp.senders_ranks(world_size=self.world_size)),
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
                json.dump(results, f)

        # Destroy
        logger.info("[Rank %d] Finished run, destroying objects", self.my_rank)
        all_handles: list[Any] = [h for sub in tp_handles for h in sub]
        self._destroy(all_handles)

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
