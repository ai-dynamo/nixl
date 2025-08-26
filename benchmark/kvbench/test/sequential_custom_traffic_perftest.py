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
from itertools import chain
from test.custom_traffic_perftest import CTPerftest, NixlBuffer
from test.traffic_pattern import TrafficPattern
from typing import Any, Dict, List, Optional

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
        assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"

        # NixlBuffer caches buffers and reuse them if they are big enough, let's initialize them once, with the largest needed size
        self.send_buf_by_mem_type: dict[str, NixlBuffer] = {}
        self.recv_buf_by_mem_type: dict[str, NixlBuffer] = {}

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
            dsts = set(tp.receivers_ranks(from_ranks=[self.my_rank]))
            if dsts.issubset(warm_dsts):
                # All the dsts have been warmed up
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
                        len(tp.senders_ranks()),
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
                json.dump(results, f)

        # Destroy
        logger.info("[Rank %d] Finished run, destroying objects", self.my_rank)
        self._destroy(handles)

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
