"""Sequential is different from multi in that every rank processes only one TP at a time, but they can process different ones"""
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
from typing import Optional, Tuple, List

import numpy as np
from custom_traffic_perftest import CTPerftest
from common import TrafficPattern
from dist_utils import dist_utils, ReduceOp
from tabulate import tabulate
from common import NixlHandle
from nixl._api import nixl_agent

log = logging.getLogger(__name__)


class SequentialCTPerftest(CTPerftest):
    """Extends CTPerftest to handle multiple traffic patterns sequentially.
    The patterns are executed in sequence, and the results are aggregated.

    Allows testing multiple communication patterns sequentially between distributed processes.
    """

    def __init__(self, traffic_patterns: list[TrafficPattern]) -> None:
        """Initialize multi-pattern performance test.

        Args:
            traffic_patterns: List of traffic patterns to test simultaneously
        """
        self.my_rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.traffic_patterns = traffic_patterns

        log.debug(f"[Rank {self.my_rank}] Initializing Nixl agent")
        self.nixl_agent = nixl_agent(f"{self.my_rank}")
        assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"
    
    def _barrier_tp(self, tp: TrafficPattern):
        """Barrier for a traffic pattern"""
        dist_utils.barrier(tp.senders_ranks())

    def run(
        self, verify_buffers: bool = False, print_recv_buffers: bool = False
    ) -> float:
        """
        Args:
            verify_buffers: Whether to verify buffer contents after transfer
            print_recv_buffers: Whether to print receive buffer contents

        Returns:
            Total execution time in seconds

        This method initializes and executes multiple traffic patterns simultaneously,
        measures their performance, and optionally verifies the results.
        """
        # TODO ADD WARMUP
        self._warmup()
        # TODO Add verification that rank i and j have only one connection active (prevent from sending a buffer over another)

        tp_handles: list[list] = []
        tp_bufs = []
        for tp in self.traffic_patterns:
            handles, send_bufs, recv_bufs = self._prepare_tp(tp)
            tp_bufs.append((send_bufs, recv_bufs))
            tp_handles.append(handles)

        tp_ix = 0
        exec_time_by_tp: List[Optional[float]] = [None for _ in tp_handles]
        barrier_time_by_tp: List[Optional[float]] = [None for _ in tp_handles]

        for tp_ix, handles in enumerate(tp_handles): # DEBUG remove [:2]
            tp = self.traffic_patterns[tp_ix]

            if self.my_rank not in tp.senders_ranks():
                continue

            self._barrier_tp(tp) 
            if tp.sleep_before_launch_sec is not None:
                time.sleep(tp.sleep_before_launch_sec)

            # Run TP
            s = time.perf_counter()
            self._run_tp(handles, blocking=True)
            elapsed = time.perf_counter() - s
            exec_time_by_tp[tp_ix] = elapsed

            # Check that all ranks have finished TP
            barrier_start = time.perf_counter()
            self._barrier_tp(tp) 
            barrier_time_by_tp[tp_ix] = time.perf_counter() - barrier_start

            if tp.sleep_after_launch_sec is not None:
                time.sleep(tp.sleep_after_launch_sec)


        total_time_by_tp = [
            None if exec_time_by_tp[i] is None or barrier_time_by_tp[i] is None 
            else exec_time_by_tp[i] + barrier_time_by_tp[i] 
            for i in range(len(tp_handles))
        ]

        log.info(
            f"[Rank {self.my_rank}] Exec time by TP: {exec_time_by_tp}\n"
            f"Barrier time by TP: {barrier_time_by_tp}\n"
            f"Total time by TP: {total_time_by_tp}\n"
        )

        tp_sizes_gb = [
            self._get_tp_total_size(tp) / 1e9 for tp in self.traffic_patterns
        ]
        bw_by_tp = [
            tp_sizes_gb[i] / exec_time_by_tp[i] if exec_time_by_tp[i] is not None else None for i in range(len(tp_handles))
        ]

        bw_by_tp_per_rank = dist_utils.allgather_obj(bw_by_tp)
        avg_bws = [
            sum(bw[i] for bw in bw_by_tp_per_rank if bw[i] is not None) / sum(1 for bw in bw_by_tp_per_rank if bw[i] is not None)
            for i in range(len(tp_handles))
        ]
        min_bws = [
            min(bw[i] for bw in bw_by_tp_per_rank if bw[i] is not None)
            for i in range(len(tp_handles))
        ]
        max_bws = [
            max(bw[i] for bw in bw_by_tp_per_rank if bw[i] is not None)
            for i in range(len(tp_handles))
        ]

        if self.my_rank == 0:
            headers = ["TP size (GB)", "TP avg BW (GB/s)", "TP min BW (GB/s)", "TP max BW (GB/s)", "Num Senders"]
            data = [
                [tp_sizes_gb[i], avg_bws[i], min_bws[i], max_bws[i], len(tp.senders_ranks())]
                for i, tp in enumerate(self.traffic_patterns)
            ]
            log.info("\n" + tabulate(data, headers=headers, floatfmt=".2f"))

        if verify_buffers:
            for i, tp in enumerate(self.traffic_patterns):
                send_bufs, recv_bufs = tp_bufs[i]
                self._verify_tp(tp, recv_bufs, print_recv_buffers)

        for i, tp in enumerate(self.traffic_patterns):
            send_bufs, recv_bufs = tp_bufs[i]
            self._destroy(tp_handles[i], send_bufs, recv_bufs)

        # return end - start
    
    #def _get_tp_bw(self, tp: TrafficPattern, total_time: float) -> float:
