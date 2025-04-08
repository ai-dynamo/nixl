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
import json
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
from utils import format_size
import yaml
import tqdm

log = logging.getLogger(__name__)


class SequentialCTPerftest(CTPerftest):
    """Extends CTPerftest to handle multiple traffic patterns sequentially.
    The patterns are executed in sequence, and the results are aggregated.

    Allows testing multiple communication patterns sequentially between distributed processes.
    """

    def __init__(self, traffic_patterns: list[TrafficPattern], n_iters: int = 3) -> None:
        """Initialize multi-pattern performance test.

        Args:
            traffic_patterns: List of traffic patterns to test simultaneously
        """
        self.my_rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.traffic_patterns = traffic_patterns
        self.n_iters = n_iters

        log.debug(f"[Rank {self.my_rank}] Initializing Nixl agent")
        self.nixl_agent = nixl_agent(f"{self.my_rank}")
        assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"
    
    def _barrier_tp(self, tp: TrafficPattern):
        """Barrier for a traffic pattern"""
        dist_utils.barrier(tp.senders_ranks())

    def run(
        self, verify_buffers: bool = False, print_recv_buffers: bool = False, json_output_path: Optional[str] = None
    ) -> float:
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
        results = {
            "iterations_results": [],
            "metadata": {"ts": time.time()}
        }

        if self.my_rank == 0:
            log.info(f"[Rank {self.my_rank}] Preparing TPs")
        tp_handles: list[list] = []
        tp_bufs = []
        # for tp in self.traffic_patterns:
        total_tps = len(self.traffic_patterns)
        s = time.time()
        for i, tp in enumerate(self.traffic_patterns):
            try:
                if i > 0 and i % (total_tps // 10) == 0 and self.my_rank == 0:
                    log.info(f"[Rank {self.my_rank}] Preparing TPs: {(i/total_tps)*100:.1f}% complete")
            except:
                pass # DEBUG
            handles, send_bufs, recv_bufs = self._prepare_tp(tp)
            tp_bufs.append((send_bufs, recv_bufs))
            tp_handles.append(handles)
        
        results["metadata"]["prepare_tp_time"] = time.time() - s

        # Measure SOL for every matrix
        isolated_tp_starts = [None for _ in tp_handles]
        isolated_tp_ends = [None for _ in tp_handles]
        n_isolation_iters = 20
        results["metadata"]["sol_calculation_ts"] = time.time()
        for tp_ix, handles in enumerate(tp_handles):
            tp = self.traffic_patterns[tp_ix]
            for _ in range(10): # Warmup
                self._run_tp(handles, blocking=True)

            dist_utils.barrier()

            isolated_tp_starts[tp_ix] = time.time()
            for _ in range(n_isolation_iters):
                self._run_tp(handles, blocking=True)
                self._barrier_tp(tp)
            isolated_tp_ends[tp_ix] = time.time()

        isolated_tp_starts_by_ranks = dist_utils.allgather_obj(isolated_tp_starts)
        isolated_tp_ends_by_ranks = dist_utils.allgather_obj(isolated_tp_ends)
        isolated_tp_latencies = []
        for i in range(len(self.traffic_patterns)):
            starts = [isolated_tp_starts_by_ranks[rank][i] for rank in range(len(isolated_tp_starts_by_ranks))]
            ends = [isolated_tp_ends_by_ranks[rank][i] for rank in range(len(isolated_tp_ends_by_ranks))]
            starts = [x for x in starts if x is not None]
            ends = [x for x in ends if x is not None]
            if not ends or not starts:
                isolated_tp_latencies.append(None)
            else:
                isolated_tp_latencies.append((max(ends) - min(starts)) / n_isolation_iters)

        for iter_ix in range(self.n_iters):
            # WARMUP
            for _ in range(10):
                for tp_ix, handles in enumerate(tp_handles):
                    self._run_tp(handles, blocking=True)
            
            tp_starts = [None for _ in tp_handles]
            tp_ends = [None for _ in tp_handles]
            dist_utils.barrier()
            results["metadata"][f"iter_{iter_ix}_ts"] = time.time()
            for tp_ix, handles in enumerate(tp_handles):

                tp = self.traffic_patterns[tp_ix]

                if self.my_rank not in tp.senders_ranks():
                    continue

                self._barrier_tp(tp) 
                if tp.sleep_before_launch_sec is not None:
                    time.sleep(tp.sleep_before_launch_sec)

                # Run TP
                tp_starts[tp_ix] = time.time()
                self._run_tp(handles, blocking=True)
                
                # Check that all ranks have finished TP
                self._barrier_tp(tp) 
                tp_ends[tp_ix] = time.time()

                if tp.sleep_after_launch_sec is not None:
                    time.sleep(tp.sleep_after_launch_sec)

            tp_starts_by_ranks = dist_utils.allgather_obj(tp_starts)
            tp_ends_by_ranks = dist_utils.allgather_obj(tp_ends)

            tp_latencies = []
            for i in range(len(self.traffic_patterns)):
                starts = [tp_starts_by_ranks[rank][i] for rank in range(len(tp_starts_by_ranks))]
                ends = [tp_ends_by_ranks[rank][i] for rank in range(len(tp_ends_by_ranks))]
                starts = [x for x in starts if x is not None]
                ends = [x for x in ends if x is not None]
                if not ends or not starts:
                    tp_latencies.append(None)
                else:
                    tp_latencies.append(max(ends) - min(starts))

            tp_sizes_gb = [ self._get_tp_total_size(tp)/1E9 for tp in self.traffic_patterns ]

            if self.my_rank == 0:
                headers = ["TP size (GB)", "Latency (ms)", "Isolated Latency (ms)", "Num Senders"]
                data = [
                    [tp_sizes_gb[i], tp_latencies[i]*1E3, isolated_tp_latencies[i]*1E3, len(tp.senders_ranks())]
                    for i, tp in enumerate(self.traffic_patterns)
                ]
                log.info(f"Iteration {iter_ix+1}/{self.n_iters}")
                log.info("\n" + tabulate(data, headers=headers, floatfmt=".3f"))
                
            if verify_buffers:
                for i, tp in enumerate(self.traffic_patterns):
                    send_bufs, recv_bufs = tp_bufs[i]
                    self._verify_tp(tp, recv_bufs, print_recv_buffers)
            
            iter_results = [
                {"size": tp_sizes_gb[i], "latency": tp_latencies[i]*1E3, "isolated_latency": isolated_tp_latencies[i]*1E3, "num_senders": len(tp.senders_ranks())}
                for i, tp in enumerate(self.traffic_patterns)
            ]
            results["iterations_results"].append(iter_results)
        
        results["metadata"]["finished_ts"] = time.time() 
        if json_output_path and self.my_rank == 0:
            print(f"Saving results to {json_output_path}")
            with open(json_output_path, "w") as f:
                json.dump(results, f)

        for i, tp in enumerate(self.traffic_patterns):
            send_bufs, recv_bufs = tp_bufs[i]
            self._destroy(tp_handles[i], send_bufs, recv_bufs)

    
    def _write_yaml_results(self, output_path: str, headers: List[str], data: List[List], traffic_patterns: List[TrafficPattern]) -> None:
        """Write performance test results to a YAML file.
        
        Args:
            output_path: Path to save the YAML file
            headers: Column headers for the results
            data: Performance data rows
            traffic_patterns: List of traffic patterns tested
        """
        results = {
            "performance_results": {
                "timestamp": time.time(),
                "world_size": self.world_size,
                "traffic_patterns": []
            }
        }
        
        for i, tp in enumerate(traffic_patterns):
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
            with open(output_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, sort_keys=False)
            log.info(f"Results saved to YAML file: {output_path}")
        except Exception as e:
            log.error(f"Failed to write YAML results to {output_path}: {e}")

    #def _get_tp_bw(self, tp: TrafficPattern, total_time: float) -> float:
