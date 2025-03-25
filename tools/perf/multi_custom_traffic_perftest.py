from dist_utils import dist_utils
from tabulate import tabulate
from pathlib import Path
from utils import load_matrix
from os import PathLike
from common import NixlBuffer
from typing import *
import logging
from dataclasses import dataclass
import uuid
import time
import numpy as np
import torch

from nixl._api import nixl_agent

from custom_traffic_perftest import CTPerftest, TrafficPattern

log = logging.getLogger(__name__)

class MultiCTPerftest(CTPerftest):
    """Extends CTPerftest to handle multiple traffic patterns simultaneously.
    The patterns are executed in parallel, and the results are aggregated.
    
    Allows testing multiple communication patterns in parallel between distributed processes.
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

        self.send_bufs: list[Optional[NixlBuffer]] = [] # [i]=None if no send to rank i
        self.recv_bufs: list[Optional[NixlBuffer]] = [] # [i]=None if no recv from rank i
        self.dst_bufs_descs = [] # [i]=None if no recv from rank i else descriptor of the dst buffer

    def _wait(self, tp_handles: list[list]) -> list[Optional[float]]:
        """Wait for all transfers to complete and record completion times.
        
        Args:
            tp_handles: List of transfer handles for each traffic pattern
            
        Returns:
            List of completion timestamps for each traffic pattern
        """
        # Wait for transfers to complete - report end time for each tp
        tp_done_ts = [None for _ in tp_handles]
        while True:
            pending = [[] for _ in tp_handles]
            for i, handles in enumerate(tp_handles):
                for handle in handles:
                    state = self.nixl_agent.check_xfer_state(handle)
                    assert state != "ERR", "Transfer got to Error state."
                    if state != "DONE":
                        pending[i].append(handle)
                if not pending[i]:
                    tp_done_ts[i] = time.perf_counter()

            tp_handles = pending

            if not any(tp_handles):
                break
        
        return tp_done_ts

    def run(self, verify_buffers: bool = False, print_recv_buffers: bool = False) -> float:
        """Execute all traffic patterns in parallel.
        
        Args:
            verify_buffers: Whether to verify buffer contents after transfer
            print_recv_buffers: Whether to print receive buffer contents
            
        Returns:
            Total execution time in seconds
        
        This method initializes and executes multiple traffic patterns simultaneously,
        measures their performance, and optionally verifies the results.
        """
        tp_handles: list[list] = []
        tp_bufs = []
        for tp in self.traffic_patterns:
            handles, send_bufs, recv_bufs = self._prepare_tp(tp)
            tp_bufs.append((send_bufs, recv_bufs))
            tp_handles.append(handles)
        

        start_ts_by_tp = [None for _ in tp_handles]
        start = time.time()
        for i, handles in enumerate(tp_handles):
            start_ts_by_tp[i] = time.perf_counter()
            self._run_tp(handles)
            sleep = self.traffic_patterns[i].sleep_after_finish_sec
            if sleep > 0:
                time.sleep(sleep)

        end_ts_by_tp = self._wait(tp_handles)
        end = time.time()

        tp_times_sec = [end_ts_by_tp[i] - start_ts_by_tp[i] for i in range(len(tp_handles))]
        tp_sizes_gb = [self._get_tp_total_size(tp) / 1E9 for tp in self.traffic_patterns]
        tp_bandwidths_gbps = [tp_sizes_gb[i] / tp_times_sec[i] for i in range(len(tp_handles))]
        avg_tp_bw = np.mean(tp_bandwidths_gbps)

        # Metrics report
        start_times = dist_utils.allgather_obj(start)
        end_times = dist_utils.allgather_obj(end)
        avg_tp_bws = dist_utils.allgather_obj(avg_tp_bw)

        # This is the total time taken by all ranks to run all traffic patterns
        global_total_time = max(end_times) - min(start_times)
        global_avg_tp_bw = np.mean(avg_tp_bws)
    
        if self.my_rank == 0:
            headers = ["Total time (s)", "Avg pattern BW (GB/s)"]
            data = [[global_total_time, global_avg_tp_bw]]
            log.info("\n" + tabulate(data, headers=headers, floatfmt=".6f"))

        if verify_buffers:
            for i, tp in enumerate(self.traffic_patterns):
                send_bufs, recv_bufs = tp_bufs[i]
                self._verify_tp(tp, recv_bufs, print_recv_buffers)

        for i, tp in enumerate(self.traffic_patterns):
            send_bufs, recv_bufs = tp_bufs[i]
            self._destroy(tp_handles[i], send_bufs, recv_bufs)

        return end - start
