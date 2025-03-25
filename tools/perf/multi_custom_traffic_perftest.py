from dist_utils import dist_utils
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
    def __init__(self, traffic_patterns: list[TrafficPattern]):
        self.my_rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.traffic_patterns = traffic_patterns

        log.debug(f"[Rank {self.my_rank}] Initializing Nixl agent")
        self.nixl_agent = nixl_agent(f"{self.my_rank}")
        assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"

        self.send_bufs: list[Optional[NixlBuffer]] = [] # [i]=None if no send to rank i
        self.recv_bufs: list[Optional[NixlBuffer]] = [] # [i]=None if no recv from rank i  
        self.dst_bufs_descs = [] # [i]=None if no recv from rank i else descriptor of the dst buffer

    def run(self, verify_buffers: bool = False, print_recv_buffers: bool = False):
        tp_handles: list[list] = []
        tp_bufs = []
        for tp in self.traffic_patterns:
            handles, send_bufs, recv_bufs = self._prepare_tp(tp)
            tp_bufs.append((send_bufs, recv_bufs))
            tp_handles.append(handles)

        start = time.time()
        for handles in tp_handles:
            self._run_tp(handles)
        
        self._wait([h for handles in tp_handles for h in handles])
        end = time.time()

        # Metrics report
        start_times = dist_utils.allgather_obj(start)
        end_times = dist_utils.allgather_obj(end)

        # This is the total time taken by all ranks to run all traffic patterns
        global_total_time = max(end_times) - min(start_times)

        log.info(f"Total time taken to run {len(self.traffic_patterns)} traffic patterns: {end - start} seconds")

        if verify_buffers:
            for i, tp in enumerate(self.traffic_patterns):
                send_bufs, recv_bufs = tp_bufs[i]
                self._verify_tp(tp, recv_bufs, print_recv_buffers)

        self._destroy()

        return end - start