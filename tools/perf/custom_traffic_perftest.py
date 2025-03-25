from itertools import chain
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
from tabulate import tabulate

from nixl._api import nixl_agent

log = logging.getLogger(__name__)

@dataclass
class TrafficPattern:
    matrix_file: PathLike
    shards: int
    mem_type: Literal["cuda", "vram", "cpu", "dram"]
    xfer_op: Literal["WRITE", "READ"]  # Transfer operation type
    dtype: torch.dtype = torch.float32

    id: str = str(uuid.uuid4())


class CTPerftest:
    def __init__(self, traffic_pattern: TrafficPattern, iters: int = 1, warmup_iters: int = 0):
        self.my_rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.traffic_pattern = traffic_pattern
        self.iters = iters
        self.warmup_iters = warmup_iters

        log.debug(f"[Rank {self.my_rank}] Initializing Nixl agent")
        self.nixl_agent = nixl_agent(f"{self.my_rank}")
        assert "UCX" in self.nixl_agent.get_plugin_list(), "UCX plugin is not loaded"

        self.send_bufs: list[Optional[NixlBuffer]] = [] # [i]=None if no send to rank i
        self.recv_bufs: list[Optional[NixlBuffer]] = [] # [i]=None if no recv from rank i  
        self.dst_bufs_descs = [] # [i]=None if no recv from rank i else descriptor of the dst buffer

    def _share_md(self):
        """Needs to be run after init_buffers"""
        log.debug(f"[Rank {self.my_rank}] Sharing agent metadata with other ranks")
        md = self.nixl_agent.get_agent_metadata()
        mds = dist_utils.allgather_obj(md)
        for other_rank, metadata in enumerate(mds):
            if other_rank == self.my_rank:
                continue
            self.nixl_agent.add_remote_agent(metadata)
            log.debug(f"[Rank {self.my_rank}] Added remote agent {other_rank}'s metadata")
        
    def _share_recv_buf_descs(self, my_recv_bufs: list[NixlBuffer]):
        """Send descriptors of the buffers to the world as an alltoall (rank 0 get bufs[0], rank 1 get bufs[1], etc)"""

        my_recv_bufs_descs = [buf.xfer_descs if buf is not None else None for buf in my_recv_bufs]
        my_recv_bufs_serdes = [self.nixl_agent.get_serialized_descs(des) for des in my_recv_bufs_descs]

        dst_bufs_serdes = dist_utils.alltoall_obj(my_recv_bufs_serdes)
        dst_bufs_descs = [self.nixl_agent.deserialize_descs(serdes) for serdes in dst_bufs_serdes]
        return dst_bufs_descs

    def _init_buffers(self, tp: TrafficPattern) -> tuple[list[Optional[NixlBuffer]], list[Optional[NixlBuffer]]]:
    
        send_bufs = []
        recv_bufs = []
        for other_rank in range(self.world_size):
            matrix = load_matrix(tp.matrix_file)
            send_size = matrix[self.my_rank][other_rank]
            recv_size = matrix[other_rank][self.my_rank]
            send_buf = recv_buf = None
            if send_size > 0:
                send_buf = NixlBuffer(send_size, mem_type=tp.mem_type, nixl_agent=self.nixl_agent, fill_value=self.my_rank, dtype=tp.dtype)
            if recv_size > 0:
                recv_buf = NixlBuffer(recv_size, mem_type=tp.mem_type, nixl_agent=self.nixl_agent, dtype=tp.dtype)
            send_bufs.append(send_buf)
            recv_bufs.append(recv_buf)
        
        self._share_md()
        return send_bufs, recv_bufs

    def _prepare_tp(self, tp: TrafficPattern) -> list:

        send_bufs, recv_bufs = self._init_buffers(tp)
        dst_bufs_descs = self._share_recv_buf_descs(recv_bufs)

        handles = []
        for other, buf in enumerate(send_bufs):
            if buf is None:
                continue

            handle = self.nixl_agent.initialize_xfer(
                "WRITE",
                buf.xfer_descs,
                dst_bufs_descs[other],
                f"{other}",
                f"{tp.id}_{self.my_rank}_{other}"
            )
            handles.append(handle)
        
        return handles, send_bufs, recv_bufs
    
    def _run_tp(self, handles: list):
        for handle in handles:
            status = self.nixl_agent.transfer(handle)
            assert status != "ERR", "Transfer failed"

        return

    def _wait(self, handles: list):
        # Wait for transfers to complete
        while True:
            pending = []
            for handle in handles:
                state = self.nixl_agent.check_xfer_state(handle)
                assert state != "ERR", "Transfer got to Error state."
                if state != "DONE":
                    pending.append(handle)

            if not pending:
                break
            handles = pending
    
    def _destroy(self, handles: list, send_bufs: list[Optional[NixlBuffer]], recv_bufs: list[Optional[NixlBuffer]]):
        for handle in handles:
            self.nixl_agent.release_xfer_handle(handle)
        
        for other_rank in range(self.world_size):
            if other_rank == self.my_rank:
                continue
            self.nixl_agent.remove_remote_agent(f"{other_rank}")
        
        for buf in chain(send_bufs, recv_bufs):
            if buf is None:
                continue
            buf.deregister()
        
    def _verify_tp(self, tp: TrafficPattern, recv_bufs: list[Optional[NixlBuffer]], print_recv_buffers: bool = False):
        matrix = load_matrix(tp.matrix_file)
        for r, recv_buf in enumerate(recv_bufs):

            if recv_buf is None:
                if matrix[r][self.my_rank] > 0:
                    log.error(f"Rank {self.my_rank} expected {matrix[r][self.my_rank]} bytes from rank {r}, but got 0")
                    raise RuntimeError("Buffer verification failed")
                continue

            if print_recv_buffers:
                s = ""
                for b in recv_buf.bufs:
                    s += f"{b}\n"
                log.info(f"Recv buffer {r}:\n{s}")

            # recv_buf has to be filled with the rank of the sender
            # and its size has to be the same as matrix[r][my_rank]
            full_recv_buf = torch.cat([b for b in recv_buf.bufs])
            expected = torch.full_like(full_recv_buf, r)
            assert torch.all(full_recv_buf == expected), f"Vector not equal to {r}, got {full_recv_buf}"
            assert full_recv_buf.size(0) == matrix[r][self.my_rank], f"Size of vector {r} is not the same as matrix[r][{self.my_rank}], got {full_recv_buf.size(0)}"
            log.debug(f"Vector {r} verified successfully")

    def _get_tp_total_size(self, tp: TrafficPattern) -> int:
        matrix = load_matrix(tp.matrix_file)
        return np.sum(matrix, axis=(0,1))

    def run(self, verify_buffers: bool = False, print_recv_buffers: bool = False):
        handles, send_bufs, recv_bufs = self._prepare_tp(self.traffic_pattern)

        for _ in range(self.warmup_iters):
            self._run_tp(handles)
            self._wait(handles)

        start = time.perf_counter()
        for _ in range(self.iters):
            self._run_tp(handles)
            self._wait(handles)
        end = time.perf_counter()

        # Metrics report
        total_time_sec = end - start
        total_times_sec = dist_utils.allgather_obj(total_time_sec)
        mean_total_time_sec = np.mean(total_times_sec)
        mean_time_per_iter_sec = mean_total_time_sec / self.iters

        total_size_gb = self._get_tp_total_size(self.traffic_pattern) / 1E9
        alg_bw_gbps = total_size_gb / mean_time_per_iter_sec

        # Print metrics as a table
        if self.my_rank == 0:
            headers = ["Avg Total time (s)", "Avg Time/iter (s)", "Total size (GB)", "Alg BW (GB/s)"]
            data = [[mean_total_time_sec, mean_time_per_iter_sec, total_size_gb, alg_bw_gbps]]
            log.info("\n" + tabulate(data, headers=headers, floatfmt=".6f"))

        if verify_buffers:
            self._verify_tp(self.traffic_pattern, recv_bufs, print_recv_buffers)

        self._destroy(handles, send_bufs, recv_bufs)

        return end - start
    
"""
Total time for 1 iterations: 0.0014827251434326172 seconds
I0325 10:06:33.480000 2235324 torch/_subclasses/fake_tensor.py:2796] FakeTensor cache stats:
I0325 10:06:33.481000 2235324 torch/_subclasses/fake_tensor.py:2797]   cache_hits: 0
I0325 10:06:33.481000 2235324 torch/_subclasses/fake_tensor.py:2798]   cache_misses: 0
Initialized NIXL agent: 3
Total time for 1 iterations: 0.002142667770385742 seconds
I0325 10:06:33.514000 2235301 torch/_subclasses/fake_tensor.py:2796] FakeTensor cache stats:
I0325 10:06:33.515000 2235301 torch/_subclasses/fake_tensor.py:2797]   cache_hits: 0
I0325 10:06:33.516000 2235301 torch/_subclasses/fake_tensor.py:2798]   cache_misses: 0
Initialized NIXL agent: 1
Total time for 1 iterations: 0.0024819374084472656 seconds
I0325 10:06:33.533000 2235323 torch/_subclasses/fake_tensor.py:2796] FakeTensor cache stats:
I0325 10:06:33.534000 2235323 torch/_subclasses/fake_tensor.py:2797]   cache_hits: 0
I0325 10:06:33.534000 2235323 torch/_subclasses/fake_tensor.py:2798]   cache_misses: 0
Initialized NIXL agent: 2
Total time for 1 iterations: 0.002760648727416992 seconds"""