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

log = logging.getLogger(__name__)

@dataclass
class TrafficPattern:
    matrix_file: PathLike
    shards: int
    mem_type: Literal["cuda", "vram", "cpu", "dram"]
    xfer_op: Literal["WRITE", "READ"]  # Transfer operation type

    id: str = str(uuid.uuid4())


class CTPerftest:
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
        
    def share_recv_buf_descs(self, my_recv_bufs: list[NixlBuffer]):
        """Send descriptors of the buffers to the world as an alltoall (rank 0 get bufs[0], rank 1 get bufs[1], etc)"""

        my_recv_bufs_descs = [buf.xfer_descs if buf is not None else None for buf in my_recv_bufs]
        my_recv_bufs_serdes = [self.nixl_agent.get_serialized_descs(des) for des in my_recv_bufs_descs]

        dst_bufs_serdes = dist_utils.alltoall_obj(my_recv_bufs_serdes)
        dst_bufs_descs = [self.nixl_agent.deserialize_descs(serdes) for serdes in dst_bufs_serdes]
        return dst_bufs_descs

    def init_buffers(self, tp: TrafficPattern) -> tuple[list[Optional[NixlBuffer]], list[Optional[NixlBuffer]]]:
    
        send_bufs = []
        recv_bufs = []
        for other_rank in range(self.world_size):
            matrix = load_matrix(tp.matrix_file)
            send_size = matrix[self.my_rank][other_rank]
            recv_size = matrix[other_rank][self.my_rank]
            send_buf = recv_buf = None
            if send_size > 0:
                send_buf = NixlBuffer(send_size, mem_type=tp.mem_type, nixl_agent=self.nixl_agent, fill_value=self.my_rank)
            if recv_size > 0:
                recv_buf = NixlBuffer(recv_size, mem_type=tp.mem_type, nixl_agent=self.nixl_agent)
            send_bufs.append(send_buf)
            recv_bufs.append(recv_buf)
        
        self._share_md()
        return send_bufs, recv_bufs

    def prepare_tp(self, tp: TrafficPattern) -> list:

        send_bufs, recv_bufs = self.init_buffers(tp)
        dst_bufs_descs = self.share_recv_buf_descs(recv_bufs)

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
    
    def run_tp(self, handles: list):
        for handle in handles:
            status = self.nixl_agent.transfer(handle)
            assert status != "ERR", "Transfer failed"

        return

    def wait(self, handles: list):
        # Wait for transfers to complete
        while True:
            pending = []
            for handle in handles:
                state = self.nixl_agent.check_xfer_state(handle)
                assert state != "ERR", "Transfer got to Error state."
                if state == "DONE":
                    self.nixl_agent.release_xfer_handle(handle)
                else:
                    pending.append(handle)

            if not pending:
                break
            handles = pending
    
    def _destroy(self):
        for other_rank in range(self.world_size):
            if other_rank == self.my_rank:
                continue
            self.nixl_agent.remove_remote_agent(f"{other_rank}")
        

    def run(self, verify_buffers: bool = False):
        tp_handles: list[list] = []
        tp_bufs = []
        for tp in self.traffic_patterns:
            handles, send_bufs, recv_bufs = self.prepare_tp(tp)
            tp_bufs.append((send_bufs, recv_bufs))
            tp_handles.append(handles)

        start = time.time()
        for handles in tp_handles:
            self.run_tp(handles)
        
        self.wait([h for handles in tp_handles for h in handles])
        end = time.time()

        log.info(f"Total time taken to run all traffic patterns: {end - start} seconds")

        if verify_buffers:
            for i, tp in enumerate(self.traffic_patterns):
                send_bufs, recv_bufs = tp_bufs[i]
                matrix = load_matrix(tp.matrix_file)
                for r, recv_buf in enumerate(recv_bufs):
                    if recv_buf is None:
                        if matrix[r][self.my_rank] > 0:
                            log.error(f"Rank {self.my_rank} expected {matrix[r][self.my_rank]} bytes from rank {r}, but got 0")
                            raise RuntimeError("Buffer verification failed")
                        continue
                    # recv_buf has to be filled with the rank of the sender
                    # and its size has to be the same as matrix[r][my_rank]
                    full_recv_buf = torch.cat([b for b in recv_buf.bufs])
                    expected = torch.full_like(full_recv_buf, r)
                    assert torch.all(full_recv_buf == expected), f"Vector not equal to {r}, got {full_recv_buf}"
                    assert full_recv_buf.size(0) == matrix[r][self.my_rank], f"Size of vector {r} is not the same as matrix[r][{self.my_rank}], got {full_recv_buf.size(0)}"
                    log.info(f"Vector {r} verified successfully")


        self._destroy()

        return end - start