import torch
from nixl._api import nixl_agent
from dist_utils import dist_utils 
import logging

log = logging.getLogger(__name__)


class NixlBuffer:
    """Can be sharded"""
    def __init__(self, size: int, mem_type: str, nixl_agent: nixl_agent, shards=2, fill_value=0):
        if mem_type in ("cuda", "vram"):
            device = "cuda"
        elif mem_type in ("cpu", "dram"):
            device = "cpu"
        else:
            raise ValueError(f"Unsupported memory type: {mem_type}")

        log.debug(f"[Rank {dist_utils.get_rank()}] Initializing NixlBuffer with size {size}, device {device}, shards {shards}, fill_value {fill_value}")
        self.bufs = [
            torch.full((size // shards,), fill_value=fill_value, dtype=torch.uint8, device=device) for _ in range(shards)
        ]
        self.reg_descs = nixl_agent.get_reg_descs(self.bufs, mem_type=mem_type)
        self.xfer_descs = nixl_agent.get_xfer_descs(self.bufs, mem_type=mem_type)

        log.debug(f"[Rank {dist_utils.get_rank()}] Registering memory for bufs {self.bufs}")
        assert nixl_agent.register_memory(self.reg_descs) is not None, "Failed to register memory"

