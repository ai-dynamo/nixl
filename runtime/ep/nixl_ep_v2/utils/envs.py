import functools
import inspect
import os
import random
import re
import subprocess
import torch
import torch.distributed as dist
from typing import Optional, Tuple

from .nixl_bootstrap import NixlBootstrap, resolve_bootstrap_store

_local_rank = None
_local_seed = 0
_global_seed = 0

# Default NIC name for RDMA operations, configurable via environment variable
_DEFAULT_NIC_NAME = os.getenv('EP_NIC_NAME', 'mlx5_0')


def init_seed(global_seed: int) -> None:
    """
    Initialize the random seed for reproducibility. The local seed is derived from the global seed plus rank.

    Arguments:
        global_seed: the global random seed.
    """
    global _local_seed, _global_seed
    _local_seed = global_seed + dist.get_rank()
    _global_seed = global_seed
    torch.manual_seed(_local_seed)
    random.seed(_local_seed)


def get_local_seed() -> int:
    """
    Get the local random seed.

    Returns:
        seed: the local random seed.
    """
    return _local_seed


def get_global_seed() -> int:
    """
    Get the global random seed.

    Returns:
        seed: the global random seed.
    """
    return _global_seed


def dist_print(s: str = '', once_in_node: bool = False) -> None:
    """
    Print a message from all ranks, or only from rank 0 of each node, followed by a barrier.

    Arguments:
        s: the message to print.
        once_in_node: if `True`, only the first local rank in each node prints.
    """
    global _local_rank
    assert _local_rank is not None
    if not once_in_node or _local_rank == 0:
        print(s, flush=True)
    dist.barrier()


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> Tuple[int, int, dist.ProcessGroup]:
    """
    Initialize the distributed environment with NCCL backend.

    Arguments:
        local_rank: the local rank index.
        num_local_ranks: the number of local ranks.
        seed: the global random seed.

    Returns:
        rank: the global rank index.
        world_size: the total number of ranks.
        group: the communication group.
    """
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    # Set local rank
    global _local_rank
    _local_rank = local_rank

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    init_seed(seed)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def get_physical_domain_size(group: dist.ProcessGroup, *, store: Optional[dist.Store] = None,
                             bootstrap_id: Optional[str] = None, store_timeout_secs: int = 300) -> Tuple[int, int]:
    """
    Get the physical domain sizes (RDMA ranks and NVLink ranks): every rank's GPU evidence (an NVML fabric clique, else
    the host and its NVLink P2P peers) exchanged through a Store and inferred into uniform, rank-contiguous domains.

    Arguments:
        group: the communication group.
        store: with `bootstrap_id`, the Store to exchange through instead of a cohort-private TCPStore served by the group's first rank.
        bootstrap_id: the namespace in `store`.
        store_timeout_secs: how long to wait for the other ranks' records.

    Returns:
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
    """
    return get_logical_domain_size(group, store=store, bootstrap_id=bootstrap_id, store_timeout_secs=store_timeout_secs)


def get_logical_domain_size(group: dist.ProcessGroup, *, num_scaleup_ranks: Optional[int] = None,
                            store: Optional[dist.Store] = None, bootstrap_id: Optional[str] = None,
                            store_timeout_secs: int = 300) -> Tuple[int, int]:
    """
    Get the logical domain sizes (scaleout ranks and scaleup ranks): inferred as in `get_physical_domain_size`, or,
    with `num_scaleup_ranks`, contiguous blocks of that many ranks, each validated to be one NVLink domain.

    Arguments:
        group: the communication group.
        num_scaleup_ranks: the declared domain size (must divide the group size), or `None` to infer it.
        store: with `bootstrap_id`, the Store to exchange through instead of a cohort-private TCPStore served by the group's first rank.
        bootstrap_id: the namespace in `store`.
        store_timeout_secs: how long to wait for the other ranks' records.

    Returns:
        num_scaleout_ranks: the number of logical scaleout ranks.
        num_scaleup_ranks: the number of logical scaleup ranks.
    """
    bootstrap_store, namespace = resolve_bootstrap_store(group, store, bootstrap_id, 'domain-layout',
                                                         timeout_seconds=store_timeout_secs)
    cohort = NixlBootstrap(bootstrap_store, group.rank(), group.size(), namespace=namespace,
                           num_scaleup_ranks=num_scaleup_ranks, timeout_seconds=store_timeout_secs)
    cohort.barrier('layout')   # the first rank keeps the Store it serves until every peer has read the layout
    return cohort.num_scaleout_ranks, cohort.num_scaleup_ranks


def check_torch_deterministic() -> None:
    """
    Ensure PyTorch deterministic algorithms and fill_uninitialized_memory are not both enabled.
    When both are on, `torch.empty()` calls an initialization kernel that may overlap with communication streams,
    causing errors.
    """
    assert not (torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory)


@functools.lru_cache()
def get_nvlink_gbs(factor: float = 0.9) -> float:
    """
    Get the total NVLink bandwidth in GB/s, cached.

    Arguments:
        factor: the bandwidth efficiency factor.

    Returns:
        gbs: the total NVLink bandwidth in GB/s.

    Raises:
        RuntimeError: if `nvidia-smi` is not installed, fails, or lists no NVLink link; the callers divide by this value.
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'],
                                capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = r'GPU \d+:.*?(?=^GPU \d+:|^$)'
        match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
        if match is None:
            raise RuntimeError('`nvidia-smi nvlink -s` lists no NVLink link for this GPU: pass `nvlink_gbs` explicitly')

        gpu_block = match.group(0)
        link_pattern = r'Link \d+:\s*([\d\.]+) GB/s'
        link_matches = re.findall(link_pattern, gpu_block)
        if not link_matches:
            raise RuntimeError('`nvidia-smi nvlink -s` lists no NVLink link for this GPU: pass `nvlink_gbs` explicitly')
        return sum(float(bw) for bw in link_matches) * factor
    except FileNotFoundError:
        raise RuntimeError('`nvidia-smi` was not found: it ships with the NVIDIA driver, check the container, '
                           'or pass `nvlink_gbs` explicitly') from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'`nvidia-smi nvlink -s` failed ({e.stderr.strip() or e}): pass `nvlink_gbs` explicitly') from None


@functools.lru_cache()
def get_rdma_gbs(nic_name: str = _DEFAULT_NIC_NAME) -> float:
    """
    Get the RDMA bandwidth in GB/s, cached.

    Arguments:
        nic_name: the NIC device name.

    Returns:
        gbs: the RDMA bandwidth in GB/s.

    Raises:
        RuntimeError: if `ibstat` is not installed, fails, or does not list the adapter; the callers divide by this value.
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout

        pattern = rf"CA '{nic_name}'.*?Port \d+:\s*.*?Rate:\s*(\d+)"
        match = re.search(pattern, output, re.DOTALL)
        if match is None:
            raise RuntimeError(f"adapter '{nic_name}' is not in the `ibstat` report: set EP_NIC_NAME to a present adapter "
                               f"or pass `rdma_gbs` explicitly")
        rate = int(match.group(1))
        return rate / 8
    except FileNotFoundError:
        raise RuntimeError('`ibstat` was not found: install the infiniband-diags package, or pass `rdma_gbs` explicitly') from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'`ibstat` failed ({e.stderr.strip() or e}): check that the RDMA devices are visible to the process, '
                           f'or pass `rdma_gbs` explicitly') from None
