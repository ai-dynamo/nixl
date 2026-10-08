"""NIXL constructor-cohort coordination through a torch Store: the cohort's NVLink-domain layout, a session token, the
peers' metadata and a final barrier, under a per-cohort ``namespace`` (concurrent cohorts on one Store use distinct
ones; a cohort whose construction failed part-way, or whose membership changed, retries under a fresh one: the
construction round is counted per rank, and ranks at different rounds wait on different session keys)."""

from datetime import timedelta
import hashlib
import pickle
import socket
import uuid
from typing import Optional, Tuple

import torch.distributed as dist

from .nixl_fabric import inspect_topology_device, validate_topology_records


class NixlBootstrap:
    def __init__(self, store, rank: int, world: int, *, namespace: str, num_scaleup_ranks=None, timeout_seconds=300):
        if type(world) is not int or world <= 0 or type(rank) is not int or not 0 <= rank < world:
            raise ValueError("Bootstrap rank must be in the positive constructor-cohort world")
        self.store, self.rank, self.world = store, rank, world
        self.timeout = timedelta(seconds=timeout_seconds)
        self._rounds = {}
        cohort = hashlib.sha256(namespace.encode()).hexdigest()
        root = f'nixl_ep_v2/nixl/bootstrap/{cohort}'
        sequence = store.add(f'{root}/sequence/{rank}', 1)
        nonce_key = f'{root}/session/{sequence}'
        if rank == 0:
            store.set(nonce_key, uuid.uuid4().hex)
        self.session_token = self._get(nonce_key).decode('ascii')
        self._prefix = f'{root}/{self.session_token}'

        # Publish probe errors too, so that peers reject the same evidence instead of waiting for this rank
        try:
            record = inspect_topology_device()
        except Exception as error:
            record = {'error': f'{type(error).__name__}: {error}'}
        configurations = self._all_gather('topology', (world, num_scaleup_ranks, record))
        if any(config[:2] != (world, num_scaleup_ranks) for config in configurations):
            raise ValueError('Constructor cohort disagrees on world or num_scaleup_ranks')
        self.topology_records = tuple(config[2] for config in configurations)
        self.num_scaleout_ranks, self.num_scaleup_ranks = validate_topology_records(
            self.topology_records, num_scaleup_ranks)

    def _get(self, key):
        self.store.wait([key], self.timeout)
        return self.store.get(key)

    def _round(self, name):
        index = self._rounds.get(name, 0)
        self._rounds[name] = index + 1
        return f'{self._prefix}/{name}/{index}'

    def _all_gather(self, name, value):
        prefix = self._round(name)
        self.store.set(f'{prefix}/{self.rank}', pickle.dumps(value, protocol=4))
        keys = [f'{prefix}/{rank}' for rank in range(self.world)]
        self.store.wait(keys, self.timeout)
        return [pickle.loads(self.store.get(key)) for key in keys]

    def all_gather_peer_infos(self, mine):
        """Gather ``(dense_id, metadata_bytes, peer_info)`` for this cohort: a dict by dense id in cohort-rank order
        (insertion order, which callers rely on); dense ids need not equal bootstrap ranks.
        """
        records = self._all_gather('peers', mine)
        infos = {}
        for rid, metadata, info in records:
            if rid in infos:
                raise ValueError(f'Duplicate dense peer ID {rid} in constructor cohort')
            infos[rid] = (metadata, tuple(info))
        return infos

    def barrier(self, phase):
        """Cohort barrier for explicitly coordinated lifecycle steps (repeated phases get fresh keys); the leader
        waits for the peers' departure so that it may release a Store it owns afterwards.
        """
        prefix = self._round(f'barrier/{phase}')
        self.store.set(f'{prefix}/arrived/{self.rank}', b'1')
        self.store.wait([f'{prefix}/arrived/{rank}' for rank in range(self.world)], self.timeout)
        if self.rank:
            self.store.set(f'{prefix}/departed/{self.rank}', b'1')
        elif self.world > 1:
            self.store.wait([f'{prefix}/departed/{rank}' for rank in range(1, self.world)], self.timeout)


def resolve_bootstrap_store(group: Optional[dist.ProcessGroup], store: Optional[dist.Store],
                            bootstrap_id: Optional[str], group_namespace: str,
                            timeout_seconds: int = 300) -> Tuple[dist.Store, str]:
    """
    The Store a cohort coordinates its construction through, and its namespace there: an explicit `store` with a
    `bootstrap_id`, else a cohort-private `TCPStore` under `group_namespace` (or `bootstrap_id` when only that was
    given): the group's first rank serves it on a free port and its address travels through the group. The returned
    Store must outlive the cohort's use of it. Without a group both `store` and `bootstrap_id` are required.
    """
    if store is not None and bootstrap_id is not None:
        return store, bootstrap_id
    if group is None:
        raise ValueError('Bootstrapping without a process group requires both store and bootstrap_id')
    rank, world, timeout = dist.get_rank(group), dist.get_world_size(group), timedelta(seconds=timeout_seconds)
    server, address = None, [None]
    if rank == 0:
        server = dist.TCPStore(socket.gethostname(), 0, world, is_master=True, timeout=timeout, wait_for_workers=False)
        address[0] = (socket.gethostname(), server.port)
    dist.broadcast_object_list(address, src=dist.get_global_rank(group, 0), group=group)
    host, port = address[0]
    cohort_store = server if rank == 0 else dist.TCPStore(host, port, world, is_master=False, timeout=timeout)
    return cohort_store, bootstrap_id if bootstrap_id is not None else group_namespace
