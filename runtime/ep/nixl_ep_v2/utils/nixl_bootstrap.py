# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""NIXL cohort coordination through a torch Store: the cohort's NVLink domain count and size, a session token, the
peers' metadata and a final barrier, under a per-cohort ``namespace`` (concurrent cohorts on one Store use distinct
ones; a cohort whose construction failed part-way, or whose membership changed, retries under a fresh one: the
construction round is counted per rank, and ranks at different rounds wait on different session keys)."""

import hashlib
import pickle
import socket
import uuid
from datetime import timedelta
from typing import Optional, Tuple, cast

import torch.distributed as dist

from .nixl_fabric import inspect_fabric_device, validate_fabric_records


class NixlBootstrap:
    def __init__(
        self,
        store,
        rank: int,
        num_ranks: int,
        *,
        namespace: str,
        num_scaleup_ranks=None,
        timeout_seconds=300,
    ):
        self.store, self.rank, self.num_ranks = store, rank, num_ranks
        self.timeout = timedelta(seconds=timeout_seconds)
        self._rounds: dict[str, int] = {}
        cohort = hashlib.sha256(namespace.encode()).hexdigest()
        root = f"nixl_ep_v2/nixl/bootstrap/{cohort}"
        sequence = store.add(f"{root}/sequence/{rank}", 1)
        nonce_key = f"{root}/session/{sequence}"
        if rank == 0:
            store.set(nonce_key, uuid.uuid4().hex)
        try:
            self.session_token = self._get(nonce_key).decode("ascii")
        except Exception as e:
            raise RuntimeError(
                f"Cohort rank {rank} (construction round {sequence}) saw no session token from rank 0 "
                f"within {timeout_seconds}s: the cohort disagrees on its rounds or membership; "
                "construct with a fresh bootstrap_id"
            ) from e
        self._prefix = f"{root}/{self.session_token}"

        # Publish probe errors too, so that peers reject the same evidence instead of waiting for this rank
        try:
            record = inspect_fabric_device()
        except Exception as error:
            record = {"error": f"{type(error).__name__}: {error}"}
        configurations = self._all_gather(
            "fabric", (num_ranks, num_scaleup_ranks, record)
        )
        bad = [
            (r, c[:2])
            for r, c in enumerate(configurations)
            if c[:2] != (num_ranks, num_scaleup_ranks)
        ]
        if bad:
            raise ValueError(
                f"Cohort rank {rank} declared (num_ranks, num_group_scaleup_ranks)="
                f"{(num_ranks, num_scaleup_ranks)}; ranks {bad} disagree"
            )
        records = tuple(config[2] for config in configurations)
        self.num_scaleout_ranks, self.num_scaleup_ranks = validate_fabric_records(
            records, num_scaleup_ranks
        )

    def _get(self, key):
        self.store.wait([key], self.timeout)
        return self.store.get(key)

    def _round(self, name):
        index = self._rounds.get(name, 0)
        self._rounds[name] = index + 1
        return f"{self._prefix}/{name}/{index}"

    def _all_gather(self, name, value):
        prefix = self._round(name)
        self.store.set(f"{prefix}/{self.rank}", pickle.dumps(value, protocol=4))
        keys = [f"{prefix}/{rank}" for rank in range(self.num_ranks)]
        self.store.wait(keys, self.timeout)
        return [pickle.loads(self.store.get(key)) for key in keys]

    def all_gather_peer_infos(self, mine):
        """Gather ``(metadata_bytes, peer_info)`` for this cohort: a dict by world rank (``peer_info[2]``) in cohort-rank
        order (insertion order, which callers rely on); world ranks need not equal cohort ranks.
        """
        records = self._all_gather("peers", mine)
        infos = {}
        for metadata, info in records:
            world_rank = info[2]
            if world_rank in infos:
                raise ValueError(f"Duplicate world rank {world_rank} in the cohort")
            infos[world_rank] = (metadata, tuple(info))
        return infos

    def barrier(self, phase):
        """Cohort barrier for explicitly coordinated lifecycle steps (repeated phases get fresh keys); the leader
        waits for the peers' departure so that it may release a Store it owns afterwards.
        """
        prefix = self._round(f"barrier/{phase}")
        self.store.set(f"{prefix}/arrived/{self.rank}", b"1")
        self.store.wait(
            [f"{prefix}/arrived/{rank}" for rank in range(self.num_ranks)], self.timeout
        )
        if self.rank:
            self.store.set(f"{prefix}/departed/{self.rank}", b"1")
        elif self.num_ranks > 1:
            self.store.wait(
                [f"{prefix}/departed/{rank}" for rank in range(1, self.num_ranks)],
                self.timeout,
            )


def resolve_bootstrap_store(
    group: Optional[dist.ProcessGroup],
    store: Optional[dist.Store],
    bootstrap_id: Optional[str],
    group_namespace: str,
    timeout_seconds: int = 300,
) -> Tuple[dist.Store, str]:
    """
    The Store a cohort coordinates its construction through, and its namespace there: an explicit `store` with a
    `bootstrap_id`, else a cohort-private `TCPStore` under `group_namespace` (or `bootstrap_id` when only that was
    given): the group's first rank serves it on a free port and its address travels through the group. The returned
    Store must outlive the cohort's use of it. Without a group both `store` and `bootstrap_id` are required.
    """
    if store is not None and bootstrap_id is not None:
        return store, bootstrap_id
    if group is None:
        raise ValueError(
            "Bootstrapping without a process group requires both store and bootstrap_id"
        )
    rank, num_ranks = group.rank(), group.size()
    host, timeout = socket.gethostname(), timedelta(seconds=timeout_seconds)
    cohort_store = None
    address: list[Optional[Tuple[str, int]]] = [None]
    if rank == 0:
        cohort_store = dist.TCPStore(
            host, 0, num_ranks, is_master=True, timeout=timeout, wait_for_workers=False
        )
        address[0] = (host, cohort_store.port)
    dist.broadcast_object_list(address, src=dist.get_global_rank(group, 0), group=group)
    if cohort_store is None:
        cohort_store = dist.TCPStore(
            *cast(Tuple[str, int], address[0]),
            num_ranks,
            is_master=False,
            timeout=timeout,
        )
    return cohort_store, bootstrap_id if bootstrap_id is not None else group_namespace
