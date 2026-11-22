# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Cohort discovery and metadata exchange through temporary global-rank Store keys."""

import pickle
import uuid
from contextlib import contextmanager
from datetime import timedelta

from .nixl_fabric import inspect_fabric_device, validate_fabric_records


@contextmanager
def exchange_records(
    store,
    rank,
    peers,
    value,
    *,
    key_prefix="NIXL_EP",
    timeout_seconds=300,
):
    """Exchange records with reciprocal peers; delete this rank's record after all peers read it.

    Use globally unique rank IDs within the Store. Complete the previous exchange on all participating
    ranks before reusing its keys; concurrent exchanges must not share a rank and key prefix.
    """
    peers = tuple(peers)
    timeout = timedelta(seconds=timeout_seconds)
    publication = uuid.uuid4().hex
    key = f"{key_prefix}/{rank}"
    acknowledgements = [
        f"{key_prefix}/read/{rank}/{publication}/{peer}" for peer in peers
    ]
    store.set(key, pickle.dumps((publication, value), protocol=4))
    try:
        keys = [f"{key_prefix}/{peer}" for peer in peers]
        if keys:
            store.wait(keys, timeout)
        records = {}
        for peer, peer_key in zip(peers, keys):
            # A peer may withdraw after wait(); this non-pickle sentinel reads without blocking or creating a key.
            missing = b"\x00"
            encoded = store.compare_set(peer_key, missing, missing)
            if encoded == missing:
                raise RuntimeError(
                    f"Peer {peer} withdrew its Store metadata during exchange"
                )
            peer_publication, record = pickle.loads(encoded)
            records[peer] = record
            store.set(f"{key_prefix}/read/{peer}/{peer_publication}/{rank}", b"1")
        if acknowledgements:
            store.wait(acknowledgements, timeout)
        yield records
    finally:
        store.delete_key(key)
        for acknowledgement in acknowledgements:
            store.delete_key(acknowledgement)


class NixlBootstrap:
    """Coordinate global ranks; omitted world ranks use cohort indices for a private Store."""

    def __init__(
        self,
        store,
        rank: int,
        num_ranks: int,
        *,
        world_ranks=None,
        num_scaleup_ranks=None,
        timeout_seconds=300,
    ):
        self.store, self.rank = store, rank
        self.timeout_seconds = timeout_seconds
        self.timeout = timedelta(seconds=timeout_seconds)
        self._rounds: dict[str, int] = {}
        self.world_ranks = tuple(
            range(num_ranks) if world_ranks is None else world_ranks
        )
        if (
            len(self.world_ranks) != num_ranks
            or len(set(self.world_ranks)) != num_ranks
        ):
            raise ValueError(
                "world_ranks must contain one distinct global rank per cohort member"
            )
        self.world_rank = self.world_ranks[rank]
        self.peers = tuple(peer for peer in self.world_ranks if peer != self.world_rank)

        # Publish probe errors too, so that peers reject the same evidence instead of waiting for this rank
        try:
            record = inspect_fabric_device()
        except Exception as error:
            record = {"error": f"{type(error).__name__}: {error}"}
        configurations = self._all_gather(
            "fabric", (num_ranks, num_scaleup_ranks, record, uuid.uuid4().hex)
        )
        self._generation = configurations[0][3]
        bad = [
            (r, c[:2])
            for r, c in enumerate(configurations)
            if c[:2] != (num_ranks, num_scaleup_ranks)
        ]
        if bad:
            raise ValueError(
                f"Cohort rank {rank} declared (num_ranks, num_scaleup_ranks)="
                f"{(num_ranks, num_scaleup_ranks)}; ranks {bad} disagree"
            )
        records = tuple(config[2] for config in configurations)
        self.num_scaleout_ranks, self.num_scaleup_ranks = validate_fabric_records(
            records, num_scaleup_ranks
        )
        # A subsequent discovery may reuse the same global-rank keys.
        self.barrier("fabric")

    def _round(self, name):
        index = self._rounds.get(name, 0)
        self._rounds[name] = index + 1
        return f"NIXL_EP/bootstrap/{name}/{index}"

    def _all_gather(self, name, value):
        with exchange_records(
            self.store,
            self.world_rank,
            self.peers,
            value,
            key_prefix=self._round(name),
            timeout_seconds=self.timeout_seconds,
        ) as records:
            records[self.world_rank] = value
            return [records[rank] for rank in self.world_ranks]

    def barrier(self, phase):
        """Cohort barrier for explicitly coordinated lifecycle steps (repeated phases get fresh keys); the leader
        waits for the peers' departure, then removes the keys and may release a Store it owns afterwards.
        """
        # A failed discovery may leave barrier keys behind; each new discovery has its own generation.
        prefix = self._round(f"barrier/{self._generation}/{phase}")
        arrived = [f"{prefix}/arrived/{rank}" for rank in self.world_ranks]
        self.store.set(f"{prefix}/arrived/{self.world_rank}", b"1")
        self.store.wait(arrived, self.timeout)
        if self.rank:
            self.store.set(f"{prefix}/departed/{self.world_rank}", b"1")
        else:
            departed = [f"{prefix}/departed/{rank}" for rank in self.world_ranks[1:]]
            if departed:
                self.store.wait(departed, self.timeout)
            # Every peer has finished reading arrivals and has no further Store operations in this barrier.
            for key in arrived + departed:
                self.store.delete_key(key)
