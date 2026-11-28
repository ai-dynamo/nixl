# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal rank mapping, membership masks and communication mode selection."""

from typing import Mapping, Sequence, Tuple

from .nixl_fabric import group_fabric_records

# EPTopology.mode: direct = 0, hybrid = 1.
_MODE_NAMES = ("direct", "hybrid")


def _mode_code(mode: str) -> int:
    assert mode in _MODE_NAMES, f"mode must be direct or hybrid, got {mode!r}"
    return _MODE_NAMES.index(mode)


def mode_of(scaleout_mask: int) -> str:
    """Exactly one active NVLink domain selects direct mode, regardless of its index."""
    return "direct" if bin(scaleout_mask).count("1") == 1 else "hybrid"


def masks_of(
    active_ranks: Sequence[int], rank_mapping: Mapping[int, int], max_positions: int
) -> Tuple[int, int]:
    """Translate public rank IDs to a product of prepared physical domains and positions."""
    active = list(active_ranks)
    if not active or any(
        type(rank) is not int or rank not in rank_mapping for rank in active
    ):
        raise ValueError(
            "Active ranks must be nonempty integers; connect the requested ranks before activation"
        )
    slots = {rank_mapping[rank] for rank in active}
    domains = {slot // max_positions for slot in slots}
    positions = {slot % max_positions for slot in slots}
    if slots != {d * max_positions + p for d in domains for p in positions}:
        raise ValueError(
            "Active ranks must form a product of discovered NVLink domains and internal positions"
        )
    return sum(1 << d for d in domains), sum(1 << p for p in positions)


def build_rank_mapping(records, snapshots, max_domains, max_positions):
    """Preserve established slots and assign new peers to free physical-domain slots.

    Records are keyed by public world rank. Snapshots contain established
    rank-to-slot assignments, excluding provisional singleton coordinates.
    Claims for ranks absent from records do not reserve retired slots. Active
    membership's product shape is validated separately during activation.
    """
    if any(type(n) is not int or n <= 0 for n in (max_domains, max_positions)):
        raise ValueError("Topology capacities must be positive integers")
    capacity = max_domains * max_positions
    if any(type(rank) is not int or not 0 <= rank < capacity for rank in records):
        raise ValueError(
            "World ranks must be integers within the declared maximum world"
        )
    domains = group_fabric_records(records)
    if len(domains) > max_domains or any(
        len(domain) > max_positions for domain in domains
    ):
        raise ValueError(
            "Discovered NVLink domains exceed the declared topology capacity"
        )

    mapping, occupied = {}, {}
    for snapshot in snapshots:
        for rank, slot in snapshot.items():
            if rank not in records:
                continue
            if type(slot) is not int or not 0 <= slot < capacity:
                raise ValueError(
                    "Established topology slot exceeds the declared capacity"
                )
            if rank in mapping and mapping[rank] != slot:
                raise ValueError("Conflicting topology assignments for one world rank")
            if slot in occupied and occupied[slot] != rank:
                raise ValueError(
                    "Established topology assigns one slot to multiple ranks"
                )
            mapping[rank], occupied[slot] = slot, rank

    assigned_domains = {}
    for index, ranks in enumerate(domains):
        claims = {mapping[rank] // max_positions for rank in ranks if rank in mapping}
        if len(claims) > 1:
            raise ValueError("Established topology splits one physical NVLink domain")
        if claims:
            domain = claims.pop()
            if domain in assigned_domains.values():
                raise ValueError(
                    "Established topology combines different NVLink domains"
                )
            assigned_domains[index] = domain

    available_domains = iter(
        sorted(set(range(max_domains)) - set(assigned_domains.values()))
    )
    for index, ranks in enumerate(domains):
        domain = assigned_domains.get(index)
        if domain is None:
            domain = next(available_domains)
        available_slots = iter(
            slot
            for slot in range(domain * max_positions, (domain + 1) * max_positions)
            if slot not in occupied
        )
        for rank in ranks:
            if rank not in mapping:
                mapping[rank] = next(available_slots)
    return dict(sorted(mapping.items()))
