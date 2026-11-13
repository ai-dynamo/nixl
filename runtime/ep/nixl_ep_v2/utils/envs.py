# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""API shell; production docstrings below are API review contracts.

Calls compute local defaults without hardware discovery or Store access: use the
declared domain size, or assume one domain.
"""
from typing import Optional, Tuple

import torch.distributed as dist


def get_logical_domain_size(
    group: dist.ProcessGroup,
    *,
    num_scaleup_ranks: Optional[int] = None,
    store: Optional[dist.Store] = None,
    bootstrap_id: Optional[str] = None,
    store_timeout_secs: int = 300,
) -> Tuple[int, int]:
    """
    Get the logical domain sizes (scaleout ranks and scaleup ranks): every rank's GPU evidence (an NVML fabric clique, else
    the host and its NVLink P2P peers) exchanged through a Store and inferred into uniform, rank-contiguous domains, or,
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
    count = group.size()
    positions = num_scaleup_ranks or count
    if positions <= 0 or count % positions:
        raise ValueError("num_scaleup_ranks must divide the group size")
    return (count // positions, positions)
