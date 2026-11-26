# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank and expert mapping for the elastic test's correctness checks."""

import torch


def rank_mapping(buffer):
    """Committed public-rank to internal-slot mapping used by the running kernels."""
    return {
        rank: slot
        for rank, slot in enumerate(buffer.runtime.get_rank_mapping())
        if slot >= 0
    }


def rank_coordinates(buffer, world_rank=None):
    rank = buffer.world_rank if world_rank is None else world_rank
    return divmod(rank_mapping(buffer)[rank], buffer.max_num_scaleup_ranks)


def topology_experts(buffer, topk_idx):
    """Translate public expert owners for the topology-indexed reduction reference."""
    mapping = torch.tensor(buffer.runtime.get_rank_mapping(), device=topk_idx.device)
    slots = buffer.num_experts_per_rank
    owners = mapping[topk_idx.clamp_min(0).long() // slots]
    return torch.where(
        (topk_idx >= 0) & (owners >= 0),
        owners * slots + topk_idx % slots,
        -1,
    ).to(topk_idx.dtype)
