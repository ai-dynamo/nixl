# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import math
from typing import Optional, Tuple, Union

import torch


def generate_pre_combine_data(
    src_token_global_idx: torch.Tensor,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    *,
    num_ranks: int,
    seed: int,
) -> torch.Tensor:
    """
    Generate data needed for combine from `src_token_global_idx`.
    Recall that `src_token_global_idx = src_rank_idx * num_max_tokens_per_rank + src_token_local_idx`.
    The generated data (denoted as `y`) of the i-th token has a shape of [num_topk, hidden], with

    `y[j, k] = sin((token_seeds * P % max_seed + 1) / max_seed * (k + 1) + sin(seed))`

    where `P=131071` is a large prime, `token_seeds` is calculated via `token_seeds = src_token_global_idx[i] * num_topk + j`,
    and `max_seed = num_ranks * num_max_tokens_per_rank * num_topk`.

    Arguments:
    - `src_token_global_idx`: Source token global indices, `[num_tokens]`
    - `num_ranks`, `seed`: the reference configuration supplied by the test

    Returns:
    - Generated data, `[num_tokens, num_topk, hidden]`
    """
    token_seeds = src_token_global_idx.unsqueeze(1) * num_topk + torch.arange(
        num_topk, device=src_token_global_idx.device
    ).unsqueeze(
        0
    )  # [num_tokens, num_topk]
    max_seed = num_ranks * num_max_tokens_per_rank * num_topk
    result = torch.sin(
        (((token_seeds * 131071 % max_seed).float() + 1) / max_seed).unsqueeze(-1)
        * torch.arange(
            1, hidden + 1, device=src_token_global_idx.device, dtype=torch.float32
        ).broadcast_to(1, 1, hidden)
        + math.sin(float(seed))
    )
    return result.to(torch.bfloat16)


def ordered_accumulate(
    data: torch.Tensor, initial_value: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Accumulate `data` in order along the num_topk dimension.

    Arguments:
    - `data`: Data to be accumulated, `[num_tokens, num_topk, hidden]`
    - `initial_value`: Initial value for accumulation, `[num_tokens, hidden]`

    Returns:
    - Result, `[num_tokens, hidden]`
    """
    num_topk = data.shape[1]
    if initial_value is None:
        result = torch.zeros(
            (data.shape[0], data.shape[2]), dtype=torch.float32, device=data.device
        )
    else:
        result = initial_value.clone()
    for i in range(num_topk):
        result += data[:, i, :].float()
    return result.to(data.dtype)


def combine(
    y: torch.Tensor,
    topk_idx: torch.Tensor,
    num_scaleout_ranks: int,
    num_scaleup_ranks: int,
    num_experts: int,
    bias: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
    reduce_in_local: bool,
    reduce_in_scaleup: bool,
) -> torch.Tensor:
    """
    The reference implementation of (possibly multi-level reduction) combining tokens.

    Arguments:
    - `y`: Input tokens to be combined, `[num_tokens, num_topk, hidden]`
    - `topk_idx`: `[num_tokens, num_topk]`
    - `reduce_in_local` and `reduce_in_scaleup`: Whether to do reduction within rank and within scale-up group.
       - `(True, True)` -> Hybrid combine
       - `(True, False)` -> Non-hybrid combine
       - `(False, False)` -> Equivalent to `allow_multiple_reduction` is False
       Pay attention that `reduce_in_scaleup` = `True` or `False` is NOT equivalent even if `num_scaleout_ranks == 1` due to `bias` handling.

    Returns:
    - Combined result, `[num_tokens, hidden]`
    """
    num_ranks = num_scaleout_ranks * num_scaleup_ranks
    num_tokens, hidden = y.shape[0], y.shape[2]
    num_topk = y.shape[1]
    assert not (
        not reduce_in_local and reduce_in_scaleup
    ), "Invalid reduction configuration"

    def grouped_reduce(
        data_to_reduce: torch.Tensor, group_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform in-place grouped reduction on `data_to_reduce` according to `group_id`.
        The summation within each group are performed in strict order along the `num_topk` dimension.
        The result for each group is stored at the rightmost token of that group, and other tokens are set to zero.

        Arguments:
        - `data_to_reduce`: Data to be reduced, `[num_tokens, num_topk, hidden]`
        - `group_id`: group IDs for each token, `[num_tokens, num_topk]`
        """
        # Shuffle to make tokens with the same group_id contiguous
        group_id, src_indices = torch.sort(group_id, dim=-1, stable=True)
        # transformed_src_indices[i, j] = i * num_topk + src_indices[i, j]
        transformed_src_indices = (
            src_indices
            + torch.arange(0, num_tokens, device=y.device).unsqueeze(-1) * num_topk
        ).flatten()
        data_to_reduce = data_to_reduce.view(-1, hidden)[transformed_src_indices].view(
            num_tokens, num_topk, hidden
        )
        # Perform segmented reduce within each group
        cur_accum_buf = torch.zeros(
            (num_tokens, hidden), dtype=torch.float32, device=y.device
        )
        for i in range(num_topk):
            is_segment_break = (
                torch.full((num_tokens,), True, dtype=torch.bool, device=y.device)
                if i == num_topk - 1
                else group_id[:, i] != group_id[:, i + 1]
            )
            cur_accum_buf += data_to_reduce[:, i, :].float()
            # For one token, if `is_segment_break` is True,
            # save the accumulated value and clear the buffer, otherwise, clear `data_to_reduce[:, i, :]`
            segment_break_token_indices = torch.where(is_segment_break)[0]
            data_to_reduce[segment_break_token_indices, i] = cur_accum_buf[
                segment_break_token_indices
            ].to(data_to_reduce.dtype)
            cur_accum_buf[segment_break_token_indices] = 0.0
            non_segment_break_token_indices = torch.where(~is_segment_break)[0]
            data_to_reduce[non_segment_break_token_indices, i] = 0.0
        # Unshuffle
        # noinspection PyShadowingNames
        result = torch.empty_like(data_to_reduce)
        result.view(-1, hidden)[transformed_src_indices] = data_to_reduce.view(
            -1, hidden
        )
        return result.view(num_tokens, num_topk, hidden)

    num_experts_per_rank = num_experts // num_ranks
    if reduce_in_local:
        y = grouped_reduce(y, topk_idx // num_experts_per_rank)
    if reduce_in_scaleup:
        y = grouped_reduce(y, topk_idx // (num_experts_per_rank * num_scaleup_ranks))
    bias_sum = (
        bias[0].float() + bias[1].float()
        if isinstance(bias, tuple)
        else bias.float() if bias is not None else None
    )
    result = ordered_accumulate(y, bias_sum)
    return result
