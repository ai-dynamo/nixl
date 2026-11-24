# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the last rows of role-specific AFD arenas on four NVLink peers.

Run with python tests/elastic/test_afd_sizing.py from runtime/ep.
"""

import json
from unittest import TestCase

import torch
import torch.distributed as dist

from nixl.runtime import ep  # type: ignore[attr-defined]
from nixl.runtime.ep.utils.envs import init_dist

CAPACITY = 513
HIDDEN = 4096
TOPK = 4
EXPERTS_PER_FFN = 4
ALIGNMENT = 16
SCALE_BLOCK = 128
ARENA_ALIGNMENT = 2 * 1024 * 1024


def align_rows(count):
    return (count + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def inputs(source, ffn_count, fp8, skew):
    tokens = torch.arange(CAPACITY)
    channels = torch.arange(HIDDEN)
    rows = ((tokens[:, None] + channels[None, :] + source) % 7 + 1).float()
    # An exact key for every source row, including the last token in the bucket.
    rows[:, 0] = source
    rows[:, 1] = tokens // 256
    rows[:, 2] = tokens // 16 % 16
    rows[:, 3] = tokens % 16
    rows = rows.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
    scales = None
    if fp8:
        groups = torch.arange(HIDDEN // SCALE_BLOCK)
        scales = ((groups[None, :] + tokens[:, None] + source) % 4 + 1).float() / 16
    if skew:
        selected = torch.arange(TOPK) + (ffn_count - 1) * EXPERTS_PER_FFN
    else:
        selected = torch.tensor([0, 1, EXPERTS_PER_FFN, 2 * EXPERTS_PER_FFN])
    ids = selected.to(ep.topk_idx_t).repeat(CAPACITY, 1)
    weights = torch.tensor([0.25, 0.5, 1.0, 0.125], dtype=torch.float32).repeat(
        CAPACITY, 1
    )
    return rows, scales, ids, weights


def decode(rows, scales):
    values = rows.float()
    if scales is not None:
        values *= scales.repeat_interleave(SCALE_BLOCK, dim=1)
    return values


def reply(values, expert, weight):
    return ((values + expert % EXPERTS_PER_FFN + 1) * weight).bfloat16()


def reference(rows, scales, ids, weights, ffn_count, allow_multiple_reduction):
    values = decode(rows, scales)
    total = torch.zeros_like(values)
    for worker in range(ffn_count):
        local = torch.zeros_like(values)
        for slot in range(TOPK):
            expert = ids[:, slot, None]
            coeff = weights[:, slot, None].masked_fill(
                expert // EXPERTS_PER_FFN != worker, 0
            )
            local += reply(values, expert, coeff).float()
        total += local.bfloat16().float() if allow_multiple_reduction else local
    return total.bfloat16()


def prepare_replies(worker, attention, ffn_count, received, do_cpu_sync, fp8, skew):
    payload, ids, weights, handle, _ = received
    rows, scales = payload if fp8 else (payload, None)
    assert ids is None and weights.shape == (len(rows),)
    expected_compact = (
        len(attention) * CAPACITY if not skew or worker == ffn_count - 1 else 0
    )
    assert handle.num_recv_tokens_device.item() == expected_compact
    assert handle.recv_count_is_exact == do_cpu_sync

    source_inputs = [inputs(source, ffn_count, fp8, skew) for source in attention]
    replies = torch.full(rows.shape, float("nan"), dtype=torch.bfloat16)
    start = 0
    host_counts = []
    for local_expert in range(EXPERTS_PER_FFN):
        expert = worker * EXPERTS_PER_FFN + local_expert
        bits, scales_expected, weights_expected = [], [], []
        for codes, sf, routes, coeff in source_inputs:
            token, slot = torch.where(routes == expert)
            bits.append(codes.view(torch.uint8 if fp8 else torch.int16)[token])
            weights_expected.append(coeff[token, slot])
            if fp8:
                scales_expected.append(sf[token])
        count = sum(len(part) for part in bits)
        end = int(handle.psum_num_recv_tokens_per_expert[local_expert].item())
        assert end == start + count
        assert handle.num_unaligned_recv_tokens_per_expert[local_expert].item() == count

        actual = rows[start:end]
        key = actual[:, :4].float().int()
        order = (
            key[:, 0] * 4096 + key[:, 1] * 256 + key[:, 2] * 16 + key[:, 3]
        ).argsort()
        actual_bits = actual.view(torch.uint8 if fp8 else torch.int16)
        assert torch.equal(actual_bits[order], torch.cat(bits))
        assert torch.equal(weights[start:end][order], torch.cat(weights_expected))
        actual_scales = scales[start:end] if fp8 else None
        if fp8:
            assert torch.equal(actual_scales[order], torch.cat(scales_expected))
        replies[start:end] = reply(
            decode(actual, actual_scales), expert, weights[start:end, None]
        )
        host_counts.append(align_rows(count))
        start = align_rows(end)

    if do_cpu_sync:
        assert len(rows) == start
        assert handle.num_recv_tokens_per_expert_list == host_counts
    else:
        expected_allocation = align_rows(
            len(attention) * CAPACITY * min(TOPK, EXPERTS_PER_FFN)
            + (ALIGNMENT - 1) * EXPERTS_PER_FFN
        )
        assert len(rows) == expected_allocation
        assert not handle.num_recv_tokens_per_expert_list

    return replies, {
        "expanded_activation_bytes": rows.nbytes,
        "expanded_scale_bytes": scales.nbytes if scales is not None else 0,
        "expanded_weight_bytes": weights.nbytes,
        "expanded_reply_bytes": replies.nbytes,
        "compact_reverse_metadata_bytes": handle.recv_src_metadata.nbytes,
    }


def run(local_rank, layout, fp8):
    rank, world, group = init_dist(local_rank, len(layout))
    attention = [peer for peer, role in enumerate(layout) if role == "A"]
    ffn = [peer for peer, role in enumerate(layout) if role == "F"]
    role = ep.AFDRole.ATTENTION if rank in attention else ep.AFDRole.FFN
    wire_format = "fp8" if fp8 else "bf16"
    maxima = {"max_num_scaleout_ranks": 1, "max_num_scaleup_ranks": world}
    for multiple in (False, True):
        sizing = {
            **maxima,
            "allow_multiple_reduction": multiple,
            "group": group,
            "num_max_tokens_per_rank": CAPACITY,
            "hidden": HIDDEN,
            "num_topk": TOPK,
        }
        union = ep.ElasticBuffer.get_buffer_size_hint(**sizing)
        options = {
            **sizing,
            "role": role,
            "use_fp8_dispatch": fp8,
            "max_attention_ranks": len(attention),
            "max_ffn_ranks": len(ffn),
        }
        token_bytes = ep.ElasticBuffer.get_buffer_size_hint(**options)
        assert ARENA_ALIGNMENT < token_bytes < union
        construction = {
            **options,
            "num_experts_per_rank": EXPERTS_PER_FFN,
            "explicitly_destroy": True,
        }
        with TestCase().assertRaisesRegex(
            (RuntimeError, ValueError), "too small|smaller|must cover"
        ):
            ep.ElasticBuffer(**construction, num_bytes=token_bytes - ARENA_ALIGNMENT)
        buf = ep.ElasticBuffer(**construction, num_bytes=token_bytes)
        registered_bytes = buf.get_local_peer_info()[3]
        workspace_bytes = registered_bytes - token_bytes
        assert workspace_bytes > 0 and workspace_bytes % ARENA_ALIGNMENT == 0
        assert buf.num_bytes == token_bytes
        allocation_metrics = {}
        rounds = [(True, True), (False, True)]
        if len(ffn) > 1:
            rounds.append((False, False))
        for do_cpu_sync, skew in rounds:
            if rank in attention:
                rows, scales, ids, weights = inputs(rank, len(ffn), fp8, skew)
                payload = (rows, scales) if fp8 else rows
                receipt, _ = buf.dispatch_send(
                    payload, ids, weights, do_expand=True, expert_alignment=ALIGNMENT
                )
                result, _, _ = buf.combine_recv(receipt)
                torch.testing.assert_close(
                    result,
                    reference(rows, scales, ids, weights, len(ffn), multiple),
                    rtol=0,
                    atol=0,
                )
            else:
                received = buf.dispatch_recv(
                    with_weights=True,
                    do_expand=True,
                    expert_alignment=ALIGNMENT,
                    do_cpu_sync=do_cpu_sync,
                    num_max_tokens_per_rank=1,
                )
                replies, metrics = prepare_replies(
                    ffn.index(rank),
                    attention,
                    len(ffn),
                    received,
                    do_cpu_sync,
                    fp8,
                    skew,
                )
                buf.combine_send(replies, received[-2])
                allocation_metrics[
                    f"{'host' if do_cpu_sync else 'device'}_{'skew' if skew else 'fanout'}"
                ] = metrics
        measurements = [None] * world
        dist.all_gather_object(
            measurements,
            {
                "rank": rank,
                "role": role.value,
                "control_workspace_bytes": workspace_bytes,
                "registered_token_bytes": token_bytes,
                "registered_bytes": registered_bytes,
                "union_token_bytes": union,
                "token_savings_bytes": union - token_bytes,
                "ffn_tensors": allocation_metrics,
            },
            group=group,
        )
        assert len({item["control_workspace_bytes"] for item in measurements}) == 1
        if rank == 0:
            print(
                json.dumps(
                    {
                        "layout": layout,
                        "wire_format": wire_format,
                        "allow_multiple_reduction": multiple,
                        "capacity": CAPACITY,
                        "ranks": measurements,
                    }
                ),
                flush=True,
            )
        torch.cuda.synchronize()
        dist.barrier(group=group)
        buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(run, args=("AFAA", False), nprocs=4)
    torch.multiprocessing.spawn(run, args=("FAFF", True), nprocs=4)
