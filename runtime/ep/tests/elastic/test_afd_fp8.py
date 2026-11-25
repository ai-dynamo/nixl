# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact FP8 AFD with BF16 replies on four GPUs in one NVLink domain.

Run with python tests/elastic/test_afd_fp8.py from runtime/ep.
"""

from unittest import TestCase

import torch
import torch.distributed as dist

from nixl.runtime import ep  # type: ignore[attr-defined]
from nixl.runtime.ep.utils.envs import init_dist

ATTENTION = (0, 2)
FFN = (1, 3)
HIDDEN = 512
CAPACITY = 4
TOPK = 3
EXPERTS_PER_FFN = 4
SCALE_BLOCK = 128


def inputs(source, round_idx):
    tokens = torch.arange(3 if source == 0 or round_idx else 2)
    channels = torch.arange(HIDDEN)
    blocks = torch.arange(HIDDEN // SCALE_BLOCK)
    values = ((channels[None, :] + tokens[:, None] + source) % 9 - 4).float()
    # This channel identifies each source row and is exactly representable in E4M3FN.
    values[:, 0] = 1 + source * CAPACITY + tokens
    rows = values.to(torch.float8_e4m3fn)
    scales = (
        1
        + blocks[None, :]
        + (source * CAPACITY + tokens[:, None] + round_idx) * len(blocks)
    ).float() / 32
    first = (source // 2 + tokens) % len(FFN)
    second = torch.where(tokens == 2, first, (first + 1) % len(FFN))
    ids = torch.stack(
        (
            first * EXPERTS_PER_FFN,
            first * EXPERTS_PER_FFN + 1,
            second * EXPERTS_PER_FFN + 2,
        ),
        dim=1,
    ).to(ep.topk_idx_t)
    weights = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32).repeat(len(rows), 1)
    return rows, scales, ids, weights


def dequantize(rows, scales):
    return (
        rows.float().reshape(len(rows), HIDDEN // SCALE_BLOCK, SCALE_BLOCK)
        * scales.unsqueeze(-1)
    ).reshape(len(rows), HIDDEN)


def expert_value(values, expert_ids):
    return values * (1 + expert_ids % 3) + expert_ids


def reference(rows, scales, ids, weights):
    values = dequantize(rows, scales)
    result = torch.zeros_like(values)
    for worker in range(len(FFN)):
        partial = torch.zeros_like(values)
        for slot in range(TOPK):
            expert_ids = ids[:, slot, None]
            coeff = weights[:, slot, None].masked_fill(
                expert_ids // EXPERTS_PER_FFN != worker, 0
            )
            partial += expert_value(values, expert_ids) * coeff
        # Round each worker's contribution at the BF16 wire boundary.
        result += partial.bfloat16().float()
    return result.bfloat16()


def worker_reply(rank, round_idx, rows, scales, ids, weights, count):
    worker = FFN.index(rank)
    expected = []
    for source in ATTENTION:
        x, sf, routes, coeff = inputs(source, round_idx)
        local = routes // EXPERTS_PER_FFN == worker
        keep = local.any(dim=1)
        local_ids = (routes % EXPERTS_PER_FFN).masked_fill(~local, -1)
        expected.append(
            (x.view(torch.uint8)[keep], sf[keep], local_ids[keep], coeff[keep])
        )
    expected_bits, expected_scales, expected_ids, expected_weights = map(
        torch.cat, zip(*expected, strict=True)
    )

    assert count == len(expected_bits)
    assert rows.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert scales.shape == (len(rows), HIDDEN // SCALE_BLOCK)
    assert scales.is_contiguous()
    bits = rows[:count].view(torch.uint8)
    received_order = bits[:, 0].argsort()
    expected_order = expected_bits[:, 0].argsort()
    assert torch.equal(bits[received_order], expected_bits[expected_order])
    assert torch.equal(scales[:count][received_order], expected_scales[expected_order])
    assert torch.equal(ids[:count][received_order], expected_ids[expected_order])
    valid = expected_ids[expected_order] >= 0
    assert torch.equal(
        weights[:count][received_order][valid], expected_weights[expected_order][valid]
    )

    values = dequantize(rows[:count], scales[:count])
    partial = torch.zeros_like(values)
    for slot in range(TOPK):
        local_ids = ids[:count, slot, None]
        expert_ids = worker * EXPERTS_PER_FFN + local_ids
        coeff = weights[:count, slot, None].masked_fill(local_ids < 0, 0)
        partial += expert_value(values, expert_ids) * coeff
    replies = torch.full(rows.shape, float("nan"), dtype=torch.bfloat16)
    replies[:count] = partial.bfloat16()
    return replies


def run(local_rank):
    rank, _, group = init_dist(local_rank, len(ATTENTION) + len(FFN))
    role = ep.AFDRole.ATTENTION if rank in ATTENTION else ep.AFDRole.FFN
    options = {
        "group": group,
        "role": role,
        "use_fp8_dispatch": True,
        "hidden": HIDDEN,
        "num_topk": TOPK,
        "num_experts_per_rank": EXPERTS_PER_FFN,
        "num_max_tokens_per_rank": CAPACITY,
        "explicitly_destroy": True,
    }
    with TestCase().assertRaisesRegex(ValueError, "wire configuration"):
        ep.ElasticBuffer(**(options | {"use_fp8_dispatch": rank in ATTENTION}))
    with TestCase().assertRaisesRegex(ValueError, "divisible by 128"):
        ep.ElasticBuffer(**(options | {"hidden": HIDDEN + 16}))

    buf = ep.ElasticBuffer(**options)
    for round_idx, do_cpu_sync in enumerate((True, False)):
        if rank in ATTENTION:
            rows, scales, ids, weights = inputs(rank, round_idx)
            if round_idx == 0:
                with TestCase().assertRaisesRegex(RuntimeError, "configured wire"):
                    buf.dispatch_send(rows, ids, weights)
                for invalid_scales in (scales[:, :-1], scales.bfloat16()):
                    with TestCase().assertRaises(RuntimeError):
                        buf.dispatch_send((rows, invalid_scales), ids, weights)
            handle, _ = buf.dispatch_send((rows, scales), ids, weights)
            combined, _, _ = buf.combine_recv(handle)
            torch.testing.assert_close(
                combined, reference(rows, scales, ids, weights), rtol=0, atol=0
            )
        else:
            (rows, scales), ids, weights, handle, _ = buf.dispatch_recv(
                with_weights=True, do_cpu_sync=do_cpu_sync, num_max_tokens_per_rank=1
            )
            count = int(handle.num_recv_tokens_device.item())
            assert len(rows) == (count if do_cpu_sync else len(ATTENTION) * CAPACITY)
            buf.combine_send(
                worker_reply(rank, round_idx, rows, scales, ids, weights, count), handle
            )
    torch.cuda.synchronize()
    dist.barrier(group=group)
    buf.destroy()
    dist.destroy_process_group()
    if rank == 0:
        print("Compact FP8 AFD with BF16 replies passed", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.spawn(run, nprocs=len(ATTENTION) + len(FFN))
