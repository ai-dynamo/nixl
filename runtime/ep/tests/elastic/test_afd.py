# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-domain eager BF16 AFD. Run with four visible GPUs.

python tests/elastic/test_afd.py
python tests/elastic/test_afd.py --layouts AF,AFA --perf-repeats 10
"""

import argparse
import statistics
import time
from unittest import TestCase

import torch
import torch.distributed as dist

from nixl.runtime import ep  # type: ignore[attr-defined]
from nixl.runtime.ep.utils.envs import init_dist

HIDDEN = 7168
TOPK = 3
EXPERTS_PER_FFN = 4
BUCKET = 8


def inputs(source, round_idx, attention, ffn, weighted=True):
    """Regenerate each source independently, including empty producers and idle FFNs."""
    count = 1 + (source + round_idx) % 4
    if round_idx in (0, 3) or (round_idx == 2 and source == attention[0]):
        count = 0
    if round_idx == 6:
        count = BUCKET
    rows = torch.arange(count, dtype=torch.int32)
    channels = torch.arange(HIDDEN, dtype=torch.int32) % 4
    x = (
        4 * (1 + BUCKET * source + rows[:, None] + channels + round_idx % 3)
    ).bfloat16()
    first = (rows + source + round_idx) % len(ffn)
    other = (first + 1) % len(ffn)
    if round_idx == 4:
        first = torch.zeros_like(rows)
        other = first
    ids = torch.stack((4 * first, 4 * first + 1, 4 * other + 2), dim=1)
    weights = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32).repeat(count, 1)
    if not weighted or round_idx == 5:
        weights.fill_(1.0)
    return x, ids.to(ep.topk_idx_t), weights


def reference(source, round_idx, attention, ffn, weighted):
    x, ids, weights = inputs(source, round_idx, attention, ffn, weighted)
    # Sum all selected expert contributions in original token order.
    out = torch.zeros_like(x, dtype=torch.float32)
    for k in range(TOPK):
        out += (x.float() + 4 * ids[:, k, None]) * weights[:, k, None]
    return out.bfloat16()


def check_worker(
    me,
    round_idx,
    attention,
    ffn,
    received,
    local_ids,
    weights,
    weighted,
    *,
    bounded=False,
):
    ordinal = ffn.index(me)
    rows, indices, coeffs = [], [], []
    for source in attention:
        x, ids, w = inputs(source, round_idx, attention, ffn, weighted)
        local = ids // EXPERTS_PER_FFN == ordinal
        keep = local.any(dim=1)
        rows.append(x[keep])
        indices.append((ids % EXPERTS_PER_FFN).masked_fill(~local, -1)[keep])
        coeffs.append(w[keep])

    x, ids, w = map(torch.cat, (rows, indices, coeffs))
    allocated = len(received)
    assert allocated == (len(attention) * BUCKET if bounded else len(x))
    assert len(local_ids) == allocated and (
        weights is None or len(weights) == allocated
    )
    received, local_ids = received[: len(x)], local_ids[: len(x)]
    if weights is not None:
        weights = weights[: len(x)]
    # Compact receive coalesces multiple experts on one FFN. Wire arrival order is free.
    order = received[:, 0].argsort()
    expected_order = x[:, 0].argsort()
    assert received.dtype == torch.bfloat16
    assert torch.equal(received[order], x[expected_order]), (
        me,
        round_idx,
        received.shape,
        x.shape,
    )
    assert torch.equal(local_ids[order], ids[expected_order]), (me, round_idx)
    valid = ids[expected_order] >= 0
    if weighted:
        assert torch.equal(weights[order][valid], w[expected_order][valid]), (
            me,
            round_idx,
        )
    else:
        assert weights is None
        weights = torch.ones_like(local_ids, dtype=torch.float32)

    out = torch.zeros_like(received, dtype=torch.float32)
    for k in range(TOPK):
        expert = ordinal * EXPERTS_PER_FFN + local_ids[:, k, None]
        weight = weights[:, k, None].masked_fill(local_ids[:, k, None] < 0, 0)
        out += (received.float() + 4 * expert) * weight
    # NaN tail rows expose a combine that consumes the allocation instead of the device count.
    replies = torch.full((allocated, HIDDEN), float("nan"), dtype=torch.bfloat16)
    replies[: len(x)] = out.bfloat16()
    return replies, len(x)


def check_fused_parity(rank, group, attention, ffn, directional_result):
    # Regular EP reserves expert slots on every rank, including attention ranks.
    fused = ep.ElasticBuffer(
        group=group,
        num_max_tokens_per_rank=BUCKET,
        hidden=HIDDEN,
        num_topk=TOPK,
        num_experts_per_rank=EXPERTS_PER_FFN,
        explicitly_destroy=True,
    )
    x, logical_ids, weights = inputs(rank, 1, attention, ffn)
    physical = torch.tensor(ffn)[logical_ids // EXPERTS_PER_FFN] * EXPERTS_PER_FFN
    physical = (physical + logical_ids % EXPERTS_PER_FFN).to(ep.topk_idx_t)
    if rank in ffn:
        x, physical, weights = x[:0], physical[:0], weights[:0]
    rows, ids, w, handle, _ = fused.dispatch(x, physical, weights)
    if rank in ffn:
        replies, _ = check_worker(rank, 1, attention, ffn, rows, ids, w, True)
    else:
        assert len(rows) == 0
        replies = rows
    result, _, _ = fused.combine(replies, handle)
    if rank in attention:
        assert torch.equal(result, directional_result)
    fused.destroy()


def wrong_role(fn, *args):
    with TestCase().assertRaisesRegex(
        (ValueError, RuntimeError, AssertionError), "(?i)role|attention|ffn"
    ):
        fn(*args)


def benchmark(buf, me, attention, ffn, repeats):
    """Local round duration. Eager receive includes the host count read."""
    x, ids, weights = inputs(me, 5, attention, ffn)
    samples = []
    for i in range(repeats + 2):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        if me in attention:
            handle, _ = buf.dispatch_send(x, ids, weights)
            buf.combine_recv(handle)
        else:
            received, _, _, handle, _ = buf.dispatch_recv()
            buf.combine_send(received, handle)
        stop.record()
        stop.synchronize()
        if i >= 2:
            samples.append(1000 * start.elapsed_time(stop))
    all_samples = [None] * dist.get_world_size()
    dist.all_gather_object(all_samples, samples)
    if me == 0:
        worst = [max(all_samples[a][i] for a in attention) for i in range(repeats)]
        print(
            f"AFD-BF16 attention={attention} ffn={ffn} attention_round_us={worst} "
            f"median_us={statistics.median(worst):.2f}",
            flush=True,
        )


def run(me, layout, repeats, weighted):
    rank, _, group = init_dist(me, len(layout))
    attention = [r for r, role in enumerate(layout) if role == "A"]
    ffn = [r for r, role in enumerate(layout) if role == "F"]
    role = ep.AFDRole.ATTENTION if rank in attention else ep.AFDRole.FFN
    config = {
        "group": group,
        "role": role,
        "num_max_tokens_per_rank": BUCKET,
        "hidden": HIDDEN,
        "num_topk": TOPK,
        "num_experts_per_rank": EXPERTS_PER_FFN,
        "explicitly_destroy": True,
    }
    if layout == "AF" and weighted:
        with TestCase().assertRaisesRegex(ValueError, "attention.*FFN"):
            ep.ElasticBuffer(**(config | {"role": ep.AFDRole.ATTENTION}))
        # This fails during native AFD configuration after peer registration.
        with TestCase().assertRaisesRegex((RuntimeError, ValueError), "hidden"):
            ep.ElasticBuffer(**(config | {"hidden": HIDDEN + 16}))
    buf = ep.ElasticBuffer(**config)
    if layout == "AF" and weighted:
        size = ep.ElasticBuffer.get_buffer_size_hint(
            group,
            BUCKET,
            HIDDEN,
            TOPK,
            role=role,
            max_num_scaleout_ranks=1,
            max_num_scaleup_ranks=len(layout),
        )
        assert size == buf.num_bytes
        peer = 1 - rank
        for operation in (
            lambda: buf.connect_ranks([peer]),
            lambda: buf.disconnect_ranks([peer]),
            lambda: buf.stage_topology(active_ranks=[rank]),
            lambda: buf.stage_topology_forced("hybrid"),
        ):
            with TestCase().assertRaisesRegex(ValueError, "fixed"):
                operation()
        x, ids, weights = inputs(attention[0], 1, attention, ffn)
        # Call the binding directly so the Python role guard cannot mask a native error.
        if rank in attention:
            with TestCase().assertRaisesRegex(RuntimeError, "Wrong AFD role"):
                buf.runtime.dispatch_recv(BUCKET, 1, 1, True)
            with TestCase().assertRaisesRegex(RuntimeError, "Wrong AFD role"):
                buf.runtime.combine_send(x, ids, ids, True, BUCKET, 1)
        else:
            with TestCase().assertRaisesRegex(RuntimeError, "Wrong AFD role"):
                buf.runtime.dispatch_send(x, ids, weights, BUCKET, 1, 1)
            with TestCase().assertRaisesRegex(RuntimeError, "Wrong AFD role"):
                buf.runtime.combine_recv(ids, None, None, BUCKET, 1)
    print(
        f"AFD-ARENA layout={layout} rank={rank} role={role} token_bytes={buf.num_bytes}",
        flush=True,
    )
    streams = [torch.cuda.current_stream(), torch.cuda.Stream()]
    directional_result = None
    for r in range(7):
        # Both orderings of the local bucket bounds, including a full-capacity source
        # with a one-token FFN bucket. The peer's bound is unknown at launch.
        send_bucket = 4 if r == 1 else BUCKET
        recv_bucket = 1 if r == 6 else 4 if r == 4 else BUCKET
        bounded = r in (3, 4, 6)
        stream = streams[r % 2]
        stream.wait_stream(streams[(r - 1) % 2])
        with torch.cuda.stream(stream):
            if rank in attention:
                x, ids, weights = inputs(rank, r, attention, ffn, weighted)
                if r == 1 and rank == attention[0]:
                    time.sleep(0.1)
                if r == 0:
                    wrong_role(buf.dispatch_recv)
                    with TestCase().assertRaisesRegex(ValueError, "bucket"):
                        buf.dispatch_send(x, ids, num_max_tokens_per_rank=BUCKET + 1)
                    with TestCase().assertRaises(ValueError):
                        buf.dispatch_send(x, ids, do_expand=True)
                    with TestCase().assertRaises(ValueError):
                        buf.dispatch_send(x.float(), ids)
                handle, _ = buf.dispatch_send(
                    x,
                    ids,
                    weights if weighted and r != 5 else None,
                    num_max_tokens_per_rank=send_bucket,
                )
                if r == 0:
                    wrong_role(buf.combine_send, x, handle)
                    with TestCase().assertRaisesRegex(
                        (ValueError, RuntimeError), "pending"
                    ):
                        buf.dispatch_send(x, ids)
                if r == 4 and rank == attention[-1]:
                    time.sleep(0.1)
                assert handle.num_max_tokens_per_rank == send_bucket
                result, _, _ = buf.combine_recv(handle)
                if r == 0:
                    with TestCase().assertRaisesRegex(
                        (ValueError, RuntimeError), "consumed"
                    ):
                        buf.combine_recv(handle)
                if r == 1 and layout == "FAFA":
                    directional_result = result
                assert torch.equal(
                    result, reference(rank, r, attention, ffn, weighted)
                ), (rank, r)
            else:
                if r == 2 and rank == ffn[-1]:
                    time.sleep(0.1)
                if r == 0:
                    wrong_role(buf.dispatch_send, *inputs(me, r, attention, ffn))
                    with TestCase().assertRaisesRegex(ValueError, "bucket"):
                        buf.dispatch_recv(num_max_tokens_per_rank=BUCKET + 1)
                    with TestCase().assertRaises(ValueError):
                        buf.dispatch_recv(do_expand=True)
                received, ids, weights, handle, _ = buf.dispatch_recv(
                    with_weights=weighted or r == 5,
                    do_cpu_sync=not bounded,
                    num_max_tokens_per_rank=recv_bucket,
                )
                assert handle.num_max_tokens_per_rank == recv_bucket
                assert handle.recv_count_is_exact == (not bounded)
                if bounded:
                    assert not handle.num_recv_tokens_per_expert_list
                if r == 0:
                    wrong_role(buf.combine_recv, handle)
                    with TestCase().assertRaisesRegex(
                        (ValueError, RuntimeError), "pending"
                    ):
                        buf.dispatch_recv()
                replies, count = check_worker(
                    rank,
                    r,
                    attention,
                    ffn,
                    received,
                    ids,
                    weights,
                    weighted or r == 5,
                    bounded=bounded,
                )
                if r == 5 and rank == ffn[0]:
                    time.sleep(0.1)
                buf.combine_send(replies, handle)
                assert handle.num_recv_tokens_device.item() == count
                if r == 0:
                    with TestCase().assertRaisesRegex(
                        (ValueError, RuntimeError), "consumed"
                    ):
                        buf.combine_send(replies, handle)
    torch.cuda.synchronize()
    if repeats:
        benchmark(buf, rank, attention, ffn, repeats)
    dist.barrier()
    buf.destroy()
    if layout == "FAFA":
        check_fused_parity(rank, group, attention, ffn, directional_result)
    dist.destroy_process_group()
    if rank == 0:
        print(f"AFD {layout} weighted={weighted} passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", default="AF,FAF,AFA,FAFA")
    parser.add_argument("--perf-repeats", type=int, default=5)
    args = parser.parse_args()
    for layout in args.layouts.split(","):
        assert set(layout) == {"A", "F"}, layout
        torch.multiprocessing.spawn(
            run, args=(layout, args.perf_repeats, True), nprocs=len(layout)
        )
        if layout == "AF":
            torch.multiprocessing.spawn(
                run, args=(layout, 0, False), nprocs=len(layout)
            )
