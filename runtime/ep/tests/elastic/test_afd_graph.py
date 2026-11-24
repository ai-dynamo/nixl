# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay a fixed ordinary-copy AFD decode graph with changing contents and routes."""

import statistics

import torch
import torch.distributed as dist
from test_afd_plugin import ALIGNMENT, HIDDEN, inputs

from nixl.runtime import ep  # type: ignore[attr-defined]
from nixl.runtime.ep.utils.envs import init_dist

CAPACITY = 4


def run_case(buf, rank, fp8, expand, bucket):
    x, scale, ids, weights = inputs(bucket, fp8)
    base_x = x.clone()
    routes = ids.clone()
    base_sf = scale.clone() if fp8 else None
    base_weights = weights.clone()
    source = rank == 1
    alignment = ALIGNMENT if expand else 1
    options = {
        "num_max_tokens_per_rank": bucket,
        "do_expand": expand,
        "expert_alignment": alignment,
    }

    def forward():
        if source:
            payload = (x, scale) if fp8 else x
            handle, _ = buf.dispatch_send(payload, ids, weights, **options)
            return buf.combine_recv(handle)[0]
        payload, local_ids, w, handle, _ = buf.dispatch_recv(
            with_weights=True, do_cpu_sync=False, **options
        )
        if fp8:
            data, sf = payload
            rows = data.float() * sf.repeat_interleave(128, 1)
        else:
            rows = payload.float()
        index = torch.arange(len(rows))
        offset = 0 if rank == 0 else 2
        if expand:
            reply = torch.full_like(rows, float("nan"))
            ends = handle.psum_num_recv_tokens_per_expert
            for expert in range(2):
                start = (
                    ends[expert] - handle.num_unaligned_recv_tokens_per_expert[expert]
                )
                select = ((index >= start) & (index < ends[expert]))[:, None]
                reply = torch.where(
                    select, (rows + offset + expert + 1) * w[:, None], reply
                )
        else:
            reply = torch.zeros_like(rows)
            for slot in range(3):
                expert = local_ids[:, slot, None]
                reply += torch.where(
                    expert >= 0, (rows + offset + expert + 1) * w[:, slot, None], 0
                )
            reply.masked_fill_(
                index[:, None] >= handle.num_recv_tokens_device, float("nan")
            )
        buf.combine_send(reply.bfloat16(), handle)
        return handle.num_recv_tokens_device

    def check(actual):
        if source:
            values = x.float()
            if fp8:
                values *= scale.repeat_interleave(128, 1)
            expected = torch.zeros_like(values)
            for slot in range(3):
                expert = ids[:, slot, None]
                expected += torch.where(
                    expert >= 0, (values + expert + 1) * weights[:, slot, None], 0
                )
            torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
        else:
            lo = 0 if rank == 0 else 2
            expected = ((ids >= lo) & (ids < lo + 2)).any(dim=1).sum()
            assert actual.item() == expected.item()

    for _ in range(2):
        check(forward())
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    dist.barrier()
    samples = {"eager": [], "graph": []}
    for phase in range(5):
        ids.copy_(routes)
        if phase in (0, 3):
            ids.fill_(-1)
        elif phase == 2:
            ids.copy_((routes + 1) % 4)
            ids[0].fill_(-1)
        x.copy_((base_x.float() + phase).to(x.dtype))
        weights.copy_(base_weights.roll(phase, 1))
        if fp8:
            scale.copy_(base_sf + phase)
        for mode, durations in samples.items():
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            dist.barrier()
            start.record()
            if mode == "eager":
                actual = forward()
            else:
                graph.replay()
                actual = output
            end.record()
            end.synchronize()
            check(actual)
            elapsed = torch.tensor(start.elapsed_time(end) * 1000, dtype=torch.float32)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            durations.append(elapsed.item())
    if source:
        print(
            f"AFD graph fp8={fp8} expanded={expand} bucket={bucket} rank-max median us "
            f"eager={statistics.median(samples['eager']):.2f} "
            f"replay={statistics.median(samples['graph']):.2f}",
            flush=True,
        )


def run(rank):
    _, _, group = init_dist(rank, 3)
    for fp8 in (False, True):
        buf = ep.ElasticBuffer(
            group=group,
            role=ep.AFDRole.ATTENTION if rank == 1 else ep.AFDRole.FFN,
            use_fp8_dispatch=fp8,
            num_max_tokens_per_rank=CAPACITY,
            hidden=HIDDEN,
            num_topk=3,
            num_experts_per_rank=2,
            explicitly_destroy=True,
        )
        for expand in (False, True):
            for bucket in (1, CAPACITY):
                run_case(buf, rank, fp8, expand, bucket)
        dist.barrier()
        buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(run, nprocs=3)
