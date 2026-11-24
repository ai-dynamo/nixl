# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the plugin's ordinary-copy worker interface on a three-GPU FAF topology."""

import sys
from itertools import product
from pathlib import Path

import torch
import torch.distributed as dist

from nixl.runtime import ep  # type: ignore[attr-defined]
from nixl.runtime.ep.utils.envs import init_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.afd import nixlep  # noqa: E402

HIDDEN = 7168
ALIGNMENT = 4


def inputs(n, fp8):
    token = torch.arange(n, dtype=torch.float32)[:, None]
    group = torch.arange(HIDDEN // 128)[None, :] % 3
    data = (1 + token + group).repeat_interleave(128, dim=1)
    scale = (1 + token + group).float() if fp8 else None
    data = data.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
    # Alternate the expert that receives the singleton selection on the second FFN.
    ids = torch.cat((token * 0, token * 0 + 1, 2 + token % 2), 1).to(ep.topk_idx_t)
    weights = (
        torch.tensor([0.25, 0.5, 0.125], dtype=torch.float32).expand(n, 3).contiguous()
    )
    return data, scale, ids, weights


def run(rank):
    _, _, group = init_dist(rank, 3)
    source = rank == 1
    for fp8, multiple in product((False, True), repeat=2):
        buf = ep.ElasticBuffer(
            group=group,
            role=ep.AFDRole.ATTENTION if source else ep.AFDRole.FFN,
            use_fp8_dispatch=fp8,
            allow_multiple_reduction=multiple,
            num_max_tokens_per_rank=4,
            hidden=HIDDEN,
            num_topk=3,
            num_experts_per_rank=2,
            explicitly_destroy=True,
        )
        for sync in (True, False):
            for n, bucket in ((0, 1), (1, 2), (3, 4), (0, 2), (1, 1)):
                x, scale, ids, weights = inputs(n, fp8)
                decoded = x.float()
                if fp8:
                    decoded *= scale.repeat_interleave(128, 1)
                if source:
                    handle = nixlep.source_dispatch(
                        buf, x, ids, weights, scale, bucket, ALIGNMENT
                    )
                    result = nixlep.source_collect(buf, handle)
                    expected = torch.zeros_like(decoded)
                    for slot in range(3):
                        expected += (
                            (
                                (decoded + ids[:, slot, None] + 1)
                                * weights[:, slot, None]
                            )
                            .bfloat16()
                            .float()
                        )
                    torch.testing.assert_close(
                        result, expected.bfloat16(), rtol=0, atol=0
                    )
                else:
                    worker = nixlep.worker_receive(
                        buf, bucket, ALIGNMENT, do_cpu_sync=sync
                    )
                    assert worker.receipt.num_recv_tokens_device.item() == n
                    assert worker.expert_counts.is_cuda and worker.expert_ends.is_cuda
                    assert (worker.recv_sf is not None) == fp8
                    assert worker.recv_x.dtype == x.dtype
                    assert worker.recv_topk_weights.shape == (len(worker.recv_x),)
                    replies = torch.full(
                        worker.recv_x.shape, float("nan"), dtype=torch.bfloat16
                    )
                    start = 0
                    counts = []
                    for local_expert in range(2):
                        expert = (0 if rank == 0 else 2) + local_expert
                        tokens, slots = torch.where(ids == expert)
                        count = len(tokens)
                        end = int(worker.expert_ends[local_expert].item())
                        assert end == start + count
                        assert worker.expert_counts[local_expert].item() == count
                        rows = worker.recv_x[start:end].float()
                        order = rows[:, 0].argsort()
                        torch.testing.assert_close(
                            rows[order], x.float()[tokens], rtol=0, atol=0
                        )
                        if fp8:
                            scales = worker.recv_sf[start:end]
                            torch.testing.assert_close(
                                scales[order], scale[tokens], rtol=0, atol=0
                            )
                            rows *= scales.repeat_interleave(128, 1)
                        w = worker.recv_topk_weights[start:end]
                        torch.testing.assert_close(
                            w[order], weights[tokens, slots], rtol=0, atol=0
                        )
                        replies[start:end] = (rows + expert + 1) * w[:, None]
                        start = (end + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
                        counts.append((count + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT)
                    assert worker.host_counts == (counts if sync else [])
                    nixlep.worker_reply(buf, replies, worker.receipt)
        dist.barrier()
        buf.destroy()
    dist.destroy_process_group()
    if source:
        print("AFD BF16 and FP8 expert-major plugin contract passed", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.spawn(run, nprocs=3)
