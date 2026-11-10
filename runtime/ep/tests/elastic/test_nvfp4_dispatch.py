# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Byte-exact NVFP4 dispatch and BF16 combine, eager and with fresh graph inputs.

Run on one or several NVLink domains; production topology selects the mode.
No quantizer, performance measurement, or wire-format switching is involved.
"""
import argparse

import torch
import torch.distributed as dist
from test_adaptive_modes import capture
from test_ep import fold_expanded

import nixl.runtime.ep as nixl_ep_v2
from nixl.runtime.ep.utils.envs import dist_print, init_dist
from nixl.runtime.ep.utils.refs import dispatch as ref_dispatch


def run(buffer, args):
    rank, world = dist.get_rank(), dist.get_world_size()
    assert args.hidden % 64 == 0 and args.num_experts % world == 0
    assert 0 < args.num_topk <= args.num_experts
    n = max(1, args.num_tokens - rank)
    token = rank * args.num_tokens + torch.arange(n, device="cuda")
    topk = (
        token[:, None] * 7 + torch.arange(args.num_topk, device="cuda")
    ) % args.num_experts
    topk = topk.to(nixl_ep_v2.topk_idx_t)
    weights = torch.ones_like(topk, dtype=torch.float32)
    codes = torch.empty((n, args.hidden // 2), dtype=torch.uint8, device="cuda")
    scales = torch.empty((n, args.hidden // 16), dtype=torch.uint8, device="cuda")
    common = dict(topk_idx=topk, topk_weights=weights)

    def fill(salt):
        for tensor, factor in ((codes, 17), (scales, 31)):
            tensor.copy_(
                (
                    token[:, None] * factor
                    + torch.arange(tensor.shape[1], device="cuda")
                    + salt
                ).to(torch.uint8)
            )

    def reference():
        code_ref = ref_dispatch(codes, topk, weights, args.num_tokens, args.num_experts)
        scale_ref = ref_dispatch(
            scales, topk, weights, args.num_tokens, args.num_experts
        )
        assert torch.equal(code_ref[3], scale_ref[3])
        return code_ref[0], scale_ref[0], code_ref[3]

    def check(output, expected):
        received, _, _, handle, _ = output
        count = handle.num_recv_tokens_device.item()
        src = handle.recv_src_metadata[:count, 0]
        order = src.argsort()
        code_rows, scale_rows = received[0], received[1].view(torch.uint8)
        assert (
            received[0].dtype == torch.uint8
            and received[1].dtype == torch.float8_e4m3fn
        )
        assert (
            code_rows.shape[1] == args.hidden // 2
            and scale_rows.shape[1] == args.hidden // 16
        )
        if handle.do_expand:
            slots = handle.recv_src_metadata[:count, 2:]
            valid = slots >= 0
            safe = slots.clamp_min(0)
            for tensor in (code_rows, scale_rows):
                for expert in range(args.num_experts // world):
                    begin = handle.psum_num_recv_tokens_per_expert[expert].item()
                    end = (
                        (begin + handle.expert_alignment - 1) // handle.expert_alignment
                    ) * handle.expert_alignment
                    assert not tensor[begin:end].any(), "nonzero expanded padding"
            code_rows = fold_expanded(code_rows, safe, valid)
            scale_rows = fold_expanded(scale_rows, safe, valid)
        else:
            code_rows, scale_rows = code_rows[:count], scale_rows[:count]
        assert torch.equal(src[order], expected[2]), "received source rows differ"
        assert torch.equal(code_rows[order], expected[0]), "NVFP4 code bytes differ"
        assert torch.equal(scale_rows[order], expected[1]), "NVFP4 scale bytes differ"

    def combine(output):
        received, idx, _, handle, _ = output
        src = handle.recv_src_metadata[:, 0]
        value = (src.clamp_min(0) % 13 + 1).to(torch.bfloat16)
        if handle.do_expand:
            count = handle.num_recv_tokens_device.item()
            local = value[:count, None, None].expand(-1, args.num_topk, args.hidden)
            # The final extra row absorbs -1 metadata indices.
            x = torch.empty(
                (received[0].shape[0] + 1, args.hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            x[handle.recv_src_metadata[:count, 2:].flatten()] = local.reshape(
                -1, args.hidden
            )
            x = x[:-1]
        else:
            x = (
                (value * (idx >= 0).sum(dim=1))
                .to(torch.bfloat16)[:, None]
                .expand(-1, args.hidden)
                .contiguous()
            )
        result, _ = buffer.combine(x, handle)
        return result

    expected_combine = (
        ((token % 13 + 1) * args.num_topk)
        .to(torch.bfloat16)[:, None]
        .expand(-1, args.hidden)
    )
    fill(0)
    expected = reference()
    for expand in (False, True):
        output = buffer.dispatch(
            (codes, scales.view(torch.float8_e4m3fn)),
            **common,
            do_expand=expand,
            do_zero_padding=expand,
            expert_alignment=16 if expand else 1
        )
        check(output, expected)
        assert torch.equal(
            combine(output), expected_combine
        ), "BF16 combine through NVFP4 handle differs"

    def forward():
        output = buffer.dispatch((codes, scales), **common, do_cpu_sync=False)
        return output, combine(output)

    graph, (output, combined) = capture(forward)
    for salt in (37, 83):
        fill(salt)
        expected = reference()
        output[0][0].zero_()
        output[0][1].view(torch.uint8).zero_()
        combined.zero_()
        graph.replay()
        check(output, expected)
        assert torch.equal(combined, expected_combine), "graph BF16 combine differs"
    del output, combined, graph
    torch.cuda.synchronize()
    dist.barrier()

    for x, extra in (
        (codes, {}),
        ((codes, scales[:, :-1]), {}),
        ((codes, scales.to(torch.float16)), {}),
        ((codes, scales), {"use_tma_aligned_col_major_sf": True}),
    ):
        try:
            buffer.dispatch(x, **common, **extra)
        except (AssertionError, RuntimeError) as error:
            assert "NVFP4" in str(error), str(error)
        else:
            raise AssertionError("invalid NVFP4 input was accepted")


def test_loop(local_rank, num_processes, args):
    _, world, group = init_dist(local_rank, num_processes)
    assert args.num_experts > 0 and args.num_experts % world == 0
    buffer = nixl_ep_v2.ElasticBuffer(
        group,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        num_experts_per_rank=args.num_experts // world,
        num_topk=args.num_topk,
        use_fp8_dispatch=args.fp8_capacity,
        prefer_overlap_with_compute=True,
        explicitly_destroy=True,
    )
    run(buffer, args)
    dist_print("NVFP4 byte transport and BF16 combine passed", once_in_node=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Total physical expert slots across the world",
    )
    parser.add_argument(
        "--fp8-capacity", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
