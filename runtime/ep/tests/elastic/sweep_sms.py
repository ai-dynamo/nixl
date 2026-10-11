# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Active-SM sweep of the adaptive kernels inside ONE process group: the buffer is constructed once, then every
(token count, active SM count) point runs a dispatch and a combine, checks them against the NCCL reference and
times both main kernels and both epilogues with Kineto (the same method as `test_ep.py`). The active count is
changed between points through `dispatch(num_sms=...)` (a stage + commit of the topology descriptor); the launch
grid comes from `NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS` (set it above the largest count of the list, identical on every rank).

Bench only (not part of the library's tests). Usage (one task per node, 4 processes each):
    NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS=62 python tests/elastic/sweep_sms.py --num-processes 4 --num-sms-list 8,16,26,40,62 \
        --num-tokens-list 16,64,128,1024,4096 --fp8-dispatch 0 --prefer-overlap-with-compute 1 --reps 2
Each rank prints one `SWEEP ...` line per point and rep; `adaptive-runs/launch_shape/sweep_table.py` summarises them.
"""
import argparse
import os
import torch
import torch.distributed as dist

import nixl_ep_v2
from nixl_ep_v2.utils.math import per_token_cast_back, per_token_cast_to_fp8
from nixl_ep_v2.utils.gate import get_unbalanced_scores
from nixl_ep_v2.utils.envs import init_dist, dist_print
from nixl_ep_v2.utils.refs import dispatch as ref_dispatch
from nixl_ep_v2.utils.refs import combine as ref_combine
from nixl_ep_v2.utils.refs import generate_pre_combine_data, ordered_accumulate
from nixl_ep_v2.utils.testing import bench_kineto


def parse_int_list(s: str):
    return [int(v) for v in s.replace(' ', ',').split(',') if v]


def check_dispatch(recv_x, recv_topk_idx, recv_topk_weights, handle, num_recv_tokens, ref, num_ranks: int, use_fp8: bool):
    ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, ref_recv_src_token_idx, ref_num_recv_tokens_per_rank = ref
    src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
    sorted_metadata = torch.sort(src_token_global_idx)
    assert torch.equal(ref_recv_src_token_idx, sorted_metadata.values), 'received source indices differ from the reference'
    if use_fp8:
        recv_x = (recv_x[0][:num_recv_tokens], recv_x[1][:num_recv_tokens])
    else:
        recv_x = recv_x[:num_recv_tokens]
    recv_topk_idx = recv_topk_idx[:num_recv_tokens]
    recv_topk_weights = recv_topk_weights[:num_recv_tokens]
    # The same per-source-rank bitwise comparison as test_ep.py
    for i in range(num_ranks):
        rank_start_idx = sum(ref_num_recv_tokens_per_rank[:i])
        rank_end_idx = rank_start_idx + ref_num_recv_tokens_per_rank[i]
        sorted_indices = sorted_metadata.indices[rank_start_idx:rank_end_idx]
        check_list = [(ref_recv_topk_weights, recv_topk_weights, True), (ref_recv_topk_idx, recv_topk_idx, False)]
        if use_fp8:
            check_list += [(ref_recv_x[0], recv_x[0], False), (ref_recv_x[1], recv_x[1], False)]
        else:
            check_list += [(ref_recv_x, recv_x, False)]
        ref_mask = ref_recv_topk_idx[rank_start_idx:rank_end_idx] < 0
        for ref_t, t, do_mask in check_list:
            ref_t = ref_t[rank_start_idx:rank_end_idx]
            t = t[sorted_indices]
            if do_mask:
                ref_t = ref_t.masked_fill(ref_mask, 0)
                t = t.masked_fill(ref_mask, 0)
            assert torch.equal(ref_t, t), 'dispatch data differ from the reference'


# noinspection PyUnboundLocalVariable,PyShadowingNames
@torch.inference_mode()
def sweep_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    assert args.num_experts % num_ranks == 0, f'{args.num_experts} experts do not divide by {num_ranks} ranks'
    use_fp8 = bool(args.fp8_dispatch)
    # Zero copy: a one-slot arena of the sweep's wire; every dispatch writes slot 0 in place (the handle's metadata lives in
    # the slot, so the timed dispatches overwrite what the timed combines read: fine for timing, the checks run first)
    buffer = nixl_ep_v2.ElasticBuffer(group,
                                      num_experts_per_rank=args.num_experts // num_ranks,
                                      num_max_tokens_per_rank=args.num_max_tokens_per_rank, hidden=args.hidden,
                                      num_topk=args.num_topk, use_fp8_dispatch=use_fp8, zc_use_fp8_dispatch=use_fp8,
                                      num_zero_copy_slots=1 if args.zero_copy else 0,
                                      deterministic=False, allow_multiple_reduction=True,
                                      prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
                                      explicitly_destroy=True,
                                      num_gpu_timeout_secs=args.num_gpu_timeout_secs,
                                      num_cpu_timeout_secs=args.num_cpu_timeout_secs)
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_topology_ranks()
    mode = buffer.get_topology()[0]
    # Zero copy is a direct-mode feature: the library refuses it on a hybrid world at the first dispatch; say so up front
    assert not (args.zero_copy and mode == 'hybrid'), 'zero copy is a direct-mode feature: run the sweep without --zero-copy on a hybrid (multi-domain) world'
    num_experts, num_topk, hidden = args.num_experts, args.num_topk, args.hidden
    sms_list = [s for s in parse_int_list(args.num_sms_list) if s <= buffer.num_launch_sms]
    skipped = [s for s in parse_int_list(args.num_sms_list) if s > buffer.num_launch_sms]
    dist_print(f'SWEEP-CONFIG tag={args.tag} world={num_scaleout_ranks}x{num_scaleup_ranks} mode={mode} '
               f'launch_sms={buffer.num_launch_sms} po={args.prefer_overlap_with_compute} fp8={int(use_fp8)} '
               f'zero_copy={int(args.zero_copy)} dynamic_take={buffer.dynamic_take} acquire_flags={int(buffer.acquire_flags)} '
               f'hidden={hidden} topk={num_topk} experts={num_experts} tokens={args.num_tokens_list} sms={sms_list} '
               f'skipped_above_grid={skipped} reps={args.reps} num_tests={args.num_tests}', once_in_node=True)

    for num_tokens_nominal in parse_int_list(args.num_tokens_list):
        # The same convention as test_ep.py: rank r gets `tokens - r` tokens (at least one)
        num_tokens = max(1, num_tokens_nominal - rank_idx)
        num_max_tokens_per_rank = max(args.num_max_tokens_per_rank, num_tokens_nominal)
        torch.manual_seed(args.seed + rank_idx)
        scores = get_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, 1.0, False)
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        topk_idx = topk_idx.to(nixl_ep_v2.topk_idx_t)
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        x_in = per_token_cast_to_fp8(x) if use_fp8 else x

        ref = None
        if not args.skip_check:
            ref = ref_dispatch(x_in, topk_idx, topk_weights, num_max_tokens_per_rank, num_experts)
            ref_y = generate_pre_combine_data(rank_idx * num_max_tokens_per_rank + torch.arange(num_tokens, device='cuda'),
                                              num_max_tokens_per_rank, num_topk, hidden)
            ref_y[topk_idx == -1] = 0
            # allow_multiple_reduction: the hybrid combine reduces inside the scale-up domain first
            ref_combined_y = ref_combine(ref_y, topk_idx, num_scaleout_ranks, num_scaleup_ranks, num_experts, None,
                                         True, buffer.combine_is_hybrid)
            torch.cuda.synchronize()

        for rep in range(args.reps):
            order = sms_list if rep % 2 == 0 else sms_list[::-1]
            for num_sms in order:
                # Zero copy is compact: expert alignment 1 (the main kernels do not depend on the alignment; the copy epilogue's padding does)
                dispatch_args = dict(x=x_in, topk_idx=topk_idx, topk_weights=topk_weights, num_sms=num_sms,
                                     num_max_tokens_per_rank=num_max_tokens_per_rank,
                                     expert_alignment=1 if args.zero_copy else args.expert_alignment,
                                     do_handle_copy=True, do_cpu_sync=True)
                if args.zero_copy:
                    dispatch_args.update(zero_copy=True, zero_copy_slot=0)
                recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(**dispatch_args)
                _, num_active_sms = buffer.get_topology()
                assert num_active_sms == num_sms, f'{num_active_sms=} after dispatch(num_sms={num_sms})'
                recv_x_bf16 = per_token_cast_back(recv_x[0], recv_x[1]) if use_fp8 else recv_x
                num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()

                # Combine input from the received routing (as test_ep.py)
                src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
                local_y = generate_pre_combine_data(src_token_global_idx, num_max_tokens_per_rank, num_topk, hidden)
                local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0
                input_for_combine = torch.empty_like(recv_x_bf16, dtype=torch.bfloat16, device='cuda')
                input_for_combine[:num_recv_tokens] = ordered_accumulate(local_y)
                combine_args = dict(x=input_for_combine, topk_weights=recv_topk_weights, handle=handle, num_sms=num_sms)
                combined_x, combined_topk_weights, _ = buffer.combine(**combine_args)

                if ref is not None:
                    check_dispatch(recv_x, recv_topk_idx, recv_topk_weights, handle, num_recv_tokens, ref, num_ranks, use_fp8)
                    assert torch.equal(combined_x, ref_combined_y), 'combined data differ from the reference'
                    assert torch.equal(combined_topk_weights, topk_weights), 'combined top-k weights differ'

                t_dispatch, t_copy = bench_kineto(lambda: buffer.dispatch(**dispatch_args),
                                                  kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                                                  num_tests=args.num_tests, barrier_comm_profiling=True, barrier=buffer.barrier)
                t_combine, t_reduce = bench_kineto(lambda: buffer.combine(**combine_args),
                                                   kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),
                                                   num_tests=args.num_tests, barrier_comm_profiling=True, barrier=buffer.barrier)
                print(f'SWEEP tag={args.tag} rank={rank_idx}/{num_ranks} world={num_scaleout_ranks}x{num_scaleup_ranks} '
                      f'mode={mode} po={args.prefer_overlap_with_compute} fp8={int(use_fp8)} tokens={num_tokens_nominal} '
                      f'sms={num_sms} active={num_active_sms} rep={rep} checked={int(ref is not None)} '
                      f'dispatch_us={t_dispatch * 1e6:.2f} copy_us={t_copy * 1e6:.2f} '
                      f'combine_us={t_combine * 1e6:.2f} reduce_us={t_reduce * 1e6:.2f} zc={int(args.zero_copy)}', flush=True)
                dist.barrier()
        dist_print(f'SWEEP-DONE tag={args.tag} tokens={num_tokens_nominal}', once_in_node=True)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active-SM sweep of the adaptive EP kernels (bench only)')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--tag', type=str, default='sweep')
    parser.add_argument('--num-sms-list', type=str, default='8,12,16,20,26,32,40,48,62')
    parser.add_argument('--num-tokens-list', type=str, default='16,64,128,1024,4096')
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--expert-alignment', type=int, default=128)
    parser.add_argument('--fp8-dispatch', type=int, default=0)
    parser.add_argument('--prefer-overlap-with-compute', type=int, default=1)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--num-tests', type=int, default=30, help='Kineto iterations per point (as test_ep.py)')
    parser.add_argument('--zero-copy', type=int, default=0, help='1: a one-slot zero-copy arena, dispatch(zero_copy=True, zero_copy_slot=0) (direct mode only: refused on a hybrid world)')
    parser.add_argument('--skip-check', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-gpu-timeout-secs', type=int, default=100)
    parser.add_argument('--num-cpu-timeout-secs', type=int, default=100)
    args = parser.parse_args()
    torch.multiprocessing.spawn(sweep_loop, args=(args.num_processes, args), nprocs=args.num_processes)
