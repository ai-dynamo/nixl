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
Epilogue timing (debug). Three independent measurements of one dispatch and one combine per iteration, with a
10 ms sleep + barrier before each call so ranks start together and nothing overlaps:

  wall     CUDA events around the library call (main kernel + epilogue kernel + launch gaps), event clock
  stamps   `%globaltimer` stamps compiled in with NIXL_EP_ADAPTIVE_PROFILE=1: main kernel SM 0 entry -> end
           (dispatch: SM 0 past the end barrier; combine: last block done), epilogue block 0 past the
           grid dependency -> last block done
  kineto   the library's own `bench_kineto` (the numbers tests/elastic/test_ep.py prints). The epilogues are
           launched with PDL, so their Kineto duration includes the time their blocks wait for the main kernel.

`--slots 1` prints one SLOTS line per rank as well: the phase deltas and the accumulators and counts of the slot map in
layout.cuh (a row is (name, from slot, to slot); from slot -1 an accumulator in us, -2 a count; `-` for slots the current
mode never writes). `--fp8 1` dispatches an FP8 payload.

    NIXL_EP_ADAPTIVE_PROFILE=1 python tests/elastic/profile_epilogues.py --num-processes 4 --num-sms 26 --num-tokens-used 4096
    NIXL_EP_ADAPTIVE_PROFILE=1 NIXL_EP_ADAPTIVE_MODE=hybrid python tests/elastic/profile_epilogues.py --slots 1 --fp8 1 --prefer-overlap-with-compute 0 --num-processes 4 --num-tokens 16 --num-tokens-used 16
"""
import argparse
import os
import torch
import torch.distributed as dist

from nixl.runtime import ep
from nixl.runtime.ep.utils.envs import init_dist
from nixl.runtime.ep.utils.gate import get_unbalanced_scores
from nixl.runtime.ep.utils.math import per_token_cast_to_fp8
from nixl.runtime.ep.utils.testing import bench_kineto

# Profile slots (see impls/adaptive/*.cuh): dispatch main entry 0 -> end barrier done 7, copy epilogue 12->13,
# combine main 14->15, reduce epilogue 16->17
S_DISP_ENTRY, S_DISP_END, S_COPY_START, S_COPY_END = 0, 7, 12, 13
S_COMB_ENTRY, S_COMB_END, S_RED_START, S_RED_END = 14, 15, 16, 17

DISPATCH_PHASES = [  # (name, from slot, to slot); from slot -1: an accumulator, -2: a count (see stamp_delta)
    ('bar0', 0, 1), ('count', 1, 8), ('reduce', 8, 2), ('xchg', 2, 5),
    ('x_put', 2, 20), ('x_rank', 20, 21), ('x_exp', 21, 22), ('x_local', 22, 5),
    ('put', 18, 19),  # the count put of the thread issuing to domain 1 (on domain 0) or to domain 0 (elsewhere)
    ('so_last', 1, 10), ('fwd_last', 1, 11), ('t0_end', 1, 6), ('all_end', 1, 9), ('endbar', 9, 7), ('total', 0, 7),
    ('s_loop', -1, 32), ('s_ldst', -1, 33), ('s_own', -1, 34), ('s_rel', -1, 35), ('s_put', -1, 36), ('s_ring', -1, 37),
    ('s_nput', -2, 38), ('s_nring', -2, 39),
]
# x_*: the hybrid exchange split; s_*: SM 0's first scale-out sender warp per token; combine rows: its first forward warp per row
COMBINE_PHASES = [('sync0', 14, 9), ('su_end', 14, 27), ('fwd_end', 14, 31),
                  ('put', -1, 40), ('ring', -1, 41), ('nring', -2, 42), ('stw', -1, 43),
                  ('reduce', -1, 45), ('stage', -1, 46), ('tailw', -1, 44),
                  ('nloc', -2, 48), ('loc', -1, 47), ('nrdma', -2, 50), ('rdma', -1, 49), ('total', 14, 15)]


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float('nan')


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=0)
    assert args.num_experts % num_ranks == 0
    buffer = ep.ElasticBuffer(group, num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                                   num_experts_per_rank=args.num_experts // num_ranks,
                                   allow_multiple_reduction=True, prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
                                   explicitly_destroy=True)
    num_tokens, hidden, num_topk, num_experts = args.num_tokens_used, args.hidden, args.num_topk, args.num_experts
    num_sms = args.num_sms or buffer.get_theoretical_num_sms(num_topk)
    scores = get_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, 1.0, False)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(ep.topk_idx_t)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    use_fp8 = bool(args.fp8)
    if use_fp8:
        x = per_token_cast_to_fp8(x)
    dispatch_args = dict(x=x, topk_idx=topk_idx, topk_weights=topk_weights, num_sms=num_sms,
                         num_max_tokens_per_rank=args.num_tokens, do_cpu_sync=True)
    has_stamps = os.environ.get('NIXL_EP_ADAPTIVE_PROFILE') == '1'

    def settle():
        torch.cuda.synchronize()
        torch.cuda._sleep(int(2e7))  # ~10 ms, hides launch skew between ranks
        buffer.barrier()

    def stamp_delta(prof, a, b):
        if a == -2:
            return float(prof[b]) if prof[b] != 0 else None
        if a < 0:
            return prof[b] / 1000.0 if prof[b] != 0 else None
        return (prof[b] - prof[a]) / 1000.0 if prof[a] != 0 and prof[b] != 0 else None

    rows = []
    rows_d, rows_c = [], []
    e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    combine_args = None
    for it in range(args.iters):
        settle()
        e[0].record()
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(**dispatch_args)
        e[1].record()
        torch.cuda.synchronize()
        prof_d = buffer.runtime.read_profile() if has_stamps else None
        combine_x = torch.randn((recv_x[0] if use_fp8 else recv_x).shape[0], hidden, dtype=torch.bfloat16, device='cuda')
        combine_args = dict(x=combine_x, topk_weights=recv_topk_weights, handle=handle, num_sms=num_sms)
        settle()
        e[2].record()
        buffer.combine(**combine_args)
        e[3].record()
        torch.cuda.synchronize()
        prof_c = buffer.runtime.read_profile() if has_stamps else None
        if it < 3:  # JIT / warm-up
            continue
        row = dict(disp_wall=e[0].elapsed_time(e[1]) * 1e3, comb_wall=e[2].elapsed_time(e[3]) * 1e3)
        if has_stamps:
            row.update(disp_main=stamp_delta(prof_d, S_DISP_ENTRY, S_DISP_END),
                       copy_epi=stamp_delta(prof_d, S_COPY_START, S_COPY_END),
                       disp_total=stamp_delta(prof_d, S_DISP_ENTRY, S_COPY_END),
                       comb_main=stamp_delta(prof_c, S_COMB_ENTRY, S_COMB_END),
                       reduce_epi=stamp_delta(prof_c, S_RED_START, S_RED_END),
                       comb_total=stamp_delta(prof_c, S_COMB_ENTRY, S_RED_END))
            rows_d.append({name: stamp_delta(prof_d, a, b) for name, a, b in DISPATCH_PHASES})
            rows_c.append({name: stamp_delta(prof_c, a, b) for name, a, b in COMBINE_PHASES})
        rows.append(row)

    # The library's own numbers for the same calls
    kd_main, kd_epi = bench_kineto(lambda: buffer.dispatch(**dispatch_args),
                                   kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                                   barrier_comm_profiling=True, barrier=buffer.barrier)
    kc_main, kc_epi = bench_kineto(lambda: buffer.combine(**combine_args),
                                   kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),
                                   barrier_comm_profiling=True, barrier=buffer.barrier)

    mode = buffer.get_topology()[0]
    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return f'{median(vals):8.1f}' if vals else '       -'
    def med_row(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return f'{median(vals):7.1f}' if vals else '      -'
    torch.cuda.synchronize()
    for r in range(num_ranks):
        if r == rank_idx:
            print(f'EPI rank {rank_idx}/{num_ranks} mode={mode} sms={num_sms} tokens={num_tokens} | '
                  f'dispatch wall {med("disp_wall")} main {med("disp_main")} copy_epi {med("copy_epi")} '
                  f'entry->epi_end {med("disp_total")} kineto {kd_main * 1e6:7.1f}/{kd_epi * 1e6:6.1f} | '
                  f'combine wall {med("comb_wall")} main {med("comb_main")} reduce_epi {med("reduce_epi")} '
                  f'entry->epi_end {med("comb_total")} kineto {kc_main * 1e6:7.1f}/{kc_epi * 1e6:6.1f}', flush=True)
            if args.slots:
                print(f'SLOTS rank {rank_idx:3d}/{num_ranks} mode={mode} sms={num_sms} tokens={num_tokens} '
                      f'{"fp8" if use_fp8 else "bf16"} | dispatch ' +
                      ' '.join(f'{name} {med_row(rows_d, name)}' for name, _, _ in DISPATCH_PHASES) +
                      f' wall {med("disp_wall")} | combine ' +
                      ' '.join(f'{name} {med_row(rows_c, name)}' for name, _, _ in COMBINE_PHASES) +
                      f' wall {med("comb_wall")}', flush=True)
        dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-sms', type=int, default=0)
    parser.add_argument('--num-tokens', type=int, default=4096, help='buffer max tokens per rank')
    parser.add_argument('--num-tokens-used', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--iters', type=int, default=23)
    parser.add_argument('--slots', type=int, default=0, help='also print the SLOTS line (phase deltas, accumulators, counts)')
    parser.add_argument('--fp8', type=int, default=0, help='FP8 dispatch payload (1) or BF16 (0)')
    parser.add_argument('--prefer-overlap-with-compute', type=int, default=1)
    args = parser.parse_args()
    assert not args.slots or os.environ.get('NIXL_EP_ADAPTIVE_PROFILE') == '1', 'set NIXL_EP_ADAPTIVE_PROFILE=1 for --slots'
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
