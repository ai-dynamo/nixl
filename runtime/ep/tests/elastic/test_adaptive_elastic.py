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
Elastic membership of the adaptive EP kernels on the NIXL transport (RDMA + NVLink, hybrid mode): a plan of live worlds
(phases: lists of dense rank ids `scaleout_rank_idx * max_num_scaleup_ranks + scaleup_rank_idx`, each a product of
NVLink domains and positions, see `ElasticBuffer`) on a job whose processes all stay alive. Per phase, every process
stages the topology descriptor (`EPTopology`) with the phase's ranks in its masks (`scaleout_mask`, `scaleup_mask`) and
commits it, connections kept (with `--reconnect-on-leave` a leave is a symmetric `disconnect_ranks`, a rejoin a symmetric
`connect_ranks`). Participants run one eager dispatch (CPU sync) + combine routed over the LIVE experts, checked without
any collective (every rank regenerates every other rank's routing from a seed), then replay the CUDA graph captured at
the first phase they took part in against the reference of the current phase (selections of experts owned by ranks that
left are dropped). Non-participants mirror every collective and touch no kernel. `--perf-phases` and `--feed-pairs` add
the PERF and TRANS timing lines.

Launch: one invocation per node (`--num-processes` GPUs each) under the node-level `WORLD_SIZE` / `RANK` of `init_dist`,
no process group; the rank 0 process serves a TCPStore on `MASTER_ADDR` / `MASTER_PORT` (127.0.0.1:8361) for the whole
run and keeps it alive until every peer has torn down (if a previous run still holds the port, choose another).
`--store-timeout-secs` bounds every Store wait, `--num-cpu-timeout-secs` only the dispatch's CPU sync. `--num-experts`
is the expert slot count of the DECLARED maximum world and must divide by its rank count: every rank owns that many
slots in every world (fixed expert slots), and a world's expert count is its bounds product times the slots per rank
(192 over 2 x 4 = 24 slots per rank). `--max-scaleup-ranks 72 --position-offset 64` places a 1 x 8 job at positions
64..71 of a 1 x 72 world (72 ranks: `--num-experts 288` = 4 slots per rank).
    python tests/elastic/test_adaptive_elastic.py --num-processes 4 --num-tokens 512 --hidden 7168 --num-topk 8 --num-experts 192
    python tests/elastic/test_adaptive_elastic.py --plan '[[0,1,2,3,4,5,6,7],[0,1,2,3],[0,1,2,3,4,5,6,7]]' ...
"""
import argparse
import json
import os
import random
import time
from datetime import timedelta
from statistics import median_high as median
import torch
import torch.distributed as dist
from typing import Callable, Dict, List, Optional, Tuple

from nixl.runtime import ep
from nixl.runtime.ep.utils.nixl_bootstrap import NixlBootstrap
from nixl.runtime.ep.utils.refs import combine as ref_combine, generate_pre_combine_data, ordered_accumulate
from nixl.runtime.ep.utils.math import calc_diff, per_token_cast_to_fp8
from nixl.runtime.ep.utils.testing import bench, bench_kineto

from test_adaptive_modes import combine_recipe


def init_test_environment(local_rank: int, num_local_ranks: int, seed: int) -> Tuple[int, int]:
    """`init_dist`'s node-level `WORLD_SIZE`/`RANK` convention without a process group: this process's index in the
    cohort of every invocation and the cohort size; selects the GPU and seeds per rank."""
    num_nodes, node_rank = int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('RANK', 0))
    assert num_nodes > 0 and 0 <= node_rank < num_nodes
    assert 0 <= local_rank < num_local_ranks
    assert not dist.is_initialized(), 'This test uses a standalone Store, not a process group'
    rank_idx, num_ranks = node_rank * num_local_ranks + local_rank, num_nodes * num_local_ranks
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.manual_seed(seed + rank_idx)
    random.seed(seed + rank_idx)
    return rank_idx, num_ranks


def make_buffer(args: argparse.Namespace, rank_idx: int, num_ranks: int, store: dist.Store, **placement) -> ep.ElasticBuffer:
    return ep.ElasticBuffer(rank_idx=rank_idx, num_ranks=num_ranks, store=store,
                                 num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                                 allow_multiple_reduction=bool(args.allow_multiple_reduction),
                                 prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
                                 num_nixl_channels=args.num_nixl_channels, explicitly_destroy=True,
                                 deterministic=args.deterministic,
                                 num_gpu_timeout_secs=args.num_gpu_timeout_secs,
                                 num_cpu_timeout_secs=args.num_cpu_timeout_secs,
                                 store_timeout_secs=args.store_timeout_secs, **placement)


def default_plan(max_su: int, present: List[int]) -> List[List[int]]:
    """The full world, then each candidate (a domain alone, uniform and holed position shrinks) followed by the full
    world again. The full world is the set of PRESENT ranks. Every world is legal: with fixed expert slots per rank, a
    world's expert count is its bounds product times the slots, nothing has to divide."""
    domains = sorted(set(r // max_su for r in present))
    positions = sorted(set(r % max_su for r in present))

    def ids(ds, ks):
        return sorted(d * max_su + k for d in ds for k in ks)

    all_ranks = ids(domains, positions)
    assert all_ranks == sorted(present), f'The present ranks {sorted(present)} are not a product of domains and positions'
    phases = [all_ranks]
    candidates = []
    if len(domains) > 1:
        candidates.append(ids(domains[:1], positions))                     # the first domain alone
    if len(positions) > 1:
        candidates.append(ids(domains, positions[:len(positions) // 2]))    # every domain, half the positions
        candidates.append(ids(domains, positions[:-1]))                    # every domain, one position fewer
        candidates.append(ids(domains, positions[1:]))                     # every domain, the first position out (a hole)
        candidates.append(ids(domains, positions[1::2]))                   # every domain, every other position
    if len(domains) > 1:
        candidates.append(ids(domains[-1:], positions))                    # the last domain alone (bounds keep the holes)
    for c in candidates:
        if c != all_ranks and c not in phases:
            phases.append(c)
            phases.append(all_ranks)
    return phases


class Case:
    """Deterministic data: rank `r` in phase `p` routes by `seed(p, r)` over the live experts and token `t` carries
    `payload(global idx)`, so every rank can rebuild what it should have received from whom."""

    def __init__(self, buffer: ep.ElasticBuffer, args: argparse.Namespace):
        self.buffer = buffer
        self.args = args
        self.max_so, self.max_su = buffer.max_num_scaleout_ranks, buffer.max_num_scaleup_ranks
        self.num_max_tokens_per_rank = args.num_tokens
        self.hidden, self.num_topk = args.hidden, args.num_topk
        # Fixed expert slots per rank: a world's expert count is its bounds product times the slots
        self.num_experts_per_rank = buffer.num_experts_per_rank
        live_so, live_su = buffer.get_topology_ranks()
        num_experts = live_so * live_su * self.num_experts_per_rank
        self.num_sms = buffer.get_theoretical_num_sms(num_experts, self.num_topk) if args.num_sms == 0 else args.num_sms

    def bounds(self, phase: List[int]) -> Tuple[int, int]:
        return max(r // self.max_su for r in phase) + 1, max(r % self.max_su for r in phase) + 1

    def flat_idx(self, rank_id: int, phase: List[int]) -> int:
        # The kernels encode a token's source as `scaleout * num_scaleup_bound + scaleup` (bound-relative)
        _, su_bound = self.bounds(phase)
        return (rank_id // self.max_su) * su_bound + rank_id % self.max_su

    def num_experts(self, phase: List[int]) -> int:
        so_bound, su_bound = self.bounds(phase)
        return so_bound * su_bound * self.num_experts_per_rank

    def live_expert_mask(self, phase: List[int]) -> torch.Tensor:
        epr = self.num_experts_per_rank
        live_flat = torch.tensor([self.flat_idx(r, phase) for r in phase], device='cuda')
        owner = torch.arange(self.num_experts(phase), device='cuda') // epr
        return torch.isin(owner, live_flat)

    def routing(self, phase_idx: int, rank_id: int, phase: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(topk_idx, topk_weights)` of rank `rank_id` in phase `phase_idx`: distinct live experts, positive weights"""
        num_tokens = max(1, self.num_max_tokens_per_rank - rank_id)
        g = torch.Generator(device='cuda')
        g.manual_seed(self.args.seed * 1000003 + phase_idx * 4099 + rank_id + 1)
        scores = torch.rand(num_tokens, self.num_experts(phase), generator=g, device='cuda')
        scores[:, ~self.live_expert_mask(phase)] = -1.0
        return self.select_slots(scores, phase)

    def select_slots(self, scores: torch.Tensor, phase: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k over the live slots; when the world has fewer live slots than `num_topk`, the rest is padded with -1"""
        num_selected = min(self.num_topk, len(phase) * self.num_experts_per_rank)
        weights, idx = torch.topk(scores, num_selected, dim=-1, largest=True, sorted=False)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        if num_selected < self.num_topk:
            idx = torch.nn.functional.pad(idx, (0, self.num_topk - num_selected), value=-1)
            weights = torch.nn.functional.pad(weights, (0, self.num_topk - num_selected), value=0)
        return idx.to(ep.topk_idx_t), weights.float().contiguous()

    def global_idx(self, rank_id: int, phase: List[int]) -> torch.Tensor:
        num_tokens = max(1, self.num_max_tokens_per_rank - rank_id)
        return self.flat_idx(rank_id, phase) * self.num_max_tokens_per_rank + torch.arange(num_tokens, device='cuda')

    def payload(self, global_idx: torch.Tensor) -> torch.Tensor:
        return self.pre_combine_data(global_idx.clamp_min(0), 1)[:, 0, :]

    def pre_combine_data(self, global_idx: torch.Tensor, num_topk: Optional[int] = None) -> torch.Tensor:
        # The payload's modulus is the declared maximum world size, which every process and every cohort agrees on
        return generate_pre_combine_data(global_idx, self.num_max_tokens_per_rank,
                                         self.num_topk if num_topk is None else num_topk, self.hidden,
                                         num_ranks=self.max_so * self.max_su, seed=self.args.seed)

    def expected_receive(self, phase_idx: int, phase: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """What this rank receives in the phase, sorted by source global index: `(global_idx, x, local_topk_idx, weights)`"""
        me = self.buffer.rank_id
        my_flat = self.flat_idx(me, phase)
        epr = self.num_experts_per_rank
        chunks = []
        for src in phase:
            topk_idx, topk_weights = self.routing(phase_idx, src, phase)
            owner = topk_idx // epr
            mine = (owner == my_flat).any(dim=1)
            gidx = self.global_idx(src, phase)[mine]
            local_idx = torch.where(owner[mine] == my_flat, topk_idx[mine] - my_flat * epr, torch.full_like(topk_idx[mine], -1))
            chunks.append((gidx, self.payload(gidx), local_idx, topk_weights[mine]))
        gidx = torch.cat([c[0] for c in chunks])
        order = torch.argsort(gidx)
        return (gidx[order], torch.cat([c[1] for c in chunks])[order],
                torch.cat([c[2] for c in chunks])[order], torch.cat([c[3] for c in chunks])[order])


def check_dispatch(case: Case, phase_idx: int, phase: List[int], recv_x, recv_topk_idx, recv_topk_weights, handle, tag: str):
    num_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    assert handle.recv_count_is_exact and num_recv == handle.num_recv_tokens, (num_recv, handle.num_recv_tokens)
    gidx = handle.recv_src_metadata[:num_recv, 0]
    order = torch.argsort(gidx)
    exp_gidx, exp_x, exp_idx, exp_w = case.expected_receive(phase_idx, phase)
    assert gidx.numel() == exp_gidx.numel(), f'[{tag}] received {gidx.numel()} tokens, expected {exp_gidx.numel()}'
    assert torch.equal(gidx[order], exp_gidx), f'[{tag}] received source tokens differ from the expectation'
    assert torch.equal(recv_x[:num_recv][order], exp_x), f'[{tag}] received payload mismatch'
    assert torch.equal(recv_topk_idx[:num_recv][order].to(torch.int64), exp_idx.to(torch.int64)), f'[{tag}] local expert indices mismatch'
    assert torch.equal(recv_topk_weights[:num_recv][order], exp_w), f'[{tag}] weights mismatch'
    return num_recv


def run_eager(case: Case, phase_idx: int, phase: List[int], tag: str):
    buffer, args = case.buffer, case.args
    me = buffer.rank_id
    topk_idx, topk_weights = case.routing(phase_idx, me, phase)
    x = case.payload(case.global_idx(me, phase))
    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
        x=x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_sms=case.num_sms,
        num_max_tokens_per_rank=case.num_max_tokens_per_rank, num_experts=case.num_experts(phase),
        expert_alignment=1, do_handle_copy=True, do_cpu_sync=True)
    torch.cuda.synchronize()
    num_recv = check_dispatch(case, phase_idx, phase, recv_x, recv_topk_idx, recv_topk_weights, handle, tag)

    # Combine: the pre-combine data is a function of the source token, reduced within the rank first
    local_y = case.pre_combine_data(handle.recv_src_metadata[:num_recv, 0])
    local_y[recv_topk_idx[:num_recv] == -1] = 0
    y = ordered_accumulate(local_y)
    combined_x, combined_topk_weights, _ = buffer.combine(
        x=y, topk_weights=recv_topk_weights, handle=handle, num_sms=case.num_sms)
    torch.cuda.synchronize()

    so_bound, su_bound = case.bounds(phase)
    ref_y = case.pre_combine_data(case.global_idx(me, phase))
    ref_y[topk_idx == -1] = 0   # a padded selection (fewer live slots than top-k) contributes nothing
    ref = ref_combine(ref_y, topk_idx, so_bound, su_bound, case.num_experts(phase), None,
                      *combine_recipe(case.buffer, bool(args.allow_multiple_reduction)))
    assert torch.equal(combined_x, ref), f'[{tag}] combined_x mismatch, diff={calc_diff(combined_x, ref)}'
    assert torch.equal(combined_topk_weights, topk_weights), f'[{tag}] combined weights mismatch'
    return num_recv


def run_perf(case: Case, phase_idx: int, phase: List[int], mode: str, args: argparse.Namespace) -> None:
    """Kernel timings of the LIVE world of a phase as `test_ep.py` measures them (Kineto per kernel, L2 flushed, a
    live-world barrier per iteration) plus the CUDA-event wall time of one dispatch + combine pair: is a world that GREW
    or SHRANK into this shape as fast as one that started in it? One `PERF` line per live rank"""
    buffer = case.buffer
    me = buffer.rank_id
    pc = PairCase(case, phase_idx, phase, args)
    t_d, copy_d = bench_kineto(lambda: buffer.dispatch(**pc.dispatch_args),
                               kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                               barrier_comm_profiling=True, barrier=buffer.barrier)
    t_c, red_c = bench_kineto(lambda: buffer.combine(**pc.combine_args),
                              kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),
                              barrier_comm_profiling=True, barrier=buffer.barrier)
    wall, _, _ = bench(pc.pair, num_warmups=5, num_tests=20, flush_l2=True)
    torch.cuda.synchronize()
    so_bound, su_bound = case.bounds(phase)
    print(f'PERF tag={args.perf_tag} mode={mode} phase={phase_idx} live={len(phase)} bounds={so_bound}x{su_bound} rank={me} '
          f'tokens={pc.num_tokens} dispatch_us={t_d * 1e6:.1f} copy_us={copy_d * 1e6:.1f} combine_us={t_c * 1e6:.1f} '
          f'reduce_us={red_c * 1e6:.1f} pair_wall_us={wall * 1e6:.1f}', flush=True)


class PairCase:
    """Prebuilt inputs of one dispatch + combine pair on the LIVE world of a phase (`--perf-tokens` per rank, routed over
    the live experts, BF16 or FP8); the constructor's dispatch is this rank's first kernel in the phase (`first_event`)"""

    def __init__(self, case: Case, phase_idx: int, phase: List[int], args: argparse.Namespace):
        buffer = case.buffer
        me = buffer.rank_id
        n = args.perf_tokens if args.perf_tokens > 0 else case.num_max_tokens_per_rank
        assert n <= case.num_max_tokens_per_rank, (n, case.num_max_tokens_per_rank)
        self.case, self.num_tokens = case, n
        g = torch.Generator(device='cuda')
        g.manual_seed(args.seed * 7919 + phase_idx * 104729 + me + 12345)
        scores = torch.rand(n, case.num_experts(phase), generator=g, device='cuda')
        scores[:, ~case.live_expert_mask(phase)] = -1.0
        topk_idx, topk_weights = case.select_slots(scores, phase)
        x = torch.randn(n, case.hidden, dtype=torch.bfloat16, device='cuda', generator=g)
        if args.perf_fp8:
            x = per_token_cast_to_fp8(x)
        self.dispatch_args = dict(x=x, topk_idx=topk_idx, topk_weights=topk_weights,
                                  num_sms=case.num_sms,
                                  num_max_tokens_per_rank=case.num_max_tokens_per_rank, num_experts=case.num_experts(phase),
                                  expert_alignment=1, do_handle_copy=True, do_cpu_sync=True)
        self.first_event = torch.cuda.Event(enable_timing=True)
        self.first_event.record()
        recv_x, _, recv_topk_weights, handle, _ = buffer.dispatch(**self.dispatch_args)
        num_recv_rows = (recv_x[0] if isinstance(recv_x, tuple) else recv_x).size(0)
        self.y = torch.randn(num_recv_rows, case.hidden, dtype=torch.bfloat16, device='cuda', generator=g)
        self.combine_args = dict(x=self.y, topk_weights=recv_topk_weights, handle=handle, num_sms=case.num_sms)
        buffer.combine(**self.combine_args)
        torch.cuda.synchronize()

    def pair(self):
        r = self.case.buffer.dispatch(**self.dispatch_args)
        self.case.buffer.combine(x=self.y, topk_weights=r[2], handle=r[3], num_sms=self.case.num_sms)


def run_feed(pc: PairCase, num_pairs: int) -> Tuple[List[torch.cuda.Event], List[torch.cuda.Event], List[float]]:
    """`num_pairs` back-to-back pairs with CUDA events around each: the start events, the end events, the per-pair us"""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
    for s, e in zip(starts, ends):
        s.record()
        pc.pair()
        e.record()
    torch.cuda.synchronize()
    return starts, ends, [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]


class GraphCase:
    """One graph of dispatch (no CPU sync) + combine, captured at the first phase this rank takes part in (`full_phase`)
    and replayed in every later phase against the reference of that phase (dropped selections included)."""

    def __init__(self, case: Case, full_phase: List[int], capture_phase_of: Dict[int, List[int]], barrier_fn: Callable[[], None]):
        self.case, self.captured_at = case, list(full_phase)
        # The phase every rank captured at: its graph replays the routing sampled there
        self.capture_phase_of = capture_phase_of
        # The caller's barrier over the live world (the process test's spans several cohorts)
        self.barrier = barrier_fn
        me = case.buffer.rank_id
        self.topk_idx, self.topk_weights = case.routing(10_000, me, full_phase)
        self.x = case.payload(case.global_idx(me, full_phase))
        self.handle = self.graph = self.outputs = None

    def forward(self):
        """One dispatch + combine; returns the outputs and the handle, which only `capture` keeps"""
        case = self.case
        # The expert count of the world this forward runs in (the capture world, or the live world of a warm-up)
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = case.buffer.dispatch(
            x=self.x, topk_idx=self.topk_idx, topk_weights=self.topk_weights,
            num_sms=case.num_sms,
            num_max_tokens_per_rank=case.num_max_tokens_per_rank, num_experts=case.num_experts(case.buffer.get_active_ranks()),
            expert_alignment=1, do_handle_copy=True, do_cpu_sync=False)
        src = handle.recv_src_metadata[:, 0]
        local_y = case.pre_combine_data(src.clamp_min(0))
        local_y[recv_topk_idx == -1] = 0
        y = ordered_accumulate(local_y)
        combined_x, combined_topk_weights, _ = case.buffer.combine(
            x=y, topk_weights=recv_topk_weights, handle=handle, num_sms=case.num_sms)
        return combined_x, combined_topk_weights, handle

    def capture(self, accompany: bool = False):
        # Two eager warm-ups (JIT, first-touch allocations) between barriers, then the capture; the other live ranks run
        # them in lockstep (`accompany`: warm-ups and barriers only, for a rank that has its graph), idle ranks mirror the barriers
        torch.cuda.synchronize(); self.barrier()
        for _ in range(2):
            self.forward()
        torch.cuda.synchronize(); self.barrier()
        if not accompany:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                combined_x, combined_topk_weights, self.handle = self.forward()
            self.outputs = (combined_x, combined_topk_weights)
        torch.cuda.synchronize(); self.barrier()

    def replay_and_check(self, phase: List[int], tag: str):
        case = self.case
        me = case.buffer.rank_id
        self.graph.replay()
        torch.cuda.synchronize()
        combined_x, combined_topk_weights = self.outputs

        # Selections of experts owned by ranks outside the phase are dropped by the kernels
        so_bound, su_bound = case.bounds(phase)
        epr = case.num_experts_per_rank
        live_flat = torch.tensor([case.flat_idx(r, phase) for r in phase], device='cuda')
        live_sel = (self.topk_idx >= 0) & torch.isin(self.topk_idx // epr, live_flat)
        topk_idx = torch.where(live_sel, self.topk_idx, torch.full_like(self.topk_idx, -1))
        # The source encoding of THIS phase (captured with the full world's bounds, the kernels re-derive it)
        ref_y = case.pre_combine_data(case.global_idx(me, phase))
        ref_y[topk_idx == -1] = 0
        ref = ref_combine(ref_y, topk_idx, so_bound, su_bound, case.num_experts(phase), None,
                          *combine_recipe(case.buffer, bool(case.args.allow_multiple_reduction)))
        ref_w = torch.where(live_sel, self.topk_weights, torch.zeros_like(self.topk_weights))
        assert torch.equal(combined_x, ref), f'[{tag}] graph combined_x mismatch, diff={calc_diff(combined_x, ref)}'
        assert torch.equal(combined_topk_weights, ref_w), f'[{tag}] graph combined weights mismatch'
        # Exact receive count: every live source replays the routing of ITS capture phase; a token reaches me iff one of
        # its selections is owned by me in THIS phase's layout. Zero is legitimate after a bounds change
        my_flat = case.flat_idx(me, phase)
        expected = 0
        for src in phase:
            src_idx, _ = case.routing(10_000, src, self.capture_phase_of[src])
            expected += int(((src_idx // epr) == my_flat).any(dim=1).sum().item())
        num_recv = self.handle.num_recv_tokens_device.item()
        assert num_recv == expected, \
            f'[{tag}] graph receive count {num_recv}, expected {expected}; ' \
            f'psum per position {self.handle.psum_num_recv_tokens_per_scaleup_rank.tolist()}, ' \
            f'captured at {self.captured_at}, rank id {me}'
        return num_recv

    def release(self):
        del self.outputs, self.graph
        self.outputs = self.graph = None
        torch.cuda.synchronize()


def switch_world(buffer: ep.ElasticBuffer, infos: Dict[int, Tuple[bytes, Tuple[int, int, int, int]]],
                 phase: List[int], mode: str, num_active_sms: int, barrier_fn: Callable[[], None],
                 reconnect_on_leave: bool = False, timings: Optional[Dict[str, float]] = None) -> None:
    """
    Collective: make `phase` the live world (participants and non-participants alike; `barrier_fn` spans all of them).
    By default only the masks (`scaleout_mask`, `scaleup_mask`) of the topology descriptor (`EPTopology`) change and
    every connection is kept; with `reconnect_on_leave` a leave is a symmetric `disconnect_ranks` and a rejoin a symmetric
    `connect_ranks` (one retained UCX endpoint pair per cycle). `timings`: host ms per step, peer counts, two CUDA events
    around stage + commit.
    """
    t = timings if timings is not None else {}
    t0 = time.perf_counter()
    me = buffer.rank_id
    participant = me in phase
    connected = set(buffer.get_connected_ranks())
    if reconnect_on_leave:
        needed = (set(phase) - {me}) if participant else set()
    else:
        needed = connected | (set(phase) - {me})
    joining = sorted(needed - connected)
    t['num_connect'] = len(joining)
    t1 = time.perf_counter()
    if joining:
        buffer.connect_ranks(joining, remote_mds=[infos[r][0] for r in joining], peer_infos=[infos[r][1] for r in joining])
    t2 = time.perf_counter()
    torch.cuda.synchronize(); barrier_fn()
    t3 = time.perf_counter()
    t['ev_stage'] = torch.cuda.Event(enable_timing=True); t['ev_stage'].record()
    buffer.stage_topology(mode, num_active_sms=num_active_sms, active_ranks=phase)
    t4 = time.perf_counter()
    buffer.commit_topology()
    t['ev_commit'] = torch.cuda.Event(enable_timing=True); t['ev_commit'].record()
    t5 = time.perf_counter()
    torch.cuda.synchronize(); barrier_fn()
    t6 = time.perf_counter()
    assert buffer.is_participant == participant
    assert buffer.get_active_ranks() == sorted(phase)

    # Real leave: the masks of the topology descriptor (`EPTopology`) exclude the peers, so drop them symmetrically
    leaving = sorted(connected - needed) if reconnect_on_leave else []
    t['num_disconnect'] = len(leaving)
    t7 = time.perf_counter()
    if leaving:
        buffer.disconnect_ranks(leaving)
        buffer.commit_topology()
    t8 = time.perf_counter()
    torch.cuda.synchronize(); barrier_fn()
    t9 = time.perf_counter()
    t.update(connect=(t2 - t1) * 1e3, stage=(t4 - t3) * 1e3, commit=(t5 - t4) * 1e3, disconnect=(t8 - t7) * 1e3,
             barrier=((t3 - t2) + (t6 - t5) + (t9 - t8)) * 1e3, total=(t9 - t0) * 1e3)


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks = init_test_environment(local_rank, num_local_ranks, args.seed)
    # The job's Store, served by the rank 0 process for the whole run; every Store wait is bounded by `--store-timeout-secs`
    store = dist.TCPStore(os.getenv('MASTER_ADDR', '127.0.0.1'), int(os.getenv('MASTER_PORT', '8361')),
                          world_size=num_ranks, is_master=rank_idx == 0,
                          timeout=timedelta(seconds=args.store_timeout_secs))
    # The test's own cohort over the job: barriers, the peers' transport metadata, the validated NVLink-domain layout
    cohort = NixlBootstrap(store, rank_idx, num_ranks, namespace='adaptive-elastic-test',
                           timeout_seconds=args.store_timeout_secs)

    def barrier():
        # A Store barrier orders nothing on the device: finish this rank's queued CUDA work first
        torch.cuda.synchronize()
        cohort.barrier('test')

    def cohort_print(message: str):
        if local_rank == 0:
            print(message, flush=True)
        barrier()

    # Where the job sits in the declared maximum world; the constructor validates these coordinates against the layout
    group_so, group_su = cohort.num_scaleout_ranks, cohort.num_scaleup_ranks
    # `--num-experts` is the expert slot count of the declared maximum world; the slots per rank are fixed from it
    max_num_ranks = group_so * (args.max_scaleup_ranks or group_su)
    assert args.num_experts % max_num_ranks == 0, f'{args.num_experts} experts do not divide by the {max_num_ranks} ranks of the declared world'
    buffer = make_buffer(args, rank_idx, num_ranks, store, bootstrap_id='adaptive-elastic-buffer',
                         max_num_scaleup_ranks=args.max_scaleup_ranks, num_experts_per_rank=args.num_experts // max_num_ranks,
                         scaleout_rank_idx=rank_idx // group_su, scaleup_rank_idx=args.position_offset + rank_idx % group_su)
    max_so, max_su = buffer.max_num_scaleout_ranks, buffer.max_num_scaleup_ranks
    mode, num_active_sms = buffer.get_topology()
    present = buffer.get_active_ranks()   # the whole job, wherever it sits in the declared maximum world
    plan = json.loads(args.plan) if args.plan else default_plan(max_su, present)
    modes = args.modes.split(',') if args.modes else ['hybrid'] + (['direct'] if args.also_direct and max_so == 1 else [])
    assert all(m in ('hybrid', 'direct') for m in modes) and ('direct' not in modes or max_so == 1), f'Bad modes {modes} for {max_so} domain(s)'
    cohort_print(f'Elastic test: world {max_so}x{max_su}, rank id {buffer.rank_id} = ({buffer.scaleout_rank_idx}, {buffer.scaleup_rank_idx}), '
                 f'initial mode {mode}, active SMs {num_active_sms}, launch grid {buffer.num_launch_sms}, '
                 f'connected {buffer.get_connected_ranks()}, live {buffer.get_active_ranks()}\n'
                 f'Plan: {plan}\nModes: {modes}')
    # Every rank captures at the FIRST phase it takes part in; the graph must then survive growth and shrinkage
    full = sorted(present)
    for phase in plan:
        assert all(0 <= r < max_so * max_su for r in phase) and len(set(phase)) == len(phase)
        assert set(phase) <= set(present), f'Phase {phase} names ranks that are not in the job {present}'
        buffer.masks_of(phase)   # product-set check, raises otherwise

    # Every rank's transport metadata, kept to reconnect returning peers (a real controller would use a store)
    me = buffer.rank_id
    infos = cohort.all_gather_peer_infos((me, buffer.get_local_metadata(), buffer.get_local_peer_info()))
    case = Case(buffer, args)
    # Phases whose live world gets the kernel timings (`run_perf`), e.g. the phase a world grew or shrank into
    perf_phases = set(int(p) for p in args.perf_phases.split(',') if p.strip() != '')
    assert all(0 <= p < len(plan) for p in perf_phases), f'--perf-phases {sorted(perf_phases)} outside the plan of {len(plan)} phases'

    # The phase each rank captures its graph at: the first phase of the plan it takes part in (reversed: the first wins)
    capture_phase_of = {r: list(phase) for phase in reversed(plan) for r in phase}

    feed = args.feed_pairs   # > 0: time every switch under a synthetic feed of dispatch + combine pairs
    for mode in modes:
        cohort_print(f'==== mode {mode} ====')
        graph, captured = None, set()
        pre = None    # (PairCase, end events, times) of the pairs this rank ran at the end of the previous phase
        for phase_idx, phase in enumerate(plan):
            tim: Dict[str, float] = {}
            switch_world(buffer, infos, phase, mode, num_active_sms, barrier, args.reconnect_on_leave, timings=tim)
            so_bound, su_bound = case.bounds(phase)
            tag = f'{mode} phase {phase_idx} {phase} (bounds {so_bound}x{su_bound})'
            pc = None
            if feed > 0 and phase_idx > 0:
                # The switch as an engine sees it: the last pair of the old world, the switch, the first pairs of the new
                prev = plan[phase_idx - 1]
                role = ('stay' if me in prev else 'join') if me in phase else ('leave' if me in prev else 'idle')
                gpu_gap = first = after = float('nan')
                first_list = ''
                if buffer.is_participant:
                    pc = PairCase(case, phase_idx, phase, args)
                    _, _, times = run_feed(pc, feed)
                    first_list = ','.join(f'{v:.0f}' for v in times[:3])
                    after = median(times[3:]) if len(times) > 3 else median(times)
                    if pre is not None:
                        gpu_gap = pre[1][-1].elapsed_time(pc.first_event)   # ms, same stream and device
                before = median(pre[2]) if pre is not None else float('nan')
                dev_ms = tim['ev_stage'].elapsed_time(tim['ev_commit'])
                print(f'TRANS tag={args.perf_tag} mode={mode} from={phase_idx - 1} to={phase_idx} rank={me} role={role} '
                      f'live_before={len(prev)} live_after={len(phase)} num_connect={tim["num_connect"]} num_disconnect={tim["num_disconnect"]} '
                      f'connect_ms={tim["connect"]:.2f} stage_ms={tim["stage"]:.3f} commit_ms={tim["commit"]:.3f} '
                      f'disconnect_ms={tim["disconnect"]:.2f} barrier_ms={tim["barrier"]:.2f} switch_ms={tim["total"]:.2f} '
                      f'commit_dev_ms={dev_ms:.3f} gpu_gap_ms={gpu_gap:.2f} before_us={before:.0f} first_us={first_list} after_us={after:.0f}',
                      flush=True)

            # First-time participants capture now; the other live ranks run the warm-ups with them, idle ranks mirror the barriers
            new_capturers = sorted(set(phase) - captured)
            if me in new_capturers:
                graph = GraphCase(case, phase, capture_phase_of, barrier)
                graph.capture()
            elif new_capturers and buffer.is_participant:
                graph.capture(accompany=True)
            elif new_capturers:
                for _ in range(3):
                    barrier()
            captured |= set(new_capturers)

            if buffer.is_participant:
                num_recv = run_eager(case, phase_idx, phase, tag)
                num_recv_graph = graph.replay_and_check(phase, tag)
                # One line per phase from the lowest live rank (local rank 0 may sit the phase out)
                if me == min(phase):
                    print(f' > [{tag}] eager: {num_recv} tokens received, bitwise match; '
                          f'graph replay (captured at phase {graph.captured_at}): {num_recv_graph} tokens received, bitwise match',
                          flush=True)
                if phase_idx in perf_phases:
                    run_perf(case, phase_idx, phase, mode, args)
                barrier()
            else:
                # Mirror the participants' collectives
                barrier()
            barrier()
            # The feed's last pairs of this phase, for the next switch (participants only)
            pre = None
            if feed > 0 and buffer.is_participant and phase_idx + 1 < len(plan):
                if pc is None:
                    pc = PairCase(case, phase_idx, phase, args)
                starts, ends, times = run_feed(pc, feed)
                pre = (pc, ends, times)
            barrier()
        if graph is not None:
            graph.release()

    # Back to the full world before tearing down
    switch_world(buffer, infos, full, modes[0], num_active_sms, barrier, args.reconnect_on_leave)
    cohort_print('Adaptive elastic test passed')
    buffer.destroy()
    # Rank 0 owns the TCPStore server; keep it alive until all peers finish teardown.
    barrier()


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--num-processes', type=int, default=4, help='processes to spawn (one per GPU)')
    parser.add_argument('--num-sms', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true', help='deterministic dispatch output (the handle sort runs inside the captured graphs too)')
    parser.add_argument('--num-nixl-channels', type=int, default=0)
    parser.add_argument('--num-gpu-timeout-secs', type=int, default=100)
    parser.add_argument('--store-timeout-secs', type=int, default=1800, help='bound of every Store wait (client connection, test barriers, constructor bootstrap)')
    parser.add_argument('--num-tokens', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=192,
                        help='expert slots of the declared maximum world (the fixed slots per rank are this divided by its rank count)')
    parser.add_argument('--allow-multiple-reduction', type=int, default=1)
    parser.add_argument('--prefer-overlap-with-compute', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--position-offset', type=int, default=0, help='the scale-up position of the first process in the declared maximum world')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive EP kernels: elastic membership on the NIXL transport')
    add_common_arguments(parser)
    parser.add_argument('--plan', type=str, default='', help='JSON list of phases (lists of dense rank ids); default: a built-in plan')
    parser.add_argument('--perf-phases', type=str, default='', help='comma-separated phases whose live world gets kernel timings (PERF lines)')
    parser.add_argument('--perf-tokens', type=int, default=0, help='tokens per live rank in the timed phases (default: --num-tokens)')
    parser.add_argument('--perf-fp8', type=int, default=0, choices=(0, 1), help='FP8 dispatch payload in the timed phases (default BF16)')
    parser.add_argument('--perf-tag', type=str, default='run', help='label of the PERF and TRANS lines')
    parser.add_argument('--feed-pairs', type=int, default=0,
                        help='> 0: time every world switch under this many dispatch + combine pairs before and after it (TRANS lines)')
    parser.add_argument('--also-direct', action='store_true', help='also run the plan in direct mode (one NVLink domain only)')
    parser.add_argument('--modes', type=str, default='', help='comma-separated modes to run (hybrid, direct); default: hybrid, plus direct with --also-direct')
    parser.add_argument('--reconnect-on-leave', action='store_true', help='leave = symmetric disconnect_ranks, rejoin = connect_ranks (default: masks only)')
    parser.add_argument('--num-cpu-timeout-secs', type=int, default=100)
    parser.add_argument('--max-scaleup-ranks', type=int, default=0, help='declared maximum positions per domain (0 = the job\'s), e.g. 72 for an NVL72 domain')
    args = parser.parse_args()
    assert args.position_offset == 0 or args.max_scaleup_ranks > 0, '--position-offset needs --max-scaleup-ranks'
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
