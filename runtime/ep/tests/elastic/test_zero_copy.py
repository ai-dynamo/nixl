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
Zero-copy dispatch (adaptive kernels) on the NIXL transport, checked against the staged dispatch of the same inputs on
the same buffer: the compact rows agree bitwise after a sort by source token (`check_compact`), the combine through the
zero-copy handle agrees bitwise with the CPU reference (`check_combine`). Zero copy is a DIRECT-mode feature (one NVLink
domain: the kernel writes the slot in place); `dispatch(zero_copy=True)` is refused while the topology is hybrid (T13 checks
the refusal) and hybrid zero copy is not tested.
Cases (`--cases`): bf16 / fp8 / nvfp4 (an arena laid out for that wire) = T1 fresh call, T2 combine, T3 slot reuse, T4 two slots, T5 alternating staged and
zero-copy calls, T6 batch growth, T7 a CUDA graph (no CPU sync) replayed, T8 two graphs on two slots, T11 the active
SM count per call; negatives = T13, the host refusals; all = bf16 + fp8 + nvfp4 + negatives (default).
--worlds 1 adds T10 (world changes with the reset, one domain of >= 4 ranks); --tag-wrap 1 adds T9 (511 consecutive
fresh calls across two tag periods); --max-num-scaleup-ranks 8 declares a domain larger than the group.
Usage: python tests/elastic/test_zero_copy.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 8 --num-experts 256
"""
import argparse
import torch
import torch.distributed as dist

from nixl.runtime import ep
from nixl.runtime.ep.utils.envs import init_dist, dist_print
from nixl.runtime.ep.utils.gate import get_unbalanced_scores
from nixl.runtime.ep.utils.math import per_token_cast_to_fp8, calc_diff
from nixl.runtime.ep.utils.refs import combine as ref_combine
from nixl.runtime.ep.utils.refs import generate_pre_combine_data, ordered_accumulate

TAG_PERIOD = 254   # `WorkspaceLayout::zc_call_tag`: ((index >> 1) % 127) + 1, pinned by the static_asserts in layout.cuh


def make_buffer(group, args, use_fp8: bool, one_domain: bool = True, use_nvfp4: bool = False, **overrides):
    """A buffer with two zero-copy slots, in direct mode on one NVLink domain (`one_domain`); `overrides` replace constructor arguments"""
    num_ranks = dist.get_world_size()
    assert args.num_experts % num_ranks == 0, f'{args.num_experts} experts do not divide by {num_ranks} ranks'
    kwargs = dict(num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                  num_topk=args.num_topk, use_fp8_dispatch=use_fp8, zc_use_fp8_dispatch=use_fp8, zc_use_nvfp4_dispatch=use_nvfp4,
                  num_experts_per_rank=args.num_experts // num_ranks,
                  allow_multiple_reduction=True,
                  prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
                  explicitly_destroy=True,
                  num_gpu_timeout_secs=args.num_gpu_timeout_secs,
                  num_cpu_timeout_secs=args.num_cpu_timeout_secs,
                  max_num_scaleup_ranks=args.max_num_scaleup_ranks,
                  num_zero_copy_slots=2)
    kwargs.update(overrides)
    buffer = ep.ElasticBuffer(group, **kwargs)
    if one_domain:
        num_scaleout_ranks, _ = buffer.get_topology_ranks()
        assert num_scaleout_ranks == 1, 'The zero-copy test needs one NVLink domain (direct mode)'
        buffer.set_topology_forced('direct', buffer.get_topology()[1])   # one domain is direct by derivation; explicit against NIXL_EP_ADAPTIVE_MODE
    return buffer


def make_inputs(args, use_fp8: bool, seed_offset: int = 0, num_tokens: int = 0, num_experts: int = 0, num_ranks: int = 0, use_nvfp4: bool = False):
    """Per-rank inputs, `num_tokens - rank` tokens (at least one) unless given, over the group's experts unless given; a unique seed per rank and call"""
    num_tokens = num_tokens if num_tokens > 0 else max(1, args.num_tokens - dist.get_rank())
    torch.manual_seed(args.seed + seed_offset + 1000 * dist.get_rank())
    scores = get_unbalanced_scores(num_tokens, num_experts or args.num_experts, num_ranks or dist.get_world_size(), args.num_topk, 1.0, False)
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(ep.topk_idx_t)
    if use_nvfp4:
        # Opaque NVFP4 bytes (two e2m1 codes per byte, one e4m3 scale per 16 elements): the dispatch moves them as bytes
        x = (torch.randint(0, 256, (num_tokens, args.hidden // 2), dtype=torch.uint8, device='cuda'),
             torch.randint(0, 256, (num_tokens, args.hidden // 16), dtype=torch.uint8, device='cuda').view(torch.float8_e4m3fn))
    else:
        x = torch.randn((num_tokens, args.hidden), dtype=torch.bfloat16, device='cuda')
        x = per_token_cast_to_fp8(x) if use_fp8 else x
    return x, topk_idx, topk_weights


def split_x(recv_x):
    return recv_x if isinstance(recv_x, tuple) else (recv_x, None)


def u8(t):
    """Float8 payloads and scales as bytes before any index or compare (random e4m3 bytes include NaN patterns; bitwise is the test)"""
    return t.view(torch.uint8) if t is not None and t.dtype == torch.float8_e4m3fn else t


def sort_by_source(handle, num_recv_tokens: int):
    """The compact rows in source-token order, the canonical order of both paths"""
    src = handle.recv_src_metadata[:num_recv_tokens, 0]
    assert src.unique().numel() == num_recv_tokens, 'source token indices must be unique'
    return torch.sort(src).indices, torch.sort(src).values


def check_compact(tag, staged, zc, use_fp8: bool, quiet: bool = False):
    """Both dispatches compact: the sorted rows of every output bitwise equal (`quiet`: no group-wide print)"""
    (s_x, s_idx, s_w, s_handle), (z_x, z_idx, z_w, z_handle) = staged, zc
    n = s_handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    assert n == z_handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    assert torch.equal(s_handle.psum_num_recv_tokens_per_scaleup_rank, z_handle.psum_num_recv_tokens_per_scaleup_rank)
    assert torch.equal(s_handle.psum_num_recv_tokens_per_expert, z_handle.psum_num_recv_tokens_per_expert), \
        f'[{tag}] per-expert prefix mismatch: {s_handle.psum_num_recv_tokens_per_expert} vs {z_handle.psum_num_recv_tokens_per_expert}'
    s_perm, s_src = sort_by_source(s_handle, n)
    z_perm, z_src = sort_by_source(z_handle, n)
    assert torch.equal(s_src, z_src), f'[{tag}] the sets of received tokens differ'
    s_data, s_sf = map(u8, split_x(s_x))
    z_data, z_sf = map(u8, split_x(z_x))
    assert torch.equal(s_data[:n][s_perm], z_data[:n][z_perm]), f'[{tag}] recv_x mismatch'
    if s_sf is not None:   # the FP8 and NVFP4 wires carry scaling factors
        assert torch.equal(s_sf[:n][s_perm], z_sf[:n][z_perm]), f'[{tag}] recv_sf mismatch'
    assert torch.equal(s_idx[:n][s_perm], z_idx[:n][z_perm]), f'[{tag}] recv_topk_idx mismatch'
    assert torch.equal(s_w[:n][s_perm], z_w[:n][z_perm]), f'[{tag}] recv_topk_weights mismatch'
    # The slot-relative field of the metadata: the same master lane per token (the slot itself may differ)
    assert torch.equal(s_handle.recv_src_metadata[:n, 1][s_perm] % s_handle.topk_idx.shape[1],
                       z_handle.recv_src_metadata[:n, 1][z_perm] % z_handle.topk_idx.shape[1]), f'[{tag}] master lane mismatch'
    if not quiet:
        dist_print(f' > [{tag}] compact outputs match ({n} received tokens)', once_in_node=True)
    return n


def reference_combine(args, topk_idx, hybrid: bool = False):
    """The CPU combine reference of this rank's tokens on one domain (the mode's recipe: `hybrid` reduces within the domain first)"""
    num_tokens = topk_idx.shape[0]
    ref_y = generate_pre_combine_data(dist.get_rank() * args.num_tokens + torch.arange(num_tokens, device='cuda'),
                                      args.num_tokens, args.num_topk, args.hidden)
    ref_y[topk_idx == -1] = 0
    return ref_combine(ref_y, topk_idx, 1, dist.get_world_size(), args.num_experts, None, True, hybrid)


def check_combine(tag, buffer, handle, recv_topk_idx, recv_topk_weights, topk_idx, args, num_sms):
    """Combine through the handle against the CPU reference, bitwise"""
    num_max_tokens_per_rank, hidden, num_topk = args.num_tokens, args.hidden, args.num_topk
    n = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    src = handle.recv_src_metadata[:n, 0]
    local_y = generate_pre_combine_data(src, num_max_tokens_per_rank, num_topk, hidden)
    local_y[recv_topk_idx[:n] == -1] = 0
    y = ordered_accumulate(local_y)
    combined_x, combined_w, _ = buffer.combine(x=y, handle=handle, topk_weights=recv_topk_weights[:n], num_sms=num_sms)
    ref = reference_combine(args, topk_idx, buffer.combine_is_hybrid)
    assert torch.equal(combined_x, ref), f'[{tag}] combine mismatch, diff={calc_diff(combined_x, ref)}'
    dist_print(f' > [{tag}] combine through the handle matches the reference', once_in_node=True)


def combine_from_handle(buffer, args, num_sms, handle, recv_topk_idx, recv_topk_weights):
    """The combine input from the handle's metadata over all its rows (the worst-case count without a CPU sync), then the combine"""
    src = handle.recv_src_metadata[:, 0]
    local_y = generate_pre_combine_data(src.clamp_min(0), args.num_tokens, args.num_topk, args.hidden)
    local_y[recv_topk_idx == -1] = 0
    y = ordered_accumulate(local_y)
    return buffer.combine(x=y, topk_weights=recv_topk_weights, handle=handle, num_sms=num_sms)[:2]


def make_graph_forward(buffer, args, num_sms, inputs, slot: int, state: dict):
    """[zero-copy dispatch into `slot` without CPU sync + the combine from its handle] as one callable; `state` gets the handle"""
    x, topk_idx, topk_weights = inputs
    graph_args = dict(topk_idx=topk_idx, topk_weights=topk_weights, num_sms=num_sms,
                      num_max_tokens_per_rank=args.num_tokens, do_cpu_sync=False, do_handle_copy=True)

    def forward():
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
            x=x, zero_copy=True, zero_copy_slot=slot, **graph_args)
        state.update(handle=handle, recv_x=recv_x, recv_topk_idx=recv_topk_idx, recv_topk_weights=recv_topk_weights)
        return combine_from_handle(buffer, args, num_sms, handle, recv_topk_idx, recv_topk_weights)
    return forward


def capture(forward, num_warmups: int = 2):
    """Eager warm-ups, then one graph of `forward`"""
    torch.cuda.synchronize(); dist.barrier()
    for _ in range(num_warmups):
        forward()
    torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = forward()
    torch.cuda.synchronize(); dist.barrier()
    return graph, outputs


def run_zero_copy(group, args, use_fp8: bool, use_nvfp4: bool = False):
    """T1 to T8 and T11 on one buffer with two slots"""
    wire = 'nvfp4' if use_nvfp4 else ('fp8' if use_fp8 else 'bf16')
    dist_print(f'==== {wire}/zero-copy ====', once_in_node=True)
    buffer = make_buffer(group, args, use_fp8, use_nvfp4=use_nvfp4)
    assert buffer.has_zero_copy and buffer.num_zero_copy_slots == 2, 'the arena was not configured'
    num_sms = buffer.get_topology()[1] if args.num_sms == 0 else args.num_sms

    def common_of(topk_idx, topk_weights, sms=num_sms):
        return dict(topk_idx=topk_idx, topk_weights=topk_weights, num_sms=sms,
                    num_max_tokens_per_rank=args.num_tokens, do_cpu_sync=True)

    def staged_of(inputs, sms=num_sms):
        x, topk_idx, topk_weights = inputs
        return buffer.dispatch(x=x, **common_of(topk_idx, topk_weights, sms))[:4]

    def zero_copy_of(inputs, slot: int, sms=num_sms):
        x, topk_idx, topk_weights = inputs
        return buffer.dispatch(x=x, zero_copy=True, zero_copy_slot=slot, **common_of(topk_idx, topk_weights, sms))[:4]

    def rows_of(outputs):
        recv_x, recv_topk_idx, recv_topk_weights, handle = outputs
        n = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        data, sf = map(u8, split_x(recv_x))
        rows = dict(recv_x=data[:n].clone(), recv_topk_idx=recv_topk_idx[:n].clone(),
                    recv_topk_weights=recv_topk_weights[:n].clone(), recv_src_metadata=handle.recv_src_metadata[:n, :2].clone())
        if sf is not None:
            rows['recv_sf'] = sf[:n].clone()
        return rows

    def assert_rows_intact(tag, outputs, before):
        now = rows_of(outputs)
        for name, a in before.items():
            assert torch.equal(a, now[name]), f'[{tag}] {name} of the other slot changed'

    # T1 fresh zero copy into slot 0; T2 the combine through its handle
    inputs_a = make_inputs(args, use_fp8, use_nvfp4=use_nvfp4)
    staged_a = staged_of(inputs_a)
    zc_a = zero_copy_of(inputs_a, 0)
    torch.cuda.synchronize()
    check_compact(f'{wire}/T1 fresh slot 0', staged_a, zc_a, use_fp8)
    check_combine(f'{wire}/T2 combine via the zero-copy handle', buffer, zc_a[3], zc_a[1], zc_a[2], inputs_a[1], args, num_sms)

    # T3 slot reuse after the combine; the outputs are views of the slot (the same storage as T1's)
    inputs_3 = make_inputs(args, use_fp8, use_nvfp4=use_nvfp4, seed_offset=3)
    staged_3 = staged_of(inputs_3)
    zc_3 = zero_copy_of(inputs_3, 0)
    torch.cuda.synchronize()
    check_compact(f'{wire}/T3 slot 0 reused', staged_3, zc_3, use_fp8)
    check_combine(f'{wire}/T3 combine after the reuse', buffer, zc_3[3], zc_3[1], zc_3[2], inputs_3[1], args, num_sms)
    for name, a, b in (('recv_x', split_x(zc_a[0])[0], split_x(zc_3[0])[0]), ('recv_sf', split_x(zc_a[0])[1], split_x(zc_3[0])[1]),
                       ('recv_topk_idx', zc_a[1], zc_3[1]), ('recv_topk_weights', zc_a[2], zc_3[2]),
                       ('recv_src_metadata', zc_a[3].recv_src_metadata, zc_3[3].recv_src_metadata)):
        assert a is None or a.data_ptr() == b.data_ptr(), f'[{wire}/T3] {name} is not a view of slot 0 ({a.data_ptr():#x} vs {b.data_ptr():#x})'
    dist_print(f' > [{wire}/T3] the outputs of both dispatches into slot 0 are views of the slot (same storage)', once_in_node=True)

    # T4 two slots: a second batch into slot 1 leaves slot 0 intact
    before_slot0 = rows_of(zc_3)
    inputs_b = make_inputs(args, use_fp8, use_nvfp4=use_nvfp4, seed_offset=7)
    staged_b = staged_of(inputs_b)
    zc_b = zero_copy_of(inputs_b, 1)
    torch.cuda.synchronize()
    assert_rows_intact(f'{wire}/T4 slot 0 after a slot-1 dispatch', zc_3, before_slot0)
    check_compact(f'{wire}/T4 slot 1', staged_b, zc_b, use_fp8)
    dist_print(f' > [{wire}/T4] the slot-1 dispatch left slot 0 intact', once_in_node=True)

    # T5 eight alternating staged / zero-copy calls, the slots in turn
    for i in range(4):
        inputs_i = make_inputs(args, use_fp8, use_nvfp4=use_nvfp4, seed_offset=10 + i)
        staged_i = staged_of(inputs_i)
        zc_i = zero_copy_of(inputs_i, i % 2)
        torch.cuda.synchronize()
        check_compact(f'{wire}/T5 alternating call {i} slot {i % 2}', staged_i, zc_i, use_fp8)

    # T6 batch growth: a 128-token call (the last rank one token: most data warps never wait for the bases), then the full batch
    last_rank = dist.get_rank() == dist.get_world_size() - 1
    inputs_small = make_inputs(args, use_fp8, use_nvfp4=use_nvfp4, seed_offset=20, num_tokens=1 if last_rank else min(128, args.num_tokens))
    staged_small = staged_of(inputs_small)
    zc_small = zero_copy_of(inputs_small, 0)
    torch.cuda.synchronize()
    check_compact(f'{wire}/T6 small batch ({inputs_small[1].shape[0]} tokens here)', staged_small, zc_small, use_fp8)
    inputs_full = make_inputs(args, use_fp8, use_nvfp4=use_nvfp4, seed_offset=21)
    staged_full = staged_of(inputs_full)
    zc_full = zero_copy_of(inputs_full, 0)
    torch.cuda.synchronize()
    check_compact(f'{wire}/T6 full batch after the small one', staged_full, zc_full, use_fp8)

    # T7 a graph of [zero-copy dispatch into slot 0 without CPU sync + combine], replayed three times
    def check_graph_combine(tag, inputs, combined_x, combined_w):
        ref = reference_combine(args, inputs[1])
        assert torch.equal(combined_x, ref), f'[{tag}] combine mismatch, diff={calc_diff(combined_x, ref)}'
        assert torch.equal(combined_w, inputs[2]), f'[{tag}] weights mismatch'

    state_a = {}
    graph_a, (combined_a, weights_a) = capture(make_graph_forward(buffer, args, num_sms, inputs_a, 0, state_a))
    for _ in range(3):
        graph_a.replay()
    torch.cuda.synchronize()
    check_graph_combine(f'{wire}/T7 graph replay', inputs_a, combined_a, weights_a)
    assert not state_a['handle'].recv_count_is_exact
    dist_print(f' > [{wire}/T7] graph-captured zero-copy dispatch (no CPU sync) + combine replays correctly', once_in_node=True)

    # T8 two graphs on the two slots replayed alternately, then an eager combine through each graph's handle
    state_b = {}
    graph_b, (combined_b, weights_b) = capture(make_graph_forward(buffer, args, num_sms, inputs_b, 1, state_b))
    for _ in range(2):
        graph_a.replay()
        graph_b.replay()
    torch.cuda.synchronize()
    check_graph_combine(f'{wire}/T8 graph A (slot 0) after alternate replays', inputs_a, combined_a, weights_a)
    check_graph_combine(f'{wire}/T8 graph B (slot 1) after alternate replays', inputs_b, combined_b, weights_b)
    for tag, inputs, state in ((f'{wire}/T8 eager combine via graph A handle', inputs_a, state_a),
                               (f'{wire}/T8 eager combine via graph B handle', inputs_b, state_b)):
        eager_x, eager_w = combine_from_handle(buffer, args, num_sms, state['handle'], state['recv_topk_idx'], state['recv_topk_weights'])
        torch.cuda.synchronize()
        check_graph_combine(tag, inputs, eager_x, eager_w)
    dist_print(f' > [{wire}/T8] two graphs on two slots replay alternately and combine eagerly through their handles', once_in_node=True)
    del graph_a, graph_b
    torch.cuda.synchronize()

    # T11 the active SM count per call (14, 26, 14); each call checked and combined at its count before the next change
    for i, sms in enumerate((14, 26, 14)):
        sms = min(sms, buffer.num_launch_sms)
        inputs_i = make_inputs(args, use_fp8, use_nvfp4=use_nvfp4, seed_offset=30 + i)
        staged_i = staged_of(inputs_i, sms)
        zc_i = zero_copy_of(inputs_i, i % 2, sms)
        torch.cuda.synchronize()
        assert buffer.get_topology() == ('direct', sms), buffer.get_topology()
        check_compact(f'{wire}/T11 num_sms={sms} slot {i % 2}', staged_i, zc_i, use_fp8)
        check_combine(f'{wire}/T11 combine at num_sms={sms}', buffer, zc_i[3], zc_i[1], zc_i[2], inputs_i[1], args, sms)

    dist.barrier()
    buffer.destroy()
    dist.barrier()


def run_worlds(group, args):
    """T10: world changes on one domain of four or more ranks; every rank commits every world, idle, under a CPU barrier"""
    tag = 'bf16/T10'
    dist_print(f'==== {tag} worlds ====', once_in_node=True)
    num_ranks = dist.get_world_size()
    assert num_ranks >= 4, f'[{tag}] the world test needs at least four ranks on one domain (got {num_ranks})'
    buffer = make_buffer(group, args, use_fp8=False)
    assert buffer.has_zero_copy and buffer.num_zero_copy_slots == 2, 'the arena was not configured'
    num_sms = buffer.get_topology()[1] if args.num_sms == 0 else args.num_sms
    me = buffer.rank_id                     # one domain at domain 0: the dense rank id is the position
    full = buffer.get_active_ranks()
    assert full == list(range(num_ranks)), full
    epr = buffer.num_experts_per_rank

    def live_reference(inputs, live):
        # Selections of a dead owner are dropped; the rest reduce in the direct recipe over the live bound
        _, topk_idx, topk_weights = inputs
        num_tokens = topk_idx.shape[0]
        live_positions = torch.tensor(live, device=topk_idx.device, dtype=topk_idx.dtype)
        live_sel = (topk_idx >= 0) & torch.isin(topk_idx // epr, live_positions)
        live_topk_idx = torch.where(live_sel, topk_idx, torch.full_like(topk_idx, -1))
        bound = max(live) + 1
        ref_y = generate_pre_combine_data(me * args.num_tokens + torch.arange(num_tokens, device='cuda'),
                                          args.num_tokens, args.num_topk, args.hidden)
        ref_y[~live_sel] = 0
        ref = ref_combine(ref_y, live_topk_idx, 1, bound, bound * epr, None, True, False)
        return ref, torch.where(live_sel, topk_weights, torch.zeros_like(topk_weights))

    def check_combined(tag_w, inputs, combined_x, combined_w, live):
        torch.cuda.synchronize()
        ref, ref_w = live_reference(inputs, live)
        assert torch.equal(combined_x, ref), f'[{tag_w}] combine mismatch, diff={calc_diff(combined_x, ref)}'
        assert torch.equal(combined_w, ref_w), f'[{tag_w}] weights mismatch'

    # Exactly one fresh zero-copy call (the eager run) before the first world change, then the capture (runs no kernel)
    inputs_g = make_inputs(args, use_fp8=False, seed_offset=40)
    state = {}
    forward = make_graph_forward(buffer, args, num_sms, inputs_g, 0, state)
    torch.cuda.synchronize(); dist.barrier()
    combined_x, combined_w = forward()
    check_combined(f'{tag} eager run of the graph forward, full world', inputs_g, combined_x, combined_w, full)
    committed_sms = buffer.get_topology()[1]
    torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        combined_x, combined_w = forward()
    torch.cuda.synchronize(); dist.barrier()
    assert not state['handle'].recv_count_is_exact

    def commit_world(live):
        torch.cuda.synchronize(); dist.barrier()
        buffer.set_active_ranks(live)
        torch.cuda.synchronize(); dist.barrier()
        assert buffer.get_active_ranks() == live and buffer.get_topology() == ('direct', committed_sms), \
            (buffer.get_active_ranks(), buffer.get_topology())
        assert buffer.is_participant == (me in live)

    def eager_checked(tag_w, live, seed_offset):
        # An eager zero-copy call (slot 1) against the staged reference in the same world, then its combine
        inputs = make_inputs(args, use_fp8=False, seed_offset=seed_offset)
        x, topk_idx, topk_weights = inputs
        common = dict(topk_idx=topk_idx, topk_weights=topk_weights, num_sms=num_sms,
                      num_max_tokens_per_rank=args.num_tokens, do_cpu_sync=True)
        staged = buffer.dispatch(x=x, **common)[:4]
        zc = buffer.dispatch(x=x, zero_copy=True, zero_copy_slot=1, **common)[:4]
        torch.cuda.synchronize()
        n = check_compact(f'{tag_w} eager zero copy vs staged', staged, zc, use_fp8=False, quiet=True)
        eager_x, eager_w = combine_from_handle(buffer, args, num_sms, zc[3], zc[1], zc[2])
        check_combined(f'{tag_w} eager combine', inputs, eager_x, eager_w, live)
        return n

    # (live positions, an eager checked call first?, graph replays): a hole at 1, the last dropped, 0 dropped, full again
    hole, no_last, no_first = [p for p in full if p != 1], full[:-1], full[1:]
    plan = ((hole, True, 1), (no_last, False, 2), (no_first, False, 3), (full, True, 1))
    for world_idx, (live, eager, num_replays) in enumerate(plan):
        commit_world(live)
        bound = max(live) + 1
        if me in live:
            n_eager = eager_checked(f'{tag} world {live}', live, 41 + world_idx) if eager else -1
            for r in range(num_replays):
                graph.replay()
                check_combined(f'{tag} world {live} replay {r}', inputs_g, combined_x, combined_w, live)
                num_recv_tokens = state['handle'].num_recv_tokens_device.item()
                assert num_recv_tokens == state['handle'].psum_num_recv_tokens_per_scaleup_rank[-1].item()
                assert 0 <= num_recv_tokens <= bound * args.num_tokens
            if me == live[0]:
                print(f' > [{tag}] world {live}: ' + (f'eager zero copy vs staged ({n_eager} rows here) + combine, ' if eager else '')
                      + f'{num_replays} replay(s) of the full-world graph, all bitwise', flush=True)
        else:
            assert not buffer.is_participant
        dist.barrier()
    dist_print(f' > [{tag}] worlds {hole} -> {no_last} -> {no_first} -> {full} passed (one eager call before the first change)',
               once_in_node=True)

    del graph, combined_x, combined_w
    torch.cuda.synchronize()
    dist.barrier()
    buffer.destroy()
    dist.barrier()


def run_tag_wrap(group, args):
    """T9: 511 consecutive fresh zero-copy calls on a fresh buffer, across two periods of the call tag"""
    tag = 'bf16/T9'
    dist_print(f'==== {tag} tag wrap ====', once_in_node=True)
    buffer = make_buffer(group, args, use_fp8=False)    # fresh: call i has index i
    assert buffer.has_zero_copy and buffer.num_zero_copy_slots == 2, 'the arena was not configured'
    num_sms = buffer.get_topology()[1] if args.num_sms == 0 else args.num_sms
    num_calls = 2 * TAG_PERIOD + 3
    named = {0, 1, 2, TAG_PERIOD - 1, TAG_PERIOD, TAG_PERIOD + 1, 2 * TAG_PERIOD, 2 * TAG_PERIOD + 1, 2 * TAG_PERIOD + 2}
    checked = named | set(range(0, num_calls, 16))
    fill = make_inputs(args, use_fp8=False, seed_offset=50)

    def common_of(topk_idx, topk_weights, do_cpu_sync):
        return dict(topk_idx=topk_idx, topk_weights=topk_weights, num_sms=num_sms,
                    num_max_tokens_per_rank=args.num_tokens, do_cpu_sync=do_cpu_sync)

    num_checked = 0
    for i in range(num_calls):
        slot = i % 2
        if i in checked:
            inputs = make_inputs(args, use_fp8=False, seed_offset=60 + i)
            x, topk_idx, topk_weights = inputs
            staged = buffer.dispatch(x=x, **common_of(topk_idx, topk_weights, True))[:4]
            zc = buffer.dispatch(x=x, zero_copy=True, zero_copy_slot=slot, **common_of(topk_idx, topk_weights, True))[:4]
            torch.cuda.synchronize()
            check_compact(f'{tag} call {i} (tag {(i >> 1) % (TAG_PERIOD // 2) + 1}, parity {i & 1}) slot {slot}',
                          staged, zc, use_fp8=False, quiet=i not in named)
            num_checked += 1
        else:
            # Unchecked: no CPU sync, the launches queue back to back
            x, topk_idx, topk_weights = fill
            buffer.dispatch(x=x, zero_copy=True, zero_copy_slot=slot, **common_of(topk_idx, topk_weights, False))
    torch.cuda.synchronize()
    dist_print(f' > [{tag}] {num_calls} consecutive fresh zero-copy calls (two tag periods of {TAG_PERIOD} + 3), '
               f'{num_checked} of them checked against the staged reference, passed', once_in_node=True)

    dist.barrier()
    buffer.destroy()
    dist.barrier()


def run_negatives(group, args):
    """T13: the host refusals, try/except on every rank; a refusal for another reason than the intended assert fails"""
    dist_print('==== negatives ====', once_in_node=True)

    def refuse(name, call, expected: str):
        try:
            call()
        except (AssertionError, RuntimeError) as e:
            assert expected in str(e), f'[negatives] {name}: refused for another reason than "{expected}": {e}'
            dist_print(f' > [negatives] {name}: refused ({type(e).__name__})', once_in_node=True)
        else:
            raise RuntimeError(f'[negatives] {name} was accepted')

    refuse('deterministic=True with slots', lambda: make_buffer(group, args, use_fp8=False, deterministic=True),
           'not available with deterministic mode')
    x, topk_idx, topk_weights = make_inputs(args, use_fp8=False, seed_offset=90)
    fp8_x = per_token_cast_to_fp8(x)
    plain = make_buffer(group, args, use_fp8=False, num_zero_copy_slots=0)    # one buffer alive at a time
    assert not plain.has_zero_copy and plain.num_zero_copy_slots == 0
    num_sms = plain.get_topology()[1] if args.num_sms == 0 else args.num_sms
    common = dict(topk_idx=topk_idx, topk_weights=topk_weights, num_sms=num_sms,
                  num_max_tokens_per_rank=args.num_tokens, do_cpu_sync=True)
    refuse('zero_copy=True without an arena', lambda: plain.dispatch(x=x, zero_copy=True, **common), 'needs an arena')
    dist.barrier(); plain.destroy(); dist.barrier()
    buffer = make_buffer(group, args, use_fp8=False)
    assert buffer.has_zero_copy and buffer.num_zero_copy_slots == 2
    # Zero copy is a direct-mode feature: refused while the topology is hybrid (forced on one domain, test-only), then back to direct
    buffer.set_topology_forced('hybrid', num_sms); torch.cuda.synchronize(); dist.barrier()
    refuse('zero copy while the topology is hybrid', lambda: buffer.dispatch(x=x, zero_copy=True, **common), 'direct-mode feature')
    buffer.set_topology_forced('direct', num_sms); torch.cuda.synchronize(); dist.barrier()
    refuse('zero_copy_slot=2 on a two-slot arena', lambda: buffer.dispatch(x=x, zero_copy=True, zero_copy_slot=2, **common), 'is outside the arena')
    refuse('the FP8 wire on a BF16 arena', lambda: buffer.dispatch(x=fp8_x, zero_copy=True, **common), 'does not match the zero-copy arena')
    refuse('use_tma_aligned_col_major_sf under zero copy', lambda: buffer.dispatch(x=x, zero_copy=True, use_tma_aligned_col_major_sf=True, **common),
           'row-major scaling factors')
    refuse('num_max_tokens_per_rank above the arena\'s', lambda: buffer.dispatch(x=x, zero_copy=True, **{**common, 'num_max_tokens_per_rank': args.num_tokens + 1}),
           'exceeds the zero-copy arena')
    # An NVFP4 payload on a BF16 arena: another wire (the arena is laid out for one; `--cases nvfp4` runs an NVFP4 arena)
    fp4_x = (torch.randint(0, 256, (x.shape[0], args.hidden // 2), dtype=torch.uint8, device='cuda'),
             torch.randint(0, 256, (x.shape[0], args.hidden // 16), dtype=torch.uint8, device='cuda'))
    refuse('an NVFP4 payload on a BF16 arena', lambda: buffer.dispatch(x=fp4_x, zero_copy=True, **common), 'does not match the zero-copy arena')
    # The refusals left nothing behind: one zero-copy call against its staged twin
    staged = buffer.dispatch(x=x, **common)[:4]
    zc = buffer.dispatch(x=x, zero_copy=True, **common)[:4]
    torch.cuda.synchronize()
    check_compact('negatives/zero copy after the refusals', staged, zc, use_fp8=False)
    dist.barrier(); buffer.destroy(); dist.barrier()
    # An NVFP4 arena: the BF16 and FP8 payloads are the other wires; both flags at once are refused at construction
    refuse('zc_use_fp8_dispatch with zc_use_nvfp4_dispatch', lambda: make_buffer(group, args, use_fp8=True, use_nvfp4=True), 'exclude each other')
    buffer = make_buffer(group, args, use_fp8=False, use_nvfp4=True)
    refuse('a BF16 payload on an NVFP4 arena', lambda: buffer.dispatch(x=x, zero_copy=True, **common), 'does not match the zero-copy arena')
    refuse('an FP8 payload on an NVFP4 arena', lambda: buffer.dispatch(x=fp8_x, zero_copy=True, **common), 'does not match the zero-copy arena')
    fx, fidx, fw = make_inputs(args, use_fp8=False, use_nvfp4=True, seed_offset=91)
    fcommon = dict(topk_idx=fidx, topk_weights=fw, num_sms=num_sms, num_max_tokens_per_rank=args.num_tokens, do_cpu_sync=True)
    staged = buffer.dispatch(x=fx, **fcommon)[:4]
    zc = buffer.dispatch(x=fx, zero_copy=True, **fcommon)[:4]
    torch.cuda.synchronize()
    check_compact('negatives/NVFP4 zero copy on its arena after the refusals', staged, zc, use_fp8=False)
    dist.barrier(); buffer.destroy(); dist.barrier()


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    if args.cases in ('bf16', 'all'):
        run_zero_copy(group, args, use_fp8=False)
    if args.cases in ('fp8', 'all'):
        run_zero_copy(group, args, use_fp8=True)
    if args.cases in ('nvfp4', 'all'):
        run_zero_copy(group, args, use_fp8=False, use_nvfp4=True)
    if args.cases in ('negatives', 'all'):
        run_negatives(group, args)
    if args.worlds:
        run_worlds(group, args)
    if args.tag_wrap:
        run_tag_wrap(group, args)
    extras = (', worlds' if args.worlds else '') + (', tag wrap' if args.tag_wrap else '')
    dist_print(f'Zero-copy test passed ({args.cases}{extras})', once_in_node=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive EP kernels on NIXL: zero-copy dispatch')
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-sms', type=int, default=0, help='active blocks of the fixed grid (0: the committed count)')
    parser.add_argument('--num-gpu-timeout-secs', type=int, default=100)
    parser.add_argument('--num-cpu-timeout-secs', type=int, default=100)
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256, help="expert slots of the cohort's world (the fixed slots per rank are this divided by the cohort size)")
    parser.add_argument('--prefer-overlap-with-compute', type=int, default=1,
                        help='1 (default): 7 channels per SM at hidden 7168, 18-warp blocks (the perf shape); 0: 4 channels, 12 warps')
    parser.add_argument('--max-num-scaleup-ranks', type=int, default=0, help='declared maximum domain size (0: the group)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cases', choices=('bf16', 'fp8', 'nvfp4', 'negatives', 'all'), default='all',
                        help='all = bf16 + fp8 + nvfp4 + negatives')
    parser.add_argument('--worlds', type=int, default=0, choices=(0, 1), help='1: T10, world changes with zero copy (one domain of 4 or more ranks, BF16)')
    parser.add_argument('--tag-wrap', type=int, default=0, choices=(0, 1), help='1: T9, 511 consecutive fresh zero-copy calls (BF16)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
