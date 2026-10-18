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
NVFP4 on the dispatch wire (adaptive kernels on NIXL): caller-quantised rows, e2m1 codes `[tokens, hidden // 2]` uint8 and
e4m3 block scales `[tokens, hidden // 16]`, moved as bytes in the record's hidden and scale-factor slots; the combine stays BF16.
Every check is bitwise. The reference is `nixl.runtime.ep.utils.refs.dispatch` called once per payload (the codes, the scales as
uint8) with the same routing (`ref_dispatch_nvfp4`), the received rows sorted by source token as in test_ep.py; the scales are
compared as uint8 views throughout.
Cases (`--cases main`, in the mode the buffer starts in: direct on one domain, `NIXL_EP_ADAPTIVE_MODE=hybrid` for the forwarder
relay on one domain, hybrid on two domains = T4):
  T1 compact dispatch with the CPU sync: codes, scales, top-k indices and weights bitwise
  T2 expanded dispatch (zero padding, alignment 128): the padding rows of codes, scales and weights zero, the fold of both tensors
  T9 the BF16 combine through the compact (T1) and the expanded (T2) NVFP4 handles against the CPU reference (the handle's
     `channel_linked_list` and `token_metadata_at_forward` have no other reader); the same check runs through the T3 handle
     (compact only), the T6 handles after the deterministic sort and the N6 handles at 8 channels per SM
  T3 no CPU sync (worst-case rows, sliced to the count); the scales passed as uint8; the combine through its handle
  T5 a CUDA graph of the dispatch (no CPU sync) replayed twice, T3's checks after each replay
  T7 quantised data: BF16 rows generated from the global token index, quantised by the test's reference quantiser (g = 1); the
     received bytes equal the quantisation of the regenerated source rows, and dequantise within the NVFP4 error of the source
  T8 the BF16 and the FP8 wires on the same buffer, bitwise against `ref_dispatch`
  T6 a `deterministic=True` buffer: the received rows in source order without a sort, the combine through both handles after the
     sort (its `channel_linked_list` rewrite is the one post-dispatch mutation of combine input), a second dispatch bitwise, compact
     and expanded
`--cases negatives`: N1 uint8 `x` without scales; N2 scales of width hidden / 32, and FP8-shaped int32 scales with a uint8
payload; N3 hidden 7200; N4 `use_tma_aligned_col_major_sf=True`; N5 2-byte scales; N6 a BF16 dispatch after an NVFP4 one on a
`prefer_overlap_with_compute=True` buffer at hidden 7168 (8 vs 7 channels per SM; the NVFP4 dispatch at 8 channels is checked,
compact and expanded, and the BF16 combine runs through both 8-channel handles: the only 8-channel combine of the suite).
`--bench --wire bf16|fp8|nvfp4`: that wire's compact dispatch checked, then timed as test_ep.py does (bench_kineto, 30
iterations, a barrier before each; one `* EP: r/N | dispatch: ... us | copy: ... us` line per rank).
Usage (one node, 4 GPUs):
    python tests/elastic/test_nvfp4_dispatch.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 8 --num-experts 256
"""
import argparse
import torch
import torch.distributed as dist
from typing import Optional, Tuple

from nixl.runtime import ep
from nixl.runtime.ep.utils.envs import init_dist, dist_print
from nixl.runtime.ep.utils.gate import get_unbalanced_scores
from nixl.runtime.ep.utils.math import align, calc_diff, count_bytes, per_token_cast_to_fp8, safe_div
from nixl.runtime.ep.utils.refs import dispatch as ref_dispatch
from nixl.runtime.ep.utils.refs import combine as ref_combine
from nixl.runtime.ep.utils.refs import generate_pre_combine_data, ordered_accumulate
from nixl.runtime.ep.utils.testing import bench_kineto

from test_ep import fold_expanded


# Test-local reference NVFP4 quantiser (not package code): `value ~= e2m1 * e4m3_block_scale * global_scale`, one e4m3 scale per
# 16 elements, two codes per byte with the low nibble the even element, ties to even, saturating at 6
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_to_e2m1_magnitude(q_abs: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=q_abs.device)
    hi = torch.bucketize(q_abs, values).clamp(max=7)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = q_abs - values[lo], values[hi] - q_abs
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return torch.where(pick_hi, hi, lo).to(torch.uint8)


def per_block_cast_to_nvfp4(x: torch.Tensor, global_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """`[m, n]` -> (`[m, n // 2]` uint8 packed e2m1, `[m, n // 16]` uint8 e4m3 block scales = `amax / 6` rounded to e4m3)"""
    assert x.dim() == 2 and x.shape[1] % 16 == 0
    m, n = x.shape
    blocks = x.float().view(m, n // 16, 16) * (1.0 / global_scale)
    amax = blocks.abs().amax(dim=2, keepdim=True)
    scale = (amax / 6.0).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    q = blocks * (1.0 / torch.where(scale_f == 0, torch.ones_like(scale_f), scale_f))
    mag_idx = _round_to_e2m1_magnitude(q.abs())
    nibbles = (mag_idx | ((q < 0).to(torch.uint8) << 3)).view(m, n)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()
    return packed, scale.view(torch.uint8).view(m, n // 16).contiguous()


def per_block_cast_back_nvfp4(x_fp4: torch.Tensor, x_sf: torch.Tensor, global_scale: float = 1.0) -> torch.Tensor:
    """Exact dequantisation of `per_block_cast_to_nvfp4`'s output to float32, `[m, n]`"""
    m, half_n = x_fp4.shape
    lut = torch.tensor(_E2M1_VALUES + tuple(-v for v in _E2M1_VALUES), dtype=torch.float32, device=x_fp4.device)
    nibbles = torch.stack([x_fp4 & 0xF, x_fp4 >> 4], dim=-1).view(m, half_n * 2)
    values = lut[nibbles.long()]
    scales = x_sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return values * scales * global_scale


def make_buffer(group, args, **overrides):
    """A buffer at the model hidden (no NVFP4 construction flag), `overrides` replace constructor arguments"""
    num_ranks = dist.get_world_size()
    kwargs = dict(num_experts_per_rank=args.num_experts // num_ranks,
                  num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                  allow_multiple_reduction=True,
                  prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
                  explicitly_destroy=True,
                  num_gpu_timeout_secs=args.num_gpu_timeout_secs,
                  num_cpu_timeout_secs=args.num_cpu_timeout_secs)
    kwargs.update(overrides)
    return ep.ElasticBuffer(group, **kwargs)


class Setup:
    """The shape, the routing (test_ep's unbalanced scores) and the common dispatch arguments of one buffer"""
    def __init__(self, buffer, args):
        self.buffer, self.args = buffer, args
        self.num_scaleout_ranks, self.num_scaleup_ranks = buffer.get_topology_ranks()
        self.num_experts = self.num_scaleout_ranks * self.num_scaleup_ranks * buffer.num_experts_per_rank
        self.num_local_experts = buffer.num_experts_per_rank
        self.num_tokens = max(1, args.num_tokens - dist.get_rank())
        self.num_sms = buffer.get_theoretical_num_sms(args.num_topk) if args.num_sms == 0 else args.num_sms
        scores = get_unbalanced_scores(self.num_tokens, self.num_experts, dist.get_world_size(), args.num_topk, 1.0, False)
        self.topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
        self.topk_idx = topk_idx.to(ep.topk_idx_t)
        self.common = dict(topk_idx=self.topk_idx, topk_weights=self.topk_weights, num_sms=self.num_sms,
                           num_max_tokens_per_rank=args.num_tokens)
        dist_print(f'Config: ranks {self.num_scaleout_ranks} x {self.num_scaleup_ranks}, mode {buffer.get_topology()[0]}, '
                   f'experts {args.num_topk}/{self.num_experts}, tokens {self.num_tokens} (max {args.num_tokens}), '
                   f'hidden {args.hidden}, #SM {self.num_sms}', once_in_node=True)

    def nvfp4_inputs(self, sf_dtype=torch.float8_e4m3fn):
        """Random codes and scales (any byte is a legal code pair or scale byte for a byte transport)"""
        x_fp4 = torch.randint(0, 256, (self.num_tokens, self.args.hidden // 2), dtype=torch.uint8, device='cuda')
        x_sf = torch.randint(0, 256, (self.num_tokens, self.args.hidden // 16), dtype=torch.uint8, device='cuda').view(sf_dtype)
        return x_fp4, x_sf

    def ref_dispatch_nvfp4(self, x_fp4, x_sf):
        """`ref_dispatch` once per payload with the same routing; both are sorted by source token, so they line up row for row"""
        ref_fp4, ref_idx, ref_w, ref_src, ref_counts = ref_dispatch(x_fp4, self.topk_idx, self.topk_weights, self.args.num_tokens, self.num_experts)
        ref_sf, _, _, ref_src_sf, _ = ref_dispatch(x_sf.view(torch.uint8), self.topk_idx, self.topk_weights, self.args.num_tokens, self.num_experts)
        assert torch.equal(ref_src, ref_src_sf)
        return (ref_fp4, ref_sf), ref_idx, ref_w, ref_src, ref_counts

    def ref_dispatch(self, x):
        return ref_dispatch(x, self.topk_idx, self.topk_weights, self.args.num_tokens, self.num_experts)


def u8(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Scales and FP8 payloads as bytes before any index or compare"""
    return t.view(torch.uint8) if t is not None and t.dtype == torch.float8_e4m3fn else t


def split_x(recv_x):
    return recv_x if isinstance(recv_x, tuple) else (recv_x, None)


def num_received(handle) -> int:
    return handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()


def check_rows(tag, ref, recv_x, recv_topk_idx, recv_topk_weights, src, n: int):
    """The received rows (the first `n`) sorted by source token against the reference, every tensor bitwise; the weights
    masked at the reference's -1 slots as test_ep.py does"""
    ref_data, ref_sf = split_x(ref[0])
    ref_idx, ref_w, ref_src = ref[1], ref[2], ref[3]
    assert src.shape[0] == n
    order = torch.sort(src)
    assert torch.equal(order.values, ref_src), f'[{tag}] the source tokens differ from the reference'
    perm = order.indices
    data, sf = split_x(recv_x)
    assert torch.equal(u8(data[:n])[perm], u8(ref_data)), f'[{tag}] data rows differ'
    if ref_sf is not None:
        assert torch.equal(u8(sf[:n])[perm], u8(ref_sf)), f'[{tag}] scale rows differ'
    if recv_topk_idx is not None:
        assert torch.equal(recv_topk_idx[:n][perm], ref_idx), f'[{tag}] top-k indices differ'
    mask = ref_idx < 0
    assert torch.equal(recv_topk_weights[:n][perm].masked_fill(mask, 0), ref_w.masked_fill(mask, 0)), f'[{tag}] top-k weights differ'
    dist_print(f' > [{tag}] {n} received rows match the reference', once_in_node=True)


def fold(tag, expanded_x, expanded_w, expanded_handle, n: int, expert_alignment: int, num_local_experts: int):
    """T2: the expanded output's padding rows zero, then one row per received token (`fold_expanded`); returns the compact tensors"""
    indices = expanded_handle.recv_src_metadata[:n, 2:]
    mask = indices >= 0
    safe = indices.clone()
    safe[~mask] = 0
    data, sf = split_x(expanded_x)
    for expert_idx in range(num_local_experts):
        start = expanded_handle.psum_num_recv_tokens_per_expert[expert_idx].item()
        end = align(start, expert_alignment)
        assert (u8(data[start:end]) == 0).all() and (expanded_w[start:end] == 0).all(), f'[{tag}] padding of expert {expert_idx} not zero'
        assert sf is None or (u8(sf[start:end]) == 0).all(), f'[{tag}] scale padding of expert {expert_idx} not zero'
    folded = fold_expanded(u8(data), safe, mask)
    folded = (folded, fold_expanded(u8(sf), safe, mask)) if sf is not None else folded
    return folded, expanded_w[safe]


def check_combine(tag, s: Setup, compact, expanded=None, buffer=None):
    """T9: the BF16 combine through the NVFP4 handles (compact and, when given, expanded) against the CPU reference, bitwise,
    built as test_ep.py builds its combine inputs (the GEMM replaced by `generate_pre_combine_data` at the model hidden). The
    compact handle may be a no-CPU-sync one (worst-case rows: only the first `n` are filled, as in test_ep.py); `buffer` names
    the owner of the handles when it is not `s.buffer` (T6's deterministic buffer)"""
    args = s.args
    buffer = s.buffer if buffer is None else buffer
    num_max_tokens_per_rank, hidden, num_topk = args.num_tokens, args.hidden, args.num_topk
    recv_x, recv_topk_idx, recv_topk_weights, handle = compact
    n = num_received(handle)
    src = handle.recv_src_metadata[:n, 0]
    local_y = generate_pre_combine_data(src, num_max_tokens_per_rank, num_topk, hidden)
    local_y[recv_topk_idx[:n] == -1] = 0
    x = torch.empty((split_x(recv_x)[0].shape[0], hidden), dtype=torch.bfloat16, device='cuda')
    x[:n] = ordered_accumulate(local_y)
    recipe = (True, True) if buffer.combine_is_hybrid else (True, False)
    ref_y = generate_pre_combine_data(dist.get_rank() * num_max_tokens_per_rank + torch.arange(s.num_tokens, device='cuda'),
                                      num_max_tokens_per_rank, num_topk, hidden)
    ref_y[s.topk_idx == -1] = 0
    ref = ref_combine(ref_y, s.topk_idx, s.num_scaleout_ranks, s.num_scaleup_ranks, s.num_experts, None, *recipe)
    combined_x, combined_w, _ = buffer.combine(x=x, topk_weights=recv_topk_weights, handle=handle, num_sms=s.num_sms)
    assert torch.equal(combined_x, ref), f'[{tag}] combined_x differs (diff {calc_diff(combined_x, ref)})'
    assert torch.equal(combined_w, s.topk_weights), f'[{tag}] combined weights differ'
    if expanded is None:
        dist_print(f' > [{tag}] the combine through the compact NVFP4 handle matches the reference', once_in_node=True)
        return
    e_recv_x, e_recv_topk_weights, e_handle = expanded
    assert n == num_received(e_handle)
    e_src = e_handle.recv_src_metadata[:n, 0]
    e_y = generate_pre_combine_data(e_src, num_max_tokens_per_rank, num_topk, hidden)
    # The extra last row takes the -1 slots
    e_x = torch.empty((split_x(e_recv_x)[0].shape[0] + 1, hidden), dtype=torch.bfloat16, device='cuda')
    e_x[e_handle.recv_src_metadata[:n, 2:].flatten()] = e_y.view(-1, hidden)
    e_x = e_x[:-1]
    reduced_x, reduced_w, _ = buffer.combine(x=e_x, topk_weights=e_recv_topk_weights, handle=e_handle, num_sms=s.num_sms)
    assert torch.equal(reduced_x, ref), f'[{tag}] reduced combined_x differs (diff {calc_diff(reduced_x, ref)})'
    assert torch.equal(reduced_w, s.topk_weights), f'[{tag}] reduced combined weights differ'
    dist_print(f' > [{tag}] the combine through the compact and the expanded NVFP4 handles matches the reference', once_in_node=True)


def run_main(group, args):
    buffer = make_buffer(group, args)
    s = Setup(buffer, args)
    hidden = args.hidden
    x_fp4, x_sf = s.nvfp4_inputs()
    ref = s.ref_dispatch_nvfp4(x_fp4, x_sf)

    # T1 compact
    recv_x, recv_idx, recv_w, handle, _ = buffer.dispatch(x=(x_fp4, x_sf), **s.common)
    n = num_received(handle)
    assert recv_x[0].dtype == torch.uint8 and recv_x[1].dtype == torch.float8_e4m3fn
    assert recv_x[0].shape == (n, hidden // 2) and recv_x[1].shape == (n, hidden // 16) and recv_x[1].is_contiguous()
    check_rows('T1 compact', ref, recv_x, recv_idx, recv_w, handle.recv_src_metadata[:n, 0], n)

    # T2 expanded with zero padding
    e_recv_x, e_recv_idx, e_recv_w, e_handle, _ = buffer.dispatch(x=(x_fp4, x_sf), do_expand=True, do_zero_padding=True,
                                                                  expert_alignment=128, **s.common)
    assert e_recv_idx is None and num_received(e_handle) == n
    folded_x, folded_w = fold('T2', e_recv_x, e_recv_w, e_handle, n, 128, s.num_local_experts)
    check_rows('T2 expanded', ref, folded_x, None, folded_w, e_handle.recv_src_metadata[:n, 0], n)

    # T9 the combine on both handles
    check_combine('T9 combine', s, (recv_x, recv_idx, recv_w, handle), (e_recv_x, e_recv_w, e_handle))

    # T3 no CPU sync, the scales as uint8; the combine through the worst-case-sized handle (compact only)
    recv = buffer.dispatch(x=(x_fp4, x_sf.view(torch.uint8)), do_cpu_sync=False, **s.common)
    assert num_received(recv[3]) == n and recv[0][0].shape[0] >= n
    check_rows('T3 no CPU sync', ref, recv[0], recv[1], recv[2], recv[3].recv_src_metadata[:n, 0], n)
    check_combine('T3 combine', s, (recv[0], recv[1], recv[2], recv[3]))

    # T5 a CUDA graph of the dispatch, replayed twice
    def forward():
        return buffer.dispatch(x=(x_fp4, x_sf), do_cpu_sync=False, **s.common)
    torch.cuda.synchronize(); dist.barrier()
    for _ in range(2):
        forward()
    torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_x, g_idx, g_w, g_handle, _ = forward()
    torch.cuda.synchronize(); dist.barrier()
    for replay in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert num_received(g_handle) == n
        check_rows(f'T5 graph replay {replay + 1}', ref, g_x, g_idx, g_w, g_handle.recv_src_metadata[:n, 0], n)
        dist.barrier()

    # T7 quantised data: rows regenerated from the global token index on the receiver, quantised the same way
    def source_rows(src_token_global_idx):
        return generate_pre_combine_data(src_token_global_idx, args.num_tokens, 1, hidden)[:, 0]
    x_bf16 = source_rows(dist.get_rank() * args.num_tokens + torch.arange(s.num_tokens, device='cuda'))
    q_fp4, q_sf = per_block_cast_to_nvfp4(x_bf16)
    recv_x, recv_idx, recv_w, handle, _ = buffer.dispatch(x=(q_fp4, q_sf.view(torch.float8_e4m3fn)), **s.common)
    n7 = num_received(handle)
    src = handle.recv_src_metadata[:n7, 0]
    want_fp4, want_sf = per_block_cast_to_nvfp4(source_rows(src))
    assert torch.equal(recv_x[0], want_fp4) and torch.equal(u8(recv_x[1]), want_sf), '[T7] received bytes differ from the quantised source rows'
    got = per_block_cast_back_nvfp4(recv_x[0], u8(recv_x[1]))
    err = (got - source_rows(src).float()).abs().max().item()
    assert err <= 0.25, f'[T7] dequantised rows off by {err}'   # |x| <= 1: scale <= e4m3(1 / 6), half an e2m1 step at most 1 x scale
    dist_print(f' > [T7 quantised] {n7} rows dequantise to the source within {err:.4f} (diff {calc_diff(got, source_rows(src).float()):.2e})', once_in_node=True)

    # T8 the BF16 and the FP8 wires through the same buffer
    x_bf16 = torch.randn((s.num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    for wire, x in (('bf16', x_bf16), ('fp8', per_token_cast_to_fp8(x_bf16))):
        recv = buffer.dispatch(x=x, **s.common)
        check_rows(f'T8 {wire}', s.ref_dispatch(x), recv[0], recv[1], recv[2], recv[3].recv_src_metadata[:, 0], num_received(recv[3]))
    dist.barrier(); buffer.destroy(); dist.barrier()

    # T6 deterministic: the rows arrive in source order, the combine runs through both sorted handles (the sort rewrites
    # `channel_linked_list` in place), a second dispatch is bitwise equal (compact and expanded)
    det = make_buffer(group, args, deterministic=True)
    recv_x, recv_idx, recv_w, handle, _ = det.dispatch(x=(x_fp4, x_sf), **s.common)
    n6 = num_received(handle)
    src = handle.recv_src_metadata[:n6, 0]
    assert torch.equal(src, ref[3]), '[T6] not in source order'
    check_rows('T6 deterministic', ref, recv_x, recv_idx, recv_w, src, n6)
    e_recv_x, _, e_recv_w, e_handle, _ = det.dispatch(x=(x_fp4, x_sf), do_expand=True, do_zero_padding=True, expert_alignment=128, **s.common)
    folded_x, folded_w = fold('T6 expanded', e_recv_x, e_recv_w, e_handle, n6, 128, s.num_local_experts)
    check_rows('T6 deterministic expanded', ref, folded_x, None, folded_w, e_handle.recv_src_metadata[:n6, 0], n6)
    check_combine('T6 combine', s, (recv_x, recv_idx, recv_w, handle), (e_recv_x, e_recv_w, e_handle), buffer=det)
    twice = det.dispatch(x=(x_fp4, x_sf), **s.common)
    assert torch.equal(recv_x[0], twice[0][0]) and torch.equal(u8(recv_x[1]), u8(twice[0][1])), '[T6] second dispatch differs'
    assert torch.equal(recv_idx, twice[1]) and torch.equal(recv_w, twice[2]), '[T6] second dispatch differs'
    e_twice = det.dispatch(x=(x_fp4, x_sf), do_expand=True, do_zero_padding=True, expert_alignment=128, **s.common)
    valid = e_handle.recv_src_metadata[:n6, 2:]
    valid = valid[valid >= 0]
    assert torch.equal(e_recv_x[0][valid], e_twice[0][0][valid]) and torch.equal(u8(e_recv_x[1])[valid], u8(e_twice[0][1])[valid]), '[T6] second expanded dispatch differs'
    dist_print(' > [T6 deterministic] a second dispatch is bitwise equal, compact and expanded', once_in_node=True)
    dist.barrier(); det.destroy(); dist.barrier()


def run_negatives(group, args):
    """The host and Python refusals, try/except on every rank; a refusal for another reason than the intended assert fails"""
    dist_print('==== negatives ====', once_in_node=True)

    def refuse(name, call, expected: str):
        try:
            call()
        except (AssertionError, RuntimeError) as e:
            assert expected in str(e), f'[negatives] {name}: refused for another reason than "{expected}": {e}'
            dist_print(f' > [negatives] {name}: refused ({type(e).__name__})', once_in_node=True)
        else:
            raise RuntimeError(f'[negatives] {name} was accepted')

    buffer = make_buffer(group, args)
    s = Setup(buffer, args)
    num_tokens, hidden = s.num_tokens, args.hidden
    x_fp4, x_sf = s.nvfp4_inputs()
    zeros = lambda cols, dtype: torch.zeros((num_tokens, cols), dtype=dtype, device='cuda')
    refuse('N1 uint8 x without scales', lambda: buffer.dispatch(x=x_fp4, **s.common), 'NVFP4 dispatch takes')
    refuse('N2 scales of width hidden / 32', lambda: buffer.dispatch(x=(x_fp4, zeros(hidden // 32, torch.uint8)), **s.common), 'NVFP4 dispatch takes')
    refuse('N2 FP8-shaped int32 scales with a uint8 payload', lambda: buffer.dispatch(x=(x_fp4, zeros(hidden // 128, torch.int32)), **s.common), 'NVFP4 dispatch takes')
    refuse('N3 hidden 7200', lambda: buffer.dispatch(x=(zeros(3600, torch.uint8), zeros(450, torch.float8_e4m3fn)), **s.common), 'NVFP4 dispatch takes')
    refuse('N4 column-major scales', lambda: buffer.dispatch(x=(x_fp4, x_sf), use_tma_aligned_col_major_sf=True, **s.common), 'NVFP4 scales are returned row-major only')
    refuse('N5 2-byte scales', lambda: buffer.dispatch(x=(x_fp4, zeros(hidden // 16, torch.int16)), **s.common), 'NVFP4 dispatch takes')
    dist.barrier(); buffer.destroy(); dist.barrier()

    # N6: the channel count is fixed by the first dispatch's wire; at hidden 7168 under `prefer_overlap_with_compute` NVFP4 takes
    # 8 channels per SM and BF16 7 (the NVFP4 dispatch at 8 channels, compact and expanded, and the BF16 combine through both
    # 8-channel handles are the positive half of the case: the suite's only 8-channel combine)
    if hidden != 7168:
        dist_print(f' > [negatives] N6 skipped at hidden {hidden} (the 8 vs 7 channel split is the 7168 shape)', once_in_node=True)
        return
    overlap = make_buffer(group, args, prefer_overlap_with_compute=True)
    s = Setup(overlap, args)
    x_fp4, x_sf = s.nvfp4_inputs()
    recv = overlap.dispatch(x=(x_fp4, x_sf), **s.common)
    ref6, n = s.ref_dispatch_nvfp4(x_fp4, x_sf), num_received(recv[3])
    check_rows('N6 NVFP4 at prefer_overlap_with_compute=True', ref6, recv[0], recv[1], recv[2], recv[3].recv_src_metadata[:n, 0], n)
    e_recv = overlap.dispatch(x=(x_fp4, x_sf), do_expand=True, do_zero_padding=True, expert_alignment=128, **s.common)
    assert e_recv[1] is None and num_received(e_recv[3]) == n
    folded_x, folded_w = fold('N6 expanded', e_recv[0], e_recv[2], e_recv[3], n, 128, s.num_local_experts)
    check_rows('N6 expanded at 8 channels per SM', ref6, folded_x, None, folded_w, e_recv[3].recv_src_metadata[:n, 0], n)
    check_combine('N6 combine at 8 channels per SM', s, (recv[0], recv[1], recv[2], recv[3]), (e_recv[0], e_recv[2], e_recv[3]))
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    refuse('N6 BF16 after NVFP4 on an 8-channel buffer', lambda: overlap.dispatch(x=x_bf16, **s.common), 'fix the channel count')
    dist.barrier(); overlap.destroy(); dist.barrier()


def run_bench(group, args):
    """One wire's compact dispatch checked, then timed as test_ep.py:225-249 does"""
    buffer = make_buffer(group, args)
    s = Setup(buffer, args)
    if args.wire == 'nvfp4':
        x = s.nvfp4_inputs()
        ref = s.ref_dispatch_nvfp4(*x)
    else:
        x = torch.randn((s.num_tokens, args.hidden), dtype=torch.bfloat16, device='cuda')
        x = per_token_cast_to_fp8(x) if args.wire == 'fp8' else x
        ref = s.ref_dispatch(x)
    dispatch_args = dict(x=x, **s.common)
    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(**dispatch_args)
    n = num_received(handle)
    check_rows(f'bench {args.wire}', ref, recv_x, recv_topk_idx, recv_topk_weights, handle.recv_src_metadata[:n, 0], n)

    dst_scaleout_rank_idx = s.topk_idx // (s.num_experts // s.num_scaleout_ranks)
    num_scaleout_send_tokens = sum((dst_scaleout_rank_idx == i).any(dim=1).sum().item()
                                   for i in range(s.num_scaleout_ranks if s.num_scaleout_ranks > 1 else 0))
    num_bytes_per_token = safe_div(count_bytes(recv_x, recv_topk_idx, recv_topk_weights), recv_topk_idx.size(0))
    num_scaleup_bytes = num_bytes_per_token * n
    num_scaleout_bytes = num_bytes_per_token * num_scaleout_send_tokens
    dist_print(f' > bench wire={args.wire} tokens={s.num_tokens} bytes per received token={num_bytes_per_token:.0f} '
               f'(per-wire bytes: quote the microseconds across wires)', once_in_node=True)
    t, copy_t = bench_kineto(lambda: buffer.dispatch(**dispatch_args),
                             kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                             barrier_comm_profiling=True, barrier=buffer.barrier)
    dist_print(f'   * EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
               f'dispatch: '
               f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
               f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
               f'copy: {2 * n * num_bytes_per_token / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')
    dist.barrier(); buffer.destroy(); dist.barrier()


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    assert args.num_experts % num_ranks == 0, f'{args.num_experts} experts do not divide by {num_ranks} ranks'
    assert args.hidden % 64 == 0 or args.bench or args.cases == 'negatives'
    if args.bench:
        run_bench(group, args)
    else:
        if args.cases in ('main', 'all'):
            run_main(group, args)
        if args.cases in ('negatives', 'all'):
            run_negatives(group, args)
    dist_print(f'NVFP4 dispatch test passed ({"bench " + args.wire if args.bench else args.cases})', once_in_node=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive EP kernels on NIXL: the NVFP4 dispatch wire')
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-sms', type=int, default=0, help='active blocks of the fixed grid (0: the theoretical count)')
    parser.add_argument('--num-gpu-timeout-secs', type=int, default=100)
    parser.add_argument('--num-cpu-timeout-secs', type=int, default=100)
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256, help="expert slots of the cohort's world (the fixed slots per rank are this divided by the cohort size)")
    parser.add_argument('--prefer-overlap-with-compute', type=int, default=0, help='0 (default, as test_ep.py): the 4-channel block on every wire')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cases', choices=('main', 'negatives', 'all'), default='all')
    parser.add_argument('--bench', action='store_true', help='time one wire\'s compact dispatch instead of the cases')
    parser.add_argument('--wire', choices=('bf16', 'fp8', 'nvfp4'), default='nvfp4', help='the wire of --bench')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
