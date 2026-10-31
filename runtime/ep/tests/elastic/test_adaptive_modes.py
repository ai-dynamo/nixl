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
Adaptive EP kernels: mode switching, handle sizing, world switching and CUDA-graph retention.

On a single live NVLink domain both modes are legal: `direct` (one flat team; what the library derives there) and
`hybrid` (legal everywhere); where this test runs the other kernel it forces it (`set_topology_forced`,
`stage_topology_forced`: the tests' override). The test therefore
  1. runs the regular correctness test (first case) in each mode, eagerly, against the `torch.distributed` reference;
  2. checks the handle sizing on one eager dispatch without CPU sync: tensors stored for the declared maximum world
     and the whole launch grid, the live shapes reported, the exact receive count in the fixed slot of the prefix sum;
  3. captures ONE CUDA graph of dispatch (no CPU sync) + combine, replays it, checks the result against the
     reference; then rewrites the topology descriptor to the other mode WITHOUT re-capturing, replays the
     same graph again and checks again. Same kernels, same launch parameters, same buffers: only the
     topology descriptor (`EPTopology`) changed. Every rank flips collectively between replays, as a real resize would;
  4. captures one graph at the full world, commits a sub-world (half the positions; across domains the first domain
     alone) in direct and in hybrid mode, replays the SAME graph on the ranks still in, checks against the smaller
     world's reference and brings every rank back;
  5. replays `EPHandle.deterministic_sort` inside a graph (`test_deterministic_sort.py`), then runs the fixed-slot
     graph matrix: for every (expanded output, deterministic, expert alignment) combination each rank captures its
     graph at the first world it takes part in (a small world, then the full one) and replays it in every later
     world (a shrink; across domains a growth to two domains; the full world), each with the world's routing, an
     empty round and a rerouted round, checked with the elastic test's `Case` / `GraphCase`.
On several NVLink domains only the hybrid-legal parts run (1 in hybrid, 2, 4, 5); `--max-num-scaleup-ranks 8` on 4 GPUs
sizes the buffer and the handles for a world the group does not fill.

Usage (one node, 4 GPUs):
    python tests/elastic/test_adaptive_modes.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 8 --num-experts 256
"""
import argparse
from typing import Any, Dict, List, Set

import torch
import torch.distributed as dist
from test_deterministic_sort import check_deterministic_sort
from test_ep import test_dispatch_combine

from nixl.runtime import ep  # type: ignore[attr-defined]
from nixl.runtime.ep.utils.envs import dist_print, init_dist
from nixl.runtime.ep.utils.gate import get_unbalanced_scores
from nixl.runtime.ep.utils.math import calc_diff, ceil_div
from nixl.runtime.ep.utils.refs import combine as ref_combine
from nixl.runtime.ep.utils.refs import generate_pre_combine_data, ordered_accumulate


def combine_recipe(
    buffer: ep.ElasticBuffer, allow_multiple_reduction: bool, do_expand: bool = False
):
    if not allow_multiple_reduction:
        # The combine input is already reduced within the rank, as in `test_ep.py`; the expanded layout is not
        return (not do_expand, False)
    return (True, True) if buffer.combine_is_hybrid else (True, False)


def make_graph_case(buffer: ep.ElasticBuffer, args: argparse.Namespace):
    num_max_tokens_per_rank, hidden = args.num_tokens, args.hidden
    num_tokens = max(1, args.num_tokens - dist.get_rank())
    num_topk = args.num_topk
    # Fixed expert slots per rank: the expert count of the world the graph is captured in
    capture_so, capture_su = buffer.get_topology_ranks()
    num_experts = capture_so * capture_su * buffer.num_experts_per_rank
    num_sms = (
        buffer.get_theoretical_num_sms(num_topk) if args.num_sms == 0 else args.num_sms
    )

    # Fixed routing and data (the graph bakes in the addresses; the contents stay the same across replays)
    scores = get_unbalanced_scores(
        num_tokens, num_experts, capture_so * capture_su, num_topk, 1.0, False
    )
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    topk_idx = topk_idx.to(ep.topk_idx_t)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    # The combine input is a function of the dispatch output (which rows arrived); it is produced inside the
    # graph from the handle's metadata so that a replay in another mode stays self-consistent
    dispatch_args = dict(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_sms=num_sms,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=1,
        do_handle_copy=True,
        do_cpu_sync=False,
    )

    def forward():
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
            **dispatch_args
        )
        # `num_recv_tokens` is only known on the GPU without a CPU sync: build the combine input for the
        # worst-case-sized output, rows past the real count are never read
        src_token_global_idx = handle.recv_src_metadata[:, 0]
        local_y = generate_pre_combine_data(
            src_token_global_idx.clamp_min(0), num_max_tokens_per_rank, num_topk, hidden
        )
        local_y[recv_topk_idx == -1] = 0
        y = ordered_accumulate(local_y)
        combined_x, combined_topk_weights, _ = buffer.combine(
            x=y, topk_weights=recv_topk_weights, handle=handle, num_sms=num_sms
        )
        return combined_x, combined_topk_weights, handle

    def reference(num_scaleout_ranks: int, num_scaleup_ranks: int):
        # The live world's slots are `[0, ranks * slots)`: a captured selection beyond them (its owner left the world)
        # is dropped by the kernels; the source encoding and the reduction order follow the live bounds and the mode
        live_num_experts = (
            num_scaleout_ranks * num_scaleup_ranks * buffer.num_experts_per_rank
        )
        live_sel = (topk_idx >= 0) & (topk_idx < live_num_experts)
        live_topk_idx = torch.where(live_sel, topk_idx, torch.full_like(topk_idx, -1))
        rank_idx = (
            buffer.scaleout_rank_idx * num_scaleup_ranks + buffer.scaleup_rank_idx
        )
        ref_y = generate_pre_combine_data(
            rank_idx * num_max_tokens_per_rank
            + torch.arange(num_tokens, device="cuda"),
            num_max_tokens_per_rank,
            num_topk,
            hidden,
        )
        ref_y[~live_sel] = 0
        return (
            ref_combine(
                ref_y,
                live_topk_idx,
                num_scaleout_ranks,
                num_scaleup_ranks,
                live_num_experts,
                None,
                *combine_recipe(buffer, bool(args.allow_multiple_reduction)),
            ),
            torch.where(live_sel, topk_weights, torch.zeros_like(topk_weights)),
        )

    def check(tag: str, combined_x: torch.Tensor, combined_topk_weights: torch.Tensor):
        torch.cuda.synchronize()
        ref, ref_weights = reference(*buffer.get_topology_ranks())
        assert torch.equal(
            combined_x, ref
        ), f"[{tag}] combined_x mismatch, diff={calc_diff(combined_x, ref)}"
        assert torch.equal(
            combined_topk_weights, ref_weights
        ), f"[{tag}] weights mismatch"
        dist_print(
            f" > [{tag}] mode={buffer.get_topology()[0]} bitwise match with the reference",
            once_in_node=True,
        )

    return forward, check


def capture(forward):
    # Warm up eagerly (JIT compile, first-touch allocations), then capture one graph
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(2):
        forward()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = forward()
    torch.cuda.synchronize()
    dist.barrier()
    return graph, outputs


def run_graph_retention(
    buffer: ep.ElasticBuffer,
    args: argparse.Namespace,
    first_mode: str,
    second_mode: str,
):
    if buffer.get_topology_ranks()[0] > 1:
        return  # both modes are legal on one NVLink domain only
    forward, check = make_graph_case(buffer, args)

    buffer.set_topology_forced(first_mode)
    graph, (combined_x, combined_topk_weights, _) = capture(forward)

    graph.replay()
    check(f"replay in {first_mode}", combined_x, combined_topk_weights)
    dist.barrier()

    # Flip the topology descriptor (`EPTopology`, its `mode`) collectively and replay the SAME graph
    buffer.set_topology_forced(second_mode)
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    check(f"replay in {second_mode} (same graph)", combined_x, combined_topk_weights)
    dist.barrier()

    # And back
    buffer.set_topology_forced(first_mode)
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    check(
        f"replay back in {first_mode} (same graph)", combined_x, combined_topk_weights
    )
    dist.barrier()

    # Release everything allocated under capture before the graph goes away
    del combined_x, combined_topk_weights
    del graph
    torch.cuda.synchronize()


def run_world_retention(buffer: ep.ElasticBuffer, args: argparse.Namespace):
    # One graph captured at the full world keeps working after a commit of the topology descriptor (`EPTopology`) to a
    # sub-world in either mode (1 x N: half the positions; S x N: the first domain alone, across a domain boundary) and back
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_topology_ranks()
    so_idx, su_idx = buffer.scaleout_rank_idx, buffer.scaleup_rank_idx
    if num_scaleout_ranks == 1:
        assert num_scaleup_ranks >= 2
        capture_world, capture_mode = (1, num_scaleup_ranks), "direct"
        sub_worlds = [
            ((1, num_scaleup_ranks // 2), "direct"),
            ((1, num_scaleup_ranks // 2), "hybrid"),
        ]
    else:
        capture_world, capture_mode = (num_scaleout_ranks, num_scaleup_ranks), "hybrid"
        sub_worlds = [
            ((1, num_scaleup_ranks), "hybrid"),
            ((1, num_scaleup_ranks), "direct"),
        ]
    num_active_sms = buffer.get_topology()[1]
    forward, check = make_graph_case(buffer, args)

    def participates(world) -> bool:
        return so_idx < world[0] and su_idx < world[1]

    def commit_world(mode: str, world):
        # A change of world is committed while every rank is idle (CPU barrier around it)
        torch.cuda.synchronize()
        dist.barrier()
        active_ranks = [
            buffer.rank_id_of(d, k) for d in range(world[0]) for k in range(world[1])
        ]
        buffer.stage_topology_forced(
            mode, num_active_sms=num_active_sms, active_ranks=active_ranks
        )
        buffer.commit_topology()
        torch.cuda.synchronize()
        dist.barrier()
        assert buffer.get_topology_ranks() == world and buffer.get_topology()[0] == mode
        assert buffer.is_participant == participates(world)

    def replay_in_world(tag: str, world):
        if participates(world):
            graph.replay()  # noqa: F821
            check(tag, combined_x, combined_topk_weights)  # noqa: F821
            # The exact receive count of the live world is on the device, in a topology-independent slot
            num_recv_tokens = handle.num_recv_tokens_device.item()
            assert (
                num_recv_tokens
                == handle.psum_num_recv_tokens_per_scaleup_rank[world[1] - 1].item()
            )
            # Zero is legitimate: the captured selections may all point at slots the sub-world does not have
            assert 0 <= num_recv_tokens <= world[0] * world[1] * args.num_tokens
        else:
            assert not buffer.is_participant
            # Mirror the participants' collectives (`check` prints through `dist_print`, which ends in a barrier)
            dist.barrier()
        dist.barrier()

    buffer.set_topology_forced(capture_mode, num_active_sms)
    graph, (combined_x, combined_topk_weights, handle) = capture(forward)
    replay_in_world(f"full world {capture_world}, {capture_mode}", capture_world)

    for world, mode in sub_worlds:
        commit_world(mode, world)
        replay_in_world(f"sub-world {world}, {mode} (same graph)", world)

    # Everyone back: the leaving ranks rejoin with a consistent barrier state
    commit_world(capture_mode, capture_world)
    replay_in_world(
        f"full world {capture_world} again, {capture_mode} (same graph)", capture_world
    )

    del combined_x, combined_topk_weights, graph
    torch.cuda.synchronize()


def check_handle_sizing(buffer: ep.ElasticBuffer, args: argparse.Namespace):
    # Tensors stored for the declared maximum world and the whole launch grid, live shapes reported, the exact count in its slot
    forward, _ = make_graph_case(buffer, args)
    _, _, handle = forward()
    torch.cuda.synchronize()
    max_so, max_su = buffer.max_num_scaleout_ranks, buffer.max_num_scaleup_ranks
    live_so, live_su = buffer.get_topology_ranks()
    num_active_sms = buffer.get_topology()[1]
    num_channels = handle.token_metadata_at_forward.shape[0]
    channels_per_sm = num_channels // num_active_sms
    num_max_channels = buffer.num_launch_sms * channels_per_sm
    max_linked_list_len = (
        max_so * (args.num_tokens + num_max_channels) + num_max_channels
    )

    def capacity(t: torch.Tensor) -> int:
        return t.untyped_storage().nbytes() // t.element_size()

    assert not handle.recv_count_is_exact
    assert handle.psum_num_recv_tokens_per_scaleup_rank.shape == (max_su + 1,)
    assert (
        handle.num_recv_tokens_device.item()
        == handle.psum_num_recv_tokens_per_scaleup_rank[live_su - 1].item()
    )
    assert (handle.psum_num_recv_tokens_per_scaleup_rank[live_su:max_su] == 0).all()
    assert (
        handle.num_recv_tokens == args.num_tokens * max_so * max_su
    ), "worst case of the maximum world"
    assert num_channels == channels_per_sm * num_active_sms
    assert (
        handle.token_metadata_at_forward.shape[1]
        == live_so * ceil_div(args.num_tokens, num_channels) + 1
    )
    assert handle.channel_linked_list.shape[2] == live_su
    assert capacity(handle.token_metadata_at_forward) >= max_linked_list_len * (
        2 + 2 * args.num_topk
    )
    assert capacity(handle.channel_linked_list) >= max_linked_list_len * max_su
    # The expert-indexed handles have the fixed slot count of this rank in every world
    assert (
        capacity(handle.psum_num_recv_tokens_per_expert)
        == buffer.num_experts_per_rank + 1
    )
    assert (
        capacity(handle.num_unaligned_recv_tokens_per_expert)
        == buffer.num_experts_per_rank
    )
    dist_print(
        f" > handle sizing: live {live_so}x{live_su} of max {max_so}x{max_su}, "
        f"{num_channels} channels ({channels_per_sm}/SM), exact count {handle.num_recv_tokens_device.item()} "
        f"in a {handle.num_recv_tokens}-row output",
        once_in_node=True,
    )


def run_fixed_slot_graph_cases(buffer: ep.ElasticBuffer, args: argparse.Namespace):
    # Capture small, grow, shrink, change the routing and replay empty rounds, over the (expanded output, deterministic,
    # expert alignment) combinations, with the elastic test's `Case` and `GraphCase`
    from test_adaptive_elastic import (  # not at the top: that module imports `combine_recipe` from this one
        Case,
        GraphCase,
        switch_world,
    )

    max_su = buffer.max_num_scaleup_ranks
    full = buffer.get_active_ranks()
    domains = sorted(set(r // max_su for r in full))
    positions = sorted(set(r % max_su for r in full))

    def ids(ds, ks):
        return sorted(d * max_su + k for d in ds for k in ks)

    # The smallest world whose slots cover the top-k, the full world, a shrink (across domains: the first domain with
    # four then three positions, then two domains with three), the full world again
    small = min(
        len(positions), max(1, ceil_div(args.num_topk, buffer.num_experts_per_rank))
    )
    phases = [ids(domains[:1], positions[:small]), full]
    if len(domains) >= 2 and len(positions) >= 4:
        phases.extend(
            [
                ids(domains[:1], positions[:4]),
                ids(domains[:1], positions[:3]),
                ids(domains[:2], positions[:3]),
            ]
        )
    elif len(positions) > 1:
        phases.append(ids(domains[:1], positions[:-1]))
    phases.append(full)
    # Every rank captures at the first phase it takes part in
    capture_phase_of: Dict[int, List[int]] = {}
    for phase in phases:
        for rank in phase:
            capture_phase_of.setdefault(rank, phase)

    def barrier():
        torch.cuda.synchronize()
        dist.barrier()

    infos = (
        buffer._gather_group_peer_infos()
    )  # what `switch_world` reconnects from; the group is connected already
    num_active_sms = buffer.get_topology()[1]
    original_deterministic = buffer.deterministic
    me = buffer.rank_id
    for do_expand, deterministic, alignment in (
        (False, False, 1),
        (False, True, 1),
        (True, False, 1),
        (True, True, 1),
        (True, False, 8),
        (True, True, 8),
    ):
        buffer.deterministic = deterministic
        case_args = argparse.Namespace(**vars(args))
        case_args.num_sms = num_active_sms
        case_args.do_expand, case_args.expert_alignment = do_expand, alignment
        # Every slot applies the same fake expert function in this case; multiplicity must be preserved
        case_args.replicated_expert = do_expand and deterministic and alignment > 1
        case = Case(buffer, case_args)
        graph: Any = None
        captured: Set[int] = set()
        for phase_idx, phase in enumerate(phases):
            switch_world(
                buffer, infos, phase, num_active_sms, barrier
            )  # the library derives the mode, checked inside
            new_capturers = set(phase) - captured
            if me in new_capturers:
                graph = GraphCase(case, phase, capture_phase_of, barrier)
                graph.capture()
            elif new_capturers and buffer.is_participant:
                graph.capture(accompany=True)
            elif new_capturers:
                for _ in range(3):
                    barrier()
            captured |= new_capturers
            if buffer.is_participant:
                tag = f"fixed slots expand={do_expand} deterministic={deterministic} alignment={alignment} phase={phase_idx}"
                graph.replay_and_check(phase, tag, routing_phase_idx=20_000 + phase_idx)
                graph.replay_and_check(phase, tag + " empty", empty=True)
                graph.replay_and_check(
                    phase, tag + " rerouted", routing_phase_idx=30_000 + phase_idx
                )
            barrier()
        graph.release()
    buffer.deterministic = original_deterministic
    dist_print(" > fixed-slot graph matrix passed", once_in_node=True)


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    # `--num-experts` is the expert count of the cohort's world; the slots per rank are fixed from it
    assert (
        args.num_experts % num_ranks == 0
    ), f"{args.num_experts} experts do not divide by {num_ranks} ranks"
    buffer = ep.ElasticBuffer(
        group,
        num_experts_per_rank=args.num_experts // num_ranks,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        deterministic=args.deterministic,
        allow_multiple_reduction=bool(args.allow_multiple_reduction),
        prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
        explicitly_destroy=True,
        num_gpu_timeout_secs=args.num_gpu_timeout_secs,
        max_num_scaleout_ranks=args.max_num_scaleout_ranks,
        max_num_scaleup_ranks=args.max_num_scaleup_ranks,
        num_cpu_timeout_secs=args.num_cpu_timeout_secs,
    )
    num_scaleout_ranks, _ = buffer.get_topology_ranks()
    one_domain = num_scaleout_ranks == 1
    mode, num_active_sms = buffer.get_topology()
    dist_print(
        f"Adaptive kernels: launch grid {buffer.num_launch_sms}, active {num_active_sms}, initial mode {mode}, "
        f"world {buffer.get_topology_ranks()} of max ({buffer.max_num_scaleout_ranks}, {buffer.max_num_scaleup_ranks})",
        once_in_node=True,
    )

    # Settings of the shared `test_dispatch_combine` that this test fixes: first case only, no benchmark
    args.test_first_only = True
    args.skip_perf_test = True
    args.dump_profile_traces = ""
    args.ignore_local_traffic = False

    # Eager correctness in every legal mode (first case of the regular test), forced: direct needs one NVLink domain
    for mode in ("hybrid", "direct") if one_domain else ("hybrid",):
        buffer.set_topology_forced(mode, num_active_sms)
        torch.cuda.synchronize()
        dist.barrier()
        dist_print(f"==== Eager correctness, mode={mode} ====", once_in_node=True)
        test_dispatch_combine(buffer, args)

    dist_print("==== Handle sizing ====", once_in_node=True)
    check_handle_sizing(buffer, args)

    # Graph retention across the mode switch, both directions (both modes legal on one domain only)
    dist_print("==== CUDA graph retention: mode switch ====", once_in_node=True)
    run_graph_retention(buffer, args, "direct", "hybrid")
    run_graph_retention(buffer, args, "hybrid", "direct")

    dist_print("==== CUDA graph retention: world switch ====", once_in_node=True)
    run_world_retention(buffer, args)

    dist_print(
        "==== Deterministic sort in a graph, fixed-slot graph matrix ====",
        once_in_node=True,
    )
    check_deterministic_sort()
    run_fixed_slot_graph_cases(buffer, args)

    dist_print("Adaptive mode-switch test passed", once_in_node=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive EP kernels: mode switch + CUDA graph retention"
    )
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument("--num-gpu-timeout-secs", type=int, default=100)
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=100)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="expert slots of the cohort's world (the fixed slots per rank are this divided by the cohort size)",
    )
    parser.add_argument("--allow-multiple-reduction", type=int, default=1)
    parser.add_argument("--prefer-overlap-with-compute", type=int, default=0)
    parser.add_argument("--max-num-scaleout-ranks", type=int, default=0)
    parser.add_argument("--max-num-scaleup-ranks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    # Arguments the shared `test_dispatch_combine` reads
    parser.add_argument("--do-cpu-sync", type=int, default=1)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    parser.add_argument("--precise-unbalanced-ratio", action="store_true")
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
