# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Adaptive EP kernels: mode switching, handle sizing, world switching and CUDA-graph retention.

On a single active NVLink domain both modes are legal: `direct` (one flat team; what the library derives there) and
`hybrid` (legal everywhere). `ModeTestBuffer` owns the forced-mode methods and the `NIXL_EP_ADAPTIVE_MODE`
initial-mode override; production buffers always select their mode from the active domains. The test therefore
  1. runs the regular correctness test (first case) in each mode, eagerly, against the `torch.distributed` reference;
  2. checks the handle sizing on one eager dispatch without CPU sync: tensors stored for the declared maximum world
     and the whole launch grid, the extent shapes reported, the exact receive count in the fixed slot of the prefix sum;
  3. captures ONE CUDA graph of dispatch (no CPU sync) + combine, replays it, checks received payloads and the
     combined result against the reference; then rewrites the topology descriptor to the other mode WITHOUT re-capturing, replays the
     same graph again and checks again. Same kernels, same launch parameters, same buffers: only the
     topology descriptor (`EPTopology`) changed. Every rank flips collectively between replays, as a real resize would;
  4. captures one graph per output layout (nonexpanded and expanded with alignment 16), commits a sub-world
     (half the positions; across domains the first domain alone) in direct and hybrid mode, and brings every rank back.
     The same graph receives routed, empty and rerouted inputs in each world; output capacity, local-slot statistics
     and metadata retain their shapes and addresses. Expanded payloads and zero padding are checked as well.
     Sub-worlds below the supported active-GPU minimum are skipped.
On several NVLink domains only the hybrid-legal parts run (1 in hybrid, 2, 4); `--max-num-scaleup-ranks 8` on 4 GPUs
sizes the buffer and the handles for a world the group does not fill.

Usage (one node, 4 GPUs):
    python tests/elastic/test_adaptive_modes.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 8 --num-experts 256
"""
import argparse
import os

import torch
import torch.distributed as dist
from test_ep import (
    add_test_dispatch_combine_arguments,
    check_expanded_slot_mapping,
    test_dispatch_combine,
)

import nixl.runtime.ep as nixl_ep_v2
from nixl.runtime.ep.utils.envs import dist_print, get_logical_domain_size, init_dist
from nixl.runtime.ep.utils.gate import get_unbalanced_scores
from nixl.runtime.ep.utils.math import align, calc_diff
from nixl.runtime.ep.utils.refs import combine as ref_combine
from nixl.runtime.ep.utils.refs import generate_pre_combine_data, ordered_accumulate


class ModeTestBuffer(nixl_ep_v2.ElasticBuffer):
    """Test-only overrides; topology updates retain the production staging, validation and commit rules."""

    @staticmethod
    def mode_of(scaleout_mask):
        forced = os.environ.get("NIXL_EP_ADAPTIVE_MODE", "direct")
        assert forced in (
            "direct",
            "hybrid",
        ), f"NIXL_EP_ADAPTIVE_MODE={forced!r}: direct or hybrid"
        return (
            forced
            if nixl_ep_v2.ElasticBuffer.mode_of(scaleout_mask) == "direct"
            else "hybrid"
        )

    def stage_topology_forced(self, mode, num_active_sms=0, active_ranks=None):
        """Stage an explicit mode; a later ordinary stage derives the mode again."""
        self._stage(mode=mode, num_active_sms=num_active_sms, active_ranks=active_ranks)

    def set_topology_forced(self, mode, num_active_sms=0):
        """Force the committed world's mode without overwriting a pending topology."""
        assert (
            not self.runtime.has_staged_topology()
        ), "Commit the staged world before forcing the committed mode"
        self.stage_topology_forced(mode, num_active_sms)
        self.commit_topology()


def combine_recipe(buffer: nixl_ep_v2.ElasticBuffer):
    # Without `allow_multiple_reduction` the combine input is already reduced within the rank, as in `test_ep.py`
    return (
        True,
        buffer.allow_multiple_reduction and buffer.get_topology()[0] == "hybrid",
    )


def make_graph_case(
    buffer: nixl_ep_v2.ElasticBuffer,
    args: argparse.Namespace,
    do_expand: bool = False,
    expert_alignment: int = 1,
):
    num_max_tokens_per_rank, hidden = args.num_tokens, args.hidden
    num_tokens = max(1, args.num_tokens - dist.get_rank())
    num_topk = args.num_topk
    so_extent, su_extent = buffer.get_topology_extent()
    num_experts = so_extent * su_extent * buffer.num_experts_per_rank
    num_sms = (
        buffer.get_theoretical_num_sms(num_topk) if args.num_sms == 0 else args.num_sms
    )

    # Fixed addresses; world-retention tests update routing in place between replays.
    scores = get_unbalanced_scores(
        num_tokens, num_experts, buffer.num_ranks, num_topk, 1.0, False
    )
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    topk_idx = topk_idx.to(nixl_ep_v2.topk_idx_t)
    slots_per_rank = buffer.num_experts_per_rank
    stats_guard = 104729
    stats_storage = torch.full(
        (slots_per_rank + 1,), stats_guard, dtype=torch.int, device="cuda"
    )
    stats = stats_storage[:-1]
    routing_round = None
    fixed_layout = None

    def routing(rank, extent, round_idx):
        # Route only even slots, leaving physical capacity unused; pad when fewer eligible slots than top-k.
        count = max(1, args.num_tokens - rank)
        eligible = (extent[0] * extent[1] * slots_per_rank + 1) // 2
        choice = torch.arange(num_topk, device="cuda").unsqueeze(0)
        token = torch.arange(count, device="cuda").unsqueeze(1)
        idx = 2 * ((token * num_topk + choice + rank + round_idx) % eligible)
        return torch.where((choice < eligible) & (round_idx != 1), idx, -1).to(
            nixl_ep_v2.topk_idx_t
        )

    def prepare(round_idx):
        nonlocal routing_round
        routing_round = round_idx
        topk_idx.copy_(
            routing(dist.get_rank(), buffer.get_topology_extent(), round_idx)
        )
        topk_weights.copy_((topk_idx >= 0).float())

    def payload(src_global_idx):
        # Stable world-rank numbering, independent of the committed extent used by the metadata
        return generate_pre_combine_data(
            src_global_idx.to(torch.int64),
            num_max_tokens_per_rank,
            1,
            hidden,
            num_ranks=buffer.max_num_scaleout_ranks * buffer.max_num_scaleup_ranks,
        )[:, 0, :]

    x = payload(
        buffer.world_rank * num_max_tokens_per_rank
        + torch.arange(num_tokens, device="cuda")
    )

    # The combine input is a function of the dispatch output (which rows arrived); it is produced inside the
    # graph from the handle's metadata so that a replay in another mode stays self-consistent
    dispatch_args = dict(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_sms=num_sms,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment,
        do_handle_copy=True,
        do_cpu_sync=False,
        do_expand=do_expand,
        do_zero_padding=do_expand,
        cumulative_local_expert_recv_stats=stats,
    )

    def forward():
        stats.zero_()
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
            **dispatch_args
        )
        # `num_recv_tokens` is only known on the GPU without a CPU sync: build the combine input for the
        # worst-case-sized output, rows past the real count are never read
        if do_expand:
            y = recv_x
        else:
            src_token_global_idx = handle.recv_src_metadata[:, 0]
            local_y = generate_pre_combine_data(
                src_token_global_idx.clamp_min(0),
                num_max_tokens_per_rank,
                num_topk,
                hidden,
            )
            local_y[recv_topk_idx == -1] = 0
            y = ordered_accumulate(local_y)
        combined_x, _ = buffer.combine(x=y, handle=handle)
        return combined_x, handle, recv_x

    def reference(num_scaleout_ranks: int, num_scaleup_ranks: int):
        # Seeded as captured (the process group's world size); the reduction order follows the extent and the mode
        ref_y = (
            x.unsqueeze(1).expand(-1, num_topk, -1).clone()
            if do_expand
            else generate_pre_combine_data(
                dist.get_rank() * num_max_tokens_per_rank
                + torch.arange(num_tokens, device="cuda"),
                num_max_tokens_per_rank,
                num_topk,
                hidden,
            )
        )
        num_live_slots = (
            num_scaleout_ranks * num_scaleup_ranks * buffer.num_experts_per_rank
        )
        live_idx = torch.where(topk_idx < num_live_slots, topk_idx, -1)
        ref_y[live_idx == -1] = 0
        reduce_local, reduce_scaleup = combine_recipe(buffer)
        return ref_combine(
            ref_y,
            live_idx,
            num_scaleout_ranks,
            num_scaleup_ranks,
            num_live_slots,
            None,
            buffer.allow_multiple_reduction if do_expand else reduce_local,
            reduce_scaleup,
        )

    def check(
        tag: str,
        combined_x: torch.Tensor,
        handle: nixl_ep_v2.EPHandle,
        recv_x: torch.Tensor,
    ):
        nonlocal fixed_layout
        torch.cuda.synchronize()
        layout = tuple(
            (tensor.shape, tensor.data_ptr())
            for tensor in (
                recv_x,
                stats,
                handle.psum_num_recv_tokens_per_expert,
                handle.num_unaligned_recv_tokens_per_expert,
            )
        )
        if fixed_layout is None:
            fixed_layout = layout
        assert (
            layout == fixed_layout
        ), f"[{tag}] replay changed output or local-slot storage"
        assert (
            handle.psum_num_recv_tokens_per_expert.shape
            == stats.shape
            == (slots_per_rank,)
        )
        assert handle.num_unaligned_recv_tokens_per_expert.shape == (slots_per_rank,)
        assert (
            stats_storage[-1].item() == stats_guard
        ), f"[{tag}] statistics overran fixed local slots"
        assert torch.equal(stats, handle.num_unaligned_recv_tokens_per_expert)
        max_tokens = (
            num_max_tokens_per_rank
            * buffer.max_num_scaleout_ranks
            * buffer.max_num_scaleup_ranks
        )
        expanded_capacity = align(
            max_tokens * min(num_topk, slots_per_rank)
            + (expert_alignment - 1) * slots_per_rank,
            expert_alignment,
        )
        assert recv_x.shape == (expanded_capacity if do_expand else max_tokens, hidden)
        ref = reference(*buffer.get_topology_extent())
        assert torch.equal(
            combined_x, ref
        ), f"[{tag}] combined_x mismatch, diff={calc_diff(combined_x, ref)}"
        num_recv = handle.num_recv_tokens_device.item()
        src_global_idx = handle.recv_src_metadata[:num_recv, 0].to(torch.int64)
        src_rank, src_token = (
            src_global_idx // num_max_tokens_per_rank,
            src_global_idx % num_max_tokens_per_rank,
        )
        _, num_scaleup_ranks = buffer.get_topology_extent()
        src_world_rank = (
            src_rank // num_scaleup_ranks
        ) * buffer.max_num_scaleup_ranks + src_rank % num_scaleup_ranks
        expected_x = payload(src_world_rank * num_max_tokens_per_rank + src_token)
        if do_expand:
            slots = handle.recv_src_metadata[:num_recv, 2:]
            valid = slots >= 0
            expected_expanded = expected_x.unsqueeze(1).expand(-1, num_topk, -1)
            assert torch.equal(
                recv_x[slots[valid].long()], expected_expanded[valid]
            ), f"[{tag}] expanded payload mismatch"
            ends = handle.psum_num_recv_tokens_per_expert
            for end in ends.tolist():
                assert (
                    not recv_x[end : align(end, expert_alignment)].any().item()
                ), f"[{tag}] nonzero alignment padding"
        else:
            assert torch.equal(
                recv_x[:num_recv], expected_x
            ), f"[{tag}] graph dispatched payload mismatch"
        if routing_round is not None:
            extent = buffer.get_topology_extent()
            owner = buffer.scaleout_rank_idx * extent[1] + buffer.scaleup_rank_idx
            expected_stats = torch.zeros_like(stats)
            if do_expand:
                expected_local_idx = torch.full_like(slots, -1)
            for domain in range(extent[0]):
                for position in range(extent[1]):
                    src_idx = routing(
                        domain * su_extent + position, extent, routing_round
                    )
                    local_idx = (
                        src_idx[src_idx // slots_per_rank == owner] % slots_per_rank
                    )
                    expected_stats += torch.bincount(
                        local_idx.long(), minlength=slots_per_rank
                    ).to(stats.dtype)
                    if do_expand:
                        rows = src_rank == domain * extent[1] + position
                        row_idx = src_idx[src_token[rows]]
                        expected_local_idx[rows] = torch.where(
                            row_idx // slots_per_rank == owner,
                            row_idx % slots_per_rank,
                            -1,
                        ).to(slots.dtype)
            assert torch.equal(
                stats, expected_stats
            ), f"[{tag}] fixed-slot statistics mismatch"
            if do_expand:
                check_expanded_slot_mapping(
                    slots,
                    expected_local_idx,
                    handle.psum_num_recv_tokens_per_expert,
                    expert_alignment,
                )
        dist_print(
            f" > [{tag}] mode={buffer.get_topology()[0]} bitwise match with the reference",
            once_in_node=True,
        )
        return num_recv

    return forward, check, num_sms, prepare


# The barriers `capture` runs (before the warm-ups, before the capture, after it), for idle ranks that mirror them
NUM_CAPTURE_BARRIERS = 3


def capture(forward, barrier=dist.barrier, do_capture=True):
    # Warm up eagerly (JIT compile, first-touch allocations), then capture one graph. The elastic test shares it:
    # with its Store `barrier`, and with `do_capture=False` to run the warm-ups and barriers without capturing
    torch.cuda.synchronize()
    barrier()
    for _ in range(2):
        forward()
    torch.cuda.synchronize()
    barrier()
    graph, outputs = None, None
    if do_capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = forward()
    torch.cuda.synchronize()
    barrier()
    return graph, outputs


def run_graph_retention(
    buffer: nixl_ep_v2.ElasticBuffer,
    args: argparse.Namespace,
    first_mode: str,
    second_mode: str,
):
    forward, check, _, _ = make_graph_case(buffer, args)

    buffer.set_topology_forced(first_mode)
    graph, outputs = capture(forward)

    graph.replay()
    check(f"replay in {first_mode}", *outputs)
    dist.barrier()

    # Flip the topology descriptor (`EPTopology`, its `mode`) collectively and replay the SAME graph, then back
    for mode, tag in (
        (second_mode, f"replay in {second_mode} (same graph)"),
        (first_mode, f"replay back in {first_mode} (same graph)"),
    ):
        buffer.set_topology_forced(mode)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        check(tag, *outputs)
        dist.barrier()

    # Release everything allocated under capture before the graph goes away
    del outputs
    del graph
    torch.cuda.synchronize()


def run_world_retention(
    buffer: nixl_ep_v2.ElasticBuffer,
    args: argparse.Namespace,
    do_expand: bool = False,
    expert_alignment: int = 1,
):
    # One graph captured at the full world keeps working after a commit of the topology descriptor (`EPTopology`) to a
    # sub-world in either mode (1 x N: half the positions; S x N: the first domain alone, across a domain boundary) and back
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_topology_extent()
    so_idx, su_idx = buffer.scaleout_rank_idx, buffer.scaleup_rank_idx
    if num_scaleout_ranks == 1:
        assert num_scaleup_ranks % 2 == 0
        capture_extent, capture_mode = (1, num_scaleup_ranks), "direct"
        sub_extents = [
            ((1, num_scaleup_ranks // 2), "direct"),
            ((1, num_scaleup_ranks // 2), "hybrid"),
        ]
    else:
        capture_extent, capture_mode = (num_scaleout_ranks, num_scaleup_ranks), "hybrid"
        sub_extents = [
            ((1, num_scaleup_ranks), "hybrid"),
            ((1, num_scaleup_ranks), "direct"),
        ]
    sub_extents = [
        (extent, mode)
        for extent, mode in sub_extents
        if extent[0] * extent[1] >= nixl_ep_v2._C.min_num_active_ranks
    ]
    num_active_sms = buffer.get_topology()[1]
    forward, check, _, prepare = make_graph_case(
        buffer, args, do_expand, expert_alignment
    )
    prepare(0)

    def participates(extent) -> bool:
        return so_idx < extent[0] and su_idx < extent[1]

    def commit_world(mode: str, extent):
        # The world filling `extent` is committed while every rank is idle (CPU barrier around it)
        buffer.barrier(with_cpu_sync=True)
        dist.barrier()
        active_ranks = [
            buffer.world_rank_of(d, k)
            for d in range(extent[0])
            for k in range(extent[1])
        ]
        buffer.stage_topology_forced(
            mode, num_active_sms=num_active_sms, active_ranks=active_ranks
        )
        buffer.commit_topology()
        torch.cuda.synchronize()
        dist.barrier()
        assert (
            buffer.get_topology_extent() == extent and buffer.get_topology()[0] == mode
        )
        assert buffer.is_active == participates(extent)

    def replay_in_world(tag: str, extent):
        for round_idx in range(
            3
        ):  # Routed, empty, rerouted; all reuse the same captured graph and allocations.
            if participates(extent):
                prepare(round_idx)
                graph.replay()
                num_recv_tokens = check(
                    f"{tag}, expanded={do_expand}, round={round_idx}", *outputs
                )
                handle = outputs[1]
                assert (
                    num_recv_tokens
                    == handle.psum_num_recv_tokens_per_scaleup_rank[
                        extent[1] - 1
                    ].item()
                )
                assert 0 <= num_recv_tokens <= extent[0] * extent[1] * args.num_tokens
            else:
                assert not buffer.is_active
                # Mirror the active ranks' collective inside `check`.
                dist.barrier()
            dist.barrier()

    buffer.set_topology_forced(capture_mode, num_active_sms)
    graph, outputs = capture(forward)
    replay_in_world(f"full world {capture_extent}, {capture_mode}", capture_extent)

    for extent, mode in sub_extents:
        commit_world(mode, extent)
        replay_in_world(f"sub-world {extent}, {mode} (same graph)", extent)

    # Everyone back: the leaving ranks rejoin with a consistent barrier state
    commit_world(capture_mode, capture_extent)
    replay_in_world(
        f"full world {capture_extent} again, {capture_mode} (same graph)",
        capture_extent,
    )

    outputs = graph = None
    torch.cuda.synchronize()


def check_handle_sizing(buffer: nixl_ep_v2.ElasticBuffer, args: argparse.Namespace):
    # Tensors stored for the declared maximum world and the whole launch grid, extent shapes reported, the exact count in its slot
    forward, _, num_sms, _ = make_graph_case(buffer, args)
    _, handle, _ = forward()
    torch.cuda.synchronize()
    max_so, max_su = buffer.max_num_scaleout_ranks, buffer.max_num_scaleup_ranks
    active_so, active_su = buffer.get_topology_extent()
    num_active_sms = handle.num_sms
    # `dispatch(num_sms=...)` committed it through `set_num_active_sms`: even, at most the launch grid
    assert num_active_sms == min(align(num_sms, 2), buffer.num_launch_sms)
    num_channels = handle.token_metadata_at_forward.shape[0]
    channels_per_sm = num_channels // num_active_sms
    num_max_channels = buffer.num_launch_sms * channels_per_sm
    max_linked_list_len = (
        max_so * (args.num_tokens + num_max_channels) + num_max_channels
    )

    def capacity(t: torch.Tensor) -> int:
        return t.untyped_storage().nbytes() // t.element_size()

    assert (handle.psum_num_recv_tokens_per_scaleup_rank[active_su:max_su] == 0).all()
    assert (
        handle.num_recv_tokens == args.num_tokens * max_so * max_su
    ), "worst case of the maximum world"
    assert capacity(handle.token_metadata_at_forward) >= max_linked_list_len * (
        2 + 2 * args.num_topk
    )
    assert capacity(handle.channel_linked_list) >= max_linked_list_len * max_su
    assert (
        capacity(handle.psum_num_recv_tokens_per_expert)
        == buffer.num_experts_per_rank + 1
    )
    assert (
        capacity(handle.num_unaligned_recv_tokens_per_expert)
        == buffer.num_experts_per_rank
    )
    dist_print(
        f" > handle sizing: extent {active_so}x{active_su} of max {max_so}x{max_su}, "
        f"{num_channels} channels ({channels_per_sm}/SM), exact count {handle.num_recv_tokens_device.item()} "
        f"in a {handle.num_recv_tokens}-row output",
        once_in_node=True,
    )


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    cohort_so, cohort_su = get_logical_domain_size(group)
    max_ranks = (args.max_num_scaleout_ranks or cohort_so) * (
        args.max_num_scaleup_ranks or cohort_su
    )
    assert args.num_experts > 0 and args.num_experts % max_ranks == 0
    buffer = ModeTestBuffer(
        group,
        num_experts_per_rank=args.num_experts // max_ranks,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        allow_multiple_reduction=bool(args.allow_multiple_reduction),
        prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
        explicitly_destroy=True,
        num_gpu_timeout_secs=args.num_gpu_timeout_secs,
        max_num_scaleout_ranks=args.max_num_scaleout_ranks,
        max_num_scaleup_ranks=args.max_num_scaleup_ranks,
        num_cpu_timeout_secs=args.num_cpu_timeout_secs,
    )
    num_scaleout_ranks, _ = buffer.get_topology_extent()
    one_domain = num_scaleout_ranks == 1
    mode, num_active_sms = buffer.get_topology()
    dist_print(
        f"Adaptive kernels: launch grid {buffer.num_launch_sms}, active {num_active_sms}, initial mode {mode}, "
        f"world {buffer.get_topology_extent()} of max ({buffer.max_num_scaleout_ranks}, {buffer.max_num_scaleup_ranks})",
        once_in_node=True,
    )

    # Settings of the shared `test_dispatch_combine` that this test fixes: first case only, no benchmark
    args.num_experts = num_ranks * buffer.num_experts_per_rank
    args.test_first_only = True
    args.skip_perf_test = True
    args.dump_profile_traces = ""
    args.deterministic = False

    # Eager correctness in every legal mode (first case of the regular test), forced: direct needs one NVLink domain
    for mode in ("hybrid", "direct") if one_domain else ("hybrid",):
        buffer.set_topology_forced(mode, num_active_sms)
        torch.cuda.synchronize()
        dist.barrier()
        dist_print(f"==== Eager correctness, mode={mode} ====", once_in_node=True)
        test_dispatch_combine(buffer, args, use_fp8_dispatch=False)

    dist_print("==== Handle sizing ====", once_in_node=True)
    check_handle_sizing(buffer, args)

    # Graph retention across the mode switch, both directions (both modes legal on one domain only)
    dist_print("==== CUDA graph retention: mode switch ====", once_in_node=True)
    if one_domain:  # both modes are legal on one NVLink domain only
        run_graph_retention(buffer, args, "direct", "hybrid")
        run_graph_retention(buffer, args, "hybrid", "direct")

    dist_print("==== CUDA graph retention: world switch ====", once_in_node=True)
    run_world_retention(buffer, args)
    run_world_retention(buffer, args, do_expand=True, expert_alignment=16)

    dist_print("Adaptive modes test passed", once_in_node=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive EP kernels: mode switching and CUDA-graph retention"
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
        help="physical slots in the declared maximum world",
    )
    parser.add_argument("--allow-multiple-reduction", type=int, default=1)
    parser.add_argument("--prefer-overlap-with-compute", type=int, default=0)
    parser.add_argument("--max-num-scaleout-ranks", type=int, default=0)
    parser.add_argument("--max-num-scaleup-ranks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    add_test_dispatch_combine_arguments(parser)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
