# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""BF16 regression: membership changes retain the fixed constructor grid and graph.

Run on 16 GPUs arranged as 1x16 or 2x8, using test_adaptive_elastic.py's
node-level launcher convention. No benchmark or performance claim is made.
The same dispatch/combine graph is reused while shrinking and expanding
direct worlds and, on 2x8, hybrid worlds; its addresses and launch count stay fixed.

    python tests/elastic/test_fixed_sm_replay.py --num-processes 4
"""

import argparse
import os
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from buffer_setup import rank_coordinates
from test_adaptive_elastic import (
    Case,
    GraphCase,
    add_common_arguments,
    init_test_environment,
    make_buffer,
    switch_world,
    validate_plan,
)

from nixl.runtime.ep.utils.nixl_bootstrap import NixlBootstrap


def graph_addresses(graph):
    handle = graph.handle
    return tuple(
        tensor.data_ptr()
        for tensor in (
            graph.x,
            graph.topk_idx,
            graph.topk_weights,
            graph.stats_storage,
            graph.recv_x,
            graph.outputs,
            handle.recv_src_metadata,
            handle.token_metadata_at_forward,
            handle.channel_linked_list,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.psum_num_recv_tokens_per_expert,
            handle.num_unaligned_recv_tokens_per_expert,
        )
    )


@torch.inference_mode()
def test_loop(local_rank, num_local_ranks, args):
    rank, size = init_test_environment(local_rank, num_local_ranks, args.seed)
    assert size == 16, "This regression requires 16 GPUs in 1x16 or 2x8"
    store = dist.TCPStore(
        os.getenv("MASTER_ADDR", "127.0.0.1"),
        int(os.getenv("MASTER_PORT", "8361")),
        world_size=size,
        is_master=rank == 0,
        timeout=timedelta(seconds=args.store_timeout_secs),
    )
    cohort = NixlBootstrap(
        store,
        rank,
        size,
        timeout_seconds=args.store_timeout_secs,
    )

    def barrier():
        torch.cuda.synchronize()
        cohort.barrier("test")

    buffer = make_buffer(
        args, rank, size, store, num_group_scaleup_ranks=cohort.num_scaleup_ranks
    )
    fixed_sms = buffer.num_launch_sms
    assert fixed_sms == args.num_sms
    full_extent = buffer._get_topology_extent()
    assert full_extent in ((1, 16), (2, 8)), "Expected 1x16 or 2x8 NVLink domains"
    present = buffer.get_active_ranks()
    first_domain = min(rank_coordinates(buffer, r)[0] for r in present)
    eight = sorted(
        (r for r in present if rank_coordinates(buffer, r)[0] == first_domain),
        key=lambda r: rank_coordinates(buffer, r)[1],
    )[:8]
    four = eight[:4]
    two = eight[:2]
    plan = [present]
    if full_extent == (2, 8):
        plan.extend(
            [
                [
                    rank
                    for rank in present
                    if rank_coordinates(buffer, rank)[1] < positions
                ]
                for positions in (4, 2, 1)
            ]
        )
        plan.append(present)
    plan.extend([eight, four, two, present])
    case = Case(buffer, args)
    validate_plan(case, plan, present)

    graph = GraphCase(case, present, {peer: present for peer in present}, barrier)
    graph.capture()
    captured_graph, addresses = graph.graph, graph_addresses(graph)
    region_base = buffer.runtime.get_local_peer_info()[0]
    # Replay must use the captured work after each membership update.
    with (
        patch.object(
            buffer,
            "dispatch",
            side_effect=AssertionError("eager dispatch after capture"),
        ),
        patch.object(
            buffer, "combine", side_effect=AssertionError("eager combine after capture")
        ),
    ):
        for phase_idx, phase in enumerate(plan):
            switch_world(buffer, phase, barrier, args.store_timeout_secs)
            if buffer.world_rank in buffer.get_active_ranks():
                assert buffer.num_launch_sms == fixed_sms
                assert not hasattr(graph.handle, "num_sms")
                assert (
                    graph.graph is captured_graph
                    and graph_addresses(graph) == addresses
                )
                assert buffer.runtime.get_local_peer_info()[0] == region_base
                # Repeated replays exercise reset state after each membership change.
                for replay in range(2):
                    graph.replay_and_check(
                        phase,
                        f"fixed-sm phase={phase_idx} replay={replay} num_launch_sms={fixed_sms}",
                    )
            barrier()
    graph.release()
    buffer.destroy()
    barrier()  # Keep the rank-0 Store alive through every peer's teardown.
    if rank == 0:
        print(
            "Fixed-SM membership graph replay passed (BF16 payloads, combine outputs, fixed addresses)",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=100)
    parser.set_defaults(
        num_tokens=32, num_experts=256, num_topk=8, prefer_overlap_with_compute=1
    )
    args = parser.parse_args()
    assert args.num_experts == 256 and args.num_topk == 8
    assert args.prefer_overlap_with_compute == 1 and args.position_offset == 0
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
