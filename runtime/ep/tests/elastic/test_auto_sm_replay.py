# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""BF16 regression: automatic SM changes reuse one captured dispatch/combine graph.

Run on 16 GPUs arranged as 1x16 or 2x8, using test_adaptive_elastic.py's
node-level launcher convention. No benchmark or performance claim is made.
The numerical model uses fixed test bandwidths to exercise different active
counts in direct mode and, on 2x8, while shrinking and expanding hybrid mode.

    python tests/elastic/test_auto_sm_replay.py --num-processes 4
"""

import argparse
import functools
import os
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from test_adaptive_elastic import (
    Case,
    GraphCase,
    add_common_arguments,
    init_test_environment,
    make_buffer,
    switch_world,
    validate_plan,
)

from nixl.runtime.ep.buffers import elastic as elastic_module
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
        namespace="auto-sm-test",
        timeout_seconds=args.store_timeout_secs,
    )

    def barrier():
        torch.cuda.synchronize()
        cohort.barrier("test")

    buffer = make_buffer(args, rank, size, store, bootstrap_id="auto-sm-buffer")
    assert (
        buffer.num_launch_sms >= 24
    ), "Use a launch grid of at least 24 blocks for distinct automatic estimates"
    full_extent = buffer.get_topology_extent()
    assert full_extent in ((1, 16), (2, 8)), "Expected 1x16 or 2x8 NVLink domains"
    present = buffer.get_active_ranks()
    first_domain = min(present) // buffer.max_num_scaleup_ranks
    eight = [
        first_domain * buffer.max_num_scaleup_ranks + position for position in range(8)
    ]
    four = eight[:4]
    two = eight[:2]
    plan = [present]
    if full_extent == (2, 8):
        plan.extend(
            [
                [
                    domain * buffer.max_num_scaleup_ranks + position
                    for domain in range(2)
                    for position in range(positions)
                ]
                for positions in (4, 2, 1)
            ]
        )
        plan.append(present)
    plan.extend([eight, four, two, present])
    case = Case(buffer, args)
    case.num_sms = 0  # Exercise production automatic dispatch and topology retuning.
    validate_plan(case, plan, present)
    me = buffer.world_rank

    # Test-only cached probe values; the production traffic formula remains intact.
    @functools.lru_cache()
    def rdma_bandwidth():
        return 50.0

    @functools.lru_cache()
    def nvlink_bandwidth():
        return 600.0

    with patch.object(elastic_module, "get_rdma_gbs", rdma_bandwidth), patch.object(
        elastic_module, "get_nvlink_gbs", nvlink_bandwidth
    ):
        graph = GraphCase(case, present, {peer: present for peer in present}, barrier)
        graph.capture()
        captured_graph, addresses = graph.graph, graph_addresses(graph)
        region_base = buffer.get_local_peer_info()[0]
        counts = []
        # No Python dispatch/combine call may retune or rebuild the graph after capture.
        with patch.object(
            buffer,
            "dispatch",
            side_effect=AssertionError("eager dispatch after capture"),
        ), patch.object(
            buffer, "combine", side_effect=AssertionError("eager combine after capture")
        ):
            for phase_idx, phase in enumerate(plan):
                switch_world(buffer, phase, barrier, args.store_timeout_secs)
                if buffer.is_active:
                    mode, active_sms = buffer.get_topology()
                    expected_sms = (
                        {16: 12, 8: 10, 4: 8, 2: 6}
                        if mode == "hybrid"
                        else {16: 16, 8: 18, 4: 20, 2: 30}
                    )[len(phase)]
                    expected_sms = min(buffer.num_launch_sms, expected_sms)
                    assert active_sms == expected_sms, (
                        phase_idx,
                        active_sms,
                        expected_sms,
                    )
                    assert (
                        graph.graph is captured_graph
                        and graph_addresses(graph) == addresses
                    )
                    assert buffer.get_local_peer_info()[0] == region_base
                    # Two replays also exercise reset state at the new count.
                    for replay in range(2):
                        graph.replay_and_check(
                            phase,
                            f"auto-sm phase={phase_idx} replay={replay} active_sms={active_sms}",
                        )
                    counts.append(active_sms)
                barrier()
        if all(me in phase for phase in plan):
            full_sms = 12 if full_extent == (2, 8) else 16
            assert counts[-5:] == [
                full_sms,
                18,
                20,
                min(buffer.num_launch_sms, 30),
                full_sms,
            ], counts
            if full_extent == (2, 8):
                assert counts[:5] == [12, 10, 8, 6, 12], counts
        graph.release()
    buffer.destroy()
    barrier()  # Keep the rank-0 Store alive through every peer's teardown.
    if rank == 0:
        print(
            "Automatic SM shrink/expand graph replay passed (BF16 payloads, combine outputs, fixed addresses)",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser._option_string_actions["--num-sms"].help = (
        "must be 0: exercise automatic SM selection and topology retuning"
    )
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=100)
    parser.set_defaults(
        num_tokens=32, num_experts=256, num_topk=8, prefer_overlap_with_compute=1
    )
    args = parser.parse_args()
    assert args.num_sms == 0 and args.num_experts == 256 and args.num_topk == 8
    assert args.prefer_overlap_with_compute == 1 and args.position_offset == 0
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
