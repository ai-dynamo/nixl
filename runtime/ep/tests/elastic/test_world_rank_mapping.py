# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 correctness: singleton graphs expand into interleaved public world ranks.

Run on at least two NVLink domains, with at least two GPUs per domain. Physical
process order is domain-contiguous; public rank IDs interleave those domains.
Each rank captures independently, then retains its graph through full -> one
domain -> full membership, checking payloads, expert counts and combine outputs.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from buffer_setup import rank_mapping
from test_adaptive_elastic import (
    Case,
    GraphCase,
    add_common_arguments,
    init_test_environment,
    make_buffer,
    run_eager,
    switch_world,
)

from nixl.runtime.ep.utils.nixl_bootstrap import NixlBootstrap


@torch.inference_mode()
def run(local_rank, num_local_ranks, args):
    rank, size = init_test_environment(local_rank, num_local_ranks, args.seed)
    store = dist.TCPStore(
        os.getenv("MASTER_ADDR", "127.0.0.1"),
        int(os.getenv("MASTER_PORT", "8361")),
        world_size=size,
        is_master=rank == 0,
        timeout=timedelta(seconds=args.store_timeout_secs),
    )
    cohort = NixlBootstrap(store, rank, size, timeout_seconds=args.store_timeout_secs)
    domains, positions = cohort.num_scaleout_ranks, cohort.num_scaleup_ranks
    assert domains >= 2 and positions >= 2, "Requires at least 2x2 NVLink domains"
    world = [(r % positions) * domains + r // positions for r in range(size)]
    me = world[rank]
    full = sorted(world)
    first_domain = sorted(world[:positions])
    expected_mapping = {world[r]: r for r in range(size)}
    args.num_experts = size * 2
    buffer = make_buffer(
        args,
        0,
        1,
        store,
        max_num_scaleout_ranks=domains,
        max_num_scaleup_ranks=positions,
        world_ranks=[me],
    )
    case = Case(buffer, args)
    graph = GraphCase(case, full, {peer: full for peer in full}, torch.cuda.synchronize)
    graph.capture()
    assert rank_mapping(buffer) == {me: 0}
    graph.replay_and_check([me], "independent singleton")
    cohort.barrier("singletons-captured")

    buffer.connect_ranks([peer for peer in full if peer != me])
    buffer.set_active_ranks(full)
    torch.cuda.synchronize()
    assert rank_mapping(buffer) == expected_mapping

    def barrier():
        torch.cuda.synchronize()
        cohort.barrier("phase")

    captured_graph = graph.graph
    for phase_idx, phase in enumerate((full, first_domain, full)):
        switch_world(buffer, phase, barrier, args.store_timeout_secs)
        if me in phase:
            run_eager(case, phase_idx, phase, f"permuted eager {phase_idx}")
            graph.replay_and_check(phase, f"permuted retained graph {phase_idx}")
            assert graph.graph is captured_graph
        barrier()
    graph.release()
    buffer.destroy()
    barrier()
    if rank == 0:
        print(
            "PASS: permuted public ranks, singleton capture, direct/hybrid graph reuse",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=100)
    parser.set_defaults(num_tokens=4, num_topk=2, num_processes=2)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        run, args=(args.num_processes, args), nprocs=args.num_processes
    )
