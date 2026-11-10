# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain two BF16 graph capacities and replay after same/different-stream commits.

Run on at least two NVLink domains, with at least two GPUs per domain. Each graph
keeps its own capacity while sharing a buffer through full -> one domain -> full.
"""

import argparse
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from test_adaptive_elastic import (
    Case,
    GraphCase,
    add_common_arguments,
    init_test_environment,
    make_buffer,
)

from nixl.runtime.ep.utils.nixl_bootstrap import NixlBootstrap


@torch.inference_mode()
def run(local_rank, num_local_ranks, args):
    rank, num_ranks = init_test_environment(local_rank, num_local_ranks, args.seed)
    store = dist.TCPStore(
        os.getenv("MASTER_ADDR", "127.0.0.1"),
        int(os.getenv("MASTER_PORT", "8361")),
        world_size=num_ranks,
        is_master=rank == 0,
        timeout=timedelta(seconds=args.store_timeout_secs),
    )
    cohort = NixlBootstrap(
        store,
        rank,
        num_ranks,
        namespace="graph-capacity-stream",
        timeout_seconds=args.store_timeout_secs,
    )
    buffer = make_buffer(args, rank, num_ranks, store)
    assert buffer.max_num_scaleout_ranks >= 2 and buffer.max_num_scaleup_ranks >= 2
    full = buffer.get_active_ranks()
    first_domain = list(range(buffer.max_num_scaleup_ranks))
    captured_at = {peer: full for peer in full}

    def drained_barrier():
        torch.cuda.synchronize()
        cohort.barrier("drained")

    def kernel_entries():
        return {
            path.name for path in (Path(args.jit_cache_dir) / "cache").glob("kernel.*")
        }

    graphs = []
    entries = None
    for capacity in (args.num_tokens - 1, args.num_tokens):
        case_args = argparse.Namespace(**{**vars(args), "num_tokens": capacity})
        graph = GraphCase(Case(buffer, case_args), full, captured_at, drained_barrier)
        graph.capture()
        graph.replay_and_check(full, f"capture capacity {capacity}")
        graphs.append(graph)
        if entries is None:
            entries = kernel_entries()
            assert any(key.startswith("kernel.adaptive_dispatch.") for key in entries)
        else:
            assert kernel_entries() == entries, "Second capacity created JIT entries"

    comm_stream = buffer.get_comm_stream()
    for stream_name, commit_stream in (
        ("different", torch.cuda.Stream()),
        ("same", comm_stream),
    ):
        for phase in (full, first_domain, full):
            if buffer.is_active:
                buffer.barrier(with_cpu_sync=True)
            drained_barrier()
            buffer.stage_topology(
                active_ranks=phase,
                num_active_sms=args.num_sms or buffer.num_launch_sms,
            )
            with torch.cuda.stream(commit_stream):
                buffer.commit_topology()
            if rank in phase:
                for graph in (graphs[0], graphs[1], graphs[0]):
                    # commit_topology must order the upload before communication;
                    # synchronize only after replay, never between commit and replay.
                    with torch.cuda.stream(comm_stream):
                        graph.stats.zero_()
                        graph.graph.replay()
                    comm_stream.synchronize()
                    graph.check(
                        phase,
                        f"{stream_name} stream, {phase}, capacity {graph.case.num_max_tokens_per_rank}",
                    )
            drained_barrier()
            if rank == 0:
                print(
                    f"PASS: {stream_name}-stream commit, ranks {phase}, retained A/B/A",
                    flush=True,
                )

    for graph in graphs:
        graph.release()
    buffer.barrier(with_cpu_sync=True)
    drained_barrier()
    buffer.destroy()
    cohort.barrier("destroyed")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(num_processes=2, num_tokens=17)
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=300)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    assert args.num_tokens >= 2 and args.position_offset == 0
    with tempfile.TemporaryDirectory(prefix="nixl_ep_graph_capacity_") as cache_dir:
        args.jit_cache_dir = cache_dir
        os.environ["NIXL_EP_JIT_CACHE_DIR"] = cache_dir
        torch.multiprocessing.spawn(
            run, args=(args.num_processes, args), nprocs=args.num_processes
        )
