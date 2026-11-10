# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Prepare fresh joiners while survivors replay, then reuse their graphs through grow/shrink cycles.

Launch with the same node-level RANK/WORLD_SIZE convention as test_adaptive_elastic.py.
Initial ranks warm up eagerly. Joiners compile and capture against their future topology,
without executing communication until cutover; this preserves the two-serving-GPU minimum.
The first execution on a joiner is included in completion_gap_ms. This test exercises EP
communication with synthetic expert outputs, not model loading or redistribution.

JSON SWITCH rows report the worst survivor's final-old-completion to first-new-completion
gap, including drain, host rendezvous, activation and departure detachment. Completed
replays and maximum completion gaps during preparation/retirement expose serving pauses
that mean kernel durations omit. Store coordination and correctness checking contribute
to these gaps; they are test-observed service gaps, not kernel-only latency.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

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


def resize_plan(num_ranks, max_su, initial_ranks, iterations, single_grow=False):
    initial = initial_ranks or max(2, max_su // 2 if num_ranks == max_su else max_su)
    step = 1 if num_ranks == max_su else max_su
    assert (
        2 <= initial < num_ranks
    ), "The test needs at least two serving GPUs and a joiner"
    assert (
        initial % step == 0 and num_ranks % step == 0
    ), "Multi-domain growth adds whole domains"
    assert iterations > 0
    ascending = list(range(initial, num_ranks + 1, step))
    return (
        [initial, num_ranks]
        if single_grow
        else [initial] + (ascending[1:] + ascending[-2::-1]) * iterations
    )


def barrier(store, key, rank, ranks):
    store.set(f"{key}/{rank}", b"1")
    store.wait([f"{key}/{peer}" for peer in ranks])


class ReplayTimer:
    def __init__(self, num_pairs):
        # Alternate event sets so the previous completion is not overwritten before its gap is read.
        self.events = [
            [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(num_pairs)
            ]
            for _ in range(2)
        ]
        self.next_set, self.completed, self.last_end = 0, 0, None
        for events in self.events:
            for start, end in events:
                start.record()
                end.record()
        torch.cuda.synchronize()

    def batch(self, graph):
        events = self.events[self.next_set]
        self.next_set ^= 1
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        events[-1][1].synchronize()
        gaps = []
        for _, end in events:
            if self.last_end is not None:
                gaps.append(self.last_end.elapsed_time(end))
            self.last_end = end
        self.completed += len(events)
        return dict(
            pairs=len(events),
            kernel_us=sum(start.elapsed_time(end) for start, end in events) * 1e3,
            max_completion_gap_ms=max(gaps, default=0.0),
        )


@torch.inference_mode()
def test_loop(local_rank, num_local_ranks, args):
    rank, num_ranks = init_test_environment(local_rank, num_local_ranks, args.seed)
    timeout = timedelta(seconds=args.store_timeout_secs)
    host, port = os.getenv("MASTER_ADDR", "127.0.0.1"), int(
        os.getenv("MASTER_PORT", "8361")
    )
    store = dist.TCPStore(
        host, port, world_size=num_ranks, is_master=rank == 0, timeout=timeout
    )
    cohort = NixlBootstrap(
        store,
        rank,
        num_ranks,
        namespace="elastic-warmup-test",
        timeout_seconds=args.store_timeout_secs,
    )
    max_so, max_su = cohort.num_scaleout_ranks, cohort.num_scaleup_ranks
    plan = resize_plan(
        num_ranks, max_su, args.initial_ranks, args.iterations, args.single_grow
    )
    full, initial = list(range(num_ranks)), list(range(plan[0]))
    captured_at = {}
    buffer = graph = timer = None

    def trace(event, **fields):
        if args.trace:
            print(
                "WARMUP_TRACE "
                + json.dumps(dict(rank=rank, time=time.time(), event=event, **fields)),
                flush=True,
            )

    def connect(target, prefix, peer_store):
        torch.cuda.set_device(local_rank)
        buffer.publish(store=peer_store, key_prefix=prefix)
        peers = sorted(set(target) - {rank} - set(buffer.get_connected_ranks()))
        before = timer.completed if timer is not None else 0
        start = time.monotonic()
        if peers:
            buffer.connect_ranks(
                peers,
                store=peer_store,
                key_prefix=prefix,
                timeout_secs=args.store_timeout_secs,
            )
        return dict(
            connect_ms=(time.monotonic() - start) * 1e3,
            completed_during_connect=(timer.completed if timer is not None else 0)
            - before,
        )

    def initialize(target, prefix, eager_warmup):
        nonlocal buffer, graph, timer
        buffer = make_buffer(
            args,
            0,
            1,
            store,
            bootstrap_id=f"elastic-warmup/{prefix}/{rank}",
            max_num_scaleout_ranks=max_so,
            max_num_scaleup_ranks=max_su,
            scaleout_rank_idx=rank // max_su,
            scaleup_rank_idx=rank % max_su,
        )
        stats = connect(target, prefix, store)
        buffer.stage_topology(
            active_ranks=target, num_active_sms=args.num_sms or buffer.num_launch_sms
        )
        buffer.commit_topology()
        round_idx = [0]

        def capture_barrier():
            round_idx[0] += 1
            barrier(store, f"{prefix}/capture/{round_idx[0]}", rank, target)

        graph = GraphCase(Case(buffer, args), target, captured_at, capture_barrier)
        start = time.monotonic()
        if eager_warmup:
            graph.capture()
            graph.replay_and_check(target, "initial world")
        else:
            # CUDA capture records communication; it does not execute it against the still-serving old world.
            torch.cuda.synchronize()
            graph.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph.graph):
                graph.outputs, graph.handle, graph.recv_x = graph.forward()
            torch.cuda.synchronize()
        timer = ReplayTimer(args.replays_per_batch)
        trace(
            "prepared",
            eager_warmup=eager_warmup,
            capture_ms=(time.monotonic() - start) * 1e3,
            **stats,
        )
        return stats

    def serve_until(target, ready_keys, key, preparation=None):
        totals = dict(pairs=0, kernel_us=0.0, max_completion_gap_ms=0.0)
        deadline = time.monotonic() + args.store_timeout_secs
        batch = 0
        while True:
            row = timer.batch(graph.graph)
            totals["pairs"] += row["pairs"]
            totals["kernel_us"] += row["kernel_us"]
            totals["max_completion_gap_ms"] = max(
                totals["max_completion_gap_ms"], row["max_completion_gap_ms"]
            )
            if preparation is not None and preparation.done():
                preparation.result()
            # Collective kernels require the same replay count on every serving rank.
            barrier(store, f"{key}/{batch}/done", rank, target)
            if rank == target[0]:
                assert time.monotonic() < deadline, f"Timed out waiting for {key}"
                store.set(
                    f"{key}/{batch}/continue", b"0" if store.check(ready_keys) else b"1"
                )
            if store.get(f"{key}/{batch}/continue") == b"0":
                return totals
            batch += 1

    captured_at.update({peer: initial for peer in initial})
    if rank in initial:
        initialize(initial, "initial", True)
    barrier(store, "initial-ready", rank, full)
    switches = []
    for index, (old_size, new_size) in enumerate(zip(plan, plan[1:]), 1):
        old, target = list(range(old_size)), list(range(new_size))
        participants, joiners = sorted(set(old) | set(target)), sorted(
            set(target) - set(old)
        )
        captured_at.update({peer: target for peer in joiners})
        if rank not in participants:
            continue
        key = f"switch/{index}"
        stats = dict(rank=rank, connect_ms=0.0, completed_during_connect=0)
        if rank in old:
            barrier(store, f"{key}/baseline-ready", rank, old)
            graph.stats.zero_()
            before = timer.completed
            stats["baseline"] = timer.batch(graph.graph)
            barrier(store, f"{key}/baseline-done", rank, old)
            if rank == 0:
                store.set(f"{key}/serving", b"1")

            def prepare():
                # Separate Store client: metadata waits must not block serving-thread coordination.
                connection_store = dist.TCPStore(
                    host, port, is_master=False, wait_for_workers=False, timeout=timeout
                )
                result = connect(target, key, connection_store)
                buffer.stage_topology(
                    active_ranks=target,
                    num_active_sms=args.num_sms or buffer.num_launch_sms,
                )
                connection_store.set(f"{key}/prepared/{rank}", b"1")
                return result

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(prepare)
                stats["preparation"] = serve_until(
                    old,
                    [f"{key}/prepared/{peer}" for peer in participants],
                    f"{key}/preparation",
                    future,
                )
                stats.update(future.result())
            graph.check(
                old, f"{key}/last serving replay", num_replays=timer.completed - before
            )
            # Resume serving after the oracle so its cost is outside the measured cutover interval.
            stats["pre_cutover"] = timer.batch(graph.graph)
        else:
            store.wait([f"{key}/serving"])
            stats.update(initialize(target, key, False))
            store.set(f"{key}/prepared/{rank}", b"1")

        # Only the old world drains collectively. Joiners have captured a future world but have not executed it.
        if rank in old:
            buffer.barrier(with_cpu_sync=True)
        torch.cuda.synchronize()
        barrier(store, f"{key}/drained", rank, participants)
        start = time.monotonic()
        if rank in old:
            buffer.commit_topology()
        if new_size < old_size:
            peers = buffer.get_connected_ranks()
            buffer.disconnect_ranks(
                [peer for peer in peers if rank not in target or peer not in target]
            )
            buffer.commit_topology()
        torch.cuda.synchronize()
        stats["activation_ms"] = (time.monotonic() - start) * 1e3
        barrier(store, f"{key}/activated", rank, participants)

        if rank in target:
            # Check this first replay directly, before another replay could overwrite incorrect output.
            first_end = torch.cuda.Event(enable_timing=True)
            graph.stats.zero_()
            graph.graph.replay()
            first_end.record()
            first_end.synchronize()
            stats["completion_gap_ms"] = (
                timer.last_end.elapsed_time(first_end) if rank in old else None
            )
            graph.check(target, f"{key}/first replay")
            timer.last_end = first_end
        else:
            start = time.monotonic()
            graph.release()
            buffer.destroy()
            buffer = graph = timer = None
            stats["retirement_ms"] = (time.monotonic() - start) * 1e3
            store.set(f"{key}/retired/{rank}", b"1")

        if rank in target and new_size < old_size:
            stats["retirement_serving"] = serve_until(
                target,
                [f"{key}/retired/{peer}" for peer in old if peer not in target],
                f"{key}/retirement",
            )
            graph.check(
                target,
                f"{key}/last retirement replay",
                num_replays=1 + stats["retirement_serving"]["pairs"],
            )
        store.set(f"{key}/result/{rank}", json.dumps(stats))
        barrier(store, f"{key}/finished", rank, participants)
        if rank == 0:
            rows = [
                json.loads(store.get(f"{key}/result/{peer}")) for peer in participants
            ]
            row = dict(
                switch=index,
                from_ranks=old_size,
                to_ranks=new_size,
                completion_gap_ms=max(
                    item["completion_gap_ms"]
                    for item in rows
                    if item.get("completion_gap_ms") is not None
                ),
                ranks=rows,
            )
            row["max_serving_gap_ms"] = max(
                [row["completion_gap_ms"]]
                + [
                    item[phase]["max_completion_gap_ms"]
                    for item in rows
                    for phase in (
                        "baseline",
                        "preparation",
                        "pre_cutover",
                        "retirement_serving",
                    )
                    if phase in item
                ]
            )
            switches.append(row)
            print("SWITCH " + json.dumps(row), flush=True)
            if args.max_completion_gap_ms:
                assert (
                    row["max_serving_gap_ms"] <= args.max_completion_gap_ms
                ), f'Switch {index} maximum serving gap {row["max_serving_gap_ms"]:.3f} ms exceeds {args.max_completion_gap_ms} ms'

    barrier(store, "finished", rank, full)
    if buffer is not None:
        graph.release()
        buffer.destroy()
    barrier(store, "destroyed", rank, full)
    if rank:
        store.set(f"departed/{rank}", b"1")
    else:
        store.wait([f"departed/{peer}" for peer in full[1:]])
        if args.save_output:
            with open(args.save_output, "w") as output:
                json.dump(
                    dict(config=vars(args), plan=plan, switches=switches),
                    output,
                    indent=2,
                )
        print(
            "Passed: fresh joiners prepared during serving, first replays checked, survivor graphs retained.",
            flush=True,
        )


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--initial-ranks", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--single-grow", action="store_true")
    parser.add_argument("--replays-per-batch", type=int, default=8)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=300)
    parser.add_argument("--save-output", default="")
    parser.add_argument(
        "--max-completion-gap-ms",
        type=float,
        default=0.0,
        help="bound preparation, cutover and retirement serving gaps in ms (0: report only)",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    assert (
        args.replays_per_batch > 0
        and args.position_offset == 0
        and args.max_completion_gap_ms >= 0
    )
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
