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

"""Background connection/warmup and repeated grow/shrink cycles with retained graphs.

Launch on each allocated host with the same MASTER_ADDR/MASTER_PORT and node-level
RANK/WORLD_SIZE, as in test_adaptive_elastic.py. --num-processes is GPUs per host.
Rank 0 owns the TCPStore. All processes start together; joining ranks construct
their buffers only after the old ranks start serving. Growth adds whole domains
(the host derives hybrid) or GPUs within the single domain (direct); a joiner
warms up alone, in direct mode. Leaving ranks
destroy their graphs, buffers and NIXL agents; each rejoin prepares fresh resources.
Python worker processes stay alive, and surviving ranks retain their graphs.
The default is four up/down iterations: 4..8..4 on one 8-GPU domain, or
8..16..24..32..24..16..8 on four 8-GPU domains, all within one allocation.
For one diagnostic growth on three domains, use --initial-ranks 16 --single-grow
--trace. --trace enables rank/timestamp logs in Python.

The declared maximum world is the job itself: `--num-experts` is its expert slot count and
must divide by its rank count (fixed slots per rank, as in test_adaptive_elastic.py). Every
rank stages the same active SM count, so `--num-sms 0` uses the fixed launch grid rather than
the bandwidth estimate (which follows the live world: a single rank on a joiner).
`--store-timeout-secs` bounds every Store wait and the preparation of a switch.
    python tests/elastic/test_elastic_warmup.py --num-processes 8 --num-experts 256
    python tests/elastic/test_elastic_warmup.py --num-processes 8 --num-experts 512 --initial-ranks 8   (2 hosts)

commit_return_ms: per-rank CPU time until commit_topology returns, including its internal waits.
commit_complete_ms: time from the same start until the GPU topology updates complete.
Connection establishment, registration, warmup, capture, topology staging, replay
draining, routing input updates, and test barriers are outside both measurements.
Commit times use the maximum across each switch's old/new participants. Summary statistics
use those switch maxima, separately for growth, shrinking and all switches.
teardown_ms: shrink cleanup after correctness checking, measured separately from commit.
Includes disconnect/commit, waiting for all peers to detach, and departing ranks'
graph/buffer destruction. Shrink summaries report its mean, median and maximum.
replay_gap_ms: each survivor's final old replay end to its first new replay start,
using CUDA events. Includes test coordination, routing updates and commit; excludes
the first new replay's duration and later teardown. Switches report the maximum gap.
replay_before_us: mean replay duration in one baseline batch before preparation starts.
replay_during_prepare_us: mean from serving batches until all ranks report prepared.
Includes background connections, warmup/capture, staging and the final batch's tail;
shrink has no joining-GPU warmup. Both replay means use the old topology.
CUDA events time individual replays, with one synchronization per batch. Switch rows
report the mean across all old-rank replay samples and the sample counts for each period.
This tests EP communication with synthetic expert results, not model weight moves.
"""
import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import torch
import torch.distributed as dist
from test_adaptive_elastic import (
    Case,
    GraphCase,
    add_common_arguments,
    expected_mode,
    init_test_environment,
    make_buffer,
)

from nixl.runtime.ep.utils.nixl_bootstrap import NixlBootstrap


def barrier(store, key, rank, ranks):
    store.set(f"{key}/{rank}", b"1")
    store.wait([f"{key}/{peer}" for peer in ranks])


def resize_plan(
    num_ranks, num_scaleup_ranks, initial_ranks, iterations, single_grow=False
):
    single_domain = num_ranks == num_scaleup_ranks
    initial_ranks = initial_ranks or (
        num_scaleup_ranks // 2 if single_domain else num_scaleup_ranks
    )
    step = 1 if single_domain else num_scaleup_ranks
    assert 0 < initial_ranks < num_ranks, "The test needs both serving and joining GPUs"
    assert initial_ranks % step == 0, "Multi-domain growth adds whole domains"
    assert iterations > 0
    if single_grow:
        return [initial_ranks, num_ranks]
    ascending = list(range(initial_ranks, num_ranks + 1, step))
    return [initial_ranks] + (ascending[1:] + ascending[-2::-1]) * iterations


def summarize_switches(switches):
    summary = {}
    for direction in ("grow", "shrink", "all"):
        selected = [
            row
            for row in switches
            if direction == "all" or row["direction"] == direction
        ]
        if not selected:
            continue
        summary[direction] = dict(num_switches=len(selected))
        metrics = ("commit_return_ms", "commit_complete_ms", "replay_gap_ms") + (
            ("teardown_ms",) if direction == "shrink" else ()
        )
        for metric in metrics:
            values = [row[metric] for row in selected]
            summary[direction][metric] = dict(
                mean=statistics.mean(values),
                median=statistics.median(values),
                max=max(values),
            )
    return summary


@torch.inference_mode()
def test_loop(local_rank, num_local_ranks, args):
    rank, num_ranks = init_test_environment(local_rank, num_local_ranks, args.seed)
    switch_idx = 0

    def trace(event, **fields):
        if args.trace:
            record = dict(
                time=time.time(),
                host=os.uname().nodename,
                pid=os.getpid(),
                rank=rank,
                switch=switch_idx,
                event=event,
                **fields,
            )
            os.write(2, ("WARMUP_TRACE " + json.dumps(record) + "\n").encode())

    store_host = os.getenv("MASTER_ADDR", "127.0.0.1")
    store_port = int(os.getenv("MASTER_PORT", "8361"))
    store_timeout = timedelta(seconds=args.store_timeout_secs)
    store = dist.TCPStore(
        store_host,
        store_port,
        world_size=num_ranks,
        is_master=rank == 0,
        timeout=store_timeout,
    )
    topology = NixlBootstrap(
        store,
        rank,
        num_ranks,
        namespace="elastic-warmup-test",
        timeout_seconds=args.store_timeout_secs,
    )
    max_so, max_su = topology.num_scaleout_ranks, topology.num_scaleup_ranks
    # The declared maximum world is the job; `--num-experts` is its slot count and the slots per rank are fixed from it
    assert max_so * max_su == num_ranks, (max_so, max_su, num_ranks)
    assert (
        args.num_experts % num_ranks == 0
    ), f"{args.num_experts} experts do not divide by the {num_ranks} ranks of the declared world"
    num_experts_per_rank = args.num_experts // num_ranks
    iterations = 1 if args.single_grow else args.iterations
    plan = resize_plan(
        num_ranks, max_su, args.initial_ranks, iterations, args.single_grow
    )
    initial = list(range(plan[0]))
    full = list(range(num_ranks))
    switches_per_iteration = (len(plan) - 1) // iterations
    captured_at = {
        peer: list(range(next(size for size in plan if peer < size))) for peer in full
    }
    peer_prefix = "warmup/peers/0"
    submitted = [0]
    first_start = torch.cuda.Event(enable_timing=True)
    first_end = torch.cuda.Event(enable_timing=True)
    replay_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(args.replays_per_batch)
    ]

    def mode_for(phase):
        # The mode the host must derive for `phase` (the rule, stated independently of the library)
        return expected_mode(phase, max_su)

    def replay_batch(batch):
        for start, end in replay_events:
            start.record()
            graph.graph.replay()
            end.record()
            submitted[0] += 1
        trace("replay.submitted", batch=batch, submitted=submitted[0])
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in replay_events) * 1e3

    def connect(buffer, peers, peer_store=store):
        # This also runs on a background CPU thread on the serving ranks.
        torch.cuda.set_device(local_rank)
        trace("metadata.wait.begin", peers=peers)
        if peers:
            peer_store.wait([f"{peer_prefix}/{peer}" for peer in peers])
        trace("metadata.wait.end", peers=peers)
        before = submitted[0]
        trace("connect.begin", peers=peers)
        start = time.perf_counter()
        try:
            buffer.connect_ranks(
                peers,
                store=peer_store,
                key_prefix=peer_prefix,
                timeout_secs=args.store_timeout_secs,
            )
        except Exception as error:
            trace("connect.error", peers=peers, error=repr(error))
            raise
        stats = dict(
            connect_ms=(time.perf_counter() - start) * 1e3,
            submitted_during_connect=submitted[0] - before,
        )
        trace("connect.end", peers=peers, **stats)
        return stats

    def initialize(capture_phase, is_initial):
        # Every join allocates/registers fresh resources while old ranks serve: a one-rank cohort placed in the declared world
        trace("initialize.begin", capture_ranks=len(capture_phase), initial=is_initial)
        buffer = make_buffer(
            args,
            0,
            1,
            store,
            bootstrap_id=f"elastic-warmup-buffer/{rank}",
            connect_group=False,
            scaleout_rank_idx=rank // max_su,
            scaleup_rank_idx=rank % max_su,
            max_num_scaleout_ranks=max_so,
            max_num_scaleup_ranks=max_su,
            num_experts_per_rank=num_experts_per_rank,
        )
        buffer.publish(key_prefix=peer_prefix)
        trace("initialize.published")
        args.num_sms = args.num_sms or buffer.num_launch_sms
        stats = connect(buffer, [peer for peer in capture_phase if peer != rank])
        warmup_phase = capture_phase if is_initial else [rank]
        mode = mode_for(warmup_phase)
        buffer.stage_topology(
            active_ranks=warmup_phase, num_active_sms=args.num_sms
        )  # the host derives the mode
        buffer.commit_topology()
        assert (
            buffer.get_topology()[0] == mode
        ), f"auto mode {buffer.get_topology()[0]} for {warmup_phase}, expected {mode}"
        trace("warmup.begin", active_ranks=warmup_phase, mode=mode)
        case = Case(buffer, args)
        capture_round = [0]

        def capture_barrier():
            capture_round[0] += 1
            barrier(store, f"initial-capture/{capture_round[0]}", rank, initial)

        graph = GraphCase(
            case,
            capture_phase,
            captured_at,
            capture_barrier if is_initial else lambda: None,
        )
        warmup_start = time.perf_counter()
        graph.capture()
        trace("warmup.captured")
        graph.graph.replay()
        torch.cuda.synchronize()
        stats["warmup_ms"] = (time.perf_counter() - warmup_start) * 1e3
        trace("warmup.end", **stats)
        # Initialize timing events outside every measured switch.
        first_start.record()
        first_end.record()
        for start, end in replay_events:
            start.record()
            end.record()
        torch.cuda.synchronize()
        print(
            "WARMUP "
            + json.dumps(
                dict(
                    switch=switch_idx,
                    rank=rank,
                    first_world=len(capture_phase),
                    **stats,
                )
            ),
            flush=True,
        )
        return buffer, case, graph, stats

    buffer = case = graph = None
    if rank < len(initial):
        buffer, case, graph, _ = initialize(initial, True)
        graph.replay_and_check(initial, "initial world")
        barrier(store, "initial-ready", rank, initial)
        trace("initial.checked", active_ranks=initial, mode=mode_for(initial))
    if rank == 0:
        print(
            f"Warmup test: {max_so} domains x {max_su} GPUs; {iterations} iterations; "
            f"{len(plan) - 1} switches; plan={plan}",
            flush=True,
        )
    switches = []

    for switch_idx, (old_size, new_size) in enumerate(zip(plan, plan[1:]), 1):
        # Only old/new participants coordinate this switch; future joiners stay idle.
        participants = list(range(max(old_size, new_size)))
        if rank not in participants:
            continue
        old, target = list(range(old_size)), list(range(new_size))
        is_old, is_active = rank < old_size, rank < new_size
        direction = "grow" if new_size > old_size else "shrink"
        mode = mode_for(target)
        key = f"switch/{switch_idx}"
        # A rejoin must never consume metadata published by an earlier NIXL agent.
        peer_prefix = f"warmup/peers/{switch_idx}"
        trace(
            "switch.begin",
            old_size=old_size,
            new_size=new_size,
            serving_mode=mode_for(old),
        )
        routing_phase_idx = 10_000 + switch_idx
        submitted_before = submitted[0]
        connection_stats = dict(connect_ms=0.0, submitted_during_connect=0)
        replay_before_us = replay_during_prepare_us = None
        replay_before_samples = replay_during_prepare_samples = 0
        if is_old:
            barrier(store, f"{key}/baseline-ready", rank, old)
            trace("baseline.begin")
            replay_before_us = replay_batch("baseline") / args.replays_per_batch
            replay_before_samples = args.replays_per_batch
            trace("baseline.complete", replay_before_us=replay_before_us)
            # No rank starts connection preparation until every baseline has finished.
            barrier(store, f"{key}/baseline-done", rank, old)
        if rank == 0:
            store.set(f"{key}/serving", b"1")
        if not is_old:
            store.wait([f"{key}/serving"])
            assert (
                buffer is None and case is None and graph is None
            ), "Joining ranks must start with fresh EP resources"
            buffer, case, graph, connection_stats = initialize(target, False)

        if is_old:

            def prepare_connections():
                # A separate client prevents metadata waits from blocking serving-thread Store calls.
                connection_store = dist.TCPStore(
                    store_host,
                    store_port,
                    is_master=False,
                    wait_for_workers=False,
                    timeout=store_timeout,
                )
                buffer.publish(store=connection_store, key_prefix=peer_prefix)
                connected = set(buffer.get_connected_ranks())
                peers = [
                    peer for peer in target if peer != rank and peer not in connected
                ]
                stats = connect(buffer, peers, connection_store)
                trace("topology.stage.begin", active_ranks=target)
                buffer.stage_topology(
                    active_ranks=target, num_active_sms=args.num_sms
                )  # the host derives the mode
                trace("topology.stage.end")
                connection_store.set(f"{key}/prepared/{rank}", b"1")
                return stats

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(prepare_connections)
                batch = 0
                replay_during_prepare_total_us = 0.0
                deadline = time.monotonic() + args.store_timeout_secs
                while True:
                    trace("replay.begin", batch=batch, submitted=submitted[0])
                    try:
                        replay_during_prepare_total_us += replay_batch(batch)
                        replay_during_prepare_samples += args.replays_per_batch
                    except Exception as error:
                        trace(
                            "replay.error",
                            batch=batch,
                            submitted=submitted[0],
                            error=repr(error),
                        )
                        raise
                    trace("replay.complete", batch=batch, submitted=submitted[0])
                    if future.done():
                        future.result()  # Surface connection-thread errors.
                    # Every old rank submits the SAME number of collective replays before stopping.
                    barrier(store, f"{key}/batch/{batch}/done", rank, old)
                    if rank == 0:
                        ready = store.check(
                            [f"{key}/prepared/{peer}" for peer in participants]
                        )
                        assert (
                            time.monotonic() < deadline
                        ), "Timed out preparing a topology switch"
                        if ready:
                            store.set(f"{key}/stop", b"1")
                        store.set(
                            f"{key}/batch/{batch}/decision",
                            b"switch" if ready else b"continue",
                        )
                    decision = store.get(f"{key}/batch/{batch}/decision")
                    if decision == b"switch":
                        break
                    batch += 1
                connection_stats = future.result()
                replay_during_prepare_us = (
                    replay_during_prepare_total_us / replay_during_prepare_samples
                )
        else:
            trace("topology.stage.begin", active_ranks=target)
            buffer.stage_topology(
                active_ranks=target, num_active_sms=args.num_sms
            )  # the host derives the mode
            trace("topology.stage.end")
            store.set(f"{key}/prepared/{rank}", b"1")
            store.wait([f"{key}/stop"])

        # All old replays stopped. Prepare target routing and drain BEFORE timing.
        if is_active:
            next_topk, next_weights = case.routing(routing_phase_idx, rank, target)
            graph.topk_idx.copy_(next_topk)
            graph.topk_weights.copy_(next_weights)
            del next_topk, next_weights
        torch.cuda.synchronize()
        trace("drain.complete")
        barrier(store, f"{key}/drained", rank, participants)

        stream = torch.cuda.current_stream()
        trace("commit.begin")
        commit_start = time.perf_counter()
        buffer.commit_topology()
        commit_return_ms = (time.perf_counter() - commit_start) * 1e3
        assert (
            buffer.get_topology()[0] == mode
        ), f"auto mode {buffer.get_topology()[0]} for {target}, expected {mode}"
        stream.synchronize()
        commit_complete_ms = (time.perf_counter() - commit_start) * 1e3
        trace(
            "commit.complete",
            commit_return_ms=commit_return_ms,
            commit_complete_ms=commit_complete_ms,
        )

        barrier(store, f"{key}/activated", rank, participants)
        first_replay_us = None
        replay_gap_ms = None
        if is_active:
            trace("first_replay.begin")
            first_start.record()
            graph.graph.replay()
            first_end.record()
            first_end.synchronize()
            first_replay_us = first_start.elapsed_time(first_end) * 1e3
            if is_old:
                replay_gap_ms = replay_events[-1][1].elapsed_time(first_start)
            trace("first_replay.complete", first_replay_us=first_replay_us)
            # Preserve the FIRST replay after this switch before the checker replays again.
            first_outputs = tuple(
                output.clone() if output is not None else None
                for output in graph.outputs
            )
            first_recv = graph.handle.num_recv_tokens_device.item()
        barrier(store, f"{key}/first-replay", rank, participants)
        if is_active:
            checked_recv = graph.replay_and_check(
                target, key, routing_phase_idx=routing_phase_idx
            )
            assert (
                first_recv == checked_recv
            ), f"[{key}] First replay had an incorrect receive count"
            for first, checked in zip(first_outputs, graph.outputs):
                if first is not None:
                    assert torch.equal(
                        first, checked
                    ), f"[{key}] First replay differs from the checked result"
            del first_outputs, first, checked
            trace("switch.checked")

        teardown_ms = 0.0
        if direction == "shrink":
            barrier(store, f"{key}/teardown-ready", rank, participants)
            trace("teardown.begin")
            teardown_start = time.perf_counter()
            peers = buffer.get_connected_ranks()
            to_disconnect = [
                peer for peer in peers if not is_active or peer >= new_size
            ]
            buffer.disconnect_ranks(to_disconnect)
            buffer.commit_topology()
            stream.synchronize()
            # Nobody may free a region while another rank still holds its view.
            barrier(store, f"{key}/disconnected", rank, participants)
            if not is_active:
                graph.release()
                buffer.destroy()
                graph = case = buffer = None
            teardown_ms = (time.perf_counter() - teardown_start) * 1e3
            trace("teardown.complete", teardown_ms=teardown_ms, destroyed=not is_active)

        stats = dict(
            rank=rank,
            serving_replays=submitted[0] - submitted_before,
            commit_return_ms=commit_return_ms,
            commit_complete_ms=commit_complete_ms,
            first_replay_us=first_replay_us,
            teardown_ms=teardown_ms,
            replay_gap_ms=replay_gap_ms,
            replay_before_us=replay_before_us,
            replay_during_prepare_us=replay_during_prepare_us,
            replay_before_samples=replay_before_samples,
            replay_during_prepare_samples=replay_during_prepare_samples,
            **connection_stats,
        )
        store.set(f"{key}/result/{rank}", json.dumps(stats))
        barrier(store, f"{key}/checked", rank, participants)
        if rank == 0:
            results = [
                json.loads(store.get(f"{key}/result/{peer}")) for peer in participants
            ]
            for result in results:
                print(
                    "SWITCH_RANK " + json.dumps(dict(switch=switch_idx, **result)),
                    flush=True,
                )
            result = dict(
                switch=switch_idx,
                iteration=(switch_idx - 1) // switches_per_iteration + 1,
                direction=direction,
                from_ranks=old_size,
                to_ranks=new_size,
                commit_return_ms=max(row["commit_return_ms"] for row in results),
                commit_complete_ms=max(row["commit_complete_ms"] for row in results),
                replay_gap_ms=max(
                    row["replay_gap_ms"]
                    for row in results
                    if row["replay_gap_ms"] is not None
                ),
            )
            for mean_key, count_key in (
                ("replay_before_us", "replay_before_samples"),
                ("replay_during_prepare_us", "replay_during_prepare_samples"),
            ):
                result[count_key] = sum(row[count_key] for row in results)
                result[mean_key] = (
                    sum(
                        row[mean_key] * row[count_key]
                        for row in results
                        if row[count_key]
                    )
                    / result[count_key]
                )
            if direction == "shrink":
                result["teardown_ms"] = max(row["teardown_ms"] for row in results)
            switches.append(result)
            print("SWITCH " + json.dumps(result), flush=True)

    barrier(store, "finished", rank, full)
    if rank == 0:
        for direction, summary in summarize_switches(switches).items():
            print(
                "SUMMARY " + json.dumps(dict(direction=direction, **summary)),
                flush=True,
            )
        print(
            "Elastic background warmup test passed; survivors reused graphs and every join prepared fresh EP resources",
            flush=True,
        )
    if buffer is not None:
        graph.release()
        buffer.destroy()
    barrier(store, "destroyed", rank, full)
    # Keep rank 0's Store alive until every client has returned from the final barrier.
    if rank:
        store.set(f"departed/{rank}", b"1")
    else:
        store.wait([f"departed/{peer}" for peer in full[1:]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(
        num_processes=8
    )  # GPUs per host; the default plan assumes 8-GPU domains
    parser.add_argument(
        "--initial-ranks",
        type=int,
        default=0,
        help="initial serving GPUs; default one domain, or half the GPUs for a single domain",
    )
    parser.add_argument(
        "--iterations", type=int, default=4, help="complete grow/shrink cycles"
    )
    parser.add_argument(
        "--single-grow",
        action="store_true",
        help="grow once to the allocation size; no shrink or repetitions",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="log rank/timestamp diagnostics for setup and replay",
    )
    parser.add_argument("--replays-per-batch", type=int, default=8)
    parser.add_argument("--do-expand", action="store_true")
    parser.add_argument("--expert-alignment", type=int, default=1)
    parser.add_argument(
        "--num-cpu-timeout-secs",
        type=int,
        default=300,
        help="CPU-sync timeout of the buffer (the warm-ups and graphs run without CPU sync)",
    )
    args = parser.parse_args()
    assert args.replays_per_batch > 0
    assert (
        args.position_offset == 0
    ), "The declared maximum world is the job itself (no --position-offset)"
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
