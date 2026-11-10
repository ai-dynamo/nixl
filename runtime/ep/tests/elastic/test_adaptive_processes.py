# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Processes that start later and processes that exit. Where `test_adaptive_elastic.py` keeps every process alive, here a
job is a set of independent STEPS: a step is one bootstrap cohort (`torch.multiprocessing.spawn` over the GPUs of one
node, or a subset), launched when its ranks first appear in the plan and exiting for good when they leave. Steps meet
only through an external `torch.distributed.TCPStore` (transport metadata via `ElasticBuffer.publish` /
`connect_ranks(store=...)`, phase barriers via counters, all under `--run-tag`); world ranks follow from the step's
coordinates (`--scaleout-idx`, `--position-offset`), so a later step may re-use the world ranks of one that exited.

Per phase k, every process alive at its start arrives at `phase/k/arrive`; a LEAVER makes itself a one-rank world,
drops every connection, destroys its buffer and exits without waiting for anyone; a STAYER or JOINER connects the
joiners (both sides call `connect_ranks`), commits the phase's world, drops the leavers with a second commit, runs one
eager dispatch + combine checked against the seed-regenerated routing of the active ranks and replays the graph its
process captured when it joined (joiners capture now, stayers run the warm-ups with them). One `PROC` line per process
and phase records the timeline of its join, stay or leave.

Launch: a TCPStore server runs for the whole job outside the steps (`--store-host` / `--store-port`, defaults
`STORE_HOST` / `STORE_PORT`, then 127.0.0.1:8400); a step is one invocation with `WORLD_SIZE` and `RANK` unset, started
by a launcher once phase k-1 is done (the counter `{run_tag}phase/{k-1}/done` at the size of phase k-1). Every step gets
the same plan, declared maximum world and `--num-sms` (0 = the fixed launch grid, kept for every phase and capture).
Plans sharing one server use distinct `--run-tag`s, and a failed run is never
retried under its tag (its counters, metadata and bootstrap namespaces all derive from it).
    PLAN='[[0,1,2,3],[0,1,2,3,4,5,6,7],[0,1,2,3],[0,1,2,3,4,5,6,7],[4,5,6,7],[0,1,2,3,4,5,6,7]]'
    node 0: test_adaptive_processes.py --plan "$PLAN" --scaleout-idx 0 --start-phase 0 --max-scaleout-ranks 2 ...
    node 1: test_adaptive_processes.py --plan "$PLAN" --scaleout-idx 1 --start-phase 1 --max-scaleout-ranks 2 ...   (exits at phase 2; back at 3 as new processes with --start-phase 3)
"""
import argparse
import json
import os
import time
from datetime import timedelta
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from test_adaptive_elastic import (
    Case,
    GraphCase,
    add_common_arguments,
    expected_mode,
    init_test_environment,
    make_buffer,
    run_eager,
    validate_plan,
)


def store_barrier(
    store: dist.Store, key: str, count: int, timeout_secs: float, who: str
) -> None:
    """Counting barrier over a TCPStore: add one to the key, then wait for `count` arrivals (device drained first)"""
    torch.cuda.synchronize()
    store.add(key, 1)
    deadline = time.time() + timeout_secs
    while True:
        arrived = store.add(key, 0)
        if arrived >= count:
            return
        assert (
            time.time() < deadline
        ), f"rank {who}: barrier {key}: {arrived}/{count} arrivals after {timeout_secs} s"
        time.sleep(0.02)


def capture_phase_of(plan: List[List[int]], k: int) -> Dict[int, List[int]]:
    """For every rank of phase k, the phase its CURRENT process joined at (leavers exit, so a rejoin is a fresh process)"""
    out = {}
    for r in plan[k]:
        j = k
        while j > 0 and r in plan[j - 1]:
            j -= 1
        out[r] = list(plan[j])
    return out


@torch.inference_mode()
def step_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # Timeline of this process (`PROC` lines, one per phase): the entry epoch and the host duration of every step
    t_entry = time.time()
    pc = time.perf_counter
    t0 = pc()
    # This step's own cohort (one invocation, `WORLD_SIZE` unset or 1): no process group, no rendezvous port
    rank_idx, num_ranks = init_test_environment(local_rank, num_local_ranks, args.seed)
    assert (
        num_ranks == num_local_ranks
    ), "A step is one invocation: launch it with WORLD_SIZE unset (or 1)"
    store = dist.TCPStore(
        args.store_host,
        args.store_port,
        is_master=False,
        timeout=timedelta(seconds=args.store_timeout_secs),
    )
    plan = json.loads(args.plan)
    start = args.start_phase
    # A namespace of this cohort alone (steps start concurrently and restart at the same ids)
    bootstrap_id = "adaptive-processes/" + json.dumps(
        [
            args.run_tag,
            args.scaleout_idx,
            args.position_offset,
            num_ranks,
            args.start_phase,
        ]
    )
    peer_prefix = f"{args.run_tag}EP_NIXL"
    t_init = pc()

    buffer = make_buffer(
        args,
        rank_idx,
        num_ranks,
        store,
        bootstrap_id=bootstrap_id,
        max_num_scaleout_ranks=args.max_scaleout_ranks,
        max_num_scaleup_ranks=args.max_scaleup_ranks,
        scaleout_rank_idx=args.scaleout_idx,
        scaleup_rank_idx=args.position_offset + local_rank,
    )
    t_buffer = pc()
    me = buffer.world_rank

    def barrier(key: str, count: int):
        # One key space per run.
        store_barrier(
            store, args.run_tag + key, count, args.store_timeout_secs, f"{me}"
        )

    max_so, max_su = buffer.max_num_scaleout_ranks, buffer.max_num_scaleup_ranks
    assert me in plan[start] and (
        start == 0 or me not in plan[start - 1]
    ), f"Rank {me} does not join at phase {start} of {plan}"
    buffer.publish(
        key_prefix=peer_prefix
    )  # other steps connect to us through the store
    t_publish = pc()
    startup = dict(
        init_ms=(t_init - t0) * 1e3,
        buffer_ms=(t_buffer - t_init) * 1e3,
        publish_ms=(t_publish - t_buffer) * 1e3,
    )
    mode = buffer.get_topology()[0]
    step_ids = (
        buffer.get_active_ranks()
    )  # the constructor made only this step's cohort the initial world
    assert mode == expected_mode(
        step_ids, max_su
    ), f"the constructor derived {mode} for {step_ids}"
    case = Case(buffer, args)
    validate_plan(case, plan)

    def log(msg: str):
        if local_rank == 0:
            print(f"[step {step_ids} host {os.uname().nodename}] {msg}", flush=True)

    log(
        f"joined at phase {start}: world rank {me} = ({buffer.scaleout_rank_idx}, {buffer.scaleup_rank_idx}) of "
        f"{max_so}x{max_su}, connected {buffer.get_connected_ranks()}, plan {plan}"
    )
    graph: Optional[GraphCase] = None

    for k in range(start, len(plan)):
        phase, prev = list(plan[k]), (list(plan[k - 1]) if k > 0 else [])
        joiners = sorted(set(phase) - set(prev))
        leavers = sorted(set(prev) - set(phase))
        want_mode = expected_mode(phase, max_su)
        role = "join" if me in joiners else ("leave" if me not in phase else "stay")
        ta = pc()
        buffer.barrier(with_cpu_sync=True)
        barrier(
            f"phase/{k}/arrive", len(set(prev) | set(phase))
        )  # everyone alive at the start of the phase
        tb = pc()

        if me not in phase:
            # Leaver: a one-rank world (in the mode the host derives for it; nothing launches on it), every connection
            # dropped, teardown, exit; the survivors drop us on their side without waiting
            buffer.set_active_ranks([me])
            tc = pc()
            buffer.disconnect_ranks(buffer.get_connected_ranks())
            buffer.commit_topology()
            td = pc()
            buffer.destroy()
            te = pc()
            print(
                f'PROC tag={args.run_tag or "run"} phase={k} rank={me} local={local_rank} role=leave t_entry={t_entry:.3f} '
                f"arrive_wait_ms={(tb - ta) * 1e3:.1f} shrink_commit_ms={(tc - tb) * 1e3:.2f} "
                f"disconnect_commit_ms={(td - tc) * 1e3:.2f} destroy_ms={(te - td) * 1e3:.1f}",
                flush=True,
            )
            log(f"left at phase {k}, process exits")
            return

        connected = set(buffer.get_connected_ranks())
        needed = (
            set(phase) if me in joiners else set(joiners)
        )  # joiners connect everyone, stayers the joiners
        to_connect = sorted(needed - {me} - connected)
        buffer.connect_ranks(
            to_connect, key_prefix=peer_prefix, timeout_secs=args.store_timeout_secs
        )
        tc = pc()
        barrier(f"phase/{k}/connected", len(phase))
        td = pc()

        ev0 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        buffer.stage_topology(active_ranks=phase)
        buffer.commit_topology()
        ev1 = torch.cuda.Event(enable_timing=True)
        ev1.record()
        te = pc()
        gone = sorted(set(leavers) & set(buffer.get_connected_ranks()))
        if gone:
            buffer.disconnect_ranks(gone)
            buffer.commit_topology()
        tf = pc()
        barrier(f"phase/{k}/live", len(phase))
        tg = pc()
        assert buffer.is_active and buffer.get_active_ranks() == sorted(
            phase
        ), f"active ranks {buffer.get_active_ranks()} != phase {phase}"
        phase_mode = buffer.get_topology()[0]
        assert (
            phase_mode == want_mode
        ), f"mode {phase_mode} on {phase}, expected {want_mode}"
        so_extent, su_extent = case.extent(phase)
        tag = (
            f"rank {me} {phase_mode} phase {k} {phase} (extent {so_extent}x{su_extent})"
        )

        # Joiners capture now, stayers run the warm-ups with them; every barrier goes through the store
        n_graph_barriers = [0]

        def graph_barrier():
            n_graph_barriers[0] += 1
            barrier(f"phase/{k}/graph/{n_graph_barriers[0]}", len(phase))

        captured = capture_phase_of(plan, k)
        t_first_kernel = (
            time.time()
        )  # the warm-up launches of the capture are the first kernels of this phase
        if me in joiners:
            graph = GraphCase(case, phase, captured, barrier_fn=graph_barrier)
            graph.capture()
        else:
            assert graph is not None
            graph.barrier = graph_barrier
            graph.capture_phase_of = captured
            if joiners:
                graph.capture(accompany=True)
        th = pc()

        num_recv = run_eager(case, k, phase, tag)
        ti = pc()
        num_recv_graph = graph.replay_and_check(phase, tag)
        tj = pc()
        torch.cuda.synchronize()
        print(
            f'PROC tag={args.run_tag or "run"} phase={k} rank={me} local={local_rank} role={role} mode={phase_mode} '
            f"live={len(phase)} t_entry={t_entry:.3f} t_first_kernel={t_first_kernel:.3f} "
            f'init_ms={startup["init_ms"]:.0f} buffer_ms={startup["buffer_ms"]:.0f} publish_ms={startup["publish_ms"]:.0f} '
            f"arrive_wait_ms={(tb - ta) * 1e3:.1f} num_connect={len(to_connect)} connect_ms={(tc - tb) * 1e3:.1f} "
            f"connected_wait_ms={(td - tc) * 1e3:.1f} stage_commit_ms={(te - td) * 1e3:.2f} commit_dev_ms={ev0.elapsed_time(ev1):.3f} "
            f"num_disconnect={len(gone)} disconnect_commit_ms={(tf - te) * 1e3:.2f} live_wait_ms={(tg - tf) * 1e3:.1f} "
            f"capture_ms={(th - tg) * 1e3:.0f} eager_ms={(ti - th) * 1e3:.0f} replay_ms={(tj - ti) * 1e3:.1f}",
            flush=True,
        )
        if me == min(phase):
            print(
                f" > [{tag}] joined {joiners}, left {leavers}; eager: {num_recv} tokens received, bitwise match; "
                f"graph replay (captured at phase {graph.capture_phase_of[me]}): {num_recv_graph} tokens received, bitwise match",
                flush=True,
            )
        barrier(f"phase/{k}/done", len(phase))

    log(f"plan finished at phase {len(plan) - 1}, process exits")
    buffer.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive EP kernels: processes that start later and processes that exit"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="JSON list of phases (lists of world ranks)",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        required=True,
        help="the phase this step joins at (not inferred: a rank that leaves and returns has two)",
    )
    parser.add_argument(
        "--scaleout-idx",
        type=int,
        default=0,
        help="the NVLink domain (scale-out index) of this step",
    )
    parser.add_argument(
        "--max-scaleout-ranks",
        type=int,
        default=1,
        help="declared maximum world: domains",
    )
    parser.add_argument(
        "--max-scaleup-ranks",
        type=int,
        default=4,
        help="declared maximum world: positions per domain",
    )
    parser.add_argument(
        "--store-host", type=str, default=os.environ.get("STORE_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--store-port", type=int, default=int(os.environ.get("STORE_PORT", "8400"))
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="prefix of the barrier keys (one per plan run sharing a store)",
    )
    parser.add_argument(
        "--num-cpu-timeout-secs",
        type=int,
        default=300,
        help="CPU-sync timeout of the eager dispatch",
    )
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        step_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
