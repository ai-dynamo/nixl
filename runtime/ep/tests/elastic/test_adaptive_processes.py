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
Processes that start later and processes that exit. Where `test_adaptive_elastic.py` keeps every process alive, here a
job is a set of independent STEPS: a step is one bootstrap cohort (`torch.multiprocessing.spawn` over the GPUs of one
node, or a subset), launched when its ranks first appear in the plan and exiting for good when they leave. Steps meet
only through an external `torch.distributed.TCPStore` (transport metadata via `ElasticBuffer.publish` /
`connect_ranks(store=...)`, phase barriers via counters, all under `--run-tag`); dense rank ids follow from the step's
coordinates (`--scaleout-idx`, `--position-offset`), so a later step may re-use the ids of one that exited.

Per phase k, every process alive at its start arrives at `phase/k/arrive`; a LEAVER makes itself a one-rank world,
drops every connection, destroys its buffer and exits without waiting for anyone; a STAYER or JOINER connects the
joiners (both sides call `connect_ranks`), commits the phase's world, drops the leavers with a second commit, runs one
eager dispatch + combine checked against the seed-regenerated routing of the live world and replays the graph its
process captured when it joined (joiners capture now, stayers run the warm-ups with them). One `PROC` line per process
and phase records the timeline of its join, stay or leave.

Launch: a TCPStore server runs for the whole job outside the steps (`--store-host` / `--store-port`, defaults
`STORE_HOST` / `STORE_PORT`, then 127.0.0.1:8400); a step is one invocation with `WORLD_SIZE` and `RANK` unset, started
by a launcher once phase k-1 is done (the counter `{run_tag}phase/{k-1}/done` at the size of phase k-1). Every step gets
the same plan and declared maximum world. Plans sharing one server use distinct `--run-tag`s, and a failed run is never
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
from typing import Any, Dict, List

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
)


def store_barrier(store: dist.Store, key: str, count: int, timeout_secs: float) -> None:
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
        ), f"Barrier {key}: {arrived}/{count} arrivals after {timeout_secs} s"
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

    def barrier(key: str, count: int) -> None:
        # One key space per run
        store_barrier(store, args.run_tag + key, count, args.store_timeout_secs)

    plan = json.loads(args.plan)
    me = args.scaleout_idx * args.max_scaleup_ranks + args.position_offset + local_rank
    start = args.start_phase
    assert me in plan[start] and (
        start == 0 or me not in plan[start - 1]
    ), f"Rank {me} does not join at phase {start} of {plan}"
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

    # `--num-experts` is the expert slot count of the declared maximum world; the slots per rank are fixed from it
    max_num_ranks = args.max_scaleout_ranks * args.max_scaleup_ranks
    assert (
        args.num_experts % max_num_ranks == 0
    ), f"{args.num_experts} experts do not divide by the {max_num_ranks} ranks of the declared world"
    buffer = make_buffer(
        args,
        rank_idx,
        num_ranks,
        store,
        bootstrap_id=bootstrap_id,
        max_num_scaleout_ranks=args.max_scaleout_ranks,
        max_num_scaleup_ranks=args.max_scaleup_ranks,
        num_experts_per_rank=args.num_experts // max_num_ranks,
        scaleout_rank_idx=args.scaleout_idx,
        scaleup_rank_idx=args.position_offset + local_rank,
    )
    assert me == buffer.rank_id
    t_buffer = pc()
    buffer.publish(
        key_prefix=peer_prefix
    )  # other steps connect to us through the store
    t_publish = pc()
    startup = dict(
        init_ms=(t_init - t0) * 1e3,
        buffer_ms=(t_buffer - t_init) * 1e3,
        publish_ms=(t_publish - t_buffer) * 1e3,
    )
    mode, num_active_sms = buffer.get_topology()
    step_ids = sorted(
        buffer.get_active_ranks()
    )  # the constructor made only this step's cohort the initial world
    assert mode == expected_mode(
        step_ids, args.max_scaleup_ranks
    ), f"the constructor derived {mode} for {step_ids}"
    for phase in plan:
        assert all(
            0 <= r < args.max_scaleout_ranks * args.max_scaleup_ranks for r in phase
        ) and len(set(phase)) == len(phase)
        buffer.masks_of(phase)  # product-set check

    def log(msg: str):
        if local_rank == 0:
            print(f"[step {step_ids} host {os.uname().nodename}] {msg}", flush=True)

    log(
        f"joined at phase {start}: rank id {me} = ({buffer.scaleout_rank_idx}, {buffer.scaleup_rank_idx}) of "
        f"{args.max_scaleout_ranks}x{args.max_scaleup_ranks}, connected {buffer.get_connected_ranks()}, plan {plan}"
    )
    case = Case(buffer, args)
    graph: Any = None

    for k in range(start, len(plan)):
        phase, prev = list(plan[k]), (list(plan[k - 1]) if k > 0 else [])
        joiners = sorted(set(phase) - set(prev))
        leavers = sorted(set(prev) - set(phase))
        # The host derives the mode at `stage_topology` (direct while exactly one domain is live, whatever its index,
        # hybrid otherwise); checked against the rule once committed
        want_mode = expected_mode(phase, args.max_scaleup_ranks)
        role = "join" if me in joiners else ("leave" if me not in phase else "stay")
        t_a = pc()
        barrier(
            f"phase/{k}/arrive", len(set(prev) | set(phase))
        )  # everyone alive at the start of the phase
        t_b = pc()

        if me not in phase:
            # Leaver: a one-rank world (in the mode the host derives for it; nothing launches on it), every connection
            # dropped, teardown, exit; the survivors drop us on their side without waiting
            buffer.stage_topology(num_active_sms=num_active_sms, active_ranks=[me])
            buffer.commit_topology()
            t_c = pc()
            buffer.disconnect_ranks(buffer.get_connected_ranks())
            buffer.commit_topology()
            t_d = pc()
            buffer.destroy()
            t_e = pc()
            print(
                f'PROC tag={args.run_tag or "run"} phase={k} rank={me} local={local_rank} role=leave t_entry={t_entry:.3f} '
                f"arrive_wait_ms={(t_b - t_a) * 1e3:.1f} shrink_commit_ms={(t_c - t_b) * 1e3:.2f} "
                f"disconnect_commit_ms={(t_d - t_c) * 1e3:.2f} destroy_ms={(t_e - t_d) * 1e3:.1f}",
                flush=True,
            )
            log(f"left at phase {k}, process exits")
            return

        # Stayer or joiner: joiners connect everyone, stayers connect the joiners (both sides must call it)
        connected = set(buffer.get_connected_ranks())
        to_connect = sorted(
            ((set(phase) if me in joiners else set(joiners)) - {me}) - connected
        )
        buffer.connect_ranks(
            to_connect,
            store=store,
            key_prefix=peer_prefix,
            timeout_secs=args.store_timeout_secs,
        )
        t_c = pc()
        barrier(f"phase/{k}/connected", len(phase))
        t_d = pc()

        ev0 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        buffer.stage_topology(num_active_sms=num_active_sms, active_ranks=phase)
        buffer.commit_topology()
        ev1 = torch.cuda.Event(enable_timing=True)
        ev1.record()
        t_e = pc()
        gone = sorted(set(leavers) & set(buffer.get_connected_ranks()))
        if gone:
            buffer.disconnect_ranks(gone)
            buffer.commit_topology()
        t_f = pc()
        barrier(f"phase/{k}/live", len(phase))
        t_g = pc()
        assert buffer.is_participant and buffer.get_active_ranks() == sorted(
            phase
        ), f"live world {buffer.get_active_ranks()} != phase {phase}"
        phase_mode = buffer.get_topology()[0]
        assert (
            phase_mode == want_mode
        ), f"mode {phase_mode} on {phase}, expected {want_mode}"
        so_bound, su_bound = case.bounds(phase)
        tag = f"{phase_mode} phase {k} {phase} (bounds {so_bound}x{su_bound})"

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
            graph.barrier = graph_barrier
            graph.capture_phase_of = captured
            if joiners:
                graph.capture(accompany=True)
        t_h = pc()

        num_recv = run_eager(case, k, phase, tag)
        t_i = pc()
        num_recv_graph = graph.replay_and_check(phase, tag)
        t_j = pc()
        torch.cuda.synchronize()
        print(
            f'PROC tag={args.run_tag or "run"} phase={k} rank={me} local={local_rank} role={role} mode={phase_mode} '
            f"live={len(phase)} t_entry={t_entry:.3f} t_first_kernel={t_first_kernel:.3f} "
            f'init_ms={startup["init_ms"]:.0f} buffer_ms={startup["buffer_ms"]:.0f} publish_ms={startup["publish_ms"]:.0f} '
            f"arrive_wait_ms={(t_b - t_a) * 1e3:.1f} num_connect={len(to_connect)} connect_ms={(t_c - t_b) * 1e3:.1f} "
            f"connected_wait_ms={(t_d - t_c) * 1e3:.1f} stage_commit_ms={(t_e - t_d) * 1e3:.2f} commit_dev_ms={ev0.elapsed_time(ev1):.3f} "
            f"num_disconnect={len(gone)} disconnect_commit_ms={(t_f - t_e) * 1e3:.2f} live_wait_ms={(t_g - t_f) * 1e3:.1f} "
            f"capture_ms={(t_h - t_g) * 1e3:.0f} eager_ms={(t_i - t_h) * 1e3:.0f} replay_ms={(t_j - t_i) * 1e3:.1f}",
            flush=True,
        )
        if me == min(phase):
            print(
                f" > [{tag}] joined {joiners}, left {leavers}; eager: {num_recv} tokens received, bitwise match; "
                f"graph replay (captured at phase {graph.captured_at}): {num_recv_graph} tokens received, bitwise match",
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
        help="JSON list of phases (lists of dense rank ids)",
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
