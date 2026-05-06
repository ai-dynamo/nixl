# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import nixl_ep
import torch

# Add tests directory to path to import shared utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import rank_server, store_group  # noqa: E402
from utils.utils import CudaTimer, stats, tcp_store_barrier  # noqa: E402

TCP_STORE_PORT = 9999
RANK_SERVER_PORT = 10000

# Delay between disconnect and reconnect of same ranks (MD invalidation race)
MD_INVALIDATION_DELAY = 5

CYCLE_OPS = ("init", "connect", "disconnect", "reconnect", "destroy")


@dataclass
class BufferConfig:
    rank: int
    disable_ll_nvlink: bool
    tcp_store: object
    num_ranks: int
    num_experts_per_rank: int
    num_rdma_bytes: int


def handle_sigterm(signum, frame, rank_client):
    print(
        f"SIGTERM ({signum}) received for process {os.getpid()}! "
        f"releasing rank and exiting...",
        flush=True,
    )
    rank_client.release_rank()
    sys.exit(1)


def timed_op(fn, guard=True):
    if not guard:
        return 0.0
    with CudaTimer() as t:
        fn()
    return t.elapsed_s


def create_buffer(cfg: BufferConfig):
    buf = nixl_ep.Buffer(
        rank=cfg.rank,
        disable_ll_nvlink=cfg.disable_ll_nvlink,
        explicitly_destroy=True,
        tcp_store_group=cfg.tcp_store,
    )
    buf.update_memory_buffers(
        num_ranks=cfg.num_ranks,
        num_experts_per_rank=cfg.num_experts_per_rank,
        num_rdma_bytes=cfg.num_rdma_bytes,
    )
    return buf


def bench_init(cfg: BufferConfig, buf_out: list):
    with CudaTimer() as t:
        buf_out[0] = create_buffer(cfg)
    return t.elapsed_s


def measure_loop(
    warmup: int,
    iters: int,
    barrier_fn: Callable,
    bench_fn: Callable,
    setup_fn: Optional[Callable] = None,
    teardown_fn: Optional[Callable] = None,
):
    latencies = []
    for i in range(warmup + iters):
        if setup_fn:
            setup_fn()
        barrier_fn()
        elapsed = bench_fn()
        if i >= warmup:
            latencies.append(elapsed)
        if teardown_fn:
            teardown_fn()
    return latencies


def run_cycle(cfg: BufferConfig, other_ranks: list, warmup: int, iters: int):
    def barrier():
        tcp_store_barrier(cfg.tcp_store, cfg.rank, cfg.num_ranks)

    buf: list = [None]

    steps = [
        ("init", lambda: bench_init(cfg, buf), None),
        (
            "connect",
            lambda: timed_op(
                lambda: buf[0].connect_ranks(other_ranks), guard=other_ranks
            ),
            None,
        ),
        (
            "disconnect",
            lambda: timed_op(
                lambda: buf[0].disconnect_ranks(other_ranks), guard=other_ranks
            ),
            lambda: time.sleep(MD_INVALIDATION_DELAY),
        ),
        (
            "reconnect",
            lambda: timed_op(
                lambda: buf[0].connect_ranks(other_ranks), guard=other_ranks
            ),
            None,
        ),
        ("destroy", lambda: timed_op(buf[0].destroy), None),
    ]

    results: dict[str, list[float]] = {op_name: [] for op_name, _, _ in steps}

    for i in range(warmup + iters):
        is_measure = i >= warmup
        for op_name, bench_fn, post_fn in steps:
            barrier()
            elapsed = bench_fn()
            if is_measure:
                results[op_name].append(elapsed)
            if post_fn:
                post_fn()

    return {op_name: stats(times) for op_name, times in results.items()}


def run_single_op(
    mode: str, cfg: BufferConfig, other_ranks: list, warmup: int, iters: int
):
    def barrier():
        tcp_store_barrier(cfg.tcp_store, cfg.rank, cfg.num_ranks)

    latencies = []

    if mode == "init":
        buf: list = [None]

        latencies = measure_loop(
            warmup,
            iters,
            barrier,
            bench_fn=lambda: bench_init(cfg, buf),
            teardown_fn=lambda: buf[0].destroy(),
        )

    elif mode == "connect":
        buf = [None]

        def setup():
            buf[0] = create_buffer(cfg)

        latencies = measure_loop(
            warmup,
            iters,
            barrier,
            bench_fn=lambda: timed_op(
                lambda: buf[0].connect_ranks(other_ranks), guard=other_ranks
            ),
            setup_fn=setup,
            teardown_fn=lambda: buf[0].destroy(),
        )

    elif mode == "disconnect":
        buffer = create_buffer(cfg)
        latencies = measure_loop(
            warmup,
            iters,
            barrier,
            bench_fn=lambda: timed_op(
                lambda: buffer.disconnect_ranks(other_ranks), guard=other_ranks
            ),
            setup_fn=lambda: buffer.connect_ranks(other_ranks) if other_ranks else None,
            teardown_fn=lambda: time.sleep(MD_INVALIDATION_DELAY),
        )
        buffer.destroy()

    elif mode == "reconnect":
        buffer = create_buffer(cfg)
        if other_ranks:
            buffer.connect_ranks(other_ranks)

        def reconnect_setup():
            if other_ranks:
                buffer.disconnect_ranks(other_ranks)
            time.sleep(MD_INVALIDATION_DELAY)

        latencies = measure_loop(
            warmup,
            iters,
            barrier,
            bench_fn=lambda: timed_op(
                lambda: buffer.connect_ranks(other_ranks), guard=other_ranks
            ),
            setup_fn=reconnect_setup,
        )
        buffer.destroy()

    elif mode == "destroy":
        buf = [None]

        def destroy_setup():
            buf[0] = create_buffer(cfg)
            if other_ranks:
                buf[0].connect_ranks(other_ranks)

        latencies = measure_loop(
            warmup,
            iters,
            barrier,
            bench_fn=lambda: timed_op(buf[0].destroy),
            setup_fn=destroy_setup,
        )

    return {mode: stats(latencies)}


def worker(torch_rank: int, args: argparse.Namespace):
    server_addr = args.tcp_server if args.tcp_server else "127.0.0.1"
    rank_client = rank_server.RankClient(server_addr, RANK_SERVER_PORT)
    local_rank, global_rank, _ = rank_client.get_rank()
    num_ranks = args.num_ranks

    print(
        f"Process {torch_rank} -> global_rank={global_rank}, local_rank={local_rank}",
        flush=True,
    )

    signal.signal(
        signal.SIGTERM,
        partial(handle_sigterm, rank_client=rank_client),
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    tcp_store = store_group.create_client_store(
        master_addr=server_addr,
        port=TCP_STORE_PORT,
    )

    num_experts = args.num_experts_per_rank * num_ranks
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens,
        args.hidden_dim,
        num_ranks,
        num_experts,
    )
    if local_rank == 0:
        print(f"Allocating buffer size: {num_rdma_bytes / 1e6} MB ...", flush=True)

    cfg = BufferConfig(
        rank=global_rank,
        disable_ll_nvlink=args.disable_ll_nvlink,
        tcp_store=tcp_store,
        num_ranks=num_ranks,
        num_experts_per_rank=args.num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
    )
    other_ranks = [r for r in range(num_ranks) if r != global_rank]

    common_kwargs = {
        "cfg": cfg,
        "other_ranks": other_ranks,
        "warmup": args.warmup,
        "iters": args.iters,
    }

    is_cycle = args.mode == "cycle"

    if is_cycle:
        results = run_cycle(**common_kwargs)
    else:
        results = run_single_op(mode=args.mode, **common_kwargs)

    ops = list(results.keys())

    print(f"[rank {global_rank}] mode={args.mode}:", flush=True)
    for op in ops:
        avg_t, min_t, max_t = results[op]
        print(
            f"[rank {global_rank}]   {op:12s}: "
            f"avg_t={avg_t * 1e3:.2f} ms, "
            f"min_t={min_t * 1e3:.2f} ms, "
            f"max_t={max_t * 1e3:.2f} ms",
            flush=True,
        )
    if is_cycle:
        # Sum of per-op averages = average total cycle time
        total_avg = sum(v[0] for v in results.values())
        print(
            f"[rank {global_rank}]   {'total':12s}: " f"avg_t={total_avg * 1e3:.2f} ms",
            flush=True,
        )

    for op in ops:
        tcp_store.set(f"result/{global_rank}/{op}", str(results[op][0]))

    print(f"global_rank={global_rank}, local_rank={local_rank} -> done", flush=True)

    tcp_store_barrier(tcp_store, global_rank, num_ranks)

    # Rank 0 collects and prints cross-rank averages
    if global_rank == 0:
        cross_total = 0.0
        print("Cross-rank average:", flush=True)
        for op in ops:
            vals = [float(tcp_store.get(f"result/{r}/{op}")) for r in range(num_ranks)]
            cross_avg = sum(vals) / len(vals)
            cross_total += cross_avg
            print(f"  {op:12s}: avg_t={cross_avg * 1e3:.2f} ms", flush=True)
        if is_cycle:
            print(
                f"  {'total':12s}: avg_t={cross_total * 1e3:.2f} ms",
                flush=True,
            )


def run_server():
    _store = store_group.create_master_store(port=TCP_STORE_PORT)  # noqa: F841
    rank_server.start_server(port=RANK_SERVER_PORT)


def main():
    parser = argparse.ArgumentParser(description="Control Plane Latency Test")
    parser.add_argument(
        "--mode",
        type=str,
        default="cycle",
        choices=["cycle", *CYCLE_OPS],
        help="Operation to benchmark (default: cycle)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of worker processes to launch",
    )
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=None,
        help="Total number of ranks across all nodes "
        "(default: same as --num-processes)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=128,
        help="Number of tokens (for buffer sizing)",
    )
    parser.add_argument(
        "--num-experts-per-rank",
        type=int,
        default=2,
        help="Number of experts per rank (for buffer sizing)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=7168,
        help="Hidden dimension (for buffer sizing)",
    )
    parser.add_argument(
        "--tcp-server",
        type=str,
        help="TCP server address (for both TCPStore and rank server). "
        "If not set, both will be started locally.",
    )
    parser.add_argument(
        "--disable-ll-nvlink",
        action="store_true",
        help="Disable NVLink communication for low-latency kernels",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup iterations before measurement",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Measurement iterations",
    )

    args = parser.parse_args()

    if args.num_ranks is None:
        args.num_ranks = args.num_processes

    if not args.tcp_server:
        print("Starting TCPStore and rank server locally", flush=True)
        server_process = torch.multiprocessing.Process(target=run_server, daemon=True)
        server_process.start()
        time.sleep(0.5)

    if args.num_processes == 1:
        worker(0, args)
        return

    ctx = torch.multiprocessing.spawn(
        worker,
        args=(args,),
        nprocs=args.num_processes,
        join=False,
        daemon=False,
        start_method="spawn",
    )

    failed = []
    for i, p in enumerate(ctx.processes):
        p.join()
        if p.exitcode != 0:
            failed.append((i, p.exitcode))
    if failed:
        raise RuntimeError(
            f"Worker processes failed: "
            f"{', '.join(f'worker {i} (exit code {code})' for i, code in failed)}"
        )


if __name__ == "__main__":
    main()
