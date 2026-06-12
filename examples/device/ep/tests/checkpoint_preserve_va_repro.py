# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NIXL EP checkpoint/restore VA-stability reproducer.

This script is intentionally small enough to run in two Kubernetes pods while
still exercising the NIXL EP graph-stable checkpoint API:

1. initialize a NIXL EP buffer and connect all ranks;
2. record graph-visible CUDA virtual addresses;
3. optionally capture a CUDA graph that launches a NIXL EP barrier;
4. run checkpoint_pause_preserve_va();
5. signal Dynamo snapshot-agent readiness via /snapshot-control;
6. after CRIU restore, run checkpoint_resume_preserve_va() with fresh store
   metadata, validate addresses, and replay the captured graph.

It can also be run without snapshot-agent by passing --no-snapshot-control; in
that mode pause/resume happens in-process and validates the NIXL EP APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import nixl_ep


READY_FOR_CHECKPOINT = "ready-for-checkpoint"
SNAPSHOT_COMPLETE = "snapshot-complete"
RESTORE_COMPLETE = "restore-complete"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _control_dir() -> Path:
    return Path(os.environ.get("DYN_SNAPSHOT_CONTROL_DIR", "/snapshot-control"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time()) + "\n")


def _wait_for_file(path: Path, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            print(f"[snapshot] observed {label}: {path}", flush=True)
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {label}: {path}")


def _init_dist(args: argparse.Namespace) -> tuple[int, int, int]:
    rank = _env_int("RANK", args.rank)
    world_size = _env_int("WORLD_SIZE", args.world_size)
    local_rank = _env_int("LOCAL_RANK", args.local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size, local_rank


def _make_store(args: argparse.Namespace, rank: int, world_size: int) -> dist.TCPStore:
    is_master = rank == 0
    return dist.TCPStore(
        args.store_host,
        args.store_port,
        world_size,
        is_master,
        timeout=torch.distributed.constants.default_pg_timeout,
        wait_for_workers=True,
    )


def _capture_barrier_graph(buffer: nixl_ep.Buffer) -> torch.cuda.CUDAGraph | None:
    if not torch.cuda.is_available():
        return None

    # Warm up once before capture so the captured graph contains the steady-state
    # NIXL EP barrier kernel launch with graph-visible gpu_ctx_ptr.
    buffer.barrier()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.capture_begin()
        buffer.barrier()
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


def _run_barrier_iterations(
    buffer: nixl_ep.Buffer,
    graph: torch.cuda.CUDAGraph | None,
    iterations: int,
    label: str,
) -> None:
    for i in range(iterations):
        if graph is not None:
            graph.replay()
        else:
            buffer.barrier()
        torch.cuda.synchronize()
        print(f"[{label}] barrier iteration {i + 1}/{iterations} ok", flush=True)


def _make_buffer(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    store: dist.TCPStore,
) -> nixl_ep.Buffer:
    buffer = nixl_ep.Buffer(
        rank=rank,
        low_latency_mode=True,
        explicitly_destroy=True,
        group=None,
        tcp_store_group=store,
        timeout_ms=args.timeout_ms,
    )
    rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        num_max_dispatch_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        num_ranks=world_size,
        num_experts=args.num_experts_per_rank * world_size,
    )
    print(f"[rank {rank}] rdma_bytes={rdma_bytes}", flush=True)
    buffer.update_memory_buffers(
        num_ranks=world_size,
        num_experts_per_rank=args.num_experts_per_rank,
        num_rdma_bytes=rdma_bytes,
    )
    buffer.connect_ranks(list(range(world_size)))
    return buffer


def _check_all_ranks_ok(rank: int, world_size: int, device: torch.device) -> None:
    flag = torch.ones(1, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if int(flag.item()) != world_size:
        raise RuntimeError(
            f"rank {rank}: expected all ranks ok sum {world_size}, "
            f"got {int(flag.item())}"
        )


def run(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = _init_dist(args)
    device = torch.device("cuda", local_rank)
    store = _make_store(args, rank, world_size)
    buffer = _make_buffer(args, rank, world_size, store)

    try:
        remote_ranks = [r for r in range(world_size) if r != rank]
        baseline = buffer.get_graph_visible_addresses()
        print(
            f"[rank {rank}] baseline_addresses="
            f"{json.dumps(baseline, sort_keys=True)}",
            flush=True,
        )

        graph = _capture_barrier_graph(buffer) if args.capture_graph else None
        _run_barrier_iterations(buffer, graph, args.pre_iterations, "pre")
        _check_all_ranks_ok(rank, world_size, device)

        expected = buffer.checkpoint_pause_preserve_va()
        print(
            f"[rank {rank}] paused_addresses="
            f"{json.dumps(expected, sort_keys=True)}",
            flush=True,
        )
        if expected != baseline:
            raise RuntimeError(
                "checkpoint_pause_preserve_va returned unexpected addresses"
            )

        output_dir = Path(args.output_dir)
        _write_json(output_dir / f"rank-{rank}-addresses.json", expected)

        control_dir = _control_dir()
        if args.no_snapshot_control:
            print("[snapshot] --no-snapshot-control: resuming in-process", flush=True)
        else:
            _touch(control_dir / READY_FOR_CHECKPOINT)
            print(f"[snapshot] wrote {control_dir / READY_FOR_CHECKPOINT}", flush=True)
            _wait_for_file(
                control_dir / SNAPSHOT_COMPLETE,
                args.snapshot_timeout_s,
                SNAPSHOT_COMPLETE,
            )
            _wait_for_file(
                control_dir / RESTORE_COMPLETE,
                args.snapshot_timeout_s,
                RESTORE_COMPLETE,
            )

        dist.barrier()
        buffer.checkpoint_resume_preserve_va(
            remote_ranks,
            activate=True,
            expected_addresses=expected,
        )
        dist.barrier()

        if not buffer.validate_graph_visible_addresses(expected):
            raise RuntimeError("graph-visible addresses changed after resume")
        _write_json(
            output_dir / f"rank-{rank}-addresses-after-resume.json",
            buffer.get_graph_visible_addresses(),
        )

        _run_barrier_iterations(buffer, graph, args.post_iterations, "post")
        _check_all_ranks_ok(rank, world_size, device)

        print(f"PASS rank={rank} world_size={world_size}", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist-backend", default="gloo")
    parser.add_argument("--store-host", required=True)
    parser.add_argument("--store-port", type=int, default=9999)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--num-experts-per-rank", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--pre-iterations", type=int, default=2)
    parser.add_argument("--post-iterations", type=int, default=2)
    parser.add_argument(
        "--capture-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-snapshot-control", action="store_true")
    parser.add_argument("--snapshot-timeout-s", type=float, default=900.0)
    parser.add_argument("--output-dir", default="/tmp/nixl-ep-cr")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        raise
