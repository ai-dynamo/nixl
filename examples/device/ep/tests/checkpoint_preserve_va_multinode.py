# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-node NIXL EP graph-stable checkpoint/restart reproducer.

This script is intentionally standalone so it can run inside Kubernetes pods or
other containers that already contain the built ``nixl_ep`` extension. Launch
one process per EP rank, for example:

    torchrun --nnodes 2 --nproc-per-node 1 \
        --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} \
        --master-port 29500 \
        examples/device/ep/tests/checkpoint_preserve_va_multinode.py \
        --store-master-addr ${MASTER_ADDR}

The script validates the direct NIXL EP Buffer path used by vLLM's
NixlEPAll2AllManager: VMM-backed graph-visible buffers are allocated, a
representative low-latency dispatch/combine is optionally captured in a CUDA
graph, graph-visible virtual addresses are recorded, checkpoint pause releases
non-checkpointable NIXL state while preserving those VAs, and resume rebuilds
fresh NIXL metadata before replaying the captured graph without recapture.
CUDA graph capture uses NIXL EP receive-hook mode so send and receive kernels
are launched on the capture stream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

EP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EP_ROOT))

import nixl_ep  # noqa: E402


DEFAULT_STORE_PORT = 9999


@dataclass(frozen=True)
class RankEnv:
    rank: int
    world_size: int
    local_rank: int


@dataclass
class IterationState:
    recv_x: torch.Tensor
    recv_count: torch.Tensor
    handle: tuple[Any, ...]
    combined_x: torch.Tensor


@dataclass
class CapturedIteration:
    graph: torch.cuda.CUDAGraph
    state: IterationState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate NIXL EP checkpoint_pause_preserve_va/"
            "checkpoint_resume_preserve_va across real ranks."
        )
    )
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-experts-per-rank", type=int, default=1)
    parser.add_argument("--num-topk", type=int, default=1)
    parser.add_argument("--capture-warmups", type=int, default=3)
    parser.add_argument("--skip-cuda-graph", action="store_true")
    parser.add_argument(
        "--allow-eager-fallback",
        action="store_true",
        help=(
            "If CUDA graph capture fails, continue with eager validation. "
            "By default graph capture errors fail the run."
        ),
    )
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--store-master-addr", default=os.getenv("MASTER_ADDR"))
    parser.add_argument("--store-port", type=int, default=DEFAULT_STORE_PORT)
    parser.add_argument(
        "--external-store",
        action="store_true",
        help=(
            "Connect all ranks to an already-running TCPStore instead of "
            "creating the TCPStore in rank 0."
        ),
    )
    parser.add_argument(
        "--barrier-timeout-sec",
        type=float,
        default=300.0,
        help="Timeout for TCPStore coordination barriers.",
    )
    parser.add_argument(
        "--hook-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for external checkpoint hooks. Each rank writes "
            "rank_<rank>.paused.json after checkpoint pause."
        ),
    )
    parser.add_argument(
        "--wait-for-resume-file",
        action="store_true",
        help=(
            "After writing the paused hook file, wait for "
            "rank_<rank>.resume before calling checkpoint_resume_preserve_va()."
        ),
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional sleep after checkpoint pause for external orchestration.",
    )
    return parser.parse_args()


def read_rank_env() -> RankEnv:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return RankEnv(rank=rank, world_size=world_size, local_rank=local_rank)


def create_store(args: argparse.Namespace, env: RankEnv) -> dist.TCPStore:
    master_addr = args.store_master_addr
    if master_addr is None:
        raise ValueError(
            "--store-master-addr or MASTER_ADDR is required for multi-rank runs"
        )

    is_master = env.rank == 0 and not args.external_store
    host_name = "0.0.0.0" if is_master else master_addr
    return dist.TCPStore(
        host_name=host_name,
        port=args.store_port,
        is_master=is_master,
        wait_for_workers=False,
        timeout=timedelta(seconds=args.barrier_timeout_sec),
    )


def store_barrier(
    store: dist.TCPStore,
    env: RankEnv,
    name: str,
    timeout_sec: float,
) -> None:
    prefix = f"checkpoint_preserve_va/{name}"
    store.set(f"{prefix}/{env.rank}", str(time.time()).encode())
    store.wait(
        [f"{prefix}/{rank}" for rank in range(env.world_size)],
        timedelta(seconds=timeout_sec),
    )


def gather_store_values(
    store: dist.TCPStore,
    env: RankEnv,
    name: str,
    value: str,
    timeout_sec: float,
) -> list[str]:
    prefix = f"checkpoint_preserve_va/{name}"
    store.set(f"{prefix}/{env.rank}", value.encode())
    keys = [f"{prefix}/{rank}" for rank in range(env.world_size)]
    store.wait(keys, timedelta(seconds=timeout_sec))
    return [raw.decode() for raw in store.multi_get(keys)]


def make_inputs(
    env: RankEnv,
    num_tokens: int,
    hidden_dim: int,
    num_experts_per_rank: int,
    num_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = env.world_size * num_experts_per_rank
    if num_topk != 1:
        raise ValueError("This focused reproducer currently expects --num-topk 1")
    if hidden_dim not in (2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192):
        raise ValueError(f"Unsupported NIXL EP hidden dimension: {hidden_dim}")
    if hidden_dim % 128 != 0:
        raise ValueError("--hidden-dim must be a multiple of 128")
    if (env.world_size * num_tokens) % 4 != 0:
        raise ValueError(
            "--num-tokens * WORLD_SIZE must be divisible by 4 for NIXL EP"
        )

    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    x = torch.full(
        (num_tokens, hidden_dim),
        float(env.rank + 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    x[:, -1] = (token_ids + 1).to(torch.bfloat16)

    owner_rank = token_ids % env.world_size
    expert_offset = (token_ids // env.world_size) % num_experts_per_rank
    topk_idx = (owner_rank * num_experts_per_rank + expert_offset).view(
        num_tokens, num_topk
    )
    topk_idx = topk_idx.to(nixl_ep.topk_idx_t).contiguous()
    topk_weights = torch.ones(
        (num_tokens, num_topk), device="cuda", dtype=torch.float32
    )
    expected = x.clone()

    assert int(topk_idx.max().item()) < num_experts
    return x, topk_idx, topk_weights, expected


def run_iteration(
    buffer: nixl_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    use_recv_hooks: bool = False,
) -> IterationState:
    recv_x, recv_count, handle, _event, recv_hook = buffer.dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=False,
        async_finish=False,
        return_recv_hook=use_recv_hooks,
    )
    if recv_hook is not None:
        recv_hook()
    combined_x, _event, recv_hook = buffer.combine(
        recv_x,
        topk_idx,
        topk_weights,
        handle,
        async_finish=False,
        return_recv_hook=use_recv_hooks,
    )
    if recv_hook is not None:
        recv_hook()
    return IterationState(
        recv_x=recv_x,
        recv_count=recv_count,
        handle=handle,
        combined_x=combined_x,
    )


def assert_expected(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    torch.cuda.synchronize()
    if not torch.equal(actual, expected):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{label} mismatch: max_abs_diff={max_diff}")


def warmup_for_capture(
    args: argparse.Namespace,
    env: RankEnv,
    store: dist.TCPStore,
    buffer: nixl_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    store_barrier(store, env, "capture_warmup_start", args.barrier_timeout_sec)
    with torch.cuda.stream(stream):
        for _ in range(args.capture_warmups):
            state = run_iteration(
                buffer,
                x,
                topk_idx,
                topk_weights,
                args.num_tokens,
                num_experts,
                use_recv_hooks=True,
            )
            state.combined_x.record_stream(stream)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    store_barrier(store, env, "capture_warmup_done", args.barrier_timeout_sec)


def capture_iteration(
    args: argparse.Namespace,
    env: RankEnv,
    store: dist.TCPStore,
    buffer: nixl_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> CapturedIteration:
    warmup_for_capture(args, env, store, buffer, x, topk_idx, topk_weights, num_experts)
    graph = torch.cuda.CUDAGraph()
    store_barrier(store, env, "capture_start", args.barrier_timeout_sec)
    with torch.cuda.graph(graph):
        state = run_iteration(
            buffer,
            x,
            topk_idx,
            topk_weights,
            args.num_tokens,
            num_experts,
            use_recv_hooks=True,
        )
    torch.cuda.synchronize()
    return CapturedIteration(graph=graph, state=state)


def write_pause_hook(
    args: argparse.Namespace,
    env: RankEnv,
    addresses: dict[str, int],
) -> None:
    if args.hook_dir is None:
        return
    args.hook_dir.mkdir(parents=True, exist_ok=True)
    pause_file = args.hook_dir / f"rank_{env.rank}.paused.json"
    pause_file.write_text(
        json.dumps(
            {
                "rank": env.rank,
                "world_size": env.world_size,
                "addresses": addresses,
            },
            indent=2,
            sort_keys=True,
        )
    )


def wait_for_external_resume(args: argparse.Namespace, env: RankEnv) -> None:
    if args.pause_seconds > 0:
        time.sleep(args.pause_seconds)

    if not args.wait_for_resume_file:
        return
    if args.hook_dir is None:
        raise ValueError("--wait-for-resume-file requires --hook-dir")

    resume_file = args.hook_dir / f"rank_{env.rank}.resume"
    print(
        f"[rank {env.rank}] waiting for external resume file {resume_file}",
        flush=True,
    )
    while not resume_file.exists():
        time.sleep(0.5)


def main() -> None:
    args = parse_args()
    env = read_rank_env()
    if env.world_size < 2:
        raise ValueError("Real multi-rank validation requires WORLD_SIZE >= 2")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(env.local_rank)

    store = create_store(args, env)
    store_barrier(store, env, "store_ready", args.barrier_timeout_sec)

    num_experts = env.world_size * args.num_experts_per_rank
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens,
        args.hidden_dim,
        env.world_size,
        num_experts,
    )
    buffer = nixl_ep.Buffer(
        rank=env.rank,
        explicitly_destroy=True,
        tcp_store_group=store,
        timeout_ms=args.timeout_ms,
    )
    try:
        buffer.update_memory_buffers(
            num_ranks=env.world_size,
            num_experts_per_rank=args.num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )
        all_ranks = list(range(env.world_size))
        buffer.connect_ranks(all_ranks)
        store_barrier(store, env, "connected", args.barrier_timeout_sec)

        x, topk_idx, topk_weights, expected = make_inputs(
            env,
            args.num_tokens,
            args.hidden_dim,
            args.num_experts_per_rank,
            args.num_topk,
        )

        eager_state = run_iteration(
            buffer,
            x,
            topk_idx,
            topk_weights,
            args.num_tokens,
            num_experts,
        )
        assert_expected("eager pre-pause", eager_state.combined_x, expected)
        store_barrier(store, env, "eager_pre_pause_done", args.barrier_timeout_sec)

        captured: CapturedIteration | None = None
        graph_capture_error: str | None = None
        if not args.skip_cuda_graph:
            try:
                captured = capture_iteration(
                    args,
                    env,
                    store,
                    buffer,
                    x,
                    topk_idx,
                    topk_weights,
                    num_experts,
                )
            except Exception as exc:
                graph_capture_error = repr(exc)

            statuses = gather_store_values(
                store,
                env,
                "graph_capture_status",
                "ok" if graph_capture_error is None else graph_capture_error,
                args.barrier_timeout_sec,
            )
            graph_errors = [status for status in statuses if status != "ok"]
            if graph_errors:
                graph_capture_error = graph_capture_error or graph_errors[0]
                captured = None
                if not args.allow_eager_fallback:
                    raise RuntimeError(
                        "CUDA graph capture failed on at least one rank: "
                        f"{graph_errors}"
                    )
                print(
                    f"[rank {env.rank}] CUDA graph capture failed on at least "
                    "one rank; continuing eager-only because "
                    f"--allow-eager-fallback was set: {graph_errors}",
                    flush=True,
                )
            else:
                store_barrier(
                    store,
                    env,
                    "graph_pre_pause_replay_start",
                    args.barrier_timeout_sec,
                )
                assert captured is not None
                captured.graph.replay()
                assert_expected(
                    "cuda graph pre-pause replay",
                    captured.state.combined_x,
                    expected,
                )
                store_barrier(
                    store,
                    env,
                    "graph_pre_pause_replay_done",
                    args.barrier_timeout_sec,
                )

        addresses_before_pause = buffer.get_graph_visible_addresses()
        pause_snapshot = buffer.checkpoint_pause_preserve_va()
        if pause_snapshot != addresses_before_pause:
            raise AssertionError(
                "checkpoint_pause_preserve_va returned a different address "
                "snapshot than get_graph_visible_addresses()"
            )
        write_pause_hook(args, env, pause_snapshot)
        store_barrier(store, env, "paused", args.barrier_timeout_sec)

        if env.rank == 0:
            print(
                "[rank 0] all ranks called checkpoint_pause_preserve_va(); "
                "external CRIU/Dynamo checkpoint can be taken now",
                flush=True,
            )
        wait_for_external_resume(args, env)
        store_barrier(store, env, "resume_start", args.barrier_timeout_sec)

        buffer.set_tcp_store_group(store)
        buffer.checkpoint_resume_preserve_va(
            all_ranks,
            activate=False,
            expected_addresses=pause_snapshot,
        )
        for rank in all_ranks:
            buffer.update_mask_buffer(rank, mask=False)
        if not buffer.validate_graph_visible_addresses(pause_snapshot):
            raise AssertionError("graph-visible CUDA VAs changed after resume")
        store_barrier(store, env, "resumed", args.barrier_timeout_sec)

        if captured is not None:
            store_barrier(
                store,
                env,
                "graph_post_resume_replay_start",
                args.barrier_timeout_sec,
            )
            captured.graph.replay()
            assert_expected(
                "cuda graph post-resume replay without recapture",
                captured.state.combined_x,
                expected,
            )
            validation_mode = "cuda_graph_replay"
        else:
            store_barrier(
                store,
                env,
                "eager_post_resume_start",
                args.barrier_timeout_sec,
            )
            post_resume_state = run_iteration(
                buffer,
                x,
                topk_idx,
                topk_weights,
                args.num_tokens,
                num_experts,
            )
            assert_expected(
                "eager post-resume",
                post_resume_state.combined_x,
                expected,
            )
            validation_mode = "eager"

        store_barrier(store, env, "post_resume_done", args.barrier_timeout_sec)
        summary = {
            "rank": env.rank,
            "world_size": env.world_size,
            "num_rdma_bytes": num_rdma_bytes,
            "validation_mode": validation_mode,
            "graph_capture_error": graph_capture_error,
            "addresses": pause_snapshot,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        buffer.destroy()


if __name__ == "__main__":
    main()
