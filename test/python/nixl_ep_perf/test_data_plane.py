# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data plane performance test for NIXL EP Buffer.

Measures throughput and latency of dispatch/combine operations.

Usage:
    # Dispatch only (measure dispatch BW/latency)
    python3 test_data_plane.py --num-processes=8 --mode=dispatch

    # Combine only (one dispatch, many combines)
    python3 test_data_plane.py --num-processes=8 --mode=combine

    # End-to-end (dispatch + combine cycles)
    python3 test_data_plane.py --num-processes=8 --mode=e2e

    # Custom configuration
    python3 test_data_plane.py --num-processes=8 --mode=e2e \
        --tokens=2048 --hidden=7168 --experts-per-rank=32 --topk=8
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

from mp_runner import TestResult, run_multiprocess_test, sync_all_ranks

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Import store_group for TCPStore support
try:
    from store_group import store_group
except ImportError:
    store_group = None

# Defaults
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 100


def _run_data_plane_test(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    mode: str = "e2e",
    num_experts_per_rank: int = 8,
    num_tokens: int = 512,
    hidden: int = 4096,
    topk: int = 2,
    nvlink_backend: str = "ipc",
    warmup_iters: int = DEFAULT_WARMUP,
    measure_iters: int = DEFAULT_ITERS,
    use_tcp_store: bool = False,
    node_rank: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run data plane performance test.

    Args:
        mode: "dispatch" (only dispatch), "combine" (1 dispatch + N combines),
              or "e2e" (N dispatch+combine cycles)
        use_tcp_store: Use TCPStore for metadata exchange instead of etcd
        node_rank: Node rank for log message prefix
    """
    import nixl_ep
    import numpy as np
    import torch
    import torch.distributed as dist

    # Configure logger with node prefix
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(f"[Node {node_rank}] %(message)s"))

    total_experts = num_experts_per_rank * world_size

    # Setup TCPStore if requested
    tcp_store = None
    if use_tcp_store and store_group is not None:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        tcp_store_port = kwargs.get("tcp_store_port", 9999)
        tcp_store = store_group.create_client_store(
            master_addr=master_addr,
            port=tcp_store_port,
            timeout_sec=60.0,
        )

    # Create buffer
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        num_tokens, hidden, world_size, total_experts
    )
    buffer = nixl_ep.Buffer(
        rank=rank,
        nvlink_backend=nvlink_backend,
        explicitly_destroy=True,
        enable_shrink=True,
        tcp_store_group=tcp_store,
    )
    buffer.update_memory_buffers(
        num_ranks=world_size,
        num_experts_per_rank=num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
    )

    sync_all_ranks(rank, world_size, f"{mode}_init")

    # Connect to all other ranks
    other_ranks = [r for r in range(world_size) if r != rank]
    if other_ranks:
        torch.cuda.synchronize()
        buffer.connect_ranks(other_ranks)
        torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, f"{mode}_connected")

    # Create test data
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.randint(
        0, total_experts, (num_tokens, topk), dtype=torch.int64, device="cuda"
    )
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Calculate bytes for BW measurement (FP8 dispatch, BF16 combine)
    num_fp8_bytes = hidden + hidden // 128 * 4 + 16
    num_combine_bytes = hidden * 2  # BF16
    num_dispatch_comm_bytes = 0
    num_combine_comm_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_combine_bytes * num_selections

    # Initial dispatch to get shape for combine
    recv_x, recv_count, handle_init, event, hook = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=total_experts,
        use_fp8=True,
        async_finish=False,
    )
    simulated_gemm_x = recv_x[0].to(torch.bfloat16).clone()

    # Define test functions based on mode
    def dispatch_fn():
        return buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            num_max_dispatch_tokens_per_rank=num_tokens,
            num_experts=total_experts,
            use_fp8=True,
            async_finish=False,
        )

    def combine_fn(handle):
        return buffer.combine(
            x=simulated_gemm_x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=handle,
            use_logfmt=False,
        )

    # Flush L2 cache
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # For combine mode: do ONE dispatch, reuse handle for all combines
    combine_handle = None
    if mode == "combine":
        _, _, combine_handle, _, _ = dispatch_fn()

    # Warmup
    for _ in range(warmup_iters):
        if mode == "dispatch":
            dispatch_fn()
        elif mode == "combine":
            combine_fn(combine_handle)
        else:  # e2e
            _, _, handle, _, _ = dispatch_fn()
            combine_fn(handle)

    cache.zero_()  # Flush L2
    sync_all_ranks(rank, world_size, f"{mode}_warmup")

    # Measure with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]

    for i in range(measure_iters):
        start_events[i].record()

        if mode == "dispatch":
            dispatch_fn()
        elif mode == "combine":
            combine_fn(combine_handle)
        else:  # e2e
            _, _, handle, _, _ = dispatch_fn()
            combine_fn(handle)

        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate times (skip first iteration)
    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]

    if mode == "combine":
        comm_bytes = num_combine_comm_bytes
    elif mode == "dispatch":
        comm_bytes = num_dispatch_comm_bytes
    else:  # e2e
        comm_bytes = num_dispatch_comm_bytes + num_combine_comm_bytes

    avg_t = np.average(times)
    min_t = np.min(times)
    max_t = np.max(times)

    # Calculate metrics
    bandwidth_gbps = comm_bytes / 1e9 / avg_t
    avg_latency_us = avg_t * 1e6
    tokens_per_sec = num_tokens / avg_t

    sync_all_ranks(rank, world_size, f"{mode}_measured")

    # Cleanup
    buffer.destroy()
    sync_all_ranks(rank, world_size, f"{mode}_cleanup")

    # Validate results
    passed = True
    error = None
    if np.isnan(bandwidth_gbps) or bandwidth_gbps <= 0:
        passed = False
        error = f"Invalid bandwidth: {bandwidth_gbps}"
    elif np.isnan(avg_t) or avg_t <= 0:
        passed = False
        error = f"Invalid timing: {avg_t}"

    return {
        "passed": passed,
        "error": error,
        "metrics": {
            "mode": mode,
            "bandwidth_gbps": bandwidth_gbps,
            "avg_latency_us": avg_latency_us,
            "min_latency_us": min_t * 1e6,
            "max_latency_us": max_t * 1e6,
            "tokens_per_sec": tokens_per_sec,
            "num_tokens": num_tokens,
            "hidden": hidden,
            "topk": topk,
            "total_experts": total_experts,
            "measure_iters": measure_iters,
        },
    }


def log_results(test_name: str, results: List[TestResult]):
    """Log formatted results."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    logger.info("=" * 70)
    logger.info("%s: %d/%d ranks passed", test_name, passed, total)
    logger.info("=" * 70)

    if passed == 0:
        for r in results:
            if r.error:
                logger.info("  Rank %d: %s", r.rank, r.error[:200])
        return

    # Aggregate metrics
    bw_values = []
    lat_values = []
    for r in results:
        if r.passed and r.metrics:
            bw_values.append(r.metrics.get("bandwidth_gbps", 0))
            lat_values.append(r.metrics.get("avg_latency_us", 0))

    if bw_values:
        logger.info(
            "Bandwidth (GB/s): avg=%.2f, min=%.2f, max=%.2f",
            sum(bw_values) / len(bw_values),
            min(bw_values),
            max(bw_values),
        )
    if lat_values:
        logger.info(
            "Latency (μs):     avg=%.1f, min=%.1f, max=%.1f",
            sum(lat_values) / len(lat_values),
            min(lat_values),
            max(lat_values),
        )


def main():
    parser = argparse.ArgumentParser(description="NIXL EP Data Plane Performance Test")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of processes per node")
    parser.add_argument(
        "--mode",
        type=str,
        default="e2e",
        choices=["dispatch", "combine", "e2e"],
        help="Test mode: dispatch, combine, or e2e",
    )
    parser.add_argument("--tokens", type=int, default=512, help="Number of tokens")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--experts-per-rank", type=int, default=8, help="Experts/rank")
    parser.add_argument("--topk", type=int, default=2, help="TopK value")
    parser.add_argument(
        "--nvlink-backend",
        type=str,
        default="ipc",
        choices=["nixl", "ipc", "none"],
        help="NVLink backend (none forces RDMA)",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iters"
    )
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_ITERS, help="Measure iters"
    )
    parser.add_argument("--timeout", type=int, default=300, help="Timeout (seconds)")
    parser.add_argument(
        "--skip-nic-discovery",
        action="store_true",
        help="Skip GPU-NIC topology discovery (let UCX auto-select)",
    )
    parser.add_argument(
        "--use-tcp-store",
        action="store_true",
        help="Use TCPStore for metadata exchange instead of etcd",
    )
    # Multi-node parameters
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Total number of nodes (overrides WORLD_SIZE env var, default: 1)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank of this node 0=master (overrides RANK env var, default: 0)",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=None,
        help="Master node address (overrides MASTER_ADDR env var)",
    )
    args = parser.parse_args()

    # Get multi-node configuration from environment or command line
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "0"))
    master_addr = args.master_addr if args.master_addr is not None else os.environ.get("MASTER_ADDR", "127.0.0.1")

    # Configure logger with node prefix for multi-node debugging
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(f"[Node {rank}] %(message)s"))

    # Validation
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, {world_size-1}], got {rank}")
    if world_size > 1 and rank > 0 and master_addr == "127.0.0.1":
        raise ValueError(
            "MASTER_ADDR must be set (not 127.0.0.1) for worker nodes in multi-node setup. "
            "Set MASTER_ADDR environment variable or use --master-addr flag."
        )

    # Calculate total ranks
    total_ranks = args.num_processes * world_size
    total_experts = args.experts_per_rank * total_ranks
    metadata_exchange = "TCPStore" if args.use_tcp_store else "etcd"

    logger.info("=" * 70)
    logger.info("NIXL EP Data Plane Performance Test")
    logger.info("=" * 70)
    if world_size > 1:
        logger.info("Multi-node setup:")
        logger.info("  Nodes (WORLD_SIZE): %d", world_size)
        logger.info("  This node (RANK): %d %s", rank, "(master)" if rank == 0 else "(worker)")
        logger.info("  Processes per node: %d", args.num_processes)
        logger.info("  Total ranks: %d", total_ranks)
        logger.info("  Master address: %s", master_addr)
    else:
        logger.info("Single-node setup: %d processes", args.num_processes)
    logger.info("Mode: %s", args.mode)
    logger.info("Tokens: %d, Hidden: %d, TopK: %d", args.tokens, args.hidden, args.topk)
    logger.info("Experts: %d/rank (%d total)", args.experts_per_rank, total_experts)
    logger.info("NVLink backend: %s", args.nvlink_backend)
    logger.info("Metadata exchange: %s", metadata_exchange)
    logger.info("Warmup: %d, Measure: %d iterations", args.warmup, args.iters)
    logger.info("=" * 70)

    results = run_multiprocess_test(
        test_fn=_run_data_plane_test,
        num_processes=args.num_processes,
        timeout=args.timeout,
        skip_nic_discovery=args.skip_nic_discovery,
        use_tcp_store=args.use_tcp_store,
        world_size=world_size,
        rank=rank,
        master_addr=master_addr,
        mode=args.mode,
        num_experts_per_rank=args.experts_per_rank,
        num_tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        nvlink_backend=args.nvlink_backend,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
    )

    log_results(f"Data Plane ({args.mode})", results)

    # Exit with error if any rank failed
    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
