# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Control plane performance tests for NIXL EP Buffer.

Measures latency of:
- Buffer initialization (init)
- connect_ranks()
- disconnect_ranks()
- destroy()
- Full cycle (init → connect → disconnect → reconnect → destroy)

Usage:
    # Run full cycle test with default expert counts
    python3 test_control_plane.py --num-processes=8

    # Specific expert counts
    python3 test_control_plane.py --num-processes=8 --experts-per-rank=8,32

    # Single test
    python3 test_control_plane.py --num-processes=8 --test=connect --experts-per-rank=8

    # RDMA only
    python3 test_control_plane.py --num-processes=8 --nvlink-backend=none
"""

import argparse
import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

import store_group
import torch.distributed as dist
from mp_runner import TestResult, run_multiprocess_test, sync_all_ranks

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_EXPERT_COUNTS = [8, 32]
DEFAULT_NUM_TOKENS = 512
DEFAULT_HIDDEN = 4096
DEFAULT_WARMUP = 0  # No warmup for now (bug in repeated cycles)
DEFAULT_ROUNDS = 1  # Single measurement cycle


def _run_control_plane_test(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    test_mode: str = "cycle",
    num_experts_per_rank: int = 8,
    num_tokens: int = DEFAULT_NUM_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    warmup_rounds: int = DEFAULT_WARMUP,
    measure_rounds: int = DEFAULT_ROUNDS,
    nvlink_backend: str = "ipc",
    use_tcp_store: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run control plane performance test.

    Args:
        test_mode: "init", "connect", "disconnect", "destroy", or "cycle" (all phases)
        warmup_rounds: Number of warmup rounds (default: 0 due to repeated cycle bug)
        measure_rounds: Number of measurement rounds (default: 1)
        use_tcp_store: Use TCPStore for metadata exchange instead of etcd
    """
    import nixl_ep

    # Setup TCPStore if requested
    tcp_store: Optional[dist.TCPStore] = None
    if use_tcp_store:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        tcp_store_port = kwargs.get("tcp_store_port", 9999)
        # Each worker creates a client connection to the shared TCPStore server
        tcp_store = store_group.create_client_store(
            master_addr=master_addr,
            port=tcp_store_port,
            timeout_sec=60.0,
        )

    other_ranks = [r for r in range(world_size) if r != rank]
    num_experts = num_experts_per_rank * world_size
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )

    if test_mode == "cycle":
        return _run_full_cycle(
            rank,
            world_size,
            other_ranks,
            num_experts_per_rank,
            num_rdma_bytes,
            warmup_rounds,
            measure_rounds,
            nvlink_backend,
            tcp_store,
        )
    else:
        return _run_single_op(
            rank,
            world_size,
            other_ranks,
            test_mode,
            num_experts_per_rank,
            num_rdma_bytes,
            warmup_rounds,
            measure_rounds,
            nvlink_backend,
            tcp_store,
        )


def _run_single_op(
    rank: int,
    world_size: int,
    other_ranks: List[int],
    operation: str,
    num_experts_per_rank: int,
    num_rdma_bytes: int,
    warmup_rounds: int,
    measure_rounds: int,
    nvlink_backend: str,
    tcp_store: Optional[dist.TCPStore] = None,
) -> Dict[str, Any]:
    """Run a single control plane operation test."""
    import nixl_ep
    import torch

    latencies = []

    for i in range(warmup_rounds + measure_rounds):
        is_measure = i >= warmup_rounds

        if operation == "init":
            sync_all_ranks(rank, world_size, f"init_{i}")
            torch.cuda.synchronize()
            start = time.perf_counter()

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

            torch.cuda.synchronize()
            if is_measure:
                latencies.append((time.perf_counter() - start) * 1000)

            buffer.destroy()
            sync_all_ranks(rank, world_size, f"init_cleanup_{i}")

        elif operation == "connect":
            # DEBUG: Copy elastic.py pattern - log environment before Buffer creation
            import os
            logger = logging.getLogger(__name__)
            logger.info(f"Rank {rank} (local_rank={local_rank}): UCX_NET_DEVICES={os.environ.get('UCX_NET_DEVICES', 'NOT SET')}")
            logger.info(f"Rank {rank} (local_rank={local_rank}): NIXL_ETCD_ENDPOINTS={os.environ.get('NIXL_ETCD_ENDPOINTS', 'NOT SET')}")
            logger.info(f"Rank {rank} (local_rank={local_rank}): Creating Buffer with nvlink_backend={nvlink_backend}")
            
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
            sync_all_ranks(rank, world_size, f"connect_pre_{i}")

            torch.cuda.synchronize()
            start = time.perf_counter()

            if other_ranks:
                buffer.connect_ranks(other_ranks)

            torch.cuda.synchronize()
            if is_measure:
                latencies.append((time.perf_counter() - start) * 1000)

            sync_all_ranks(rank, world_size, f"connect_post_{i}")
            buffer.destroy()
            sync_all_ranks(rank, world_size, f"connect_cleanup_{i}")

        elif operation == "disconnect":
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
            sync_all_ranks(rank, world_size, f"disconnect_pre_connect_{i}")
            if other_ranks:
                buffer.connect_ranks(other_ranks)
            sync_all_ranks(rank, world_size, f"disconnect_post_connect_{i}")

            torch.cuda.synchronize()
            start = time.perf_counter()

            if other_ranks:
                buffer.disconnect_ranks(other_ranks)

            torch.cuda.synchronize()
            if is_measure:
                latencies.append((time.perf_counter() - start) * 1000)

            sync_all_ranks(rank, world_size, f"disconnect_post_{i}")
            time.sleep(0.5)
            buffer.destroy()
            sync_all_ranks(rank, world_size, f"disconnect_cleanup_{i}")

        elif operation == "destroy":
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
            sync_all_ranks(rank, world_size, f"destroy_pre_connect_{i}")
            if other_ranks:
                buffer.connect_ranks(other_ranks)
            sync_all_ranks(rank, world_size, f"destroy_post_connect_{i}")

            torch.cuda.synchronize()
            start = time.perf_counter()

            buffer.destroy()

            torch.cuda.synchronize()
            if is_measure:
                latencies.append((time.perf_counter() - start) * 1000)

            sync_all_ranks(rank, world_size, f"destroy_cleanup_{i}")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    return {
        "passed": True,
        "metrics": {
            "operation": operation,
            "num_experts_per_rank": num_experts_per_rank,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
        },
    }


def _run_full_cycle(
    rank: int,
    world_size: int,
    other_ranks: List[int],
    num_experts_per_rank: int,
    num_rdma_bytes: int,
    warmup_rounds: int,
    measure_rounds: int,
    nvlink_backend: str,
    tcp_store: Optional[dist.TCPStore] = None,
) -> Dict[str, Any]:
    """Run full control plane cycle: init → connect → disconnect → reconnect → destroy."""
    import nixl_ep
    import torch

    init_latencies = []
    connect_latencies = []
    disconnect_latencies = []
    reconnect_latencies = []
    destroy_latencies = []

    for i in range(warmup_rounds + measure_rounds):
        is_measure = i >= warmup_rounds

        # === INIT ===
        sync_all_ranks(rank, world_size, f"cycle_init_{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()

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

        torch.cuda.synchronize()
        if is_measure:
            init_latencies.append((time.perf_counter() - start) * 1000)

        # === CONNECT ===
        sync_all_ranks(rank, world_size, f"cycle_connect_{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()

        if other_ranks:
            buffer.connect_ranks(other_ranks)

        torch.cuda.synchronize()
        if is_measure:
            connect_latencies.append((time.perf_counter() - start) * 1000)

        # === DISCONNECT ===
        sync_all_ranks(rank, world_size, f"cycle_disconnect_{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()

        if other_ranks:
            buffer.disconnect_ranks(other_ranks)

        torch.cuda.synchronize()
        if is_measure:
            disconnect_latencies.append((time.perf_counter() - start) * 1000)

        # === RECONNECT ===
        sync_all_ranks(rank, world_size, f"cycle_reconnect_{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()

        if other_ranks:
            buffer.connect_ranks(other_ranks)

        torch.cuda.synchronize()
        if is_measure:
            reconnect_latencies.append((time.perf_counter() - start) * 1000)

        # === DESTROY ===
        sync_all_ranks(rank, world_size, f"cycle_destroy_{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()

        buffer.destroy()

        torch.cuda.synchronize()
        if is_measure:
            destroy_latencies.append((time.perf_counter() - start) * 1000)

        sync_all_ranks(rank, world_size, f"cycle_cleanup_{i}")

    def stats(latencies):
        if not latencies:
            return {"avg_ms": 0, "min_ms": 0, "max_ms": 0}
        return {
            "avg_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }

    total_avg = 0.0
    if init_latencies:
        total_avg = (
            sum(init_latencies)
            + sum(connect_latencies)
            + sum(disconnect_latencies)
            + sum(reconnect_latencies)
            + sum(destroy_latencies)
        ) / len(init_latencies)

    return {
        "passed": True,
        "metrics": {
            "operation": "cycle",
            "num_experts_per_rank": num_experts_per_rank,
            "num_peers": len(other_ranks),
            "init": stats(init_latencies),
            "connect": stats(connect_latencies),
            "disconnect": stats(disconnect_latencies),
            "reconnect": stats(reconnect_latencies),
            "destroy": stats(destroy_latencies),
            "total_avg_ms": total_avg,
        },
    }


def log_cycle_results(results: List[TestResult], num_experts: int, world_size: int):
    """Log full cycle results."""
    rank0 = next((r for r in results if r.rank == 0), None)
    if not rank0 or not rank0.metrics:
        logger.info("No results from rank 0")
        return

    m = rank0.metrics
    total_experts = num_experts * world_size

    logger.info("=" * 70)
    logger.info(
        "Control Plane: %d experts/rank x %d ranks = %d total",
        num_experts,
        world_size,
        total_experts,
    )
    logger.info("=" * 70)
    logger.info(
        "%-15s %-12s %-12s %-12s", "Operation", "Avg (ms)", "Min (ms)", "Max (ms)"
    )
    logger.info("-" * 70)

    for op in ["init", "connect", "disconnect", "reconnect", "destroy"]:
        if op in m:
            s = m[op]
            logger.info(
                "%-15s %-12.2f %-12.2f %-12.2f",
                op,
                s["avg_ms"],
                s["min_ms"],
                s["max_ms"],
            )

    if "total_avg_ms" in m:
        logger.info("-" * 70)
        logger.info("%-15s %-12.2f", "TOTAL", m["total_avg_ms"])
    logger.info("=" * 70)


def log_single_op_results(
    results: List[TestResult], operation: str, num_experts: int, world_size: int
):
    """Log single operation results."""
    per_rank_avgs = []
    for r in results:
        if r.metrics and "avg_latency_ms" in r.metrics:
            per_rank_avgs.append((r.rank, r.metrics["avg_latency_ms"]))

    if not per_rank_avgs:
        logger.info("No results collected")
        return

    latencies = [avg for _, avg in per_rank_avgs]
    avg_all = sum(latencies) / len(latencies)
    min_all = min(latencies)
    max_all = max(latencies)

    logger.info(
        "%s: %.2fms avg across %d ranks (min=%.2f, max=%.2f) [%d experts/rank]",
        operation.upper(),
        avg_all,
        len(per_rank_avgs),
        min_all,
        max_all,
        num_experts,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Control plane performance tests for NIXL EP Buffer"
    )
    parser.add_argument(
        "--num-processes", type=int, default=8, help="Number of processes/ranks"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="cycle",
        choices=["init", "connect", "disconnect", "destroy", "cycle"],
        help="Which test to run (default: cycle)",
    )
    parser.add_argument(
        "--experts-per-rank",
        type=str,
        default=None,
        help="Experts per rank, comma-separated (default: 8,32)",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup rounds"
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS, help="Measurement rounds"
    )
    parser.add_argument(
        "--nvlink-backend",
        type=str,
        default="ipc",
        choices=["ipc", "nixl", "none"],
        help="NVLink backend (none forces RDMA)",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Timeout (seconds)"
    )
    parser.add_argument(
        "--skip-nic-discovery",
        action="store_true",
        help="Skip GPU-NIC topology discovery",
    )
    parser.add_argument(
        "--use-tcp-store",
        action="store_true",
        help="Use TCPStore for metadata exchange instead of etcd",
    )
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
    # CLI args override environment variables
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "0"))
    master_addr = args.master_addr if args.master_addr is not None else os.environ.get("MASTER_ADDR", "127.0.0.1")
    
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

    # Parse expert counts
    if args.experts_per_rank:
        expert_counts = [int(x.strip()) for x in args.experts_per_rank.split(",")]
    else:
        expert_counts = DEFAULT_EXPERT_COUNTS

    metadata_exchange = "TCPStore" if args.use_tcp_store else "etcd"
    
    # Calculate total rank count (processes per node * number of nodes)
    total_ranks = args.num_processes * world_size

    logger.info("=" * 70)
    logger.info("NIXL EP Control Plane Performance Test")
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
    logger.info("NVLink backend: %s", args.nvlink_backend)
    logger.info("Metadata exchange: %s", metadata_exchange)
    logger.info("Experts/rank: %s", expert_counts)
    logger.info("Test: %s", args.test)
    logger.info("Warmup: %d, Measure: %d rounds", args.warmup, args.rounds)
    logger.info("=" * 70)

    all_passed = True

    for num_experts in expert_counts:
        total_experts = num_experts * total_ranks
        logger.info(
            "\nRunning: %s (%d experts/rank, %d total)",
            args.test,
            num_experts,
            total_experts,
        )

        results = run_multiprocess_test(
            test_fn=_run_control_plane_test,
            num_processes=args.num_processes,
            timeout=args.timeout,
            skip_nic_discovery=args.skip_nic_discovery,
            test_mode=args.test,
            num_experts_per_rank=num_experts,
            warmup_rounds=args.warmup,
            measure_rounds=args.rounds,
            nvlink_backend=args.nvlink_backend,
            use_tcp_store=args.use_tcp_store,
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
        )

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        if passed == total:
            if args.test == "cycle":
                log_cycle_results(results, num_experts, total_ranks)
            else:
                log_single_op_results(
                    results, args.test, num_experts, total_ranks
                )
        else:
            logger.info("FAILED: %d/%d ranks passed", passed, total)
            for r in results:
                if not r.passed and r.error:
                    logger.info("  Rank %d: %s", r.rank, r.error[:200])
            all_passed = False

    if not all_passed:
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
