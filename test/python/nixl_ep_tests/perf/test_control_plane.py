# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for NIXL EP Buffer control plane operations.

Measures latency of:
- Buffer initialization (init)
- connect_ranks()
- disconnect_ranks()
- destroy()

Tests with varying num_experts_per_rank to understand scaling behavior.

Usage:
    # Single node (8 ranks)
    python3 test_control_plane.py --num-processes=8

    # Two nodes (16 ranks) - requires SLURM multi-node allocation
    python3 test_control_plane.py --num-processes=16

    # Specific expert counts per node
    python3 test_control_plane.py --num-processes=8 --experts-per-rank=1,8,32

    # Single test with specific expert count
    python3 test_control_plane.py --num-processes=8 --test=connect --experts-per-rank=8
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import pytest

# Add parent directory to path
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)

from utils.mp_runner import (
    TestResult,
    all_passed,
    print_results,
    run_multiprocess_test,
    sync_all_ranks,
)

# Import results collector for CI/CD integration
try:
    from perf.results_collector import ResultsCollector

    HAS_COLLECTOR = True
except ImportError:
    HAS_COLLECTOR = False


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_EXPERT_COUNTS = [2, 4, 8, 16, 32]
DEFAULT_NUM_TOKENS = 512
DEFAULT_HIDDEN = 4096
DEFAULT_WARMUP_ROUNDS = 2
DEFAULT_MEASURE_ROUNDS = 5


# ============================================================================
# P-CTRL-01: Buffer Initialization Latency
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_init_latency_fn(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    num_experts_per_rank: int = 8,
    num_tokens: int = DEFAULT_NUM_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
    measure_rounds: int = DEFAULT_MEASURE_ROUNDS,
):
    """
    P-CTRL-01: Measure Buffer initialization latency.

    Times: Buffer() + update_memory_buffers()
    """
    import torch

    import nixl_ep

    latencies = []

    # Calculate buffer size
    num_experts = num_experts_per_rank * world_size
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )

    # Warmup + measurement rounds
    for i in range(warmup_rounds + measure_rounds):
        sync_all_ranks(rank, world_size, f"init_round_{i}")

        torch.cuda.synchronize()
        start = time.perf_counter()

        buffer = nixl_ep.Buffer(
            rank=rank, nvlink_backend="ipc", explicitly_destroy=True, enable_shrink=True
        )
        buffer.update_memory_buffers(
            num_ranks=world_size,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup_rounds:
            latencies.append(elapsed_ms)

        # Cleanup
        buffer.destroy()
        sync_all_ranks(rank, world_size, f"init_cleanup_{i}")

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    return {
        "passed": True,
        "metrics": {
            "operation": "init",
            "num_experts_per_rank": num_experts_per_rank,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latencies": latencies,
        },
    }


# ============================================================================
# P-CTRL-02: Connect Latency
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_connect_latency_fn(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    num_experts_per_rank: int = 8,
    num_tokens: int = DEFAULT_NUM_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
    measure_rounds: int = DEFAULT_MEASURE_ROUNDS,
):
    """
    P-CTRL-02: Measure connect_ranks() latency.

    Times: connect_ranks([all other ranks])
    """
    import torch

    import nixl_ep

    latencies = []
    other_ranks = [r for r in range(world_size) if r != rank]

    # Calculate buffer size
    num_experts = num_experts_per_rank * world_size
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )

    # Warmup + measurement rounds
    for i in range(warmup_rounds + measure_rounds):
        # Create buffer
        buffer = nixl_ep.Buffer(
            rank=rank, nvlink_backend="ipc", explicitly_destroy=True, enable_shrink=True
        )
        buffer.update_memory_buffers(
            num_ranks=world_size,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )

        # Wait for all buffers to be registered in etcd
        sync_all_ranks(rank, world_size, f"connect_pre_{i}")

        # Measure connect
        torch.cuda.synchronize()
        start = time.perf_counter()

        if other_ranks:
            buffer.connect_ranks(other_ranks)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup_rounds:
            latencies.append(elapsed_ms)

        # Cleanup
        sync_all_ranks(rank, world_size, f"connect_post_{i}")
        buffer.destroy()
        sync_all_ranks(rank, world_size, f"connect_cleanup_{i}")

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    return {
        "passed": True,
        "metrics": {
            "operation": "connect",
            "num_experts_per_rank": num_experts_per_rank,
            "num_peers": len(other_ranks),
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latencies": latencies,
        },
    }


# ============================================================================
# P-CTRL-03: Disconnect Latency
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_disconnect_latency_fn(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    num_experts_per_rank: int = 8,
    num_tokens: int = DEFAULT_NUM_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
    measure_rounds: int = DEFAULT_MEASURE_ROUNDS,
):
    """
    P-CTRL-03: Measure disconnect_ranks() latency.

    Times: disconnect_ranks([all other ranks])

    Note: This will show invalidateRemoteMD warnings which are expected.
    """
    import torch

    import nixl_ep

    latencies = []
    other_ranks = [r for r in range(world_size) if r != rank]

    # Calculate buffer size
    num_experts = num_experts_per_rank * world_size
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )

    # Warmup + measurement rounds
    for i in range(warmup_rounds + measure_rounds):
        # Create buffer
        buffer = nixl_ep.Buffer(
            rank=rank, nvlink_backend="ipc", explicitly_destroy=True, enable_shrink=True
        )
        buffer.update_memory_buffers(
            num_ranks=world_size,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )

        # Connect
        sync_all_ranks(rank, world_size, f"disconnect_pre_connect_{i}")
        if other_ranks:
            buffer.connect_ranks(other_ranks)
        sync_all_ranks(rank, world_size, f"disconnect_post_connect_{i}")

        # Measure disconnect
        torch.cuda.synchronize()
        start = time.perf_counter()

        if other_ranks:
            buffer.disconnect_ranks(other_ranks)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup_rounds:
            latencies.append(elapsed_ms)

        # Cleanup
        sync_all_ranks(rank, world_size, f"disconnect_post_{i}")
        time.sleep(0.5)  # Allow metadata invalidation to complete
        buffer.destroy()
        sync_all_ranks(rank, world_size, f"disconnect_cleanup_{i}")

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    return {
        "passed": True,
        "metrics": {
            "operation": "disconnect",
            "num_experts_per_rank": num_experts_per_rank,
            "num_peers": len(other_ranks),
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latencies": latencies,
        },
    }


# ============================================================================
# P-CTRL-04: Destroy Latency
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_destroy_latency_fn(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    num_experts_per_rank: int = 8,
    num_tokens: int = DEFAULT_NUM_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
    measure_rounds: int = DEFAULT_MEASURE_ROUNDS,
):
    """
    P-CTRL-04: Measure destroy() latency.

    Times: destroy() after full connect cycle
    """
    import torch

    import nixl_ep

    latencies = []
    other_ranks = [r for r in range(world_size) if r != rank]

    # Calculate buffer size
    num_experts = num_experts_per_rank * world_size
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )

    # Warmup + measurement rounds
    for i in range(warmup_rounds + measure_rounds):
        # Create buffer
        buffer = nixl_ep.Buffer(
            rank=rank, nvlink_backend="ipc", explicitly_destroy=True, enable_shrink=True
        )
        buffer.update_memory_buffers(
            num_ranks=world_size,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )

        # Connect
        sync_all_ranks(rank, world_size, f"destroy_pre_connect_{i}")
        if other_ranks:
            buffer.connect_ranks(other_ranks)
        sync_all_ranks(rank, world_size, f"destroy_post_connect_{i}")

        # Measure destroy
        torch.cuda.synchronize()
        start = time.perf_counter()

        buffer.destroy()

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup_rounds:
            latencies.append(elapsed_ms)

        # Post-cleanup sync
        sync_all_ranks(rank, world_size, f"destroy_cleanup_{i}")

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    return {
        "passed": True,
        "metrics": {
            "operation": "destroy",
            "num_experts_per_rank": num_experts_per_rank,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latencies": latencies,
        },
    }


# ============================================================================
# P-CTRL-05: Full Control Plane Cycle
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_full_cycle_latency_fn(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    num_experts_per_rank: int = 8,
    num_tokens: int = DEFAULT_NUM_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
    measure_rounds: int = DEFAULT_MEASURE_ROUNDS,
):
    """
    P-CTRL-05: Measure full control plane cycle.

    Times each phase separately: init, connect, disconnect, reconnect, destroy

    The reconnect phase re-establishes connections after disconnect, which:
    1. Tests the reconnect path (important for elastic scenarios)
    2. Leaves buffer in connected state before destroy (reduces race conditions)
    """
    import torch

    import nixl_ep

    init_latencies = []
    connect_latencies = []
    disconnect_latencies = []
    reconnect_latencies = []
    destroy_latencies = []

    other_ranks = [r for r in range(world_size) if r != rank]

    # Calculate buffer size
    num_experts = num_experts_per_rank * world_size
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )

    # Warmup + measurement rounds
    for i in range(warmup_rounds + measure_rounds):
        is_measure = i >= warmup_rounds

        # === INIT ===
        sync_all_ranks(rank, world_size, f"cycle_init_{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()

        buffer = nixl_ep.Buffer(
            rank=rank, nvlink_backend="ipc", explicitly_destroy=True, enable_shrink=True
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
        # Re-establish connections after disconnect (tests elastic reconnect path)
        sync_all_ranks(rank, world_size, f"cycle_reconnect_{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()

        if other_ranks:
            buffer.connect_ranks(other_ranks)

        torch.cuda.synchronize()
        if is_measure:
            reconnect_latencies.append((time.perf_counter() - start) * 1000)

        # === DESTROY ===
        # Destroying from connected state (reduces invalidateRemoteMD warnings)
        sync_all_ranks(rank, world_size, f"cycle_destroy_{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()

        buffer.destroy()

        torch.cuda.synchronize()
        if is_measure:
            destroy_latencies.append((time.perf_counter() - start) * 1000)

        sync_all_ranks(rank, world_size, f"cycle_cleanup_{i}")

    def stats(latencies):
        return {
            "avg_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }

    return {
        "passed": True,
        "metrics": {
            "operation": "full_cycle",
            "num_experts_per_rank": num_experts_per_rank,
            "num_peers": len(other_ranks),
            "init": stats(init_latencies),
            "connect": stats(connect_latencies),
            "disconnect": stats(disconnect_latencies),
            "reconnect": stats(reconnect_latencies),
            "destroy": stats(destroy_latencies),
            "total_avg_ms": (
                sum(init_latencies)
                + sum(connect_latencies)
                + sum(disconnect_latencies)
                + sum(reconnect_latencies)
                + sum(destroy_latencies)
            )
            / len(init_latencies),
        },
    }


# ============================================================================
# Test Registry
# ============================================================================

TESTS = {
    "init": ("P-CTRL-01: Init latency", _test_init_latency_fn),
    "connect": ("P-CTRL-02: Connect latency", _test_connect_latency_fn),
    "disconnect": ("P-CTRL-03: Disconnect latency", _test_disconnect_latency_fn),
    "destroy": ("P-CTRL-04: Destroy latency", _test_destroy_latency_fn),
    "cycle": ("P-CTRL-05: Full control plane cycle", _test_full_cycle_latency_fn),
}


# ============================================================================
# Results Formatting
# ============================================================================


def format_cycle_results(results: List[TestResult], num_experts: int, world_size: int):
    """Format full cycle results into a nice table."""
    # Collect metrics from rank 0 (all ranks should have similar results)
    rank0 = next((r for r in results if r.rank == 0), None)
    if not rank0 or not rank0.metrics:
        return "No results from rank 0"

    m = rank0.metrics
    total_experts = num_experts * world_size
    lines = [
        f"\n{'='*70}",
        f"Control Plane Performance: {num_experts} experts/rank × {world_size} ranks = {total_experts} total",
        f"{'='*70}",
        f"{'Operation':<15} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}",
        f"{'-'*70}",
    ]

    for op in ["init", "connect", "disconnect", "reconnect", "destroy"]:
        if op in m:
            s = m[op]
            lines.append(
                f"{op:<15} {s['avg_ms']:<12.2f} {s['min_ms']:<12.2f} {s['max_ms']:<12.2f}"
            )

    if "total_avg_ms" in m:
        lines.append(f"{'-'*70}")
        lines.append(f"{'TOTAL':<15} {m['total_avg_ms']:<12.2f}")

    lines.append(f"{'='*70}")
    return "\n".join(lines)


def format_single_op_results(
    results: List[TestResult], operation: str, num_experts: int, world_size: int
):
    """Format single operation results aggregated across all ranks."""
    # Collect latencies from all ranks
    all_latencies = []
    per_rank_avgs = []

    for r in results:
        if r.metrics and "avg_latency_ms" in r.metrics:
            per_rank_avgs.append((r.rank, r.metrics["avg_latency_ms"]))
            # Also collect individual latencies if available
            if "latencies" in r.metrics:
                all_latencies.extend(r.metrics["latencies"])

    if not per_rank_avgs:
        return "No results collected"

    # Compute statistics across all ranks
    latencies_for_stats = [avg for _, avg in per_rank_avgs]
    avg_across_ranks = sum(latencies_for_stats) / len(latencies_for_stats)
    min_across_ranks = min(latencies_for_stats)
    max_across_ranks = max(latencies_for_stats)

    # Build detailed output
    lines = [
        f"{operation.upper()}: {avg_across_ranks:.2f}ms avg across {len(per_rank_avgs)} ranks "
        f"(min={min_across_ranks:.2f}, max={max_across_ranks:.2f}) "
        f"[{num_experts} experts/rank, {num_experts * world_size} total]"
    ]

    # Show per-rank breakdown
    per_rank_avgs.sort(key=lambda x: x[0])  # Sort by rank
    rank_times = ", ".join([f"R{rank}:{lat:.0f}" for rank, lat in per_rank_avgs])
    lines.append(f"  Per-rank (ms): {rank_times}")

    return "\n".join(lines)


def collect_all_results(
    all_results: Dict[int, Dict[str, List[TestResult]]], world_size: int
) -> Dict:
    """Collect all results into a structured format for JSON export."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "world_size": world_size,
        "results": {},
    }

    for num_experts, test_results in all_results.items():
        output["results"][num_experts] = {}
        for test_name, results in test_results.items():
            rank0 = next((r for r in results if r.rank == 0), None)
            if rank0 and rank0.metrics:
                output["results"][num_experts][test_name] = rank0.metrics

    return output


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Control plane performance tests for NIXL EP Buffer"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes/ranks (default: 8 for 1 node, use 16 for 2 nodes)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "init", "connect", "disconnect", "destroy", "cycle"],
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--experts-per-rank",
        type=str,
        default=None,
        help="Experts per rank, comma-separated (default: 2,4,8,16,32). Total experts = experts_per_rank * num_ranks",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_ROUNDS,
        help=f"Number of warmup rounds (default: {DEFAULT_WARMUP_ROUNDS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_MEASURE_ROUNDS,
        help=f"Number of measurement rounds (default: {DEFAULT_MEASURE_ROUNDS})",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for CI/CD results (default: NIXL_RESULTS_DIR env or perf/results/)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Test timeout in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Parse expert counts per node
    if args.experts_per_rank:
        expert_counts = [int(x.strip()) for x in args.experts_per_rank.split(",")]
    else:
        expert_counts = DEFAULT_EXPERT_COUNTS

    # Determine which tests to run
    if args.test == "all":
        # For 'all', just run the full cycle test (most informative)
        tests_to_run = ["cycle"]
    else:
        tests_to_run = [args.test]

    sys.stderr.write(f"\n{'#'*70}\n")
    sys.stderr.write(f"# NIXL EP Control Plane Performance Tests\n")
    sys.stderr.write(f"# Ranks: {args.num_processes}\n")
    sys.stderr.write(f"# Experts/rank: {expert_counts} (total = experts × ranks)\n")
    sys.stderr.write(f"# Tests: {tests_to_run}\n")
    sys.stderr.write(f"# Warmup: {args.warmup}, Measure: {args.rounds}\n")
    sys.stderr.write(f"{'#'*70}\n\n")

    all_results = {}

    for num_experts in expert_counts:
        all_results[num_experts] = {}

        for test_key in tests_to_run:
            test_name, test_fn = TESTS[test_key]

            total_experts = num_experts * args.num_processes
            sys.stderr.write(
                f"\nRunning: {test_name} ({num_experts} experts/rank, {total_experts} total)\n"
            )
            sys.stderr.write("-" * 60 + "\n")

            results = run_multiprocess_test(
                test_fn,
                num_processes=args.num_processes,
                timeout=args.timeout,
                num_experts_per_rank=num_experts,
                warmup_rounds=args.warmup,
                measure_rounds=args.rounds,
            )

            all_results[num_experts][test_key] = results

            if all_passed(results):
                if test_key == "cycle":
                    sys.stderr.write(
                        format_cycle_results(results, num_experts, args.num_processes)
                        + "\n"
                    )
                else:
                    sys.stderr.write(
                        format_single_op_results(
                            results, test_key, num_experts, args.num_processes
                        )
                        + "\n"
                    )
            else:
                sys.stderr.write(f"FAILED: {test_name}\n")
                print_results(results)

    # Summary table
    sys.stderr.write(f"\n{'='*105}\n")
    sys.stderr.write("SUMMARY: Control Plane Latencies (ms) by Expert Count\n")
    sys.stderr.write(f"{'='*105}\n")
    sys.stderr.write(
        f"{'Exp/Rank':<10} {'Total':<8} {'Init':<12} {'Connect':<12} {'Disconnect':<12} {'Reconnect':<12} {'Destroy':<12} {'Total(ms)':<12}\n"
    )
    sys.stderr.write(f"{'-'*105}\n")

    for num_experts in expert_counts:
        expert_results = all_results.get(num_experts, {})
        total_experts = num_experts * args.num_processes

        # Check if we have cycle results (all-in-one)
        if "cycle" in expert_results:
            results = expert_results["cycle"]
            rank0 = next((r for r in results if r.rank == 0), None)
            if rank0 and rank0.metrics:
                m = rank0.metrics
                reconnect_str = (
                    f"{m['reconnect']['avg_ms']:<12.2f}"
                    if "reconnect" in m
                    else f"{'-':<12}"
                )
                sys.stderr.write(
                    f"{num_experts:<10} "
                    f"{total_experts:<8} "
                    f"{m['init']['avg_ms']:<12.2f} "
                    f"{m['connect']['avg_ms']:<12.2f} "
                    f"{m['disconnect']['avg_ms']:<12.2f} "
                    f"{reconnect_str}"
                    f"{m['destroy']['avg_ms']:<12.2f} "
                    f"{m['total_avg_ms']:<12.2f}\n"
                )
        else:
            # Build row from individual test results
            def get_avg(test_key):
                if test_key not in expert_results:
                    return None
                results = expert_results[test_key]
                # Aggregate across all ranks
                avgs = [
                    r.metrics["avg_latency_ms"]
                    for r in results
                    if r.metrics and "avg_latency_ms" in r.metrics
                ]
                return sum(avgs) / len(avgs) if avgs else None

            init_avg = get_avg("init")
            connect_avg = get_avg("connect")
            disconnect_avg = get_avg("disconnect")
            reconnect_avg = get_avg("reconnect")
            destroy_avg = get_avg("destroy")

            # Calculate total if we have all components
            total = None
            if all(
                v is not None
                for v in [init_avg, connect_avg, disconnect_avg, destroy_avg]
            ):
                total = init_avg + connect_avg + disconnect_avg + destroy_avg
                if reconnect_avg is not None:
                    total += reconnect_avg

            # Format values (show "-" if not measured)
            fmt = lambda v: f"{v:<12.2f}" if v is not None else f"{'-':<12}"

            sys.stderr.write(
                f"{num_experts:<10} "
                f"{total_experts:<8} "
                f"{fmt(init_avg)}"
                f"{fmt(connect_avg)}"
                f"{fmt(disconnect_avg)}"
                f"{fmt(reconnect_avg)}"
                f"{fmt(destroy_avg)}"
                f"{fmt(total)}\n"
            )

    sys.stderr.write(f"{'='*105}\n")
    sys.stderr.write("Parameters:\n")
    sys.stderr.write(f"  Ranks: {args.num_processes}\n")
    sys.stderr.write(
        f"  Experts/rank: {expert_counts} (total: {[e * args.num_processes for e in expert_counts]})\n"
    )
    sys.stderr.write(f"  Tests: {tests_to_run}\n")
    sys.stderr.write(f"  Warmup: {args.warmup} rounds, Measure: {args.rounds} rounds\n")
    sys.stderr.write(f"  Timeout: {args.timeout}s\n")
    sys.stderr.write(f"{'='*105}\n\n")

    # Save results to JSON if requested
    if args.output:
        output_data = collect_all_results(all_results, args.num_processes)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        sys.stderr.write(f"Results saved to: {args.output}\n")

    # Save to collector for CI/CD tracking
    if HAS_COLLECTOR:
        try:
            collector = ResultsCollector(results_dir=args.results_dir)

            for num_experts, expert_results in all_results.items():
                for test_name, results in expert_results.items():
                    # Aggregate metrics across ranks
                    passed = sum(1 for r in results if r.passed)
                    total = len(results)

                    metrics = {}
                    if test_name == "cycle":
                        # Cycle has structured metrics
                        rank0 = next((r for r in results if r.rank == 0), None)
                        if rank0 and rank0.metrics:
                            m = rank0.metrics
                            metrics = {
                                "init_ms": m.get("init", {}).get("avg_ms", 0),
                                "connect_ms": m.get("connect", {}).get("avg_ms", 0),
                                "disconnect_ms": m.get("disconnect", {}).get(
                                    "avg_ms", 0
                                ),
                                "reconnect_ms": (
                                    m.get("reconnect", {}).get("avg_ms", 0)
                                    if "reconnect" in m
                                    else 0
                                ),
                                "destroy_ms": m.get("destroy", {}).get("avg_ms", 0),
                                "total_ms": m.get("total_avg_ms", 0),
                            }
                    else:
                        # Individual tests have simpler metrics
                        latencies = [
                            r.metrics.get("avg_latency_ms", 0)
                            for r in results
                            if r.metrics and r.passed
                        ]
                        if latencies:
                            metrics = {
                                f"{test_name}_ms": sum(latencies) / len(latencies),
                            }

                    metrics["passed_ranks"] = passed
                    metrics["total_ranks"] = total

                    config = {
                        "num_processes": args.num_processes,
                        "num_experts_per_rank": num_experts,
                        "num_tokens": DEFAULT_NUM_TOKENS,
                        "hidden_dim": DEFAULT_HIDDEN,
                    }

                    collector.record_result(
                        test_type="control_plane",
                        test_name=test_name,
                        config=config,
                        metrics=metrics,
                        passed=(passed == total),
                    )

            saved_path = collector.save()
            sys.stderr.write(f"\n[CI/CD] Results saved to: {saved_path}\n")
        except Exception as e:
            sys.stderr.write(f"\n[CI/CD] Warning: Failed to save to collector: {e}\n")


if __name__ == "__main__":
    main()
