# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for NIXL EP Buffer data plane operations.

Measures throughput and latency of:
- dispatch() - sending tokens to experts
- combine() - gathering results from experts
- Full dispatch+combine cycle

Tests with varying configurations:
- Token counts (128, 512, 2048, 4096)
- Hidden dimensions (1024, 4096, 8192)
- Expert counts (8, 16, 32)
- TopK values (1, 2, 4)

Usage:
    # Basic test (default configuration)
    python3 test_data_plane.py --num-processes=8

    # Sweep token sizes
    python3 test_data_plane.py --num-processes=8 --tokens=128,512,2048,4096

    # Sweep hidden dimensions
    python3 test_data_plane.py --num-processes=8 --hidden=1024,4096,8192

    # Full matrix test
    python3 test_data_plane.py --num-processes=8 --test=all
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import pytest

# Add parent directory to path
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)

from utils.mp_runner import (  # noqa: E402
    TestResult,
    create_buffer,
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

DEFAULT_TOKEN_COUNTS = [512]
DEFAULT_HIDDEN_DIMS = [4096]
DEFAULT_EXPERT_COUNTS = [8]
DEFAULT_TOPK = 2
DEFAULT_WARMUP_ITERS = 10
DEFAULT_MEASURE_ITERS = 100


# ============================================================================
# P-THRU-01: Dispatch Throughput
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_dispatch_throughput_fn(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    num_experts_per_rank: int = 8,
    num_tokens: int = 512,
    hidden: int = 4096,
    topk: int = 2,
    nvlink_backend: str = "ipc",
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    measure_iters: int = DEFAULT_MEASURE_ITERS,
    **kwargs,
) -> dict:
    """
    P-THRU-01/02: Measure dispatch throughput (tokens/sec).

    Uses CUDA events for precise GPU timing (matches elastic.py methodology).
    """
    import numpy as np
    import torch

    import nixl_ep  # noqa: F401

    total_experts = num_experts_per_rank * world_size

    # Create buffer
    buffer = create_buffer(
        rank,
        world_size,
        num_experts_per_rank=num_experts_per_rank,
        hidden=hidden,
        num_tokens=num_tokens,
        nvlink_backend=nvlink_backend,
    )

    sync_all_ranks(rank, world_size, "dp_init")

    # Connect to all ranks
    other_ranks = [r for r in range(world_size) if r != rank]
    if other_ranks:
        torch.cuda.synchronize()
        buffer.connect_ranks(other_ranks)
        torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "dp_connected")

    # Create test data
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.randint(
        0, total_experts, (num_tokens, topk), dtype=torch.int64, device="cuda"
    )

    # Calculate bytes per dispatch (same as elastic.py)
    num_fp8_bytes = hidden + hidden // 128 * 4 + 16
    num_dispatch_comm_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections

    def dispatch_fn():
        recv_x, recv_count, handle, event, hook = buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            num_max_dispatch_tokens_per_rank=num_tokens,
            num_experts=total_experts,
            use_fp8=True,
            async_finish=False,  # Synchronous (like elastic.py)
        )

    # Flush L2 cache
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(warmup_iters):
        dispatch_fn()

    # Flush L2
    cache.zero_()

    sync_all_ranks(rank, world_size, "dp_warmup")

    # Measure using CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]

    for i in range(measure_iters):
        start_events[i].record()
        dispatch_fn()
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate times (skip first iteration)
    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    avg_t = np.average(times)

    # Calculate metrics
    tokens_per_sec = num_tokens / avg_t
    avg_latency_us = avg_t * 1e6
    bandwidth_gbps = num_dispatch_comm_bytes / 1e9 / avg_t

    sync_all_ranks(rank, world_size, "dp_measured")

    # Cleanup
    buffer.destroy()
    sync_all_ranks(rank, world_size, "dp_cleanup")

    return {
        "passed": True,
        "metrics": {
            "tokens_per_sec": tokens_per_sec,
            "avg_latency_us": avg_latency_us,
            "bandwidth_gbps": bandwidth_gbps,
            "num_tokens": num_tokens,
            "hidden": hidden,
            "topk": topk,
            "total_experts": total_experts,
            "measure_iters": measure_iters,
        },
    }


# ============================================================================
# P-THRU-03/04: Combine Throughput
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_combine_throughput_fn(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    num_experts_per_rank: int = 8,
    num_tokens: int = 512,
    hidden: int = 4096,
    topk: int = 2,
    nvlink_backend: str = "ipc",
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    measure_iters: int = DEFAULT_MEASURE_ITERS,
    use_logfmt: bool = False,
    **kwargs,
) -> dict:
    """
    P-THRU-03/04: Measure combine throughput (tokens/sec).

    Uses CUDA events for precise GPU timing (matches elastic.py methodology).
    Note: Combine requires a dispatch first to get a handle, so we measure
    the dispatch+combine cycle and subtract the dispatch time.
    """
    import numpy as np
    import torch

    import nixl_ep  # noqa: F401

    total_experts = num_experts_per_rank * world_size

    # Create buffer
    buffer = create_buffer(
        rank,
        world_size,
        num_experts_per_rank=num_experts_per_rank,
        hidden=hidden,
        num_tokens=num_tokens,
        nvlink_backend=nvlink_backend,
    )

    sync_all_ranks(rank, world_size, "comb_init")

    # Connect to all ranks
    other_ranks = [r for r in range(world_size) if r != rank]
    if other_ranks:
        torch.cuda.synchronize()
        buffer.connect_ranks(other_ranks)
        torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "comb_connected")

    # Create test data
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.randint(
        0, total_experts, (num_tokens, topk), dtype=torch.int64, device="cuda"
    )
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Calculate bytes per combine (same as elastic.py)
    if use_logfmt:
        num_combine_bytes = hidden * 10 // 8 + hidden // 128 * 4
    else:
        num_combine_bytes = hidden * 2

    num_combine_comm_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_combine_comm_bytes += num_combine_bytes * num_selections

    # Do one dispatch to get the correct shape for simulated_gemm_x
    recv_x, recv_count, handle_init, event, hook = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=total_experts,
        use_fp8=True,
        async_finish=False,
    )

    # Pre-allocate simulated expert output with correct shape
    simulated_gemm_x = recv_x[0].to(torch.bfloat16).clone()

    # We need dispatch first to get handle, but we'll only measure combine
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
        combined_x, comb_event, comb_hook = buffer.combine(
            x=simulated_gemm_x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=handle,
            use_logfmt=use_logfmt,
        )

    # Flush L2 cache
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(warmup_iters):
        recv_x, recv_count, handle, event, hook = dispatch_fn()
        combine_fn(handle)

    # Flush L2
    cache.zero_()

    sync_all_ranks(rank, world_size, "comb_warmup")

    # Measure combine only using CUDA events
    # Strategy: measure dispatch+combine, then subtract dispatch time
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    mid_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]

    for i in range(measure_iters):
        start_events[i].record()
        recv_x, recv_count, handle, event, hook = dispatch_fn()
        mid_events[i].record()
        combine_fn(handle)
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate combine-only times (skip first iteration)
    combine_times = np.array(
        [m.elapsed_time(e) / 1e3 for m, e in zip(mid_events, end_events)]
    )[1:]
    avg_t = np.average(combine_times)

    # Calculate metrics
    tokens_per_sec = num_tokens / avg_t
    avg_latency_us = avg_t * 1e6
    bandwidth_gbps = num_combine_comm_bytes / 1e9 / avg_t

    sync_all_ranks(rank, world_size, "comb_measured")

    # Cleanup
    buffer.destroy()
    sync_all_ranks(rank, world_size, "comb_cleanup")

    return {
        "passed": True,
        "metrics": {
            "tokens_per_sec": tokens_per_sec,
            "avg_latency_us": avg_latency_us,
            "bandwidth_gbps": bandwidth_gbps,
            "num_tokens": num_tokens,
            "hidden": hidden,
            "topk": topk,
            "total_experts": total_experts,
            "use_logfmt": use_logfmt,
            "measure_iters": measure_iters,
        },
    }


# ============================================================================
# P-THRU-05: End-to-End Dispatch + Combine (elastic.py methodology)
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_e2e_throughput_fn(
    rank: int,
    world_size: int,
    local_rank: int = 0,
    num_experts_per_rank: int = 8,
    num_tokens: int = 512,
    hidden: int = 4096,
    topk: int = 2,
    nvlink_backend: str = "ipc",
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    measure_iters: int = DEFAULT_MEASURE_ITERS,
    use_logfmt: bool = False,
    **kwargs,
) -> dict:
    """
    P-THRU-05: Measure end-to-end dispatch + combine throughput.

    Uses elastic.py methodology for accurate comparison:
    - CUDA events for precise timing
    - async_finish=False (synchronous operations)
    - No intermediate sync between dispatch and combine
    - Pre-allocated simulated_gemm_x (no compute in timing loop)
    """
    import numpy as np
    import torch

    import nixl_ep  # noqa: F401

    total_experts = num_experts_per_rank * world_size

    # Create buffer
    buffer = create_buffer(
        rank,
        world_size,
        num_experts_per_rank=num_experts_per_rank,
        hidden=hidden,
        num_tokens=num_tokens,
        nvlink_backend=nvlink_backend,
    )

    sync_all_ranks(rank, world_size, "e2e_init")

    # Connect to all ranks
    other_ranks = [r for r in range(world_size) if r != rank]
    if other_ranks:
        torch.cuda.synchronize()
        buffer.connect_ranks(other_ranks)
        torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "e2e_connected")

    # Create test data (same as elastic.py)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.randint(
        0, total_experts, (num_tokens, topk), dtype=torch.int64, device="cuda"
    )
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Calculate bytes transferred (same as elastic.py lines 217-223)
    num_fp8_bytes = hidden + hidden // 128 * 4 + 16  # FP8 with scales
    if use_logfmt:
        num_combine_bytes = hidden * 10 // 8 + hidden // 128 * 4
    else:
        num_combine_bytes = hidden * 2  # BF16

    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_combine_bytes * num_selections

    # Do one dispatch to get the shape for simulated_gemm_x
    # (shape is [num_local_experts, num_max_tokens * num_ranks, hidden])
    recv_x, recv_count, handle, event, hook = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=total_experts,
        use_fp8=True,
        async_finish=False,
    )

    # Pre-allocate simulated_gemm_x with correct shape (like elastic.py line 130-131)
    # recv_x[0] is the FP8 data tensor, clone its shape in BF16
    simulated_gemm_x = recv_x[0].to(torch.bfloat16).clone()

    def test_func():
        """Run one dispatch + combine cycle (elastic.py pattern)."""
        # Dispatch with async_finish=False (synchronous internally)
        recv_x, recv_count, handle, event, hook = buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            num_max_dispatch_tokens_per_rank=num_tokens,
            num_experts=total_experts,
            use_fp8=True,
            async_finish=False,  # Match elastic.py
        )
        # Combine immediately (no intermediate sync, no compute)
        combined_x, comb_event, comb_hook = buffer.combine(
            x=simulated_gemm_x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=handle,
            use_logfmt=use_logfmt,
        )

    # Flush L2 cache (like bench() does)
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(warmup_iters):
        test_func()

    # Flush L2
    cache.zero_()

    sync_all_ranks(rank, world_size, "e2e_warmup")

    # Measure using CUDA events (like bench() in utils.py)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]

    for i in range(measure_iters):
        start_events[i].record()
        test_func()
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate times (skip first iteration like bench() does)
    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    avg_t = np.average(times)
    min_t = np.min(times)
    max_t = np.max(times)

    # Calculate metrics (same as elastic.py line 227)
    total_comm_bytes = num_dispatch_comm_bytes + num_combine_comm_bytes
    bandwidth_gbps = total_comm_bytes / 1e9 / avg_t

    # Also calculate token throughput for consistency with other tests
    tokens_per_sec = num_tokens / avg_t
    avg_latency_us = avg_t * 1e6

    sync_all_ranks(rank, world_size, "e2e_measured")

    # Cleanup
    buffer.destroy()
    sync_all_ranks(rank, world_size, "e2e_cleanup")

    return {
        "passed": True,
        "metrics": {
            "tokens_per_sec": tokens_per_sec,
            "avg_latency_us": avg_latency_us,
            "bandwidth_gbps": bandwidth_gbps,
            "min_latency_us": min_t * 1e6,
            "max_latency_us": max_t * 1e6,
            "num_tokens": num_tokens,
            "hidden": hidden,
            "topk": topk,
            "total_experts": total_experts,
            "use_logfmt": use_logfmt,
            "measure_iters": measure_iters,
        },
    }


# ============================================================================
# Result Aggregation
# ============================================================================


def aggregate_metrics(results: List[TestResult], metric_name: str) -> Dict[str, float]:
    """Aggregate a metric across all ranks."""
    values = []
    for r in results:
        if r.passed and r.metrics and metric_name in r.metrics:
            values.append(r.metrics[metric_name])

    if not values:
        return {"avg": 0, "min": 0, "max": 0}

    return {
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def print_throughput_results(test_name: str, results: List[TestResult]):
    """Print formatted throughput results."""
    if not results:
        sys.stderr.write(f"No results for {test_name}\n")
        return

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    sys.stderr.write(f"\n{'='*70}\n")
    sys.stderr.write(f"{test_name}\n")
    sys.stderr.write(f"{'='*70}\n")
    sys.stderr.write(f"Status: {passed}/{total} ranks passed\n")

    if passed == 0:
        return

    # Get first result's config
    first_metrics: Dict[str, Any] = {}
    for r in results:
        if r.passed and r.metrics:
            first_metrics = r.metrics
            break
    if first_metrics:
        sys.stderr.write(
            f"Config: {first_metrics.get('num_tokens', '?')} tokens, "
            f"{first_metrics.get('hidden', '?')} hidden, "
            f"topk={first_metrics.get('topk', '?')}, "
            f"{first_metrics.get('total_experts', '?')} total experts\n"
        )

    # Aggregate metrics
    tokens = aggregate_metrics(results, "tokens_per_sec")
    latency = aggregate_metrics(results, "avg_latency_us")
    bandwidth = aggregate_metrics(results, "bandwidth_gbps")

    sys.stderr.write("\nThroughput (tokens/sec):\n")
    sys.stderr.write(f"  Avg: {tokens['avg']:,.0f}\n")
    sys.stderr.write(f"  Min: {tokens['min']:,.0f}\n")
    sys.stderr.write(f"  Max: {tokens['max']:,.0f}\n")

    sys.stderr.write("\nLatency (μs):\n")
    sys.stderr.write(f"  Avg: {latency['avg']:,.1f}\n")
    sys.stderr.write(f"  Min: {latency['min']:,.1f}\n")
    sys.stderr.write(f"  Max: {latency['max']:,.1f}\n")

    sys.stderr.write("\nBandwidth (GB/s):\n")
    sys.stderr.write(f"  Avg: {bandwidth['avg']:.2f}\n")
    sys.stderr.write(f"  Min: {bandwidth['min']:.2f}\n")
    sys.stderr.write(f"  Max: {bandwidth['max']:.2f}\n")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Data plane performance tests for NIXL EP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test (8 experts/rank = 64 total with 8 ranks)
    python3 test_data_plane.py --num-processes=8 --experts-per-rank=8

    # RDMA-only with 16 experts/rank (128 total)
    python3 test_data_plane.py --num-processes=8 --experts-per-rank=16 --nvlink-backend=none

    # Full config: 32 experts/rank (256 total), 2048 tokens, hidden=7168
    python3 test_data_plane.py --num-processes=8 --experts-per-rank=32 --tokens=2048 --hidden=7168 --topk=8
""",
    )

    parser.add_argument(
        "--num-processes", type=int, default=8, help="Number of processes/ranks"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["dispatch", "combine", "e2e", "all"],
        help="Test to run",
    )
    parser.add_argument(
        "--tokens", type=str, default="512", help="Token counts (comma-separated)"
    )
    parser.add_argument(
        "--hidden", type=str, default="4096", help="Hidden dimensions (comma-separated)"
    )
    parser.add_argument(
        "--experts-per-rank",
        type=str,
        default="8",
        help="Experts per rank, comma-separated. Total experts = experts_per_rank * num_ranks",
    )
    parser.add_argument("--topk", type=int, default=2, help="TopK value")
    parser.add_argument(
        "--nvlink-backend",
        type=str,
        default="ipc",
        choices=["nixl", "ipc", "none"],
        help="NVLink backend: ipc (default), nixl, or none (forces RDMA)",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP_ITERS, help="Warmup iterations"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_MEASURE_ITERS,
        help="Measurement iterations",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for CI/CD results (default: NIXL_RESULTS_DIR env or perf/results/)",
    )

    args = parser.parse_args()

    # Parse comma-separated values
    token_counts = [int(x) for x in args.tokens.split(",")]
    hidden_dims = [int(x) for x in args.hidden.split(",")]
    expert_counts = [int(x) for x in args.experts_per_rank.split(",")]

    tests_to_run = []
    if args.test == "all":
        tests_to_run = ["dispatch", "combine", "e2e"]
    else:
        tests_to_run = [args.test]

    all_results = {}

    nvlink_backend = getattr(args, "nvlink_backend", "ipc")

    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("NIXL EP Data Plane Performance Tests\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f"Processes: {args.num_processes}\n")
    sys.stderr.write(f"Tests: {', '.join(tests_to_run)}\n")
    sys.stderr.write(f"Token counts: {token_counts}\n")
    sys.stderr.write(f"Hidden dims: {hidden_dims}\n")
    sys.stderr.write(f"Experts/rank: {expert_counts} (total = experts × ranks)\n")
    sys.stderr.write(f"TopK: {args.topk}\n")
    sys.stderr.write(
        f"NVLink backend: {nvlink_backend}"
        + (" (RDMA forced)" if nvlink_backend == "none" else "")
        + "\n"
    )
    sys.stderr.write(f"Warmup: {args.warmup} iters, Measure: {args.iters} iters\n")
    sys.stderr.write("=" * 70 + "\n")

    # Initialize collector early for incremental saving
    collector = None
    progress_log = None
    progress_log_path = None
    if HAS_COLLECTOR and args.results_dir:
        try:
            collector = ResultsCollector(results_dir=args.results_dir)

            # Find next available progress log number
            log_num = 1
            while True:
                progress_log_path = os.path.join(
                    args.results_dir, f"progress_log_{log_num:03d}.txt"
                )
                if not os.path.exists(progress_log_path):
                    break
                log_num += 1

            # Create new progress log file
            progress_log = open(progress_log_path, "w")
            progress_log.write(f"{'='*80}\n")
            progress_log.write(f"NIXL EP Data Plane Performance Test Run #{log_num}\n")
            progress_log.write(f"Started: {datetime.now().isoformat()}\n")
            progress_log.write(
                f"Config: {args.num_processes} ranks, topk={args.topk}, backend={nvlink_backend}\n"
            )
            progress_log.write(
                f"Tokens: {token_counts}, Hidden: {hidden_dims}, Experts/rank: {expert_counts}\n"
            )
            progress_log.write(f"{'='*80}\n\n")
            progress_log.write(
                f"{'Time':<10} | {'Test':<35} | {'Status':<6} | {'BW (GB/s)':<10} | {'Lat (μs)':<10}\n"
            )
            progress_log.write(f"{'-'*80}\n")
            progress_log.flush()
            sys.stderr.write(f"[CI/CD] Progress log: {progress_log_path}\n")
        except Exception as e:
            sys.stderr.write(f"[CI/CD] Warning: Failed to initialize collector: {e}\n")

    test_fns = {
        "dispatch": (_test_dispatch_throughput_fn, "P-THRU: Dispatch Throughput"),
        "combine": (_test_combine_throughput_fn, "P-THRU: Combine Throughput"),
        "e2e": (_test_e2e_throughput_fn, "P-THRU: End-to-End Throughput"),
    }

    for test_name in tests_to_run:
        test_fn, description = test_fns[test_name]

        for num_tokens in token_counts:
            for hidden in hidden_dims:
                for num_experts in expert_counts:
                    config_key = f"{test_name}_t{num_tokens}_h{hidden}_e{num_experts}"

                    total_experts = num_experts * args.num_processes
                    sys.stderr.write(f"\n{'='*70}\n")
                    sys.stderr.write(f"Running: {description}\n")
                    sys.stderr.write(
                        f"Config: {num_tokens} tokens, {hidden} hidden, {num_experts} experts/rank ({total_experts} total)\n"
                    )
                    sys.stderr.write(f"{'='*70}\n")

                    results = run_multiprocess_test(
                        test_fn=test_fn,
                        num_processes=args.num_processes,
                        timeout=args.timeout,
                        num_experts_per_rank=num_experts,
                        num_tokens=num_tokens,
                        hidden=hidden,
                        topk=args.topk,
                        nvlink_backend=nvlink_backend,
                        warmup_iters=args.warmup,
                        measure_iters=args.iters,
                    )

                    print_throughput_results(
                        f"{description} ({num_tokens}t, {hidden}h, {num_experts}e)",
                        results,
                    )

                    all_results[config_key] = {
                        "test": test_name,
                        "config": {
                            "num_tokens": num_tokens,
                            "hidden": hidden,
                            "num_experts_per_rank": num_experts,
                            "topk": args.topk,
                        },
                        "results": [
                            {
                                "rank": r.rank,
                                "passed": r.passed,
                                "metrics": r.metrics,
                                "error": r.error,
                            }
                            for r in results
                        ],
                    }

                    # Incremental save after each test
                    if collector:
                        try:
                            passed = sum(1 for r in results if r.passed)
                            total = len(results)

                            # Aggregate metrics
                            metrics = {}
                            if passed > 0:
                                tokens_per_sec = [
                                    r.metrics["tokens_per_sec"]
                                    for r in results
                                    if r.passed and r.metrics
                                ]
                                bandwidth_gbps = [
                                    r.metrics.get("bandwidth_gbps", 0)
                                    for r in results
                                    if r.passed and r.metrics
                                ]
                                latency_us = [
                                    r.metrics.get("avg_latency_us", 0)
                                    for r in results
                                    if r.passed and r.metrics
                                ]

                                metrics = {
                                    "throughput_tok_per_sec": (
                                        sum(tokens_per_sec) / len(tokens_per_sec)
                                        if tokens_per_sec
                                        else 0
                                    ),
                                    "bandwidth_gbps": (
                                        sum(bandwidth_gbps) / len(bandwidth_gbps)
                                        if bandwidth_gbps
                                        else 0
                                    ),
                                    "latency_us": (
                                        sum(latency_us) / len(latency_us)
                                        if latency_us
                                        else 0
                                    ),
                                    "passed_ranks": passed,
                                    "total_ranks": total,
                                }

                            config = {
                                "num_processes": args.num_processes,
                                "num_tokens": num_tokens,
                                "hidden_dim": hidden,
                                "num_experts_per_rank": num_experts,
                                "topk": args.topk,
                                "nvlink_backend": nvlink_backend,
                            }

                            collector.record_result(
                                test_type="data_plane",
                                test_name=test_name,
                                config=config,
                                metrics=metrics,
                                passed=(passed == total),
                            )

                            # Save incrementally
                            collector.save()

                            # Write to progress log
                            if progress_log:
                                status = "PASS" if passed == total else "FAIL"
                                bw = metrics.get("bandwidth_gbps", 0)
                                lat = metrics.get("latency_us", 0)
                                progress_log.write(
                                    f"{datetime.now().strftime('%H:%M:%S')} | {config_key:<35} | {status} | {bw:.1f} GB/s | {lat:.1f} μs\n"
                                )
                                progress_log.flush()
                        except Exception as e:
                            sys.stderr.write(
                                f"[CI/CD] Warning: Failed to save incremental result: {e}\n"
                            )

    # Save results to JSON if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_processes": args.num_processes,
                "tests": tests_to_run,
                "token_counts": token_counts,
                "hidden_dims": hidden_dims,
                "expert_counts": expert_counts,
                "topk": args.topk,
                "warmup_iters": args.warmup,
                "measure_iters": args.iters,
            },
            "results": all_results,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        sys.stderr.write(f"\nResults saved to: {args.output}\n")

    # Summary
    sys.stderr.write("\n" + "=" * 85 + "\n")
    sys.stderr.write("SUMMARY: Data Plane Performance\n")
    sys.stderr.write("=" * 85 + "\n")

    # Parameters section
    sys.stderr.write("Parameters:\n")
    sys.stderr.write(f"  Ranks: {args.num_processes}\n")
    sys.stderr.write(f"  Tokens: {token_counts}\n")
    sys.stderr.write(f"  Hidden: {hidden_dims}\n")
    sys.stderr.write(
        f"  Experts/rank: {expert_counts} (total: {[e * args.num_processes for e in expert_counts]})\n"
    )
    sys.stderr.write(f"  TopK: {args.topk}\n")
    sys.stderr.write(f"  NVLink backend: {nvlink_backend}\n")
    sys.stderr.write(f"  Warmup: {args.warmup} iters, Measure: {args.iters} iters\n")
    sys.stderr.write("-" * 85 + "\n")

    sys.stderr.write(
        f"{'Test':<30} {'Status':<10} {'Throughput':<15} {'Bandwidth':<12} {'Latency':<12}\n"
    )
    sys.stderr.write("-" * 85 + "\n")

    for config_key, data in all_results.items():
        results = data["results"]
        passed = sum(1 for r in results if r["passed"])
        total = len(results)

        if passed > 0:
            tokens_per_sec = [
                r["metrics"]["tokens_per_sec"]
                for r in results
                if r["passed"] and r["metrics"]
            ]
            bandwidth_gbps = [
                r["metrics"]["bandwidth_gbps"]
                for r in results
                if r["passed"] and r["metrics"] and "bandwidth_gbps" in r["metrics"]
            ]
            latency_us = [
                r["metrics"]["avg_latency_us"]
                for r in results
                if r["passed"] and r["metrics"] and "avg_latency_us" in r["metrics"]
            ]

            avg_throughput = (
                sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0
            )
            avg_bandwidth = (
                sum(bandwidth_gbps) / len(bandwidth_gbps) if bandwidth_gbps else 0
            )
            avg_latency = sum(latency_us) / len(latency_us) if latency_us else 0

            # Format throughput nicely
            if avg_throughput >= 1_000_000:
                tput_str = f"{avg_throughput/1_000_000:.2f}M tok/s"
            elif avg_throughput >= 1_000:
                tput_str = f"{avg_throughput/1_000:.1f}K tok/s"
            else:
                tput_str = f"{avg_throughput:.0f} tok/s"

            sys.stderr.write(
                f"{config_key:<30} {passed}/{total:<8} {tput_str:<15} {avg_bandwidth:.1f} GB/s{'':<4} {avg_latency:.1f} μs\n"
            )
        else:
            sys.stderr.write(
                f"{config_key:<30} {passed}/{total:<8} {'FAILED':<15} {'-':<12} {'-':<12}\n"
            )

    sys.stderr.write("=" * 85 + "\n")

    # Close progress log and print final status
    if progress_log:
        try:
            progress_log.write(f"\n{'='*80}\n")
            progress_log.write(f"Test run completed: {datetime.now().isoformat()}\n")
            progress_log.write(f"Total tests: {len(all_results)}\n")
            progress_log.write(f"{'='*80}\n")
            progress_log.close()
        except Exception:
            pass

    if collector:
        sys.stderr.write(
            f"\n[CI/CD] Results saved incrementally to: {collector.results_dir}/raw/\n"
        )
        if progress_log_path:
            sys.stderr.write(f"[CI/CD] Progress log: {progress_log_path}\n")


if __name__ == "__main__":
    main()
