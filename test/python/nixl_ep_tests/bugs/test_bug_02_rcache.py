#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BUG-02: UCX rcache Assertion Failure

================================================================================
SUMMARY
================================================================================
The UCX library's region cache (rcache) occasionally fails an assertion during
buffer cleanup, causing the process to abort. This is a reference counting
issue in UCX's memory registration cache.

Component:     UCX library
Severity:      🔴 Critical
Status:        Open - needs UCX team investigation
First seen:    December 2025
Cluster:       DFW (pool0-00290)

================================================================================
REPRODUCTION
================================================================================
1. Create buffer with 16 experts per rank (critical!)
2. Connect all 8 ranks
3. Disconnect all ranks
4. Reconnect all ranks
5. Call destroy()
6. CRASH: ~30% chance of assertion failure

================================================================================
ERROR OUTPUT
================================================================================
[pool0-00290:2841279:0:2841279] rcache.c:477 Assertion `region->refcount > 0' failed

================================================================================
INTERESTING FINDING
================================================================================
The bug is MOST reproducible with exactly 16 experts per rank:

| Experts | Failure Rate |
|---------|--------------|
| 2       | ~0%          |
| 4       | ~0%          |
| 8       | ~5%          |
| 16      | ~30%         | <-- Most reproducible
| 32      | ~10%         |

This suggests a specific memory layout or region count triggers the bug.
16 experts creates a particular number of memory regions that hits a race
condition in UCX's reference counting logic.

================================================================================
ROOT CAUSE ANALYSIS
================================================================================
The UCX rcache maintains reference counts for registered memory regions.
During cleanup with 16 experts:

1. Multiple memory regions are deregistered simultaneously
2. Race condition between deregistration and reference count decrement
3. One thread decrements refcount while another is accessing the region
4. Assertion fails: region->refcount > 0

Possible contributing factors:
- CUDA memory deregistration order
- GDR (GPU Direct RDMA) region cleanup
- Concurrent cleanup from multiple NIXL agents

================================================================================
IMPACT
================================================================================
- Intermittent test failures (~30% with 16 experts)
- CI/CD pipeline flakiness
- Cannot reliably benchmark 16-expert configurations
- Users may see random crashes in production

================================================================================
WORKAROUND
================================================================================
1. Skip 16-expert tests: --experts=2,4,8,32
2. Retry failed tests (usually succeeds on retry)
3. Run tests multiple times and accept occasional failures

================================================================================
TO VERIFY FIX
================================================================================
# Run 10 times - all should pass after fix
for i in {1..10}; do
    echo "=== Run $i ==="
    python3 tests/bugs/test_bug_02_rcache.py --num-processes=8 --timeout=300
done

Expected BEFORE fix: ~3 out of 10 runs fail with rcache assertion
Expected AFTER fix:  All 10 runs succeed

================================================================================
"""

import argparse
import os
import sys
import time

import pytest

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mp_runner import create_buffer, run_multiprocess_test, sync_all_ranks


@pytest.mark.skip(reason="Not run directly")
def _test_rcache_16_experts_fn(
    rank: int, world_size: int, local_rank: int = 0, num_experts: int = 16, **kwargs
) -> dict:
    """
    Test that triggers UCX rcache assertion with 16 experts.

    The key sequence is: connect -> disconnect -> reconnect -> destroy

    EXPECTED BEFORE FIX: ~30% chance of 'rcache.c:477 Assertion' crash
    EXPECTED AFTER FIX: Always succeeds
    """
    import torch

    import nixl_ep

    # Note: setup_worker_environment already sets CUDA_VISIBLE_DEVICES to local_rank
    # so only device 0 is visible. Don't call set_device(local_rank) - that would fail!

    sys.stderr.write(
        f"[Rank {rank}] Starting rcache test with {num_experts} experts/rank\n"
    )

    # Use shared create_buffer helper for consistency with other tests
    sys.stderr.write(f"[Rank {rank}] Creating buffer...\n")
    buffer = create_buffer(rank, world_size, num_experts_per_rank=num_experts)
    sys.stderr.write(f"[Rank {rank}] Syncing all ranks...\n")
    sync_all_ranks(rank, world_size, "init")
    sys.stderr.write(f"[Rank {rank}] Buffer created\n")

    # Connect to all other ranks
    other_ranks = [r for r in range(world_size) if r != rank]

    sys.stderr.write(f"[Rank {rank}] Connecting to {len(other_ranks)} ranks...\n")
    torch.cuda.synchronize()
    buffer.connect_ranks(other_ranks)
    torch.cuda.synchronize()
    sys.stderr.write(f"[Rank {rank}] Syncing all ranks...\n")
    sync_all_ranks(rank, world_size, "connected")
    sys.stderr.write(f"[Rank {rank}] Connected\n")

    # Disconnect from all ranks
    sys.stderr.write(f"[Rank {rank}] Disconnecting...\n")
    torch.cuda.synchronize()
    buffer.disconnect_ranks(other_ranks)
    torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "disconnected")
    sys.stderr.write(f"[Rank {rank}] Disconnected\n")

    # Reconnect - this is part of the sequence that triggers the bug
    sys.stderr.write(f"[Rank {rank}] Reconnecting...\n")
    torch.cuda.synchronize()
    buffer.connect_ranks(other_ranks)
    torch.cuda.synchronize()

    sys.stderr.write(f"[Rank {rank}] Syncing all ranks...\n")
    sync_all_ranks(rank, world_size, "reconnected")
    sys.stderr.write(f"[Rank {rank}] Reconnected\n")

    # Destroy - this is where the rcache assertion typically fails
    sys.stderr.write(
        f"[Rank {rank}] Destroying buffer (rcache assertion may occur here)...\n"
    )
    sync_all_ranks(rank, world_size, "pre_destroy")

    buffer.destroy()

    sync_all_ranks(rank, world_size, "destroyed")
    sys.stderr.write(f"[Rank {rank}] Buffer destroyed successfully!\n")

    return {
        "passed": True,
        "num_experts": num_experts,
        "message": f"Completed full cycle with {num_experts} experts without rcache assertion",
    }


def main():
    parser = argparse.ArgumentParser(
        description="BUG-02: Test UCX rcache assertion (intermittent crash)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes/ranks to spawn",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=16,
        help="Number of experts per rank (16 is most likely to trigger bug)",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of sequential runs to perform"
    )
    args = parser.parse_args()

    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("BUG-02: UCX rcache Assertion Failure\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f"Ranks: {args.num_processes}\n")
    sys.stderr.write(f"Experts per rank: {args.experts}\n")
    sys.stderr.write(f"Timeout: {args.timeout}s\n")
    sys.stderr.write(f"Sequential runs: {args.runs}\n")
    sys.stderr.write("\n")
    sys.stderr.write(
        "EXPECTED BEFORE FIX: ~30% chance of 'rcache.c:477 Assertion' crash\n"
    )
    sys.stderr.write("EXPECTED AFTER FIX:  All runs succeed\n")
    sys.stderr.write("\n")
    sys.stderr.write("NOTE: 16 experts is most likely to trigger the bug.\n")
    sys.stderr.write("      2, 4, 8, 32 experts rarely trigger it.\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("\n")

    successful_runs = 0
    failed_runs = 0

    for run_num in range(args.runs):
        if args.runs > 1:
            sys.stderr.write(f"\n{'='*70}\n")
            sys.stderr.write(f"RUN {run_num + 1}/{args.runs}\n")
            sys.stderr.write(f"{'='*70}\n\n")

        results = run_multiprocess_test(
            test_fn=_test_rcache_16_experts_fn,
            num_processes=args.num_processes,
            test_name=f"BUG-02: rcache with {args.experts} experts (run {run_num + 1})",
            timeout=args.timeout,
            num_experts=args.experts,
        )

        # Check results - TestResult is a dataclass with .passed attribute
        passed = sum(1 for r in results if r.passed)

        if passed == args.num_processes:
            successful_runs += 1
            sys.stderr.write(
                f"✅ Run {run_num + 1}: PASSED ({passed}/{args.num_processes} ranks)\n"
            )
        else:
            failed_runs += 1
            sys.stderr.write(
                f"❌ Run {run_num + 1}: FAILED ({passed}/{args.num_processes} ranks)\n"
            )

    sys.stderr.write("\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f"RESULTS: {successful_runs}/{args.runs} runs succeeded\n")
    sys.stderr.write("=" * 70 + "\n")

    if failed_runs == 0:
        sys.stderr.write(
            "✅ BUG-02 FIX VERIFIED: All runs completed without rcache assertion\n"
        )
        sys.stderr.write("   (Run more times to be sure: --runs=10)\n")
    else:
        sys.stderr.write(
            f"❌ BUG-02 REPRODUCED: {failed_runs}/{args.runs} runs failed\n"
        )
        sys.stderr.write("   The UCX rcache bug is still present.\n")
        sys.stderr.write("\n")
        sys.stderr.write("   Failure rate: {:.0%}\n".format(failed_runs / args.runs))
        sys.stderr.write("   Expected rate before fix: ~30%\n")
    sys.stderr.write("=" * 70 + "\n")

    return 0 if failed_runs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
