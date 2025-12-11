#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BUG-03: UCX rcache gdr_copy Warning

================================================================================
SUMMARY
================================================================================
UCX emits warnings about memory regions remaining in the LRU list after
operations complete. This indicates leftover memory registration entries
but does not affect functionality.

Component:     UCX library
Severity:      🟡 Low (cosmetic warning)
Status:        Open - needs investigation
First seen:    December 2025
Cluster:       DFW (pool0-00290)

================================================================================
REPRODUCTION
================================================================================
1. Run any full cycle test (init → connect → disconnect → destroy)
2. Check stderr for warning message
3. Warning appears in ~30% of runs

================================================================================
ERROR OUTPUT
================================================================================
[1765125844.886116] [pool0-00290:2842561:0] rcache.c:1392 UCX WARN rcache gdr_copy: 1 regions remained on lru list, first region: 0xad4e380

Variations:
- "1 regions remained on lru list"
- "2 regions remained on lru list"
- "4 regions remained on lru list"

================================================================================
ROOT CAUSE ANALYSIS
================================================================================
GDR (GPU Direct RDMA) copy operations register memory regions for efficient
GPU-to-NIC transfers. These warnings indicate:

1. Memory regions were not explicitly deregistered before process exit
2. UCX's lazy cleanup mechanism left entries in the LRU cache
3. Entries would eventually be cleaned up but weren't at warning time

This is NOT a memory leak - UCX will clean these up. The warning is
informational, indicating suboptimal cleanup order.

Possible causes:
- CUDA device reset before UCX cleanup
- Destroy order (NIXL vs UCX vs CUDA)
- Multiple buffers sharing the same registered regions

================================================================================
OBSERVATIONS
================================================================================
- Appears in ~30% of test runs
- More common with higher expert counts (32 > 8 > 2)
- Does NOT correlate with test failures
- Numbers vary: typically 1-4 regions
- Same warning can appear multiple times (once per rank)

================================================================================
IMPACT
================================================================================
- Clutters test output (makes logs harder to read)
- May indicate minor memory inefficiency
- NO functional impact
- NOT a memory leak

================================================================================
WORKAROUND
================================================================================
None needed - this is a cosmetic warning.

For clean test output, filter stderr:
    python3 test.py 2>&1 | grep -v "rcache gdr_copy"

================================================================================
TO VERIFY FIX
================================================================================
python3 tests/bugs/test_bug_03_gdr_copy.py --num-processes=8 2>&1 | grep -c "rcache gdr_copy"

Expected BEFORE fix: Non-zero count (~30% of the time)
Expected AFTER fix:  Count is 0

================================================================================
"""

import argparse
import os
import sys

import pytest

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mp_runner import create_buffer, run_multiprocess_test, sync_all_ranks


@pytest.mark.skip(reason="Not run directly")
def _test_gdr_copy_warning_fn(
    rank: int, world_size: int, local_rank: int = 0, num_experts: int = 32, **kwargs
) -> dict:
    """
    Test that may produce gdr_copy warnings during cleanup.

    Higher expert counts (32) are more likely to produce warnings.

    EXPECTED BEFORE FIX: ~30% chance of "rcache gdr_copy" warning in stderr
    EXPECTED AFTER FIX: No warnings
    """
    import torch

    import nixl_ep

    # Note: setup_worker_environment already sets CUDA_VISIBLE_DEVICES to local_rank
    # so only device 0 is visible. Don't call set_device(local_rank) - that would fail!

    sys.stderr.write(
        f"[Rank {rank}] Starting gdr_copy test with {num_experts} experts/rank\n"
    )

    # Use shared create_buffer helper with larger buffer to increase chance of warning
    sys.stderr.write(f"[Rank {rank}] Creating buffer...\n")
    buffer = create_buffer(
        rank, world_size, num_experts_per_rank=num_experts, num_tokens=1024
    )

    sync_all_ranks(rank, world_size, "init")

    # Connect to all other ranks
    other_ranks = [r for r in range(world_size) if r != rank]

    sys.stderr.write(f"[Rank {rank}] Connecting...\n")
    torch.cuda.synchronize()
    buffer.connect_ranks(other_ranks)
    torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "connected")
    sys.stderr.write(f"[Rank {rank}] Connected\n")

    # Disconnect
    sys.stderr.write(f"[Rank {rank}] Disconnecting...\n")
    buffer.disconnect_ranks(other_ranks)

    sync_all_ranks(rank, world_size, "disconnected")

    # Destroy - gdr_copy warning may appear here during cleanup
    sys.stderr.write(
        f"[Rank {rank}] Destroying buffer (gdr_copy warning may appear)...\n"
    )
    sync_all_ranks(rank, world_size, "pre_destroy")

    buffer.destroy()

    sync_all_ranks(rank, world_size, "destroyed")
    sys.stderr.write(
        f"[Rank {rank}] Complete - check stderr for 'rcache gdr_copy' warning\n"
    )

    return {
        "passed": True,
        "num_experts": num_experts,
        "message": "Check stderr for 'rcache gdr_copy' warning",
    }


def main():
    parser = argparse.ArgumentParser(
        description="BUG-03: Test UCX gdr_copy warning (cosmetic)"
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
        default=32,
        help="Number of experts per rank (higher = more likely to warn)",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds")
    args = parser.parse_args()

    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("BUG-03: UCX rcache gdr_copy Warning\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f"Ranks: {args.num_processes}\n")
    sys.stderr.write(f"Experts per rank: {args.experts}\n")
    sys.stderr.write(f"Timeout: {args.timeout}s\n")
    sys.stderr.write("\n")
    sys.stderr.write("This test checks for cosmetic warnings, NOT crashes.\n")
    sys.stderr.write("\n")
    sys.stderr.write("To count warnings, run:\n")
    sys.stderr.write(
        "  python3 tests/bugs/test_bug_03_gdr_copy.py 2>&1 | grep -c 'rcache gdr_copy'\n"
    )
    sys.stderr.write("\n")
    sys.stderr.write("EXPECTED BEFORE FIX: Warning count > 0 (~30% of runs)\n")
    sys.stderr.write("EXPECTED AFTER FIX:  Warning count = 0\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("\n")

    results = run_multiprocess_test(
        test_fn=_test_gdr_copy_warning_fn,
        num_processes=args.num_processes,
        test_name="BUG-03: gdr_copy warning",
        timeout=args.timeout,
        num_experts=args.experts,
    )

    # Check results
    passed = sum(1 for r in results if r.passed)

    sys.stderr.write("\n")
    sys.stderr.write("=" * 70 + "\n")
    if passed == args.num_processes:
        sys.stderr.write(f"✅ Test completed: {passed}/{args.num_processes} ranks\n")
        sys.stderr.write("\n")
        sys.stderr.write("Check the output above for warnings like:\n")
        sys.stderr.write("  'rcache gdr_copy: N regions remained on lru list'\n")
        sys.stderr.write("\n")
        sys.stderr.write("If you see such warnings, BUG-03 is present.\n")
        sys.stderr.write(
            "If no warnings, the bug may be fixed (or just didn't trigger this run).\n"
        )
    else:
        sys.stderr.write(f"❌ Test failed: {passed}/{args.num_processes} ranks\n")
        sys.stderr.write(
            "   (This is unexpected - the test should complete even with warnings)\n"
        )
    sys.stderr.write("=" * 70 + "\n")

    return 0 if passed == args.num_processes else 1


if __name__ == "__main__":
    sys.exit(main())
