#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BUG-04: NIXL invalidateRemoteMD Warning

================================================================================
SUMMARY
================================================================================
When multiple ranks call destroy() simultaneously, NIXL emits warnings about
failing to invalidate remote metadata. This is a race condition in the
distributed cleanup protocol, but the warning is benign.

Component:     NIXL library
Severity:      🟡 Low (cosmetic warning)
Status:        Open - NIXL/NIXL_EP issue
First seen:    December 2025
Cluster:       DFW (pool0-00290)

================================================================================
REPRODUCTION
================================================================================
1. Create buffers on 8 ranks
2. Connect all ranks to each other
3. All ranks call destroy() simultaneously (no barrier)
4. Multiple ranks emit warnings

================================================================================
ERROR OUTPUT
================================================================================
nixl_agent.cpp:1700] invalidateRemoteMD: error invalidating remote metadata for agent '5' with status NIXL_ERR_NOT_FOUND
nixl_agent.cpp:1700] invalidateRemoteMD: error invalidating remote metadata for agent '3' with status NIXL_ERR_NOT_FOUND
nixl_agent.cpp:1700] invalidateRemoteMD: error invalidating remote metadata for agent '7' with status NIXL_ERR_NOT_FOUND

================================================================================
ROOT CAUSE ANALYSIS
================================================================================
When rank A tries to invalidate metadata on rank B during destroy():

1. Rank A sends invalidation request to rank B
2. Meanwhile, rank B has already destroyed its NIXL agent
3. Rank A receives NIXL_ERR_NOT_FOUND
4. Warning is logged

This is EXPECTED behavior in simultaneous shutdown:
- Distributed systems can't synchronize shutdown perfectly
- Some ranks will finish destroying before others try to invalidate
- The warning indicates this race but doesn't affect correctness

The cleanup still succeeds - the warning is just informational.

================================================================================
OBSERVATIONS
================================================================================
- More warnings with more ranks (8 ranks = more warnings than 2)
- disconnect_ranks() before destroy() INCREASES warnings
- reconnect() before destroy() REDUCES warnings (keeps connections active)
- Warnings don't affect correctness - all operations complete
- Number of warnings varies between runs (0 to ~6 typically)

================================================================================
PROTOCOL ANALYSIS
================================================================================
disconnect_ranks() + destroy() sequence:
1. disconnect_ranks() removes peer info locally
2. destroy() tries to invalidate metadata on peers
3. But peers already invalidated when they saw disconnect
4. → NIXL_ERR_NOT_FOUND

reconnect() + destroy() sequence:
1. reconnect() re-establishes peer connections
2. destroy() invalidates metadata on active peers
3. Peers still exist, invalidation succeeds
4. → Fewer warnings

================================================================================
IMPACT
================================================================================
- Clutters test output
- May confuse users into thinking something is wrong
- NO functional impact - operations still complete

================================================================================
WORKAROUND
================================================================================
1. Ignore warnings in test output validation
2. For cleaner output, use reconnect() before destroy()
3. Filter stderr: grep -v "invalidateRemoteMD"

================================================================================
TO VERIFY FIX
================================================================================
python3 tests/bugs/test_bug_04_invalidate.py --num-processes=8 2>&1 | grep -c "invalidateRemoteMD"

Expected BEFORE fix: Count > 0 (typically 2-6 warnings)
Expected AFTER fix:  Count = 0

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
def _test_invalidate_warning_basic_fn(
    rank: int, world_size: int, local_rank: int = 0, num_experts: int = 8, **kwargs
) -> dict:
    """
    Test that produces invalidateRemoteMD warnings during simultaneous destroy().

    This is the "worst case" scenario that produces the most warnings:
    connect → disconnect → destroy (no reconnect)

    EXPECTED BEFORE FIX: Multiple "invalidateRemoteMD: NIXL_ERR_NOT_FOUND" warnings
    EXPECTED AFTER FIX: No warnings
    """
    import torch

    import nixl_ep

    # Note: setup_worker_environment already sets CUDA_VISIBLE_DEVICES to local_rank
    # so only device 0 is visible. Don't call set_device(local_rank) - that would fail!

    sys.stderr.write(
        f"[Rank {rank}] Starting invalidateRemoteMD test (basic scenario)\n"
    )

    # Use shared create_buffer helper for consistency with other tests
    buffer = create_buffer(rank, world_size, num_experts_per_rank=num_experts)

    sync_all_ranks(rank, world_size, "init")
    sys.stderr.write(f"[Rank {rank}] Buffer created\n")

    # Connect to all other ranks
    other_ranks = [r for r in range(world_size) if r != rank]

    torch.cuda.synchronize()
    buffer.connect_ranks(other_ranks)
    torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "connected")
    sys.stderr.write(f"[Rank {rank}] Connected to {len(other_ranks)} ranks\n")

    # Disconnect - this contributes to the warning
    torch.cuda.synchronize()
    buffer.disconnect_ranks(other_ranks)
    torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "disconnected")
    sys.stderr.write(f"[Rank {rank}] Disconnected\n")

    # Destroy simultaneously - this triggers the warnings
    # Note: We use a barrier BEFORE destroy but not during
    # This ensures all ranks are at the same point but then race to destroy
    sync_all_ranks(rank, world_size, "pre_destroy")
    sys.stderr.write(
        f"[Rank {rank}] Destroying (invalidateRemoteMD warnings may appear)...\n"
    )

    buffer.destroy()

    sync_all_ranks(rank, world_size, "destroyed")
    sys.stderr.write(f"[Rank {rank}] Destroyed - check stderr for warnings\n")

    return {
        "passed": True,
        "scenario": "basic (connect → disconnect → destroy)",
        "message": "Check stderr for 'invalidateRemoteMD' warnings",
    }


@pytest.mark.skip(reason="Not run directly")
def _test_invalidate_warning_with_reconnect_fn(
    rank: int, world_size: int, local_rank: int = 0, num_experts: int = 8, **kwargs
) -> dict:
    """
    Test with reconnect before destroy - produces FEWER warnings.

    This demonstrates the mitigation: reconnect → destroy produces fewer warnings
    than disconnect → destroy.

    EXPECTED: Fewer warnings than basic scenario
    """
    import torch

    import nixl_ep

    # Note: setup_worker_environment already sets CUDA_VISIBLE_DEVICES to local_rank
    # so only device 0 is visible. Don't call set_device(local_rank) - that would fail!

    sys.stderr.write(
        f"[Rank {rank}] Starting invalidateRemoteMD test (with reconnect)\n"
    )

    # Use shared create_buffer helper for consistency with other tests
    buffer = create_buffer(rank, world_size, num_experts_per_rank=num_experts)

    sync_all_ranks(rank, world_size, "init")
    sys.stderr.write(f"[Rank {rank}] Buffer created\n")

    # Connect to all other ranks
    other_ranks = [r for r in range(world_size) if r != rank]

    torch.cuda.synchronize()
    buffer.connect_ranks(other_ranks)
    torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "connected")
    sys.stderr.write(f"[Rank {rank}] Connected\n")

    # Disconnect
    torch.cuda.synchronize()
    buffer.disconnect_ranks(other_ranks)
    torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "disconnected")
    sys.stderr.write(f"[Rank {rank}] Disconnected\n")

    # RECONNECT - this is the mitigation
    torch.cuda.synchronize()
    buffer.connect_ranks(other_ranks)
    torch.cuda.synchronize()

    sync_all_ranks(rank, world_size, "reconnected")
    sys.stderr.write(f"[Rank {rank}] Reconnected (mitigation applied)\n")

    # Destroy - should have fewer warnings now
    sync_all_ranks(rank, world_size, "pre_destroy")
    sys.stderr.write(f"[Rank {rank}] Destroying...\n")

    buffer.destroy()

    sync_all_ranks(rank, world_size, "destroyed")
    sys.stderr.write(f"[Rank {rank}] Destroyed - should have fewer warnings\n")

    return {
        "passed": True,
        "scenario": "with reconnect (connect → disconnect → reconnect → destroy)",
        "message": "This scenario should have fewer warnings",
    }


def main():
    parser = argparse.ArgumentParser(
        description="BUG-04: Test NIXL invalidateRemoteMD warning (cosmetic)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes/ranks to spawn",
    )
    parser.add_argument(
        "--experts", type=int, default=8, help="Number of experts per rank"
    )
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds")
    parser.add_argument(
        "--scenario",
        choices=["basic", "reconnect", "both"],
        default="both",
        help="Which scenario to run",
    )
    args = parser.parse_args()

    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("BUG-04: NIXL invalidateRemoteMD Warning\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f"Ranks: {args.num_processes}\n")
    sys.stderr.write(f"Experts per rank: {args.experts}\n")
    sys.stderr.write(f"Timeout: {args.timeout}s\n")
    sys.stderr.write(f"Scenario: {args.scenario}\n")
    sys.stderr.write("\n")
    sys.stderr.write("This test checks for cosmetic warnings, NOT crashes.\n")
    sys.stderr.write("\n")
    sys.stderr.write("To count warnings, run:\n")
    sys.stderr.write(
        "  python3 tests/bugs/test_bug_04_invalidate.py 2>&1 | grep -c 'invalidateRemoteMD'\n"
    )
    sys.stderr.write("\n")
    sys.stderr.write("EXPECTED BEFORE FIX: Warning count > 0\n")
    sys.stderr.write("EXPECTED AFTER FIX:  Warning count = 0\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("\n")

    all_passed = True

    if args.scenario in ["basic", "both"]:
        sys.stderr.write("\n" + "=" * 70 + "\n")
        sys.stderr.write("SCENARIO 1: Basic (connect → disconnect → destroy)\n")
        sys.stderr.write("This scenario produces the MOST warnings\n")
        sys.stderr.write("=" * 70 + "\n\n")

        results = run_multiprocess_test(
            test_fn=_test_invalidate_warning_basic_fn,
            num_processes=args.num_processes,
            test_name="BUG-04: invalidateRemoteMD (basic)",
            timeout=args.timeout,
            num_experts=args.experts,
        )

        passed = sum(1 for r in results if r.passed)
        if passed != args.num_processes:
            all_passed = False

    if args.scenario in ["reconnect", "both"]:
        sys.stderr.write("\n" + "=" * 70 + "\n")
        sys.stderr.write(
            "SCENARIO 2: With Reconnect (connect → disconnect → reconnect → destroy)\n"
        )
        sys.stderr.write("This scenario produces FEWER warnings (mitigation)\n")
        sys.stderr.write("=" * 70 + "\n\n")

        results = run_multiprocess_test(
            test_fn=_test_invalidate_warning_with_reconnect_fn,
            num_processes=args.num_processes,
            test_name="BUG-04: invalidateRemoteMD (with reconnect)",
            timeout=args.timeout,
            num_experts=args.experts,
        )

        passed = sum(1 for r in results if r.passed)
        if passed != args.num_processes:
            all_passed = False

    sys.stderr.write("\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("RESULTS\n")
    sys.stderr.write("=" * 70 + "\n")
    if all_passed:
        sys.stderr.write("✅ All scenarios completed successfully\n")
        sys.stderr.write("\n")
        sys.stderr.write("Compare warning counts between scenarios:\n")
        sys.stderr.write("  Basic scenario:     should have MORE warnings\n")
        sys.stderr.write("  Reconnect scenario: should have FEWER warnings\n")
        sys.stderr.write("\n")
        sys.stderr.write("If both have 0 warnings, BUG-04 may be fixed!\n")
    else:
        sys.stderr.write("❌ Some scenarios failed\n")
        sys.stderr.write(
            "   (This is unexpected - scenarios should complete even with warnings)\n"
        )
    sys.stderr.write("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
