#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BUG-01: NIXL Segfault on Repeated Buffer Creation

================================================================================
SUMMARY
================================================================================
Creating a second nixl_ep.Buffer instance after destroy() causes a segfault.
This is a SINGLE-PROCESS bug - no multiprocessing needed to reproduce.

Component:     NIXL library (C++)
Severity:      🔴 Critical
Status:        Open - needs NIXL team fix
First seen:    December 2025

================================================================================
REPRODUCTION (Simple - Single Process)
================================================================================
1. Create nixl_ep.Buffer
2. Call update_memory_buffers()
3. Call buffer.destroy()
4. Create another nixl_ep.Buffer
5. Call update_memory_buffers() → SEGFAULT

================================================================================
ERROR OUTPUT
================================================================================
Round 1: Creating buffer...
Round 1: Calling update_memory_buffers...
Round 1: Destroying buffer...
Round 1: Complete
Round 2: Creating buffer...
Round 2: Calling update_memory_buffers...
Segmentation fault (core dumped)

================================================================================
ROOT CAUSE ANALYSIS
================================================================================
The NIXL C++ library has global state that is not properly reset after destroy().
The crash occurs specifically in update_memory_buffers() on the second buffer,
not in the Buffer constructor itself.

TESTED: Bug affects ALL nvlink backends (nixl, ipc, none) - confirmed Dec 2025.
This rules out backend-specific causes and points to core NIXL library issue.

INVESTIGATED (Dec 2025):
We attempted fixes that DID NOT work:
1. Adding deregisterMem() for all registered buffers before cudaFree()
2. Resetting nixl_agent_info (unique_ptr to NIXL agent)
3. Resetting nixl_ctx (unique_ptr to EP context)

The issue is DEEPER than memory registration - likely UCX/CUDA internal
state that persists across Buffer instances at the driver/library level.

Possible remaining causes:
- UCX context global state not fully cleaned up
- CUDA driver-level state persisting
- Static/global variables in UCX plugins
- etcd client connection state

================================================================================
IMPACT
================================================================================
- Cannot run warmup rounds in performance benchmarks
- Each test must spawn a fresh process
- Performance measurements less accurate (no warmup)

================================================================================
WORKAROUND
================================================================================
Use --warmup=0 --rounds=1 in performance tests.
Each test spawns fresh processes via mp_runner.py.

================================================================================
TO VERIFY FIX
================================================================================
# Ensure etcd is running first:
source ../scripts/reset_etcd.sh

# Set environment:
export NIXL_ETCD_ENDPOINTS="http://127.0.0.1:2379"

# Run test:
python3 tests/bugs/test_bug_01_segfault.py

Expected BEFORE fix: Segfault on round 2
Expected AFTER fix:  All 3 rounds complete successfully

================================================================================
"""

import os
import sys


def test_repeated_buffer_creation(backend: str = "ipc", num_rounds: int = 3):
    """
    Single-process test for BUG-01.

    EXPECTED BEFORE FIX: Segfault during round 2's update_memory_buffers()
    EXPECTED AFTER FIX: All rounds complete successfully
    """
    import nixl_ep
    import torch

    # Use GPU 0
    torch.cuda.set_device(0)

    sys.stderr.write(f"Testing with nvlink_backend='{backend}'\n")
    sys.stderr.write("-" * 40 + "\n")

    for round_num in range(num_rounds):
        sys.stderr.write(f"Round {round_num + 1}/{num_rounds}: Creating buffer...\n")

        # Create buffer
        buffer = nixl_ep.Buffer(
            rank=0,
            nvlink_backend=backend,
            explicitly_destroy=True,
            enable_shrink=False,
        )

        sys.stderr.write(
            f"Round {round_num + 1}/{num_rounds}: Calling update_memory_buffers...\n"
        )

        # This is where the segfault occurs on round 2+
        num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
            512, 4096, 1, 8  # num_tokens, hidden, num_ranks, total_experts
        )
        buffer.update_memory_buffers(
            num_ranks=1, num_experts_per_rank=8, num_rdma_bytes=num_rdma_bytes
        )

        sys.stderr.write(f"Round {round_num + 1}/{num_rounds}: Destroying buffer...\n")
        buffer.destroy()

        sys.stderr.write(f"Round {round_num + 1}/{num_rounds}: Complete ✓\n")

    sys.stderr.write("\n")
    sys.stderr.write(f"✅ Backend '{backend}': All {num_rounds} rounds completed!\n")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="BUG-01: Test repeated buffer creation"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="all",
        choices=["nixl", "ipc", "none", "all"],
        help="NVLink backend to test (default: all)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of create/destroy rounds (default: 3)",
    )
    args = parser.parse_args()

    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("BUG-01: NIXL Segfault on Repeated Buffer Creation\n")
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("\n")
    sys.stderr.write("This is a SINGLE-PROCESS test - no multiprocessing needed.\n")
    sys.stderr.write("\n")
    sys.stderr.write("EXPECTED BEFORE FIX: Segfault on round 2\n")
    sys.stderr.write("EXPECTED AFTER FIX:  All rounds complete for all backends\n")
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("\n")

    # Check etcd endpoint
    etcd_endpoint = os.environ.get("NIXL_ETCD_ENDPOINTS", "")
    if not etcd_endpoint:
        sys.stderr.write("WARNING: NIXL_ETCD_ENDPOINTS not set!\n")
        sys.stderr.write('Run: export NIXL_ETCD_ENDPOINTS="http://127.0.0.1:2379"\n')
        sys.stderr.write("\n")
    else:
        sys.stderr.write(f"NIXL_ETCD_ENDPOINTS={etcd_endpoint}\n")
        sys.stderr.write("\n")

    # Determine which backends to test
    if args.backend == "all":
        backends = ["nixl", "ipc", "none"]
    else:
        backends = [args.backend]

    results = {}

    for backend in backends:
        sys.stderr.write("\n")
        sys.stderr.write("=" * 60 + "\n")
        try:
            test_repeated_buffer_creation(backend=backend, num_rounds=args.rounds)
            results[backend] = "PASS"
        except Exception as e:
            sys.stderr.write("\n")
            sys.stderr.write(f"❌ Backend '{backend}': FAILED - {e}\n")
            results[backend] = f"FAIL: {e}"

    # Summary
    sys.stderr.write("\n")
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("SUMMARY: BUG-01 Results by Backend\n")
    sys.stderr.write("=" * 60 + "\n")
    all_passed = True
    for backend, result in results.items():
        status = "✅" if result == "PASS" else "❌"
        sys.stderr.write(f"  {status} {backend}: {result}\n")
        if result != "PASS":
            all_passed = False
    sys.stderr.write("=" * 60 + "\n")

    if all_passed:
        sys.stderr.write("✅ BUG-01 FIX VERIFIED: All backends passed!\n")
    else:
        sys.stderr.write("❌ BUG-01 REPRODUCED: Some backends failed\n")
        sys.stderr.write("   The bug is still present.\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
