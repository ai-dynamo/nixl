#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BUG-05: Intermittent CUDA Error During connect_ranks()

================================================================================
SUMMARY
================================================================================
Intermittent "CUDA error: invalid argument" failures during connect_ranks()
when running tests back-to-back. Appears to be related to stale UCX rcache
state or etcd metadata exchange timing.

Component:     NIXL EP / UCX / etcd (unclear)
Severity:      🟠 Medium (intermittent test failures)
Status:        Open - needs investigation
First seen:    December 2025
Cluster:       DFW (pool0-01755)

================================================================================
REPRODUCTION
================================================================================
1. Run test_connect_ranks multiple times in sequence
2. Error appears in ~10-20% of runs (varies)
3. Different ranks fail each time (not deterministic)
4. Re-running immediately often succeeds

Commands to reproduce:
    # Run 5 times to see failure
    for i in {1..5}; do
        echo "=== Run $i ==="
        python -m pytest test/python/nixl_ep_tests/functional/test_connection.py::test_connect_ranks -v -s 2>&1 | tail -10
    done

================================================================================
ERROR OUTPUT
================================================================================
RuntimeError: Failed: CUDA error ../examples/device/ep/csrc/nixl_ep.cpp:861 'invalid argument'
Traceback (most recent call last):
  File "test_connection.py", line 70, in _test_connect_ranks_fn
    buffer.connect_ranks(other_ranks)
  File "buffer.py", line 468, in connect_ranks
    self.runtime.connect_ranks(remote_ranks)
RuntimeError: Failed: CUDA error ../examples/device/ep/csrc/nixl_ep.cpp:861 'invalid argument'

Associated warning (often appears before failure):
[1765442432.425215] [pool0-01755:3334443:0] rcache.c:1392 UCX WARN rcache gdr_copy: 2 regions remained on lru list

================================================================================
ROOT CAUSE ANALYSIS
================================================================================
The exact cause is unclear. Possible factors:

1. UCX rcache state pollution:
   - Previous test runs leave stale GPU memory regions in UCX cache
   - "rcache gdr_copy: N regions remained on lru list" warning often precedes failure
   - New buffer creation may conflict with stale registrations

2. etcd metadata exchange timing:
   - connect_ranks() fetches metadata from etcd for remote ranks
   - If rank's metadata not yet propagated, connection may fail
   - Error at line 861 in nixl_ep.cpp suggests CUDA operation on invalid data

3. GPU context state:
   - CUDA contexts from previous runs may not be fully cleaned up
   - torch.cuda.set_device() may encounter stale state

================================================================================
OBSERVATIONS
================================================================================
- Failure rate: ~10-20% when running tests back-to-back
- Random ranks fail (not always same rank)
- Sometimes multiple ranks fail in same run
- First run after etcd reset usually succeeds
- Consecutive runs more likely to fail
- UCX "rcache gdr_copy" warning often (but not always) precedes failure

================================================================================
PATTERN ANALYSIS
================================================================================
Run 1: PASS (8/8)
Run 2: FAIL (7/8, rank 3 failed)  <- rcache warning appeared
Run 3: PASS (8/8)                 <- rcache warning appeared but passed
Run 4: FAIL (6/8, ranks 2,3 failed)
Run 5: PASS (8/8)

This suggests the failure is probabilistic, related to accumulated state.

================================================================================
IMPACT
================================================================================
- CI/CD flakiness (~10-20% test failure rate)
- Requires test retries
- Makes debugging real issues difficult
- Not a correctness bug (retry succeeds)

================================================================================
WORKAROUND
================================================================================
1. Reset etcd before each test run:
   source ./scripts/reset_etcd.sh

2. Add retry logic to tests (pytest-rerunfailures plugin)

3. Add delay between test runs (gives UCX time to clean up):
   sleep 5 between runs

================================================================================
TO VERIFY FIX
================================================================================
Run 20 consecutive tests without etcd reset - all should pass:

    for i in {1..20}; do
        echo "=== Run $i ==="
        python -m pytest test/python/nixl_ep_tests/functional/test_connection.py::test_connect_ranks -v -s 2>&1 | grep -E "(PASSED|FAILED|Result:)"
    done | tee /tmp/bug05_verify.log

    # Count results
    echo "Passed: $(grep -c PASSED /tmp/bug05_verify.log)"
    echo "Failed: $(grep -c FAILED /tmp/bug05_verify.log)"

Expected after fix: 20/20 passes

================================================================================
RELATED BUGS
================================================================================
- BUG-02: UCX rcache assertion failure (similar rcache issues)
- BUG-03: UCX rcache gdr_copy warning (warning often precedes this bug)

================================================================================
"""

import argparse
import os
import sys
import time

# Add parent paths
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)


def run_connect_test():
    """Run a single connect_ranks test and return success/failure."""
    import torch
    from utils.mp_runner import (
        all_passed,
        create_buffer,
        run_multiprocess_test,
        sync_all_ranks,
    )

    def test_fn(rank, world_size, local_rank):
        """Simple connect test."""
        buffer = create_buffer(rank, world_size)
        sync_all_ranks(rank, world_size, f"bug05_init_{time.time()}")

        try:
            other_ranks = [r for r in range(world_size) if r != rank]
            if other_ranks:
                buffer.connect_ranks(other_ranks)

            mask_status = torch.zeros(world_size, dtype=torch.int32, device="cuda")
            buffer.query_mask_buffer(mask_status)

            buffer.destroy()

            return {
                "passed": True,
                "metrics": {"mask_status": mask_status.cpu().tolist()},
            }
        except Exception:
            if buffer:
                try:
                    buffer.destroy()
                except Exception:
                    pass
            raise

    results = run_multiprocess_test(test_fn, num_processes=8)
    return all_passed(results), results


def main():
    parser = argparse.ArgumentParser(
        description="BUG-05: Intermittent CUDA error during connect_ranks"
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of test runs")
    parser.add_argument(
        "--delay", type=float, default=0, help="Delay between runs (seconds)"
    )
    args = parser.parse_args()

    sys.stderr.write(f"BUG-05: Running {args.runs} consecutive connect_ranks tests\n")
    sys.stderr.write(f"Delay between runs: {args.delay}s\n")
    sys.stderr.write("=" * 60 + "\n")

    passes = 0
    failures = 0

    for i in range(1, args.runs + 1):
        sys.stderr.write(f"\n--- Run {i}/{args.runs} ---\n")
        try:
            success, results = run_connect_test()
            if success:
                sys.stderr.write("Result: PASS (8/8)\n")
                passes += 1
            else:
                failed_ranks = [r.rank for r in results if not r.passed]
                sys.stderr.write(
                    f"Result: FAIL ({8 - len(failed_ranks)}/8) - Failed ranks: {failed_ranks}\n"
                )
                failures += 1
        except Exception as e:
            sys.stderr.write(f"Result: EXCEPTION - {e}\n")
            failures += 1

        if args.delay > 0 and i < args.runs:
            time.sleep(args.delay)

    sys.stderr.write("\n" + "=" * 60 + "\n")
    sys.stderr.write(
        f"SUMMARY: {passes}/{args.runs} passed, {failures}/{args.runs} failed\n"
    )
    sys.stderr.write(f"Failure rate: {100 * failures / args.runs:.1f}%\n")

    if failures > 0:
        sys.stderr.write("\nBUG-05 REPRODUCED: Intermittent CUDA errors detected\n")
        sys.stderr.write("This is a known issue - see bug documentation above\n")
        return 1
    else:
        sys.stderr.write("\nNo failures detected in this run (bug may still exist)\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
