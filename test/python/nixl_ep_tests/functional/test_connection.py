# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functional tests for Buffer connection management (F-CONN-01 to F-CONN-06).
"""

import os
import sys
import time

import pytest
import torch

# Add parent directory to path
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)

from utils.mp_runner import (  # noqa: E402
    all_passed,
    create_buffer,
    print_results,
    run_multiprocess_test,
    sync_all_ranks,
)

# ============================================================================
# Test Configuration
# ============================================================================

DEFAULT_NUM_EXPERTS_PER_RANK = 8
DEFAULT_NUM_RDMA_BYTES = 64 * 1024 * 1024  # 64MB


# ============================================================================
# F-CONN-01: Connect to other ranks (clean - just destroy)
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_connect_ranks_fn(rank: int, world_size: int, local_rank: int = 0):
    """F-CONN-01: Connect to other ranks and verify via query_mask_buffer."""
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    # Barrier: Wait for all ranks to register in etcd before connecting
    sync_all_ranks(rank, world_size, "conn01_init")

    try:
        # Get other ranks
        other_ranks = [r for r in range(world_size) if r != rank]

        # Connect
        if other_ranks:
            buffer.connect_ranks(other_ranks)

        # Verify via mask buffer
        mask_status = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status)

        # Check all connected ranks are unmasked (value 0 = unmasked/active)
        errors = []
        for r in other_ranks:
            if mask_status[r].item() != 0:
                errors.append(
                    f"Rank {r} should be unmasked (0), got {mask_status[r].item()}"
                )

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {
            "passed": len(errors) == 0,
            "error": "; ".join(errors) if errors else None,
            "metrics": {
                "num_connected": len(other_ranks),
                "mask_status": mask_status.cpu().tolist(),
            },
        }
    except Exception:
        if buffer is not None:
            try:
                buffer.destroy()
            except Exception:
                pass
        raise


@pytest.mark.functional
@pytest.mark.connection
def test_connect_ranks(request):
    """F-CONN-01: Connect to other ranks and verify mask buffer shows them unmasked."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(_test_connect_ranks_fn, num_processes=num_processes)
    print_results(results)
    assert all_passed(results), "Not all ranks passed connection test"


# ============================================================================
# F-CONN-WARN: Explicit disconnect pattern (demonstrates NIXL warnings)
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_disconnect_before_destroy_fn(rank: int, world_size: int, local_rank: int = 0):
    """F-CONN-WARN: Test explicit disconnect before destroy (shows benign warnings)."""
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    sync_all_ranks(rank, world_size, "conn_warn_init")

    try:
        other_ranks = [r for r in range(world_size) if r != rank]

        if other_ranks:
            buffer.connect_ranks(other_ranks)

        # Verify connection
        mask_status = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status)

        errors = []
        for r in other_ranks:
            if mask_status[r].item() != 0:
                errors.append(
                    f"Rank {r} should be unmasked (0), got {mask_status[r].item()}"
                )

        # Explicit disconnect pattern - THIS WILL SHOW WARNINGS
        sync_all_ranks(rank, world_size, "conn_warn_pre_disconnect")

        if other_ranks:
            buffer.disconnect_ranks(other_ranks)

        # Sleep to allow async metadata invalidation
        # Even with this sleep, some warnings may appear due to race conditions
        time.sleep(5)

        sync_all_ranks(rank, world_size, "conn_warn_post_disconnect")

        buffer.destroy()

        sync_all_ranks(rank, world_size, "conn_warn_done")

        return {
            "passed": len(errors) == 0,
            "error": "; ".join(errors) if errors else None,
            "metrics": {
                "num_connected": len(other_ranks),
                "mask_status": mask_status.cpu().tolist(),
                "note": "Check console for invalidateRemoteMD warnings (expected)",
            },
        }
    except Exception:
        if buffer is not None:
            try:
                buffer.destroy()
            except Exception:
                pass
        raise


@pytest.mark.functional
@pytest.mark.connection
def test_disconnect_before_destroy(request):
    """F-CONN-WARN: Explicit disconnect pattern (shows NIXL warnings - expected)."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_disconnect_before_destroy_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "Disconnect before destroy failed"


# ============================================================================
# F-CONN-02: Disconnect ranks (coordinated removal)
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_disconnect_ranks_fn(rank: int, world_size: int, local_rank: int = 0):
    """
    F-CONN-02: Coordinated disconnect - one rank leaves, others disconnect it.

    This test coordinates properly to avoid invalidateRemoteMD warnings:
    - All ranks agree on which rank is leaving (last rank)
    - Remaining ranks call disconnect_ranks() while leaving rank is STILL ALIVE
    - Leaving rank waits for disconnect to complete, THEN destroys

    The key insight: disconnect_ranks() triggers metadata invalidation.
    If the target rank has already destroyed, invalidation fails with warnings.
    By keeping the leaving rank alive during disconnect, invalidation succeeds.

    Validates:
    - disconnect_ranks() removes connection
    - query_mask_buffer shows disconnected rank as masked
    - Clean exit without warnings
    """
    import nixl_ep  # noqa: F401

    # All ranks agree: the last rank (world_size - 1) will be removed
    rank_to_remove = world_size - 1
    participating_ranks = list(range(world_size))
    remaining_ranks = [r for r in participating_ranks if r != rank_to_remove]

    if world_size < 3:
        # Need at least 3 ranks to test disconnect properly
        return {"passed": True, "metrics": {"skipped": "need at least 3 ranks"}}

    buffer = create_buffer(rank, world_size)

    # Barrier: Wait for all ranks to register in etcd
    sync_all_ranks(rank, world_size, "conn02_init")

    try:
        # Connect to all other ranks
        other_ranks = [r for r in range(world_size) if r != rank]
        buffer.connect_ranks(other_ranks)

        # Barrier: Everyone connected
        sync_all_ranks(rank, world_size, "conn02_connected")

        if rank == rank_to_remove:
            # I'm the rank being removed
            # IMPORTANT: Stay alive while others call disconnect_ranks() on me
            # so they can properly invalidate my metadata

            # Wait for others to disconnect me (they will reach this barrier after disconnect_ranks())
            sync_all_ranks(rank, world_size, "conn02_post_disconnect")

            # Small delay to ensure invalidation completes
            time.sleep(0.5)

            # NOW safe to destroy - others have already invalidated my metadata
            buffer.destroy()

            return {
                "passed": True,
                "metrics": {"role": "removed_rank", "exited_cleanly": True},
            }
        else:
            # I'm staying - disconnect the removed rank while it's still alive
            buffer.disconnect_ranks([rank_to_remove])

            # Barrier: Let rank_to_remove know we've disconnected it
            # It will wait here, then destroy after we all pass
            sync_all_ranks(rank, world_size, "conn02_post_disconnect")

            # Brief wait for disconnect to propagate
            time.sleep(0.1)

            # Verify mask buffer
            mask_status = torch.zeros(world_size, dtype=torch.int32, device="cuda")
            buffer.query_mask_buffer(mask_status)

            errors = []

            # The removed rank should be masked (non-zero)
            if mask_status[rank_to_remove].item() == 0:
                errors.append(f"Removed rank {rank_to_remove} should be masked, got 0")

            # Remaining ranks (except self) should still be unmasked
            for r in remaining_ranks:
                if r != rank and mask_status[r].item() != 0:
                    errors.append(
                        f"Remaining rank {r} should be unmasked, got {mask_status[r].item()}"
                    )

            # Cleanup
            buffer.destroy()

            return {
                "passed": len(errors) == 0,
                "error": "; ".join(errors) if errors else None,
                "metrics": {
                    "role": "remaining_rank",
                    "removed_rank": rank_to_remove,
                    "mask_status": mask_status.cpu().tolist(),
                },
            }
    except Exception:
        if buffer is not None:
            try:
                buffer.destroy()
            except Exception:
                pass
        raise


@pytest.mark.functional
@pytest.mark.connection
def test_disconnect_ranks(request):
    """F-CONN-02: Coordinated disconnect - one rank leaves, others disconnect it."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_disconnect_ranks_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "Not all ranks passed disconnect test"


# ============================================================================
# F-CONN-03: Connect self-rank filtered
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_connect_self_rank_filtered_fn(
    rank: int, world_size: int, local_rank: int = 0
):
    """
    F-CONN-03: Verify connecting to own rank is handled gracefully.

    Validates:
    - Including self in connect_ranks is a no-op or handled gracefully
    - No crash or error
    """
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    # FIX: Wait for all ranks to register in etcd before connecting
    sync_all_ranks(rank, world_size, "self_rank_test_init")

    try:
        # Include self rank in the list
        all_ranks_including_self = list(range(world_size))

        # This should either filter out self or handle gracefully
        buffer.connect_ranks(all_ranks_including_self)

        # Verify we can still query mask buffer
        mask_status = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status)

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {"passed": True, "metrics": {"handled_self_rank": True}}
    except Exception:
        if buffer is not None:
            try:
                buffer.destroy()
            except Exception:
                pass
        raise


@pytest.mark.functional
@pytest.mark.connection
def test_connect_self_rank_filtered(request):
    """F-CONN-03: Connecting own rank should be handled gracefully."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_connect_self_rank_filtered_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "Self-rank connection not handled gracefully"


# ============================================================================
# F-CONN-04: Connect already-connected rank (idempotent)
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_connect_idempotent_fn(rank: int, world_size: int, local_rank: int = 0):
    """
    F-CONN-04: Verify connecting to already-connected rank is idempotent.

    Validates:
    - Calling connect_ranks twice with same ranks doesn't cause errors
    - State remains consistent
    """
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    try:
        other_ranks = [r for r in range(world_size) if r != rank]

        if not other_ranks:
            buffer.destroy()
            return {"passed": True, "metrics": {"skipped": "single rank"}}

        # Connect first time
        buffer.connect_ranks(other_ranks)

        # Check mask status after first connect
        mask_status_1 = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status_1)

        # Connect same ranks again (should be idempotent)
        buffer.connect_ranks(other_ranks)

        # Check mask status after second connect
        mask_status_2 = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status_2)

        # Both should be the same
        same_status = torch.equal(mask_status_1, mask_status_2)

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {
            "passed": same_status,
            "error": None if same_status else "Mask status changed after reconnect",
            "metrics": {
                "idempotent": same_status,
                "status_before": mask_status_1.cpu().tolist(),
                "status_after": mask_status_2.cpu().tolist(),
            },
        }
    except Exception:
        if buffer is not None:
            try:
                buffer.destroy()
            except Exception:
                pass
        raise


@pytest.mark.functional
@pytest.mark.connection
def test_connect_idempotent(request):
    """F-CONN-04: Re-connecting already-connected ranks should be idempotent."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_connect_idempotent_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "Connect is not idempotent"


# ============================================================================
# F-CONN-05: Incremental connect
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_incremental_connect_fn(rank: int, world_size: int, local_rank: int = 0):
    """
    F-CONN-05: Connect ranks incrementally and verify all are active.

    Validates:
    - Connect subset of ranks, then connect more
    - All connected ranks show as unmasked
    """
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    # Barrier: Wait for all ranks to register in etcd before connecting
    sync_all_ranks(rank, world_size, "incr_init")

    try:
        other_ranks = [r for r in range(world_size) if r != rank]

        if len(other_ranks) < 2:
            buffer.destroy()
            return {
                "passed": True,
                "metrics": {"skipped": "need at least 2 other ranks"},
            }

        # Split into two groups
        group1 = other_ranks[: len(other_ranks) // 2]
        group2 = other_ranks[len(other_ranks) // 2 :]

        # Connect first group
        buffer.connect_ranks(group1)

        # Verify first group connected
        mask_status_1 = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status_1)

        group1_connected = all(mask_status_1[r].item() == 0 for r in group1)

        # Connect second group
        buffer.connect_ranks(group2)

        # Verify both groups connected
        mask_status_2 = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status_2)

        all_connected = all(mask_status_2[r].item() == 0 for r in other_ranks)

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {
            "passed": group1_connected and all_connected,
            "error": (
                None
                if (group1_connected and all_connected)
                else "Incremental connect failed"
            ),
            "metrics": {
                "group1": group1,
                "group2": group2,
                "group1_connected": group1_connected,
                "all_connected": all_connected,
            },
        }
    except Exception:
        if buffer is not None:
            try:
                buffer.destroy()
            except Exception:
                pass
        raise


@pytest.mark.functional
@pytest.mark.connection
def test_incremental_connect(request):
    """F-CONN-05: Connecting ranks incrementally should work."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_incremental_connect_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "Incremental connect failed"


# ============================================================================
# F-CONN-06: Barrier after connect
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_barrier_after_connect_fn(rank: int, world_size: int, local_rank: int = 0):
    """
    F-CONN-06: Verify barrier synchronizes all ranks after connect.

    Validates:
    - buffer.barrier() completes without error after connect
    - All ranks reach the barrier
    """
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    # FIX: Wait for all ranks to register in etcd before connecting
    sync_all_ranks(rank, world_size, "barrier_test_init")

    try:
        other_ranks = [r for r in range(world_size) if r != rank]

        # Connect to all other ranks
        if other_ranks:
            buffer.connect_ranks(other_ranks)

        # Barrier should synchronize all ranks
        start = time.perf_counter()
        buffer.barrier()
        barrier_time_ms = (time.perf_counter() - start) * 1000

        # If we get here without timeout, barrier succeeded

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {"passed": True, "metrics": {"barrier_time_ms": barrier_time_ms}}
    except Exception:
        if buffer is not None:
            try:
                buffer.destroy()
            except Exception:
                pass
        raise


@pytest.mark.functional
@pytest.mark.connection
def test_barrier_after_connect(request):
    """F-CONN-06: Barrier after connect should synchronize all ranks."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_barrier_after_connect_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "Barrier failed after connect"


# ============================================================================
# CLI runner for standalone execution
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run connection functional tests")
    parser.add_argument(
        "--num-processes", type=int, default=8, help="Number of processes"
    )
    parser.add_argument(
        "--etcd-server", type=str, default="http://127.0.0.1:2379", help="ETCD server"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        help="Test to run: connect, disconnect, self, idempotent, incremental, barrier, all",
    )
    args = parser.parse_args()

    os.environ["NIXL_TEST_NUM_PROCESSES"] = str(args.num_processes)

    tests = {
        "connect": (_test_connect_ranks_fn, "F-CONN-01: Connect ranks (clean)"),
        "disconnect_warn": (
            _test_disconnect_before_destroy_fn,
            "F-CONN-WARN: Explicit disconnect (shows warnings)",
        ),
        "disconnect": (_test_disconnect_ranks_fn, "F-CONN-02: Disconnect ranks"),
        "self": (_test_connect_self_rank_filtered_fn, "F-CONN-03: Self-rank filtering"),
        "idempotent": (_test_connect_idempotent_fn, "F-CONN-04: Idempotent connect"),
        "incremental": (_test_incremental_connect_fn, "F-CONN-05: Incremental connect"),
        "barrier": (_test_barrier_after_connect_fn, "F-CONN-06: Barrier after connect"),
    }

    if args.test == "all":
        for name, (fn, desc) in tests.items():
            sys.stderr.write(f"\n{'='*60}\n")
            sys.stderr.write(f"Running: {desc}\n")
            sys.stderr.write(f"{'='*60}\n")
            results = run_multiprocess_test(
                fn, num_processes=args.num_processes, etcd_server=args.etcd_server
            )
            print_results(results)
    else:
        if args.test in tests:
            fn, desc = tests[args.test]
            sys.stderr.write(f"\nRunning: {desc}\n")
            results = run_multiprocess_test(
                fn, num_processes=args.num_processes, etcd_server=args.etcd_server
            )
            print_results(results)
        else:
            sys.stderr.write(f"Unknown test: {args.test}\n")
            sys.stderr.write(f"Available: {', '.join(tests.keys())}, all\n")
