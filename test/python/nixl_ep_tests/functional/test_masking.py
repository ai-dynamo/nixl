# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functional tests for Buffer mask operations.

Test IDs: F-MASK-01 to F-MASK-04

These tests verify:
- update_mask_buffer() correctly sets mask state
- query_mask_buffer() returns correct state
- clean_mask_buffer() resets all masks
- Masked ranks are skipped during dispatch
"""

import os
import sys

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
# F-MASK-01: Update mask buffer (set masked)
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_update_mask_buffer_set_fn(rank: int, world_size: int, local_rank: int = 0):
    """
    F-MASK-01: Test update_mask_buffer to set a rank as masked.

    Validates:
    - update_mask_buffer(rank, True) marks rank as masked
    - query_mask_buffer returns non-zero for masked rank
    """
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    try:
        other_ranks = [r for r in range(world_size) if r != rank]

        # Barrier before connect
        sync_all_ranks(rank, world_size, "mask01_pre_connect")

        # Connect to all other ranks
        if other_ranks:
            buffer.connect_ranks(other_ranks)

        # Barrier after connect
        sync_all_ranks(rank, world_size, "mask01_post_connect")

        # Mask a specific rank (not self)
        if other_ranks:
            target_rank = other_ranks[0]
            buffer.update_mask_buffer(target_rank, True)  # True = masked

            # Query the mask buffer
            mask_status = torch.zeros(world_size, dtype=torch.int32, device="cuda")
            buffer.query_mask_buffer(mask_status)

            # Target should be masked (non-zero)
            target_masked = mask_status[target_rank].item() != 0

            # Cleanup - unmask before disconnect
            buffer.update_mask_buffer(target_rank, False)
        else:
            target_masked = True  # Skip for single rank

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {
            "passed": target_masked,
            "error": (
                None if target_masked else "update_mask_buffer(True) didn't mask rank"
            ),
            "metrics": {
                "target_rank": other_ranks[0] if other_ranks else None,
                "was_masked": target_masked,
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
@pytest.mark.masking
def test_update_mask_buffer_set(request):
    """F-MASK-01: update_mask_buffer(rank, True) should mark rank as masked."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_update_mask_buffer_set_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "update_mask_buffer set failed"


# ============================================================================
# F-MASK-02: Update mask buffer (unset masked)
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_update_mask_buffer_unset_fn(rank: int, world_size: int, local_rank: int = 0):
    """
    F-MASK-02: Test update_mask_buffer to unset (unmask) a rank.

    Validates:
    - update_mask_buffer(rank, False) removes mask
    - query_mask_buffer returns 0 for unmasked rank
    """
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    try:
        other_ranks = [r for r in range(world_size) if r != rank]

        if not other_ranks:
            buffer.destroy()
            return {"passed": True, "metrics": {"skipped": "single rank"}}

        # Barrier before connect
        sync_all_ranks(rank, world_size, "mask02_pre_connect")

        # Connect
        buffer.connect_ranks(other_ranks)

        # Barrier after connect
        sync_all_ranks(rank, world_size, "mask02_post_connect")

        target_rank = other_ranks[0]

        # First mask the rank
        buffer.update_mask_buffer(target_rank, True)

        # Then unmask it
        buffer.update_mask_buffer(target_rank, False)

        # Query mask status
        mask_status = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status)

        # Target should be unmasked (0)
        target_unmasked = mask_status[target_rank].item() == 0

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {
            "passed": target_unmasked,
            "error": (
                None
                if target_unmasked
                else "update_mask_buffer(False) didn't unmask rank"
            ),
            "metrics": {
                "target_rank": target_rank,
                "mask_value": mask_status[target_rank].item(),
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
@pytest.mark.masking
def test_update_mask_buffer_unset(request):
    """F-MASK-02: update_mask_buffer(rank, False) should unmask rank."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_update_mask_buffer_unset_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "update_mask_buffer unset failed"


# ============================================================================
# F-MASK-03: Clean mask buffer
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_clean_mask_buffer_fn(rank: int, world_size: int, local_rank: int = 0):
    """
    F-MASK-03: Test clean_mask_buffer resets all masks.

    Validates:
    - clean_mask_buffer() sets all masks to 0 (unmasked)
    """
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    try:
        other_ranks = [r for r in range(world_size) if r != rank]

        if not other_ranks:
            buffer.destroy()
            return {"passed": True, "metrics": {"skipped": "single rank"}}

        # Barrier before connect
        sync_all_ranks(rank, world_size, "mask03_pre_connect")

        # Connect
        buffer.connect_ranks(other_ranks)

        # Barrier after connect
        sync_all_ranks(rank, world_size, "mask03_post_connect")

        # Mask several ranks
        for r in other_ranks[: min(3, len(other_ranks))]:
            buffer.update_mask_buffer(r, True)

        # Verify some are masked
        mask_before = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_before)
        some_masked = any(
            mask_before[r].item() != 0 for r in other_ranks[: min(3, len(other_ranks))]
        )

        # Clean all masks
        buffer.clean_mask_buffer()

        # Verify all unmasked
        mask_after = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_after)
        all_unmasked = all(mask_after[r].item() == 0 for r in other_ranks)

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {
            "passed": some_masked and all_unmasked,
            "error": (
                None
                if (some_masked and all_unmasked)
                else "clean_mask_buffer didn't reset masks"
            ),
            "metrics": {
                "some_masked_before": some_masked,
                "all_unmasked_after": all_unmasked,
                "mask_before": mask_before.cpu().tolist(),
                "mask_after": mask_after.cpu().tolist(),
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
@pytest.mark.masking
def test_clean_mask_buffer(request):
    """F-MASK-03: clean_mask_buffer() should reset all masks to 0."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_clean_mask_buffer_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "clean_mask_buffer failed"


# ============================================================================
# F-MASK-04: Query mask buffer after successful operations
# ============================================================================


@pytest.mark.skip(reason="Not run directly")
def _test_query_mask_after_operations_fn(
    rank: int, world_size: int, local_rank: int = 0
):
    """
    F-MASK-04: Verify query_mask_buffer reports correct status after operations.

    This test follows the pattern from elastic.py:
    - Masking is REACTIVE, not PROACTIVE
    - After successful connect, all connected ranks should be unmasked (0)
    - query_mask_buffer() is used to detect failures AFTER operations

    Validates:
    - After connect, all ranks show as unmasked (0)
    - The mask buffer correctly tracks all connected ranks
    """
    import nixl_ep  # noqa: F401

    buffer = create_buffer(rank, world_size)

    try:
        other_ranks = [r for r in range(world_size) if r != rank]

        if not other_ranks:
            buffer.destroy()
            return {"passed": True, "metrics": {"skipped": "single rank"}}

        # Barrier before connect
        sync_all_ranks(rank, world_size, "mask04_pre_connect")

        # Connect to all other ranks
        buffer.connect_ranks(other_ranks)

        # Barrier after connect
        sync_all_ranks(rank, world_size, "mask04_post_connect")

        # Query mask buffer - all connected ranks should be unmasked (0)
        mask_status = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        buffer.query_mask_buffer(mask_status)

        # Verify all connected ranks are unmasked
        errors = []
        for r in other_ranks:
            if mask_status[r].item() != 0:
                errors.append(
                    f"Rank {r} should be unmasked (0), got {mask_status[r].item()}"
                )

        # Self should also be unmasked
        if mask_status[rank].item() != 0:
            errors.append(
                f"Self rank {rank} should be unmasked (0), got {mask_status[rank].item()}"
            )

        all_unmasked = len(errors) == 0

        # Cleanup - just destroy (matches elastic.py pattern)
        buffer.destroy()

        return {
            "passed": all_unmasked,
            "error": "\n".join(errors) if errors else None,
            "metrics": {
                "mask_status": mask_status.tolist(),
                "all_unmasked": all_unmasked,
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
@pytest.mark.masking
def test_query_mask_after_operations(request):
    """F-MASK-04: query_mask_buffer should show all ranks unmasked after successful connect."""
    num_processes = int(os.environ.get("NIXL_TEST_NUM_PROCESSES", 8))
    results = run_multiprocess_test(
        _test_query_mask_after_operations_fn, num_processes=num_processes
    )
    print_results(results)
    assert all_passed(results), "query_mask_buffer showed unexpected masked ranks"


# ============================================================================
# CLI runner
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run masking functional tests")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--etcd-server", type=str, default="http://127.0.0.1:2379")
    parser.add_argument("--test", type=str, default="all")
    args = parser.parse_args()

    os.environ["NIXL_TEST_NUM_PROCESSES"] = str(args.num_processes)

    tests = {
        "set": (_test_update_mask_buffer_set_fn, "F-MASK-01: Set mask"),
        "unset": (_test_update_mask_buffer_unset_fn, "F-MASK-02: Unset mask"),
        "clean": (_test_clean_mask_buffer_fn, "F-MASK-03: Clean mask buffer"),
        "query": (
            _test_query_mask_after_operations_fn,
            "F-MASK-04: Query mask after operations",
        ),
    }

    if args.test == "all":
        for name, (fn, desc) in tests.items():
            sys.stderr.write(f"\nRunning: {desc}\n")
            results = run_multiprocess_test(
                fn, num_processes=args.num_processes, etcd_server=args.etcd_server
            )
            print_results(results)
    elif args.test in tests:
        fn, desc = tests[args.test]
        sys.stderr.write(f"\nRunning: {desc}\n")
        results = run_multiprocess_test(
            fn, num_processes=args.num_processes, etcd_server=args.etcd_server
        )
        print_results(results)
