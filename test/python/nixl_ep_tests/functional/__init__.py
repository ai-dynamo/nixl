# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functional tests for NIXL EP Buffer.

These tests require multiple processes/GPUs and verify:
- Connection management (F-CONN-*)
- Masking operations (F-MASK-*)
- Dispatch operations (F-DISP-*)
- Combine operations (F-COMB-*)
- End-to-end flows (F-E2E-*)

Usage:
    # Run all functional tests with 8 processes
    python -m pytest functional/ -v --num-processes=8

    # Run specific test category
    python -m pytest functional/test_connection.py -v

    # Run standalone (without pytest)
    python functional/test_connection.py --num-processes=8 --test=all
"""

from utils.mp_runner import (
    DistributedBarrier,
    MultiProcessTestCase,
    TestResult,
    all_passed,
    create_buffer,
    print_results,
    run_multiprocess_test,
    sync_all_ranks,
)

__all__ = [
    "run_multiprocess_test",
    "create_buffer",
    "all_passed",
    "print_results",
    "TestResult",
    "MultiProcessTestCase",
    "sync_all_ranks",
    "DistributedBarrier",
]
