# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for NIXL EP tests.

This package contains:
- mp_runner: Multi-process test runner with GPU/UCX coordination
- test_rank_server: TCP-based rank server for distributed tests
- helpers: General test utilities
- results_reporter: Test results formatting and reporting
"""

from .mp_runner import (
    DistributedBarrier,
    MultiProcessTestCase,
    TestResult,
    all_passed,
    create_buffer,
    discover_gpu_nic_topology,
    print_results,
    run_multiprocess_test,
    setup_worker_environment,
    sync_all_ranks,
)
from .results_reporter import ResultsReporter, get_reporter, init_reporter
from .test_rank_server import RankClient, RankServer, start_test_server

__all__ = [
    # mp_runner
    "run_multiprocess_test",
    "create_buffer",
    "all_passed",
    "print_results",
    "TestResult",
    "MultiProcessTestCase",
    "sync_all_ranks",
    "DistributedBarrier",
    "setup_worker_environment",
    "discover_gpu_nic_topology",
    # test_rank_server
    "RankServer",
    "RankClient",
    "start_test_server",
    # results_reporter
    "ResultsReporter",
    "get_reporter",
    "init_reporter",
]
