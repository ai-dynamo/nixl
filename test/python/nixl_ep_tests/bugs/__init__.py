# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Bug Reproduction Tests for NIXL EP
#
# This module contains tests that reproduce known bugs in NIXL, UCX, and related libraries.
# These tests serve multiple purposes:
#
# 1. Help developers reproduce bugs consistently
# 2. Verify fixes after patches are applied
# 3. Track regression in CI/CD pipelines
# 4. Document exact conditions that trigger each bug
#
# Usage:
#   python3 -m pytest tests/bugs/ -v --timeout=120
#   python3 tests/bugs/test_bug_01_segfault.py --num-processes=8
#
# See README.md for detailed documentation of each bug.

__all__ = [
    "test_bug_01_segfault",
    "test_bug_02_rcache",
    "test_bug_03_gdr_copy",
    "test_bug_04_invalidate",
]
