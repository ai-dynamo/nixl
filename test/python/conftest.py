# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="UCX",
        help="Backend plugin name to use for tests (default: UCX)",
    )


@pytest.fixture(scope="session")
def backend_name(pytestconfig):
    return pytestconfig.getoption("--backend")
