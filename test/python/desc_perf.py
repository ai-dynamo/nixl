#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import nixl._utils as nixl_utils
from nixl._api import nixl_agent
from nixl.logging import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    desc_count = 24 * 64 * 1024
    agent = nixl_agent("test", None)
    addr = nixl_utils.malloc_passthru(256)

    addr_list = [(addr, 256, 0)] * desc_count

    start_time = time.perf_counter()

    descs = agent.get_xfer_descs(addr_list, "DRAM")

    end_time = time.perf_counter()

    assert descs.descCount() == desc_count

    logger.info(
        "Time per desc add in us: %f",
        (1000000.0 * (end_time - start_time)) / desc_count,
    )
    nixl_utils.free_passthru(addr)
