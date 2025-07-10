#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for ETCD Python runtime
Run multiple instances of this script to test distributed functionality
"""

import os
import sys

from nixl.logging import get_logger

# Add the kvbench runtime path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../kvbench/runtime"))

log = get_logger(__name__)

try:
    from etcd_rt import _EtcdDistUtils

    def test_basic_functionality():
        """Test basic rank and size functionality"""
        log.info("Testing basic functionality...")

        # Initialize runtime - modify size based on how many processes you're running
        runtime = _EtcdDistUtils(etcd_endpoints="http://localhost:2379", size=2)

        rank = runtime.get_rank()
        world_size = runtime.get_world_size()

        log.info(f"Rank: {rank}, World Size: {world_size}")

        # Test barrier
        log.info(f"Rank {rank}: Before barrier")
        runtime.barrier()
        log.info(f"Rank {rank}: After barrier")

        # Test allgather
        my_data = {"rank": rank, "message": f"Hello from rank {rank}"}
        log.info(f"Rank {rank}: Gathering data...")

        try:
            all_data = runtime.allgather_obj(my_data)
            log.info(f"Rank {rank}: Gathered data from all ranks:")
            for i, data in enumerate(all_data):
                log.info(f"  Rank {i}: {data}")
        except Exception as e:
            log.error(f"Rank {rank}: Allgather failed: {e}")

        # Test barrier again
        runtime.barrier()
        log.info(f"Rank {rank}: Test completed successfully!")

    if __name__ == "__main__":
        test_basic_functionality()

except ImportError as e:
    log.error(f"Import error: {e}")
    log.error("Make sure the etcd_runtime module is built and accessible")
    log.error("Also ensure the etcd server is running at http://localhost:2379")
    sys.exit(1)
except Exception as e:
    log.error(f"Runtime error: {e}")
    sys.exit(1)
