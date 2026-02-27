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

import multiprocessing
import time

import torch

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


def run_target():
    """
    Target mode function that runs in a subprocess.
    This posts metadata to etcd and then is killed.
    """
    logger.info("Target subprocess started")

    config = nixl_agent_config(True, True, 5555)

    # Allocate memory and register with NIXL
    agent = nixl_agent(
        "target",
        config,
    )
    tensors = [torch.ones(10, dtype=torch.float32) for _ in range(2)]

    logger.info("Target running with tensors: %s", tensors)

    reg_descs = agent.register_memory(tensors)
    if not reg_descs:
        logger.error("Target: Memory registration failed.")
        return

    agent.send_local_metadata()

    logger.info("Waiting to die")

    time.sleep(100)

    agent.deregister_memory(reg_descs)

    logger.info("Target subprocess complete successfully (should have died by now).")


if __name__ == "__main__":
    # Start the target process
    target_process = multiprocessing.Process(target=run_target)
    target_process.start()

    logger.info("Subprocess started, pausing...")

    time.sleep(5)

    config = nixl_agent_config(True, True)

    agent = nixl_agent("initiator", config)

    # Fetch remote metadata when its ready
    agent.fetch_remote_metadata("target")

    # Ensure remote metadata has arrived from fetch
    ready = False
    while not ready:
        ready = agent.check_remote_metadata("target")

    logger.info("Ready to kill, pausing...")

    time.sleep(5)
    # SIGKILL the target process to test heartbeat failure
    target_process.kill()

    logger.info("Target process killed, waiting for metadata to be invalidated")

    # Wait for metadata to be invalidated
    ready = True
    while ready:
        ready = agent.check_remote_metadata("target")

    agent.invalidate_local_metadata()

    logger.info("Test Complete.")
