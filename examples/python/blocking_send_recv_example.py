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

import argparse

import torch

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument(
        "--mode",
        type=str,
        default="initiator",
        help="Local IP in target, peer IP (target's) in initiator",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # initiator use default port
    listen_port = args.port
    if args.mode != "target":
        listen_port = 0

    if args.use_cuda:
        torch.set_default_device("cuda:0")
    else:  # To be sure this is the default
        torch.set_default_device("cpu")

    config = nixl_agent_config(True, True, listen_port)

    # Allocate memory and register with NIXL
    agent = nixl_agent(args.mode, config)
    if args.mode == "target":
        tensors = [torch.ones(10, dtype=torch.float32) for _ in range(2)]
    else:
        tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(2)]

    logger.info("Running test with %s tensors in mode %s", tensors, args.mode)

    reg_descs = agent.register_memory(tensors)
    if not reg_descs:  # Same as reg_descs if successful
        logger.error("Memory registration failed.")
        exit()

    # Target code
    if args.mode == "target":
        ready = False

        target_descs = reg_descs.trim()
        target_desc_str = agent.get_serialized_descs(target_descs)

        # Send desc list to initiator when metadata is ready
        while not ready:
            ready = agent.check_remote_metadata("initiator")

        agent.send_notif("initiator", target_desc_str)

        logger.info("Waiting for transfer")

        # Waiting for transfer
        # For now the notification is just UUID, could be any python bytes.
        # Also can have more than UUID, and check_remote_xfer_done returns
        # the full python bytes, here it would be just UUID.
        while not agent.check_remote_xfer_done("initiator", b"UUID"):
            continue
    # Initiator code
    else:
        logger.info("Initiator sending to %s", args.ip)
        agent.fetch_remote_metadata("target", args.ip, args.port)
        agent.send_local_metadata(args.ip, args.port)

        notifs = agent.get_new_notifs()

        while len(notifs) == 0:
            notifs = agent.get_new_notifs()

        target_descs = agent.deserialize_descs(notifs["target"][0])
        initiator_descs = reg_descs.trim()

        # Ensure remote metadata has arrived from fetch
        ready = False
        while not ready:
            ready = agent.check_remote_metadata("target")

        logger.info("Ready for transfer")

        xfer_handle = agent.initialize_xfer(
            "READ", initiator_descs, target_descs, "target", "UUID"
        )

        if not xfer_handle:
            logger.error("Creating transfer failed.")
            exit()

        state = agent.transfer(xfer_handle)
        if state == "ERR":
            logger.error("Posting transfer failed.")
            exit()
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                logger.error("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                break

        # Verify data after read
        for i, tensor in enumerate(tensors):
            if not torch.allclose(tensor, torch.ones(10)):
                logger.error("Data verification failed for tensor %d.", i)
                exit()
        logger.info("%s Data verification passed", args.mode)

    if args.mode != "target":
        agent.remove_remote_agent("target")
        agent.release_xfer_handle(xfer_handle)
        agent.invalidate_local_metadata(args.ip, args.port)

    agent.deregister_memory(reg_descs)

    logger.info("Test Complete.")
