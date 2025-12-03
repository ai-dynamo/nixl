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
    try:
        agent = nixl_agent(args.mode, config)
    except Exception as e:
        logger.exception("Failed to create NIXL agent: %s", e)
        exit(1)

    # Use a single 2D tensor with 10 tensors of size 16
    if args.mode == "target":
        tensor = torch.ones((12, 16), dtype=torch.float32)
    else:
        tensor = torch.zeros((12, 16), dtype=torch.float32)

    logger.info(
        "Running test with tensor shape %s in mode %s", tuple(tensor.shape), args.mode
    )

    # Register the single 2D tensor
    try:
        reg_descs = agent.register_memory(tensor)
        if not reg_descs:
            logger.error("Memory registration failed.")
            exit(1)
    except Exception as e:
        logger.exception("Memory registration failed: %s", e)
        exit(1)

    # Target code
    if args.mode == "target":
        ready = False

        # Build transfer descriptors by unraveling first dim into list of row tensors
        try:
            target_rows = [tensor[i, :] for i in range(tensor.shape[0])]
            target_descs = agent.get_xfer_descs(target_rows)
            if not target_descs:
                logger.error("Failed to build target transfer descriptors.")
                exit(1)
            target_desc_str = agent.get_serialized_descs(target_descs)
        except Exception as e:
            logger.exception("Preparing target descriptors failed: %s", e)
            exit(1)

        # Send desc list to initiator when metadata is ready
        try:
            while not ready:
                ready = agent.check_remote_metadata("initiator")
            agent.send_notif("initiator", target_desc_str)
        except Exception as e:
            logger.exception("Send of descriptors to initiator failed: %s", e)
            exit(1)

        logger.info("Waiting for transfer")

        # Waiting for transfer
        # For now the notification is just UUID, could be any python bytes.
        # Also can have more than UUID, and check_remote_xfer_done returns
        # the full python bytes, here it would be just UUID.
        try:
            while True:
                if agent.check_remote_xfer_done("initiator", b"UUID"):
                    break
        except Exception as e:
            logger.exception("Checking for transfer notification failed: %s", e)
            exit(1)

    # Initiator code
    else:
        logger.info("Initiator sending to %s", args.ip)
        try:
            agent.fetch_remote_metadata("target", args.ip, args.port)
            agent.send_local_metadata(args.ip, args.port)
        except Exception as e:
            logger.exception("Metadata exchange (fetch/send) failed: %s", e)
            exit(1)

        try:
            notifs = agent.get_new_notifs()
            while len(notifs) == 0:
                notifs = agent.get_new_notifs()
            target_descs = agent.deserialize_descs(notifs["target"][0])
        except Exception as e:
            logger.exception("Receiving target descriptors failed: %s", e)
            exit(1)

        # Build local transfer descriptors by unraveling first dim into list of row tensors
        try:
            initiator_rows = [tensor[i, :] for i in range(tensor.shape[0])]
            initiator_descs = agent.get_xfer_descs(initiator_rows)
            if not initiator_descs:
                logger.exception("Initiator's local descriptors creation failed.")
                exit(1)
        except Exception as e:
            logger.exception("Initiator's local descriptors creation failed: %s", e)
            exit(1)

        # Ensure remote metadata has arrived from fetch
        ready = False
        while not ready:
            try:
                ready = agent.check_remote_metadata("target")
            except Exception as e:
                logger.exception("Checking of target metadata failed: %s", e)
                exit(1)

        logger.info("Ready for transfer")

        try:
            xfer_handle = agent.initialize_xfer(
                "READ", initiator_descs, target_descs, "target", "UUID"
            )
        except Exception as e:
            logger.exception("Transfer handle creation failed: %s", e)
            exit(1)

        try:
            state = agent.transfer(xfer_handle)
            if state == "ERR":
                logger.error("Posting transfer failed.")
                exit(1)
        except Exception as e:
            logger.exception("Transfer post failed: %s", e)
            exit(1)

        try:
            while True:
                state = agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    logger.error("Transfer got to Error state.")
                    exit(1)
                elif state == "DONE":
                    break
        except Exception as e:
            logger.exception("Checking transfer completion failed: %s", e)
            exit(1)

        # Verify data after read
        if not torch.allclose(tensor, torch.ones((12, 16))):
            logger.error("Data verification failed.")
            exit()
        logger.info("%s Data verification passed", args.mode)

    if args.mode != "target":
        try:
            agent.remove_remote_agent("target")
            # release handle and invalidate metadata
            agent.release_xfer_handle(xfer_handle)
            agent.invalidate_local_metadata(args.ip, args.port)
        except Exception as e:
            logger.exception("Tear down (metadata/transfer handles) failed: %s", e)

    try:
        agent.deregister_memory(reg_descs)
    except Exception as e:
        logger.exception("Deregisteration of memory failed: %s", e)

    logger.info("Test Complete.")
