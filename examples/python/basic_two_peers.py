#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    # Use a single 2D tensor with 10 tensors of size 16
    if args.mode == "target":
        tensor = torch.ones((10, 16), dtype=torch.float32)
    else:
        tensor = torch.zeros((10, 16), dtype=torch.float32)

    logger.info(
        "Running test with tensor shape %s in mode %s", tuple(tensor.shape), args.mode
    )

    # Register the single 2D tensor
    reg_descs = agent.register_memory(tensor)
    if not reg_descs:
        logger.error("Memory registration failed.")
        exit(1)

    # Target code
    if args.mode == "target":
        ready = False

        # Build transfer descriptors by unraveling first dim into list of row tensors
        target_rows = [tensor[i, :] for i in range(tensor.shape[0])]
        target_descs = agent.get_xfer_descs(target_rows)
        if not target_descs:
            logger.error("Failed to build target transfer descriptors.")
            exit(1)
        target_desc_str = agent.get_serialized_descs(target_descs)

        # Send desc list to initiator when metadata is ready
        while not ready:
            ready = agent.check_remote_metadata("initiator")
        agent.send_notif("initiator", target_desc_str)

        logger.info("Waiting for transfer")

        # Waiting for transfer
        while True:
            notifs = agent.get_new_notifs()
            if "initiator" in notifs and b"Done_reading" in notifs["initiator"]:
                break

    # Initiator code
    else:
        logger.info("Initiator sending to %s", args.ip)
        agent.fetch_remote_metadata("target", args.ip, args.port)
        agent.send_local_metadata(args.ip, args.port)

        notifs = agent.get_new_notifs()
        while len(notifs) == 0:
            notifs = agent.get_new_notifs()
        target_descs = agent.deserialize_descs(notifs["target"][0])

        # Build local transfer descriptors by unraveling first dim into list of row tensors
        initiator_rows = [tensor[i, :] for i in range(tensor.shape[0])]
        initiator_descs = agent.get_xfer_descs(initiator_rows)
        if not initiator_descs:
            logger.error("Initiator's local descriptors creation failed.")
            exit(1)

        # Ensure remote metadata has arrived from fetch
        ready = False
        while not ready:
            ready = agent.check_remote_metadata("target")

        logger.info("Ready for transfer")

        xfer_handle = agent.initialize_xfer(
            "READ", initiator_descs, target_descs, "target", "Done_reading"
        )

        state = agent.transfer(xfer_handle)
        if state == "ERR":
            logger.error("Posting transfer failed.")
            exit(1)

        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                logger.error("Transfer got to Error state.")
                exit(1)
            elif state == "DONE":
                break

        # Verify data after read
        if not torch.allclose(tensor, torch.ones((10, 16))):
            logger.error("Data verification failed.")
            exit()
        logger.info("%s Data verification passed", args.mode)

    # Tear down
    if args.mode != "target":
        agent.remove_remote_agent("target")
        agent.release_xfer_handle(xfer_handle)
        agent.invalidate_local_metadata(args.ip, args.port)

    agent.deregister_memory(reg_descs)

    logger.info("Test Complete.")
