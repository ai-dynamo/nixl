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

import logging
import sys

import torch

from nixl._api import nixl_agent, nixl_agent_config
from utils import calc_memory_blocks, get_logger, parse_args

logger = get_logger(__name__)


def _handle_request(agent, xfer_desc_str):
    # Send desc list to initiator when metadata is ready.
    ready = False
    while not ready:
        ready = agent.check_remote_metadata("client")

    msgs = []
    # Handshake from client.
    logger.debug("Waiting for handshake")
    if not msgs:
        notifs = agent.get_new_notifs()
        while len(notifs) == 0:
            notifs = agent.get_new_notifs()
        msgs.extend(notifs["client"])

    msg = msgs.pop(0).decode()
    assert msg.startswith("SYN")
    seqs = msg.split(":")[-1]
    logger.info("Request received")
    agent.send_notif("client", xfer_desc_str)
    logger.debug("Waiting for transfer")

    def _step():
        if not msgs:
            # Check if we have to keep this connection.
            client_notifs = agent.update_notifs(["UCX"]).get("client", [])
            while len(client_notifs) == 0:
                notifs = agent.get_new_notifs()
                client_notifs = notifs.get("client", []).copy()
            msgs.extend(client_notifs)
        msg = msgs.pop(0).decode()
        logger.debug("Got trans message %s", msg)
        return msg

    completed_seqs = set()

    while True:
        msg = _step()
        completed_seqs.add(int(msg))
        logger.debug(msg)

        if completed_seqs == set(range(int(seqs))):
            logger.debug(
                "All sequences are completed: %s",
                completed_seqs,
            )
            break

    logger.debug("Remove remote agent and fall back to initiation")
    agent.remove_remote_agent("client")


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    tensor_size, shape_len, num_blocks = calc_memory_blocks(args)
    listen_port = args.port
    device = "cuda"

    config = nixl_agent_config(True, True, listen_port)
    agent = nixl_agent("server", config)

    # Allocate memory and register with NIXL.
    tensors = torch.ones(tensor_size * args.layers, dtype=torch.bfloat16, device=device)

    logger.debug("Tensor buffer for transfer... %s", tensors)
    size_in_bytes = tensors.nelement() * tensors.element_size()
    logger.info("Server Tensor Buffer in MB: %d", size_in_bytes / 1024 / 1024)

    block_len = shape_len * tensors.element_size()  # Bytes of tensor.
    logger.debug("block_len: %d", block_len)
    logger.debug("num_blocks: %d", num_blocks)

    logger.debug(
        "first ptr: %d, second ptr %d",
        tensors[0].data_ptr(),
        tensors[shape_len].data_ptr(),
    )
    logger.debug("distance: %d", tensors[shape_len].data_ptr() - tensors[0].data_ptr())
    logger.debug("nelement: %d", tensors.nelement())
    reg_descs = agent.get_reg_descs(tensors, "VRAM")
    success = agent.register_memory(reg_descs)

    if not success:  # Same as reg_descs if successful.
        logger.error("Memory registration failed.")
        sys.exit()

    xfer_addrs = []
    base_addr = tensors[0].data_ptr()
    for block_id in range(num_blocks * args.layers):
        offset = block_id * block_len
        addr = base_addr + offset
        xfer_addrs.append((addr, block_len, 0))

    xfer_descs = agent.get_xfer_descs(xfer_addrs, "VRAM")
    xfer_desc_str = agent.get_serialized_descs(xfer_descs)
    logger.info("Serialized xfer_desc str len: %d", len(xfer_desc_str))

    try:
        # Daemonize server process for testing until killed by hand.
        while True:
            logger.debug("Waiting for initialization")
            logger.debug(tensors)
            _handle_request(agent, xfer_desc_str)
    finally:
        agent.deregister_memory(reg_descs)
        logger.info("Test Complete.")


if __name__ == "__main__":
    main()
