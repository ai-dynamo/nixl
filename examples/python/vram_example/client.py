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
import time

import torch

from nixl._api import nixl_agent, nixl_agent_config
from utils import calc_memory_blocks, get_logger, parse_args

logger = get_logger(__name__)


def transfer(
    agent,
    size_in_bytes,
    local_prep_handle,
    remote_prep_handle,
    addrs,
    trans_blocks,
    layers,
    num_blocks,
    seq,
):
    start = time.monotonic()

    # Calculate transfer data block indices. Simply this prepares first
    # trans_blocks elements for each layer.
    indices = []
    for layer in range(layers):
        block_offset = layer * num_blocks
        indices.extend([block_offset + i for i in range(trans_blocks)])

    logger.debug("%d blocks will be transferred", len(indices))
    xfer_handle = agent.make_prepped_xfer(
        "READ",
        local_prep_handle,
        indices,
        remote_prep_handle,
        indices,
        str(seq).encode(),
        ["UCX"],
    )

    if not xfer_handle:
        logger.error("Creating transfer failed.")
        sys.exit()

    state = agent.transfer(xfer_handle)
    if state == "ERR":
        logger.error("Posting transfer failed.")
        sys.exit()

    while True:
        state = agent.check_xfer_state(xfer_handle)
        if state == "ERR":
            logger.error("Transfer got to Error state.")
            sys.exit()
        elif state == "DONE":
            break

    end = time.monotonic()
    ratio = len(indices) / len(addrs)
    logger.info(
        "Throughput: %f MiB/sec, in %f sec",
        size_in_bytes * ratio / (end - start) / 1024 / 1024,
        (end - start),
    )
    agent.release_xfer_handle(xfer_handle)


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    tensor_size, shape_len, num_blocks = calc_memory_blocks(args)

    count = args.count

    # Eumurate prefill with input_toekns
    input_tokens = args.input_tokens
    trans_blocks = input_tokens // args.block_size
    logger.info("This test will transfer %d blocks", trans_blocks)

    device = "cuda"
    config = nixl_agent_config(True, True, 0)
    agent = nixl_agent("client", config)

    # Allocate memory and register with NIXL
    tensors = torch.zeros(
        tensor_size * args.layers, dtype=torch.bfloat16, device=device
    )

    # size_in_bytes = tensors[0].nelement() * tensors[0].element_size() * len(tensors)
    size_in_bytes = tensors.nelement() * tensors.element_size()
    logger.info("Client Tensors in MB: %d", size_in_bytes / 1024 / 1024)

    block_len = shape_len * tensors.element_size()  # bytes of tensor
    logger.debug("block_len: %d", block_len)

    reg_descs = agent.get_reg_descs(tensors, "VRAM")

    success = agent.register_memory(reg_descs)
    if not success:  # Same as reg_descs if successful
        logger.error("Memory registration failed.")
        sys.exit()

    # Create data block chunk to emulate vllm 0.10.0 data transfer
    xfer_addrs = []
    base_addr = tensors[0].data_ptr()
    for block_id in range(num_blocks * args.layers):
        offset = block_id * block_len
        addr = base_addr + offset
        xfer_addrs.append((addr, block_len, 0))

    logger.info(
        "addrs info: layers: %d, elements: %d, shape: %d, block_len: %d, "
        "addrs_len: %d",
        args.layers,
        tensor_size,
        shape_len,
        block_len,
        len(xfer_addrs),
    )

    xfer_descs = agent.get_xfer_descs(xfer_addrs, "VRAM")

    logger.debug("xfer_descs: %s", xfer_descs)
    logger.debug("descCount: %d", xfer_descs.descCount())
    logger.debug("isEmpty: %s", xfer_descs.isEmpty())

    logger.info("Client sending to %s", args.ip)
    agent.fetch_remote_metadata("server", args.ip, args.port)

    # Check if remote server is available.
    ready = False
    while not ready:
        ready = agent.check_remote_metadata("server")

    agent.send_local_metadata(args.ip, args.port)

    # Prepare descriptor list.
    local_prep_handle = agent.prep_xfer_dlist(
        "NIXL_INIT_AGENT",
        xfer_addrs,
        "VRAM",
    )

    assert local_prep_handle != 0

    # Handshake to server.
    agent.send_notif("server", f"SYN:{count}".encode())

    notifs = agent.get_new_notifs()

    while len(notifs) == 0:
        notifs = agent.get_new_notifs()

    target_descs = agent.deserialize_descs(notifs["server"][0])
    logger.debug("target_descs: %s", target_descs)
    logger.debug("target descCount: %d", target_descs.descCount())
    logger.debug("target isEmpty: %s", target_descs.isEmpty())
    remote_prep_handle = agent.prep_xfer_dlist("server", target_descs, "VRAM")

    assert local_prep_handle != 0
    assert remote_prep_handle != 0

    for seq in range(count):
        logger.debug("trans with sequence: %s", seq)
        transfer(
            agent,
            size_in_bytes,
            local_prep_handle,
            remote_prep_handle,
            xfer_addrs,
            trans_blocks,
            args.layers,
            num_blocks,
            seq,
        )
        # agent.send_notif("server", str(seq).encode())

    # Verify data after read.
    # for i, tensor in enumerate(tensors):
    for i in range(args.layers):
        layer_base = i * tensor_size
        check_blocks = trans_blocks * shape_len
        if not torch.allclose(
            tensors[layer_base : layer_base + check_blocks],
            torch.ones(check_blocks, dtype=torch.bfloat16, device=device),
        ):
            logger.error("Data verification failed for tensor %d.", i)
            sys.exit()

    logger.info("Client Data verification passed")
    logger.debug(tensors)

    agent.remove_remote_agent("server")
    agent.invalidate_local_metadata(args.ip, args.port)
    logger.info("Test Complete.")


if __name__ == "__main__":
    main()
