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
import zmq

from nixl._api import nixl_agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--peer_ip", type=str, required=True)
    parser.add_argument("--mode", type=str, default="initiator")
    parser.add_argument("--peer_port", type=int, default=5555)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # zmq as side channel
    _ctx = zmq.Context()
    _socket = _ctx.socket(zmq.PAIR)
    _socket.connect(f"tcp: //{args.peer_ip}: {args.peer_port}")

    # Allocate memory and register with NIXL
    agent = nixl_agent(args.name, None)
    tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(2)]

    reg_descs = agent.register_memory(tensors)
    if not reg_descs:  # Same as reg_descs if successful
        print("Memory registration failed.")
        exit()

    # Exchange metadata
    if args.mode == "target":
        meta = agent.get_agent_metadata()
        if not meta:
            print("Acquiring metadata in target agent failed.")
            exit()

        _socket.send(meta)
        peer_name = _socket.recv()
    else:
        remote_meta = _socket.recv()
        _socket.send(args.name)  # We just need the name, not full meta

        peer_name = agent.add_remote_agent(remote_meta)
        if not peer_name:
            print("Loading target metadata in initiator agent failed.")
            exit()

    # Blocking send/recv (pull mode)
    if args.mode == "target":
        # If desired, can use send_notif instead. Also indicate
        # the notification that is expected to be received.
        targer_descs = reg_descs
        _socket.send(agent.get_serialized_descs(targer_descs))
        # For now the notification is just UUID, could be any python bytes.
        # Also can have more than UUID, and check_remote_xfer_done returns
        # the full python bytes, here it would be just UUID.
        while not agent.check_remote_xfer_done(peer_name, "UUID"):
            continue
    else:
        # If send_notif is used, get_new_notifs should listen for it,
        # or directly calling check_remote_xfer_done
        targer_descs = agent.deserialize_descs(_socket.recv())
        initiator_descs = reg_descs
        xfer_handle = agent.initialize_xfer(
            initiator_descs, targer_descs, peer_name, "UUID", "READ"
        )
        if not xfer_handle:
            print("Creating transfer failed.")
            exit()

        xfer_handle = agent.transfer(xfer_handle)
        if not xfer_handle:
            print("Posting transfer failed.")
            exit()

        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                break

    if args.mode != "target":
        agent.remove_remote_agent(peer_name)
        agent.abort_xfer(xfer_handle)

    agent.deregister_memory(reg_descs)

    print("Test Complete.")
