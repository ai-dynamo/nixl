#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nixl_cu13 import _service_api as svc

# Repeating 0..255 byte pattern used to seed the data-source buffer. It is cheap
# to build (a 256-byte source tensor) yet order-sensitive, so the sink's check
# catches dropped, duplicated, or mis-ordered bytes - not merely "something
# arrived". The transfer size below is a multiple of this period.
PATTERN_PERIOD = 256

TRANSFER_BYTES = 512 * 1024 * 1024


def print_rank(name, message):
    print(f"[{name}] {message}", flush=True)


def _known_pattern(device):
    return torch.arange(PATTERN_PERIOD, dtype=torch.uint8, device=device)


def init_agent(name, rank):
    torch.cuda.set_device(rank)

    print_rank(name, "Initializing service agent and marshal settings...")

    # Configuring marshal mode
    mode = svc.nixlMarshalCompressConfig()

    # Creating service-aware agent configs and agents
    cfg = svc.nixl_service_agent_config(mode=mode)
    agent = svc.nixl_service_agent(name, nixl_conf=cfg)

    # Getting recommended service pool sizing
    svc_mem_bytes = svc.recommendServiceMemSize(mode, maxConcurrentTransfers=1)

    # Allocating data transfer buffers
    num_bytes = TRANSFER_BYTES
    assert num_bytes % PATTERN_PERIOD == 0, "transfer size must be a multiple of the pattern period"
    data_buf = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")

    # Allocating and registering service buffers
    svc_buf = torch.empty(svc_mem_bytes, dtype=torch.uint8, device="cuda")
    agent.register_service_memory([svc_buf])

    # IMPORTANT: descriptors of 64 MiB or less are served by native NIXL, so
    # their memory must also be registered with `register_memory`. Descriptors
    # larger than that are served by the service and need no registration
    # beyond the service buffers above. The threshold applies per descriptor of
    # a transfer, so a request may need both registrations.
    #
    # Every descriptor here is 512 MiB, hence `data_buf` is left unregistered.

    print_rank(name, "Exchanging metadata...")

    dist.init_process_group(backend="gloo", rank=rank, world_size=2)

    # Getting local agent metadata
    local_md = bytes(agent.get_agent_metadata())
    dev_id = torch.cuda.current_device()

    # Gathering metadata from all agents
    gathered_md = [None] * 2
    dist.all_gather_object(gathered_md, {
        "metadata": local_md,
        "data_ptr": data_buf.data_ptr(),
        "dev_id": dev_id
    })

    return agent, gathered_md, num_bytes, data_buf


def run_initiator(name, agent, gathered_md, num_bytes, direction):
    # Adding remote agent (the peer, rank 1)
    remote_md = gathered_md[1]["metadata"]
    remote_name = agent.add_remote_agent(remote_md).decode()

    # ``local`` is always this rank's (rank 0) own buffer and ``remote`` the
    # peer's - the same two descriptor lists for both operations.  For WRITE,
    # local is the source and remote the destination; for READ the roles invert
    # (local is where the pulled data lands, remote is the peer's source), but
    # rank 0 stays the active initiator that posts, polls, and - on completion -
    # notifies the peer either way.
    local_descs = agent.get_xfer_descs([(
        gathered_md[0]["data_ptr"],
        num_bytes,
        gathered_md[0]["dev_id"]
    )], mem_type="VRAM")

    remote_descs = agent.get_xfer_descs([(
        gathered_md[1]["data_ptr"],
        num_bytes,
        gathered_md[1]["dev_id"]
    )], mem_type="VRAM")

    op = direction.upper()
    print_rank(name, f"Initiating and executing {op} with {remote_name}...")
    req = agent.initialize_xfer(
        op,
        local_descs,
        remote_descs,
        remote_agent=remote_name,
        notif_msg=b"SVC_DONE",
    )
    agent.transfer(req, notif_msg=b"SVC_DONE")

    print_rank(name, "Polling transfer state until completion...")
    while agent.check_xfer_state(req) == "PROC":
        pass

    req.release()


def run_peer(name, agent, gathered_md):
    # Adding remote agent (the initiator, rank 0)
    remote_md = gathered_md[0]["metadata"]
    remote_name = agent.add_remote_agent(remote_md).decode()

    # The passive side for both directions: the agent's progress thread services
    # an incoming WRITE (data lands in this rank's buffer) or an incoming READ
    # (this rank is the source and the service pushes its data to the
    # initiator).  Either way completion is signalled by a single notification
    # from the initiator, keyed by the initiator's name.
    print_rank(name, "Waiting for the initiator's completion notification...")

    got_notif = False
    while not got_notif:
        notifs = agent.get_new_notifs()
        got_notif = remote_name in notifs and len(notifs[remote_name]) >= 1


def worker(rank, direction):
    agent_name = f"Agent-{rank}"
    print_rank(agent_name, f"Starting worker {rank} (direction={direction})...")

    agent, gathered_md, num_bytes, data_buf = init_agent(agent_name, rank)

    # rank 0 is always the active initiator; rank 1 is always the passive peer.
    # The data source is the WRITE initiator (rank 0) or the READ peer (rank 1);
    # the other rank is the sink that receives and then verifies the bytes.
    is_initiator = rank == 0
    is_source = (direction == "write") == is_initiator

    if is_source:
        # Seed the source with the pattern the sink will check for.
        data_buf.view(-1, PATTERN_PERIOD).copy_(_known_pattern(data_buf.device))
    else:
        # Zero the landing buffer so a successful transfer is unambiguous.
        data_buf.zero_()
    torch.cuda.synchronize()

    # Barrier so the source is filled (and the sink zeroed) before the initiator
    # starts the transfer - otherwise a late zero could clobber a WRITE, or a
    # late fill could be missed by a READ.
    dist.barrier()

    if is_initiator:
        run_initiator(agent_name, agent, gathered_md, num_bytes, direction)
    else:
        run_peer(agent_name, agent, gathered_md)

    if not is_source:
        # The sink verifies the bytes it received against the known pattern.
        pattern = _known_pattern(data_buf.device)
        matches = bool((data_buf.view(-1, PATTERN_PERIOD) == pattern).all().item())
        if not matches:
            raise AssertionError(
                f"[{agent_name}] verification FAILED: received data does not match the expected pattern"
            )
        print_rank(agent_name, f"Verification PASSED: {num_bytes} received bytes match the expected pattern.")

    print_rank(agent_name, f"Worker {rank} done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIXL service-agent WRITE/READ example over two local GPUs."
    )
    parser.add_argument(
        "--direction",
        choices=("write", "read"),
        default="write",
        help="transfer direction: 'write' (default) has rank 0 push into rank 1's "
             "buffer; 'read' has rank 0 pull from rank 1's buffer instead. rank 0 is "
             "the active initiator either way - only the data-source/sink roles invert.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(worker, args=(args.direction,), nprocs=2, join=True)
