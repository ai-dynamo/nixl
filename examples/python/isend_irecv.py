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
MPI-like isend/irecv semantics over NIXL.

Usage:
    # Terminal 1 (receiver):
    python isend_irecv.py --mode receiver --ip 127.0.0.1 --port 1234

    # Terminal 2 (sender):
    python isend_irecv.py --mode sender --ip 127.0.0.1 --port 1234
"""

import argparse
import enum
import pickle
import time
from typing import Dict, Optional

import torch

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


class RequestStatus(enum.Enum):
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    ERROR = "ERROR"


class RequestType(enum.Enum):
    SEND = "SEND"
    RECV = "RECV"


class Request:
    def __init__(
        self,
        request_type: RequestType,
        remote_agent: str,
        tag: str,
        seq: int,
        xfer_handle=None,
        reg_descs=None,
        local_descs=None,
    ):
        self.type = request_type
        self.remote_agent = remote_agent
        self.tag = tag
        self.seq = seq
        self.xfer_handle = xfer_handle
        self.reg_descs = reg_descs
        self.local_descs = local_descs
        self._completed = False


class CommunicationManager:
    """Provides async isend/irecv semantics over NIXL using notifications for coordination."""

    def __init__(self, agent: nixl_agent):
        self.agent = agent
        self._notif_buffer: Dict[str, list[bytes]] = {}
        self._send_seq_counters: Dict[tuple, int] = {}
        self._recv_seq_counters: Dict[tuple, int] = {}

    def _poll_notifications(self):
        notifs = self.agent.get_new_notifs()
        for agent_name, msgs in notifs.items():
            if agent_name not in self._notif_buffer:
                self._notif_buffer[agent_name] = []
            self._notif_buffer[agent_name].extend(msgs)

    def _find_notification(
        self, src_agent: str, msg_type: str, tag: str, seq: int
    ) -> Optional[tuple]:
        self._poll_notifications()
        if src_agent not in self._notif_buffer:
            return None
        for i, raw in enumerate(self._notif_buffer[src_agent]):
            msg = pickle.loads(raw)
            if msg[0] == msg_type and msg[1] == tag and msg[2] == seq:
                self._notif_buffer[src_agent].pop(i)
                return msg
        return None

    def isend(
        self, tensor: torch.Tensor, dst_agent: str, tag: Optional[str] = None
    ) -> Request:
        """Non-blocking send. Returns Request to track via progress()."""
        if tag is None:
            tag = "default"

        key = (dst_agent, tag)
        seq = self._send_seq_counters.get(key, 0)
        self._send_seq_counters[key] = seq + 1

        reg_descs = self.agent.register_memory([tensor])
        if not reg_descs:
            raise RuntimeError(f"isend: failed to register memory for {tensor.shape}")

        local_descs = reg_descs.trim()

        return Request(
            RequestType.SEND,
            dst_agent,
            tag,
            seq,
            reg_descs=reg_descs,
            local_descs=local_descs,
        )

    def irecv(
        self, tensor: torch.Tensor, src_agent: str, tag: Optional[str] = None
    ) -> Request:
        """Non-blocking receive into pre-allocated tensor. Returns Request to track via progress()."""
        if tag is None:
            tag = "default"

        key = (src_agent, tag)
        seq = self._recv_seq_counters.get(key, 0)
        self._recv_seq_counters[key] = seq + 1

        reg_descs = self.agent.register_memory([tensor])
        if not reg_descs:
            raise RuntimeError(f"irecv: failed to register memory for {tensor.shape}")

        local_descs = reg_descs.trim()
        partial_md = self.agent.get_partial_agent_metadata(
            reg_descs, inc_conn_info=False, backends=[]
        )
        desc_str = self.agent.get_serialized_descs(local_descs)

        msg = pickle.dumps(("READY", tag, seq, partial_md, desc_str))
        self.agent.send_notif(src_agent, msg)

        return Request(
            RequestType.RECV,
            src_agent,
            tag,
            seq,
            reg_descs=reg_descs,
            local_descs=local_descs,
        )

    def _setup_send_transfer(self, request: Request, msg: tuple) -> RequestStatus:
        # msg = ("READY", tag, seq, partial_md, desc_str)
        _, _, _, partial_md, desc_str = msg
        remote_descs = self.agent.deserialize_descs(desc_str)

        try:
            added_agent_name = self.agent.add_remote_agent(partial_md)
            if isinstance(added_agent_name, bytes):
                added_agent_name = added_agent_name.decode("utf-8")
        except Exception as e:
            logger.error(f"add_remote_agent failed: {e}")
            return RequestStatus.ERROR

        if added_agent_name != request.remote_agent:
            logger.error(
                f"agent mismatch: expected {request.remote_agent}, got {added_agent_name}"
            )
            return RequestStatus.ERROR

        if not self.agent.check_remote_metadata(request.remote_agent, remote_descs):
            logger.error(f"check_remote_metadata failed for {request.remote_agent}")
            return RequestStatus.ERROR

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", request.local_descs, remote_descs, request.remote_agent
        )

        if not xfer_handle:
            logger.error(f"initialize_xfer failed for {request.remote_agent}")
            return RequestStatus.ERROR

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            logger.error(f"transfer initiation failed for {request.remote_agent}")
            return RequestStatus.ERROR

        request.xfer_handle = xfer_handle
        return RequestStatus.IN_PROGRESS

    def progress(self, request: Request) -> RequestStatus:
        """Call repeatedly until DONE or ERROR."""
        if request._completed:
            return RequestStatus.DONE

        if request.type == RequestType.SEND:
            # Wait for READY, then WRITE, then send COMPLETE
            if request.xfer_handle is None:
                msg = self._find_notification(
                    request.remote_agent, "READY", request.tag, request.seq
                )
                if msg is None:
                    return RequestStatus.IN_PROGRESS
                status = self._setup_send_transfer(request, msg)
                if status == RequestStatus.ERROR:
                    return RequestStatus.ERROR

            state = self.agent.check_xfer_state(request.xfer_handle)
            if state == "DONE":
                if request.reg_descs is not None:
                    self.agent.deregister_memory(request.reg_descs)
                msg = pickle.dumps(("COMPLETE", request.tag, request.seq))
                self.agent.send_notif(request.remote_agent, msg)
                request._completed = True
                return RequestStatus.DONE
            elif state == "ERR":
                logger.error(f"transfer failed for {request.remote_agent}")
                return RequestStatus.ERROR
            return RequestStatus.IN_PROGRESS

        elif request.type == RequestType.RECV:
            # Wait for COMPLETE
            if self._find_notification(
                request.remote_agent, "COMPLETE", request.tag, request.seq
            ):
                if request.reg_descs is not None:
                    self.agent.deregister_memory(request.reg_descs)
                request._completed = True
                return RequestStatus.DONE
            return RequestStatus.IN_PROGRESS

        logger.error(f"unknown request type: {request.type}")
        return RequestStatus.ERROR

    def batch_progress(self, requests: list[Request]) -> Dict[Request, RequestStatus]:
        return {req: self.progress(req) for req in requests}


def progress_until_done(requests, comm, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        statuses = comm.batch_progress(requests)
        if RequestStatus.ERROR in statuses.values():
            raise RuntimeError("Request failed")
        if all(s == RequestStatus.DONE for s in statuses.values()):
            return
        time.sleep(0.001)
    raise TimeoutError("Transfers did not complete in time")


def make_test_tensor(size, index, from_sender):
    """Create tensor with verifiable pattern: sender uses positive indices, receiver negative."""
    value = float(index) if from_sender else float(-(index + 1))
    return torch.full((size,), value, dtype=torch.float32, device="cuda")


def verify_tensors(recv_tensors, from_sender):
    """Verify received tensors match expected pattern. Returns True if all match."""
    for i, tensor in enumerate(recv_tensors):
        expected = make_test_tensor(tensor.shape[0], i, from_sender)
        if not torch.allclose(tensor, expected):
            logger.error(
                f"Data mismatch at index {i}: expected {expected[0].item()}, got {tensor[0].item()}"
            )
            return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="NIXL isend/irecv Example")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["sender", "receiver"]
    )
    parser.add_argument(
        "--ip", type=str, required=True, help="IP address to connect/listen"
    )
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument(
        "--bidirectional", action="store_true", help="Both sides send and receive"
    )
    parser.add_argument("--tensor-size", type=int, default=1024)
    parser.add_argument(
        "--num-transfers", type=int, default=10, help="Number of tensors to transfer"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NIXL isend/irecv Example")
    logger.info("=" * 60)

    listen_port = args.port if args.mode == "receiver" else 0
    config = nixl_agent_config(True, True, listen_port)
    agent = nixl_agent(args.mode, config)
    comm = CommunicationManager(agent)
    peer_name = "sender" if args.mode == "receiver" else "receiver"

    logger.info(f"Running as {args.mode}, peer is {peer_name}")

    # Metadata exchange
    if args.mode == "receiver":
        logger.info(f"Listening on {args.ip}:{args.port}, waiting for sender...")
        while not agent.check_remote_metadata(peer_name):
            time.sleep(0.01)
        logger.info("Sender connected")
    else:
        logger.info(f"Connecting to receiver at {args.ip}:{args.port}...")
        agent.fetch_remote_metadata(peer_name, args.ip, args.port)
        agent.send_local_metadata(args.ip, args.port)
        while not agent.check_remote_metadata(peer_name):
            time.sleep(0.01)
        logger.info("Connected to receiver")

    # Transfer
    is_sender = args.mode == "sender"
    n = args.num_transfers

    send_tensors = [make_test_tensor(args.tensor_size, i, is_sender) for i in range(n)]
    recv_tensors = [
        torch.zeros(args.tensor_size, dtype=torch.float32, device="cuda")
        for _ in range(n)
    ]

    if args.bidirectional:
        logger.info(f"Bidirectional: {n} transfers each direction")
        requests = []
        for i in range(n):
            requests.append(comm.isend(send_tensors[i], peer_name))
            requests.append(comm.irecv(recv_tensors[i], peer_name))
        progress_until_done(requests, comm)
    else:
        logger.info(f"Transferring {n} tensor(s) of size {args.tensor_size}...")
        if is_sender:
            progress_until_done([comm.isend(t, peer_name) for t in send_tensors], comm)
        else:
            progress_until_done([comm.irecv(t, peer_name) for t in recv_tensors], comm)

    # Verify received data (receiver in unidirectional, both in bidirectional)
    if args.bidirectional or not is_sender:
        if verify_tensors(recv_tensors, from_sender=not is_sender):
            logger.info(f"SUCCESS: {len(recv_tensors)} tensor(s) verified correctly")
        else:
            logger.error("FAILED: Data verification failed")
            exit(1)

    # Cleanup
    if args.mode == "sender":
        agent.remove_remote_agent(peer_name)
        agent.invalidate_local_metadata(args.ip, args.port)

    logger.info("=" * 60)
    logger.info("Example Complete")
    logger.info("=" * 60)
