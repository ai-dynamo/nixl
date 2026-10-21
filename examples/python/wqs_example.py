# SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write and read a complete WQS object without an application-side size query."""

import argparse
import ctypes
import time
import uuid

from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


def main():
    """Verify a WQS object round trip using the command-line settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/var/run/wds/wqs")
    parser.add_argument("--bytes", type=int, default=4096)
    parser.add_argument("--key", default=None)
    parser.add_argument("--pool-bytes", type=int, default=1 << 30)
    args = parser.parse_args()
    if not 0 < args.bytes < (1 << 31) - 1:
        parser.error("--bytes must be positive and smaller than INT_MAX")
    key = args.key or f"nixl-wqs-smoke/{uuid.uuid4()}"
    pattern = (bytes(range(251)) * ((args.bytes + 250) // 251))[: args.bytes]
    source = ctypes.create_string_buffer(pattern, args.bytes)
    target = ctypes.create_string_buffer(args.bytes)
    agent = nixl_agent(
        "wqs-smoke",
        nixl_agent_config(backends=[], enable_prog_thread=False),
    )
    agent.create_backend(
        "WQS",
        {"unix_path": args.socket, "staging_pool_bytes": str(args.pool_bytes)},
    )
    registrations = []
    try:
        dram = agent.register_memory(
            [
                (ctypes.addressof(source), args.bytes, 0, ""),
                (ctypes.addressof(target), args.bytes, 0, ""),
            ],
            "DRAM",
            backends=["WQS"],
        )
        registrations.append(dram)
        obj = agent.register_memory([(0, args.bytes, 1, key)], "OBJ", backends=["WQS"])
        registrations.append(obj)
        for operation, buffer in (("WRITE", source), ("READ", target)):
            local = agent.get_xfer_descs(
                [(ctypes.addressof(buffer), args.bytes, 0)], "DRAM"
            )
            handle = agent.initialize_xfer(
                operation, local, obj.trim(), "wqs-smoke", backends=["WQS"]
            )
            try:
                state = agent.transfer(handle)
                while state == "PROC":
                    time.sleep(0.001)
                    state = agent.check_xfer_state(handle)
                if state != "DONE":
                    raise RuntimeError(f"WQS {operation} failed: {state}")
            finally:
                agent.release_xfer_handle(handle)
        if target.raw != pattern:
            raise RuntimeError("WQS read data differs from the written bytes")
        logger.info(
            "PASS: key=%s, bytes=%d; no application queryMem call", key, args.bytes
        )
    finally:
        for registration in reversed(registrations):
            agent.deregister_memory(registration, backends=["WQS"])


if __name__ == "__main__":
    main()
