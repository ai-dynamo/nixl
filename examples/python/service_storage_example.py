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

"""Write compressed objects to one target, then read and validate them."""

import argparse
import os
import tempfile
import time

import torch

from nixl_cu13 import _service_api as svc


TARGET_BYTES = 8 << 30
DESCRIPTOR_COUNT = 10
DESCRIPTOR_BYTES = 512 << 20
PATTERN_BYTES = 256


def make_agent(name, backend_name="UCX", disable_callbacks=False):
    mode = svc.nixlMarshalCompressConfig(svc.nixl_marshal_compress_algo_t.ANS)
    config = svc.nixl_service_agent_config(
        mode=mode,
        capture_telemetry=True,
        backends=[backend_name],
        disable_service_notif_callbacks=disable_callbacks,
    )
    return svc.nixl_service_agent(name, nixl_conf=config), mode


def run_transfer(agent, target_name, operation, phase, local_descs, remote_descs):
    options = svc.nixlMarshalCompressOptArgs()
    handle = agent.initialize_xfer(
        operation,
        local_descs,
        remote_descs,
        remote_agent=target_name,
        marshal_opt_args=options,
        phase=phase,
    )
    try:
        status = agent.transfer(handle)
        while status == "PROC":
            status = agent.check_xfer_state(handle)
        if status != "DONE":
            raise RuntimeError(f"transfer failed: {status}")
        return agent.get_xfer_telemetry(handle).totalBytes
    finally:
        handle.release()


def pattern(device, descriptor_index):
    return (
        torch.arange(PATTERN_BYTES, dtype=torch.int16, device=device)
        + descriptor_index
    ).remainder(PATTERN_BYTES).to(torch.uint8)


def fill_source(source, descriptor_index):
    source.view(-1, PATTERN_BYTES).copy_(pattern(source.device, descriptor_index))
    torch.cuda.synchronize()


def validate_source(source, descriptor_index):
    expected = pattern(source.device, descriptor_index)
    if not bool((source.view(-1, PATTERN_BYTES) == expected).all().item()):
        raise RuntimeError(f"descriptor {descriptor_index} validation failed")


def main():
    parser = argparse.ArgumentParser(description="Storage nixlService compression example")
    parser.add_argument(
        "--memory-type", choices=("vram", "dram", "file"), default="dram"
    )
    args = parser.parse_args()

    is_file = args.memory_type == "file"
    backend_name = "GDS_MT" if is_file else "UCX"
    target_path = None
    target_reg = None

    try:
        device = torch.cuda.current_device()
        initiator, mode = make_agent(
            "initiator", backend_name, disable_callbacks=is_file
        )
        if is_file:
            target_agent = initiator
            target_name = initiator.name
        else:
            target_agent, _ = make_agent("target")
            target_name = "target"

        source = torch.empty(DESCRIPTOR_BYTES, dtype=torch.uint8, device="cuda")
        if args.memory_type == "vram":
            target = torch.empty(TARGET_BYTES, dtype=torch.uint8, device="cuda")
            target_type, target_device = "VRAM", device
        elif args.memory_type == "dram":
            target = torch.empty(TARGET_BYTES, dtype=torch.uint8)
            target_type, target_device = "DRAM", 0
        else:
            target_directory = os.environ.get("NIXL_GDS_TEST_DIR")
            if not target_directory:
                raise RuntimeError("FILE mode requires NIXL_GDS_TEST_DIR")
            target_fd, target_path = tempfile.mkstemp(
                prefix="nixl_storage_", dir=target_directory
            )
            try:
                os.ftruncate(target_fd, TARGET_BYTES)
            finally:
                os.close(target_fd)
            target_type, target_device = "FILE", 0

        service_bytes = svc.recommendServiceMemSize(
            mode, maxConcurrentTransfers=1
        )
        service_memory = torch.empty(
            service_bytes, dtype=torch.uint8, device="cuda"
        )
        initiator.register_service_memory([service_memory])
        if is_file:
            target_address = 0
            target_reg = target_agent.register_memory(
                [(0, TARGET_BYTES, 0, f"rw,direct:{target_path}")],
                mem_type="FILE",
                backends=[backend_name],
            )
        else:
            target_address = target.data_ptr()
            target_agent.register_memory([target])
            if (
                initiator.add_remote_agent(target_agent.get_agent_metadata())
                != b"target"
            ):
                raise RuntimeError("failed to load target metadata")
            if (
                target_agent.add_remote_agent(initiator.get_agent_metadata())
                != b"initiator"
            ):
                raise RuntimeError("failed to load initiator metadata")

        write_seconds = 0.0
        used_target_bytes = 0
        for index in range(DESCRIPTOR_COUNT):
            fill_source(source, index)
            local_descs = initiator.get_xfer_descs(
                [(source.data_ptr(), DESCRIPTOR_BYTES, device)], mem_type="VRAM"
            )
            remote_descs = initiator.get_xfer_descs(
                [(
                    target_address + index * DESCRIPTOR_BYTES,
                    DESCRIPTOR_BYTES,
                    target_device,
                )],
                mem_type=target_type,
            )
            start = time.perf_counter()
            used_target_bytes += run_transfer(
                initiator,
                target_name,
                "WRITE",
                svc.nixl_marshal_phase_t.PRE_TRANSFER,
                local_descs,
                remote_descs,
            )
            write_seconds += time.perf_counter() - start

        read_seconds = 0.0
        read_target_bytes = 0
        for index in range(DESCRIPTOR_COUNT):
            source.zero_()
            torch.cuda.synchronize()
            local_descs = initiator.get_xfer_descs(
                [(source.data_ptr(), DESCRIPTOR_BYTES, device)], mem_type="VRAM"
            )
            remote_descs = initiator.get_xfer_descs(
                [(
                    target_address + index * DESCRIPTOR_BYTES,
                    DESCRIPTOR_BYTES,
                    target_device,
                )],
                mem_type=target_type,
            )
            start = time.perf_counter()
            read_target_bytes += run_transfer(
                initiator,
                target_name,
                "READ",
                svc.nixl_marshal_phase_t.POST_TRANSFER,
                local_descs,
                remote_descs,
            )
            read_seconds += time.perf_counter() - start
            validate_source(source, index)

        if read_target_bytes != used_target_bytes:
            raise RuntimeError("read and write target-byte telemetry differs")

        print("\n============== Storage Results ==============")
        print(
            f"write latency: {write_seconds:.3f}s total, "
            f"{write_seconds * 1e3 / DESCRIPTOR_COUNT:.3f}ms per descriptor"
        )
        print(
            f"read latency:  {read_seconds:.3f}s total, "
            f"{read_seconds * 1e3 / DESCRIPTOR_COUNT:.3f}ms per descriptor"
        )
        print(
            f"used target memory: {used_target_bytes / (1 << 30):.3f} GiB "
            f"of {TARGET_BYTES / (1 << 30):.0f} GiB"
        )
        print(f"target memory type: {args.memory_type}")
        print("validation: PASS")
        print("============================================")
    finally:
        if target_path is not None:
            try:
                if target_reg is not None:
                    target_agent.deregister_memory(
                        target_reg, backends=[backend_name]
                    )
            finally:
                os.unlink(target_path)


if __name__ == "__main__":
    main()
