#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal single-GPU asymmetric compression benchmark.

Assumptions: etcd is running, descriptors are fixed at the service's 64 MiB
chunk size, data is intentionally compressible, and requests run sequentially.
VRAM/DRAM use UCX; FILE uses GDS_MT and NIXL_GDS_TEST_DIR. Future versions can
add configurable sizes, concurrency, random payloads, and multi-node execution.
"""

import argparse
import os
import tempfile
import time

import torch

import nixl_cu13._bindings as bindings
import nixl_cu13._service_bindings as svc


ITERATION_COUNTS = {"vram": 20, "dram": 20, "file": 5}
DESCRIPTOR_BYTES = {"vram": 8 << 30, "dram": 8 << 30, "file": 1 << 30}
BUFFER_DTYPE = torch.int32
TIMEOUT_SECONDS = 60

# Returns the configured agent, backend handle, and compression config.
def make_agent(name, backend_name, disable_callbacks=False):
    mode = svc.nixlMarshalCompressConfig(svc.nixl_marshal_compress_algo_t.ANS)
    config = svc.nixlServiceAgentConfig()
    config.mode = mode
    config.useProgThread = True
    config.captureTelemetry = True
    config.disableServiceNotifCallbacks = disable_callbacks
    agent = svc.nixlServiceAgent(name, config)
    return agent, agent.createBackend(backend_name, {}), mode


def wait_for_transfer(agent, handle):
    status = agent.postXferReq(handle)
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while status == bindings.NIXL_IN_PROG:
        if time.monotonic() >= deadline:
            raise TimeoutError("transfer timed out")
        status = agent.getXferStatus(handle)
    if status != bindings.NIXL_SUCCESS:
        raise RuntimeError(f"transfer failed: {status}")

# Returns the transferred byte count reported by telemetry.
def run_transfer(agent, backend, remote_name, operation, phase, local_desc, remote_desc):
    options = (
        svc.nixlMarshalDirectOptArgs()
        if phase is None
        else svc.nixlMarshalCompressOptArgs()
    )
    handle = agent.createXferReq(
        operation, local_desc, remote_desc, remote_name, "", [backend],
        options, phase,
    )
    try:
        wait_for_transfer(agent, handle)
        return agent.getXferTelemetry(handle).totalBytes
    finally:
        agent.releaseXferReq(handle)

# Returns storage, type, address, device, metadata, and an optional file path.
def make_target(memory_type, device, descriptor_bytes):
    if memory_type == "vram":
        storage = torch.empty(descriptor_bytes, dtype=torch.uint8, device="cuda")
        mem_type, address, dev_id, metadata = bindings.VRAM_SEG, storage.data_ptr(), device, ""
        return storage, mem_type, address, dev_id, metadata, None
    if memory_type == "dram":
        storage = torch.empty(descriptor_bytes, dtype=torch.uint8)
        return storage, bindings.DRAM_SEG, storage.data_ptr(), 0, "", None

    directory = os.environ.get("NIXL_GDS_TEST_DIR")
    if not directory:
        raise RuntimeError("FILE mode requires NIXL_GDS_TEST_DIR")
    fd, path = tempfile.mkstemp(prefix="nixl_asymmetric_", dir=directory)
    # TODO: Clean up the descriptor and path if ftruncate fails.
    os.ftruncate(fd, descriptor_bytes)
    os.close(fd)
    return None, bindings.FILE_SEG, 0, 0, f"rw,direct:{path}", path

# Returns a transfer list containing one descriptor.
def descriptor(mem_type, address, device, descriptor_bytes):
    return bindings.nixlXferDList(mem_type, [(address, descriptor_bytes, device)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory-type", choices=("vram", "dram", "file"), default="vram",
    )
    args = parser.parse_args()

    is_file = args.memory_type == "file"
    iteration_count = ITERATION_COUNTS[args.memory_type]
    descriptor_bytes = DESCRIPTOR_BYTES[args.memory_type]
    device = torch.cuda.current_device()
    source = torch.empty(descriptor_bytes // 4, dtype=BUFFER_DTYPE, device="cuda")
    for value in range(5):
        source[value::5] = (value + 1) % 5
    torch.cuda.synchronize()

    target_storage, target_type, target_address, target_device, metadata, path = (
        make_target(args.memory_type, device, descriptor_bytes)
    )
    try:
        backend_name = "GDS_MT" if is_file else "UCX"
        initiator, initiator_backend, mode = make_agent(
            "initiator", backend_name, disable_callbacks=is_file,
        )
        if is_file:
            target_agent, target_backend, remote_name = (
                initiator, initiator_backend, "initiator",
            )
        else:
            target_agent, target_backend, _ = make_agent("target", "UCX")
            remote_name = "target"

        source_reg = bindings.nixlRegDList(
            bindings.VRAM_SEG,
            [(source.data_ptr(), descriptor_bytes, device, "")],
        )
        service_bytes = svc.recommendServiceMemSize(
            mode, maxConcurrentTransfers=1,
        )
        service_memory = torch.empty(service_bytes, dtype=torch.uint8, device="cuda")
        service_reg = bindings.nixlRegDList(
            bindings.VRAM_SEG,
            [(service_memory.data_ptr(), service_bytes, device, "")],
        )
        target_reg = bindings.nixlRegDList(
            target_type,
            [(target_address, descriptor_bytes, target_device, metadata)],
        )
        initiator.registerServiceMem(service_reg, [initiator_backend])
        target_agent.registerMem(target_reg, [target_backend])

        if not is_file:
            # TODO-Eyal: change to exceptions
            assert initiator.loadRemoteMD(target_agent.getLocalMD()) == b"target"
            assert target_agent.loadRemoteMD(initiator.getLocalMD()) == b"initiator"

        compressed_bytes = 0
        write_start = time.perf_counter()
        for _ in range(iteration_count):
            compressed_bytes += run_transfer(
                initiator, initiator_backend, remote_name, bindings.NIXL_WRITE,
                svc.nixl_marshal_phase_t.PRE_TRANSFER,
                descriptor(bindings.VRAM_SEG, source.data_ptr(), device, descriptor_bytes),
                descriptor(target_type, target_address, target_device, descriptor_bytes),
            )
        write_seconds = time.perf_counter() - write_start

        source.zero_()
        torch.cuda.synchronize()
        read_start = time.perf_counter()
        for _ in range(iteration_count):
            run_transfer(
                initiator, initiator_backend, remote_name, bindings.NIXL_READ,
                svc.nixl_marshal_phase_t.POST_TRANSFER,
                descriptor(bindings.VRAM_SEG, source.data_ptr(), device, descriptor_bytes),
                descriptor(target_type, target_address, target_device, descriptor_bytes),
            )
        torch.cuda.synchronize()
        read_seconds = time.perf_counter() - read_start

        for value in range(5):
            # TODO-Eyal: change to exception
            assert torch.all(source[value::5] == (value + 1) % 5).item()

        initiator.registerMem(source_reg, [initiator_backend])
        direct_write_start = time.perf_counter()
        for _ in range(iteration_count):
            run_transfer(
                initiator, initiator_backend, remote_name, bindings.NIXL_WRITE, None,
                descriptor(bindings.VRAM_SEG, source.data_ptr(), device, descriptor_bytes),
                descriptor(target_type, target_address, target_device, descriptor_bytes),
            )
        direct_write_seconds = time.perf_counter() - direct_write_start

        source.zero_()
        torch.cuda.synchronize()
        direct_read_start = time.perf_counter()
        for _ in range(iteration_count):
            run_transfer(
                initiator, initiator_backend, remote_name, bindings.NIXL_READ, None,
                descriptor(bindings.VRAM_SEG, source.data_ptr(), device, descriptor_bytes),
                descriptor(target_type, target_address, target_device, descriptor_bytes),
            )
        torch.cuda.synchronize()
        direct_read_seconds = time.perf_counter() - direct_read_start
        for value in range(5):
            # TODO-Eyal: change to exception
            assert torch.all(source[value::5] == (value + 1) % 5).item()

        print(
            f"\n============ Results ============\n"
            f"compression writes={write_seconds:.3f}s "
            f"({iteration_count * descriptor_bytes / write_seconds / 1e9:.3f} GB/s)\n"
            f"compression reads={read_seconds:.3f}s "
            f"({iteration_count * descriptor_bytes / read_seconds / 1e9:.3f} GB/s)\n"
            f"compression_ratio={iteration_count * descriptor_bytes / compressed_bytes:.3f}x\n"
            f"direct writes={direct_write_seconds:.3f}s "
            f"({iteration_count * descriptor_bytes / direct_write_seconds / 1e9:.3f} GB/s)\n"
            f"direct reads={direct_read_seconds:.3f}s "
            f"({iteration_count * descriptor_bytes / direct_read_seconds / 1e9:.3f} GB/s)\n"
            f"validation=PASS\n"
            f"=================================\n\n"
            f"========== Test Configuration ===========\n"
            f"memory_type={args.memory_type}, iterations={iteration_count}\n"
            f"descriptor_size={descriptor_bytes // (1024 * 1024 * 1024)} GiB\n"
            f"service_mem={service_bytes / (1024 * 1024):.3f} MiB\n"
            f"========================================="
        )
        del target_storage
    finally:
        if path:
            os.unlink(path)


if __name__ == "__main__":
    main()
