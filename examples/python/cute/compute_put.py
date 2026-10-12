# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank compute -> NIXL PUT; a correctness example, not a KV-cache codec.

The bindings live in src/bindings/cute. Everything here, including the progress
loop, counter protocol and Gloo bootstrap, belongs to the application example.
"""

import argparse
import logging
import os
import time
import traceback
from datetime import timedelta

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import nixl_cute as device
import torch
import torch.distributed as dist
from cutlass.cute.runtime import from_dlpack

from nixl import nixl_agent, nixl_agent_config, nixl_thread_sync_t


@cute.jit
def complete(request, status, *, level: cutlass.Constexpr):
    """Example policy: wait on the GPU; never poll a terminal submission."""
    while status == device.IN_PROGRESS:
        status = device.get_xfer_status(request, level=level)
    return status


@cute.kernel
def compute_and_put(
    src: cute.Tensor,
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    requests: cute.Tensor,
    status: cute.Tensor,
    sequence: cutlass.Int64,
    threads: cutlass.Constexpr,
    level: cutlass.Constexpr,
    flags: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    count = cute.size(src) - 4
    for i in range(tid, count, threads):
        src[i + 1] = cutlass.Int32(i + sequence * 10007)
    cute.arch.fence_acq_rel_sys()
    cute.arch.sync_threads()  # All producers finish before the cooperative PUT.
    request = requests.iterator.toint() + tid * device.REQUEST_BYTES
    result = device.put(
        local,
        remote,
        count * 4,
        request,
        local_index=1,
        remote_index=1,
        local_offset=4,
        remote_offset=12,
        channel=2,
        flags=flags,
        level=level,
    )
    result = complete(request, result, level=level)
    status[tid] = result
    # Publish stores from every CUDA-IPC writer before sending readiness.
    cute.arch.fence_acq_rel_sys()
    cute.arch.sync_threads()
    if result == device.SUCCESS:
        result = device.atomic_add(
            7, remote, request, index=0, offset=8, channel=2, flags=flags, level=level
        )
        status[threads + tid] = complete(request, result, level=level)


@cute.kernel
def receive(
    ready: cute.Tensor,
    recv: cute.Tensor,
    observed: cute.Tensor,
    sequence: cutlass.Int64,
):
    while (
        cute.arch.load(ready.iterator + 1, cutlass.Int64, sem="acquire", scope="sys")
        < sequence * 7
    ):
        pass
    # Reading on the GPU after acquire also exercises payload visibility.
    for i in range(cute.size(observed)):
        observed[i] = recv[i + 3]


@cute.jit
def launch(
    src: cute.Tensor,
    ready: cute.Tensor,
    recv: cute.Tensor,
    observed: cute.Tensor,
    requests: cute.Tensor,
    status: cute.Tensor,
    local: cutlass.Uint64,
    remote: cutlass.Uint64,
    sequence: cutlass.Int64,
    rank: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    level: cutlass.Constexpr,
    flags: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    if cutlass.const_expr(rank == 0):
        compute_and_put(
            src, local, remote, requests, status, sequence, threads, level, flags
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)
    else:
        receive(ready, recv, observed, sequence).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("thread", "warp", "block"), default="block")
    parser.add_argument("--elements", type=int, default=1037)
    parser.add_argument("--defer", action="store_true")
    args = parser.parse_args()
    if args.elements < 1:
        parser.error("--elements must be positive")
    level, threads = {"thread": (0, 1), "warp": (1, 32), "block": (2, 128)}[args.level]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    if dist.get_world_size() != 2:
        raise ValueError("run with torchrun --standalone --nproc-per-node=2")
    rank, peer = dist.get_rank(), 1 - dist.get_rank()
    stream = torch.cuda.current_stream()
    src = torch.full((args.elements + 4,), -1, device="cuda", dtype=torch.int32)
    recv = torch.full_like(src, -1)
    ready = torch.zeros(2, device="cuda", dtype=torch.int64)
    observed = torch.empty(args.elements, device="cuda", dtype=torch.int32)
    requests = torch.zeros(
        (threads, device.REQUEST_BYTES), device="cuda", dtype=torch.uint8
    )
    status = torch.zeros(threads * 2, device="cuda", dtype=torch.int32)
    stream.synchronize()
    agent = nixl_agent(
        f"cute-{rank}",
        nixl_agent_config(
            backends=["UCX"],
            enable_prog_thread=True,
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
        ),
    )
    registration = agent.register_memory([src, recv, ready])
    coords = [
        (t.data_ptr(), t.numel() * t.element_size(), t.get_device(), agent.name)
        for t in (ready, recv)
    ]
    metadata = [None, None]
    dist.all_gather_object(metadata, (agent.get_agent_metadata(), coords))
    agent.add_remote_agent(metadata[peer][0])
    dist.barrier()
    agent.make_connection(f"cute-{peer}")
    agent.send_notif(f"cute-{peer}", b"connected")
    deadline = time.monotonic() + 60
    while b"connected" not in agent.get_new_notifs().get(f"cute-{peer}", []):
        if time.monotonic() > deadline:
            raise TimeoutError("NIXL connection handshake")
        time.sleep(0.001)
    dist.barrier()
    local = agent.prep_mem_view(agent.get_xfer_descs([ready, src]))
    remote = agent.prep_mem_view(
        agent.get_remote_descs(metadata[peer][1], mem_type="VRAM")
    )
    tensors = tuple(map(from_dlpack, (src, ready, recv, observed, requests, status)))
    cu_stream = cuda.CUstream(stream.cuda_stream)
    tail = (
        rank,
        threads,
        level,
        device.DEFER if args.defer else 0,
        cu_stream,
    )
    compiled = cute.compile(launch, *tensors, local, remote, cutlass.Int64(1), *tail)
    dist.barrier()
    for sequence in range(1, 4):
        # Constexpr launch parameters were consumed by cute.compile.
        compiled(*tensors, local, remote, cutlass.Int64(sequence), cu_stream)
        # Validation/reuse boundary only; no CPU involvement inside the kernel.
        stream.synchronize()
        assert status.count_nonzero().item() == 0, status.cpu().tolist()
        if rank == 1:
            expected = torch.arange(args.elements, device="cuda", dtype=torch.int32)
            torch.testing.assert_close(
                observed, expected + sequence * 10007, rtol=0, atol=0
            )
            assert ready.tolist() == [0, sequence * 7]
            assert recv[:3].tolist() == [-1, -1, -1] and recv[-1].item() == -1
        dist.barrier()  # Receiver has consumed this generation before reuse.
    agent.release_mem_view(remote)
    agent.release_mem_view(local)
    dist.barrier()  # All borrowed views gone before owners deregister memory.
    agent.remove_remote_agent(f"cute-{peer}")
    agent.deregister_memory(registration)
    dist.destroy_process_group()
    logging.info(
        "rank %d: compute + PUT + atomic, %s, defer=%s: PASS",
        rank,
        args.level,
        args.defer,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Do not deregister remotely borrowed buffers after an uncertain failure.
        traceback.print_exc()
        os._exit(1)
