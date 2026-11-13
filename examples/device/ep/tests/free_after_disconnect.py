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

from __future__ import annotations

import queue
import socket
import subprocess
import time
import traceback
from datetime import timedelta

import nixl_ep
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

GPU_IDS = [0, 1]
WORLD_SIZE = 2
NUM_TOKENS = 2048
HIDDEN = 4096
NUM_EXPERTS_PER_RANK = 8
TIMEOUT_SEC = 60


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gpu_free_mib(gpu_id: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_id),
        ],
        text=True,
    )
    return int(output.strip())


def _store_barrier(store: dist.TCPStore, key: str, count: int) -> None:
    store.add(key, 1)
    deadline = time.time() + TIMEOUT_SEC
    while int(store.get(key).decode()) < count:
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting at store barrier '{key}'")
        time.sleep(0.05)


def _wait_report(result_queue: mp.Queue) -> None:
    try:
        item = result_queue.get(timeout=TIMEOUT_SEC)
    except queue.Empty:
        raise RuntimeError("timed out waiting for a worker report") from None
    if item is not None:
        raise RuntimeError(item["error"])


def _worker(rank: int, port: int, result_queue: mp.Queue) -> None:
    try:
        torch.cuda.set_device(GPU_IDS[rank])

        store = dist.TCPStore(
            host_name="127.0.0.1",
            port=port,
            is_master=False,
            wait_for_workers=False,
            timeout=timedelta(seconds=TIMEOUT_SEC),
        )

        num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=NUM_TOKENS,
            hidden=HIDDEN,
            num_ranks=WORLD_SIZE,
            num_experts=NUM_EXPERTS_PER_RANK * WORLD_SIZE,
        )
        buffer = nixl_ep.Buffer(
            rank=rank,
            explicitly_destroy=True,
            tcp_store_group=store,
        )
        buffer.update_memory_buffers(
            num_ranks=WORLD_SIZE,
            num_experts_per_rank=NUM_EXPERTS_PER_RANK,
            num_rdma_bytes=num_rdma_bytes,
        )

        _store_barrier(store, "ready_to_connect", WORLD_SIZE)
        buffer.connect_ranks([1 - rank])
        torch.cuda.synchronize()
        result_queue.put(None)

        if rank == 0:
            store.wait(["rank1_killed"])
            buffer.disconnect_ranks([1])
            torch.cuda.synchronize()
            result_queue.put(None)

        # Stay alive (holding the buffer) until the parent kills the process.
        while True:
            time.sleep(1.0)
    except Exception:
        result_queue.put({"error": traceback.format_exc()})
        raise


def _kill(process: mp.Process) -> None:
    process.kill()
    process.join(timeout=TIMEOUT_SEC)
    if process.is_alive():
        raise RuntimeError(f"pid {process.pid} did not exit")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        raise SystemExit("This test requires at least 2 CUDA GPUs.")

    victim_gpu = GPU_IDS[1]
    free_before_connect = _gpu_free_mib(victim_gpu)

    ctx = mp.get_context("spawn")
    port = _pick_free_port()
    master_store = dist.TCPStore(
        host_name="127.0.0.1",
        port=port,
        is_master=True,
        wait_for_workers=False,
        timeout=timedelta(seconds=TIMEOUT_SEC),
    )
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_worker, args=(rank, port, result_queue))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()

    try:
        for _ in range(WORLD_SIZE):  # connected
            _wait_report(result_queue)

        _kill(processes[1])
        master_store.set("rank1_killed", "1")

        _wait_report(result_queue)  # disconnected

        deadline = time.time() + TIMEOUT_SEC
        while _gpu_free_mib(victim_gpu) < free_before_connect:
            assert time.time() < deadline, (
                f"GPU {victim_gpu} has {_gpu_free_mib(victim_gpu)} MiB free after "
                f"disconnect, expected {free_before_connect} MiB"
            )
            time.sleep(0.5)
    finally:
        for process in processes:
            if process.is_alive():
                _kill(process)


if __name__ == "__main__":
    main()
