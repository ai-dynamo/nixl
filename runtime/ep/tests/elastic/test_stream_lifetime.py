# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check eager output lifetimes across overlapping compute and communication.

Run on two GPUs in one NVLink domain:
    python tests/elastic/test_stream_lifetime.py
Repeat with NIXL_EP_AVOID_RECORD_STREAM=1 to cover event-owned asynchronous tensors.
"""

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist

from nixl.runtime.ep import ElasticBuffer, topk_idx_t


@torch.inference_mode()
def run(rank, port):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    assert torch.cuda.memory.get_allocator_backend() == "native"
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    tokens, hidden = 64, 512
    buffer = ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=tokens,
        hidden=hidden,
        num_topk=1,
        explicitly_destroy=True,
    )
    assert buffer.get_topology()[0] == "direct"
    compute = torch.cuda.current_stream()
    communication = buffer.get_comm_stream()
    assert compute != communication
    first_input = torch.ones((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    second_input = torch.full_like(first_input, rank + 11)
    routes = ((torch.arange(tokens, device="cuda") % 2) * 8).to(topk_idx_t).view(-1, 1)
    weights = torch.ones((tokens, 1), dtype=torch.float32, device="cuda")
    residual = torch.full_like(first_input, 3)
    initial_copy, observed = torch.empty_like(first_input), torch.empty_like(first_input)

    def dispatch(data, ready=None):
        return buffer.dispatch(
            data,
            routes,
            num_experts=16,
            topk_weights=weights,
            num_sms=4,
            previous_event=ready,
            async_with_compute_stream=ready is not None,
            allocate_on_comm_stream=True,
            do_cpu_sync=False,
        )

    def combine(data, handle, ready=None):
        return buffer.combine(
            data,
            handle,
            previous_event=ready,
            async_with_compute_stream=ready is not None,
            allocate_on_comm_stream=True,
        )

    # Compile both EP kernels and initialize arithmetic/events before the overlap window.
    for data in (first_input, second_input):
        received, indices, received_weights, handle, dispatch_event = dispatch(data)
        output, combine_event = combine(received, handle)
        torch.add(output, residual, out=observed)
        torch.cuda._sleep(1)
        for stream in (compute, communication):
            torch.cuda.Event(enable_timing=True).record(stream)
        torch.cuda.synchronize()
        assert torch.equal(output, data)
        del received, indices, received_weights, handle
        del dispatch_event, output, combine_event

    def check_output(kind):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()
        received, _indices, _weights, handle, _dispatch_event = dispatch(first_input)
        if kind == "combine":
            output, _combine_event = combine(received, handle)
        else:
            output, received = received, None
        # Each rank receives 64 compact rows; extra capacity is not initialized data.
        initial_copy.copy_(output[:tokens])
        ready = buffer.capture()

        # Hold up only the consumer; the next EP pair waits on the earlier ready event.
        torch.cuda._sleep(500_000_000)
        torch.add(output[:tokens], residual, out=observed)
        consumer_done = torch.cuda.Event(enable_timing=True)
        consumer_done.record(compute)
        del output
        # Retain asynchronous events through synchronization when they own tensors.
        second = dispatch(second_input, ready)
        next_output, _next_event = combine(second[0], second[3], ready)
        producer_done = torch.cuda.Event(enable_timing=True)
        producer_done.record(communication)
        torch.cuda.synchronize()

        assert (
            producer_done.elapsed_time(consumer_done) > 0
        ), "Test window did not overlap"
        assert torch.equal(initial_copy, first_input), f"{kind}: initial EP output"
        assert torch.equal(next_output, second_input), f"{kind}: next EP pair"
        assert torch.equal(
            observed, first_input + residual
        ), f"{kind}: premature output reuse"

    for kind in ("dispatch", "combine"):
        for attempt in range(3):
            check_output(kind)
            if rank == 0:
                print(f"PASS: {kind} output lifetime, attempt {attempt + 1}", flush=True)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(compute)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured = dispatch(first_input)
        graph_output, _ = combine(captured[0], captured[3])
    compute.wait_stream(capture_stream)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_output, first_input), "Comm-allocated graph output"
    del captured, graph_output, graph
    if rank == 0:
        print(
            "PASS: graph capture/replay with communication-stream allocations",
            flush=True,
        )

    buffer.barrier(with_cpu_sync=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=29671)
    args = parser.parse_args()
    torch.multiprocessing.spawn(run, args=(args.port,), nprocs=2)
