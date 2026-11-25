# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare serialized and caller-scheduled direct AFD with independent expert work."""

import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from nixl.runtime import ep  # type: ignore[attr-defined]
from nixl.runtime.ep.utils.envs import init_dist


def run(rank):
    _, _, group = init_dist(rank, 3)
    attention = rank == 1
    capacity, hidden = 128, 7168

    def make_buffer(slots):
        return ep.ElasticBuffer(
            group=group,
            role=ep.AFDRole.ATTENTION if attention else ep.AFDRole.FFN,
            num_slots=slots,
            num_max_tokens_per_rank=capacity,
            hidden=hidden,
            num_topk=2,
            num_experts_per_rank=1,
            explicitly_destroy=True,
        )

    buf = make_buffer(2)
    expert = torch.cuda.Stream()
    weight = torch.eye(hidden, dtype=torch.bfloat16)
    ids = torch.arange(2, dtype=ep.topk_idx_t).expand(capacity, 2).contiguous()
    batches = [torch.full((capacity, hidden), v, dtype=torch.bfloat16) for v in (1, 7)]

    def pair(pipeline, empty, delayed):
        pending = []
        outputs = []

        def finish(item):
            if attention:
                output, event = item
                if event.event is not None:
                    event.current_stream_wait()
                outputs.append(output)
            else:
                reply, handle, done = item
                torch.cuda.current_stream().wait_event(done)
                buf.combine_send(reply, handle)
                reply.record_stream(torch.cuda.current_stream())

        for slot, batch in enumerate(batches):
            exchange_slot = slot if buf.num_slots > 1 else 0
            if attention:
                rows = batch[:0] if empty == slot else batch
                handle, _ = buf.dispatch_send(
                    rows, ids[: len(rows)], slot=exchange_slot
                )
                output, _, event = buf.combine_recv(handle)
                item = output, event
            else:
                rows, _, _, handle, _ = buf.dispatch_recv(
                    slot=exchange_slot, do_cpu_sync=False
                )
                expert.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(expert):
                    if rank == 0 and delayed == slot:
                        torch.cuda._sleep(100000)
                    reply = (rows @ weight) @ weight
                    rows.record_stream(expert)
                    item = reply, handle, expert.record_event()
            if pipeline:
                pending.append(item)
            else:
                finish(item)
        for item in pending:
            finish(item)
        return outputs

    # Exercise ownership errors before any unmatched peer operation reaches the GPU.
    if attention:
        handle, _ = buf.dispatch_send(batches[0], ids, slot=1)
        try:
            buf.dispatch_send(batches[1], ids, slot=1)
        except ValueError as exc:
            assert "pending" in str(exc)
        else:
            raise AssertionError("dispatch accepted a busy slot")
        _, _, completion = buf.combine_recv(handle)
        completion.current_stream_wait()
        try:
            buf.combine_recv(handle)
        except ValueError as exc:
            assert "consumed" in str(exc)
        else:
            raise AssertionError("combine accepted a consumed handle")
    else:
        rows, _, _, handle, _ = buf.dispatch_recv(slot=1, do_cpu_sync=False)
        buf.combine_send(rows, handle)

    for empty, delayed in ((0, 1), (1, 0), (-1, -1)):
        reference = pair(False, empty, delayed)
        outputs = pair(True, empty, delayed)
        torch.cuda.synchronize()
        for slot, (result, serial) in enumerate(zip(outputs, reference)):
            torch.testing.assert_close(result, serial, rtol=0, atol=0)
            expected = batches[slot][:0] if slot == empty else 2 * batches[slot]
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    buf.destroy()
    for pipeline in (False, True):
        buf = make_buffer(2 if pipeline else 1)
        pair(pipeline, -1, -1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            captured = pair(pipeline, -1, -1)
        for replay in (False, True):
            times = []
            for step in range(6):
                batches[0].fill_(step + 2)
                batches[1].fill_(step + 11)
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                dist.barrier()
                start.record()
                if replay:
                    graph.replay()
                    outputs = captured
                else:
                    outputs = pair(pipeline, -1, -1)
                end.record()
                end.synchronize()
                for result, batch in zip(outputs, batches):
                    torch.testing.assert_close(result, 2 * batch, rtol=0, atol=0)
                elapsed = torch.tensor(start.elapsed_time(end) * 1000)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                times.append(elapsed.item())
            if attention:
                print(
                    f"pipeline={pipeline} replay={replay} pair_us={statistics.median(times):.2f}",
                    flush=True,
                )
        trace_dir = os.getenv("AFD_SLOT_TRACE_DIR")
        if trace_dir:
            dist.barrier()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU,
                ]
            ) as profiler:
                graph.replay()
                torch.cuda.synchronize()
            path = Path(trace_dir) / f"slots-{rank}-{int(pipeline)}.json"
            profiler.export_chrome_trace(str(path))
            kernels = [
                e
                for e in json.loads(path.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            ]
            gemms = [
                e
                for e in kernels
                if "gemm" in e["name"].lower()
                or "cutlass" in e["name"].lower()
                or e["name"].startswith("nvjet_")
            ]
            comms = [
                e
                for e in kernels
                if "dispatch" in e["name"].lower() or "combine" in e["name"].lower()
            ]
            windows = sorted(
                (max(g["ts"], c["ts"]), min(g["ts"] + g["dur"], c["ts"] + c["dur"]))
                for g in gemms
                for c in comms
            )
            overlap = 0.0
            end = 0.0
            for lo, hi in windows:
                overlap += max(0.0, hi - max(lo, end))
                if hi > lo:
                    end = max(end, hi)
            print(
                f"rank={rank} pipeline={pipeline} gemm_kernel_us={sum(g['dur'] for g in gemms):.2f} transport_kernel_us={sum(c['dur'] for c in comms):.2f} intersection_us={overlap:.2f}",
                flush=True,
            )
        buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(run, nprocs=3)
