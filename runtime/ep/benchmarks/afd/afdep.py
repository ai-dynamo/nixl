# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright afdEP-comm contributors (formerly afdEP).
"""afdEP benchmark arm, adapted from afdEP-comm benchmarks/comm.py at 575d2f9."""
import os
from dataclasses import asdict

import torch
import torch.distributed as dist
from afdep import Buffer, Config

COLLECT_VIEW = os.environ.get("AFDEP_BENCH_COLLECT_VIEW", "0") == "1"


class Arm:
    """One buffer and one captured exchange per token extent."""

    backend = "afdep"
    buffer_type = Buffer
    config_type = Config

    def __init__(self, name, common, overrides, args, group):
        self.name = name
        self.buffer = self.buffer_type(
            group=group, config=self.config_type(**(common | overrides))
        )
        self.config = self.buffer.config
        self.role, self.role_rank = self.buffer.role, self.buffer.role_rank
        self.graph = None
        self.result = None

    def prepare_routes(self, routes):
        return routes, {}

    def prepare(self, n, x, ids, weights, scale, graph, counts):
        buffer, expert = self.buffer, None
        routes, mapped = self.prepare_routes(ids.to(buffer.topk_idx_dtype))
        # A captured graph reads these IDs on every replay: keep them alive with it.
        self.routes = routes

        def step(verify=False):
            nonlocal expert
            if buffer.role == "ffn":
                received = buffer.worker_receive(bucket=n)
                if expert is None:
                    # Filled once outside capture/timing: every selected row
                    # contributes one, so each source receives exactly top_k.
                    expert = torch.ones(
                        received.recv_x.shape, dtype=torch.bfloat16, device="cuda"
                    )
                buffer.worker_reply(expert, received.receipt)
                if verify:
                    ends = received.recv_count_psum.cpu().tolist()
                    start = 0
                    for count, end in zip(counts, ends, strict=True):
                        assert end - start == count, (self.name, count, start, end)
                        valid_weights = received.recv_topk_weights[start:end]
                        assert torch.all(valid_weights == 1 / self.config.top_k).item()
                        a = self.config.expert_alignment
                        start = ((end + a - 1) // a) * a
            else:
                buffer.source_dispatch(x, routes, weights, scale, bucket=n, **mapped)
                self.result = buffer.source_collect(view=COLLECT_VIEW)

        step(verify=True)
        step()
        torch.cuda.synchronize()
        dist.barrier()
        self.graph = None
        if graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                step()
        self.exchange = self.graph.replay if self.graph is not None else step

    def check(self, top_k):
        if self.buffer.role == "attention":
            torch.testing.assert_close(
                self.result, torch.full_like(self.result, top_k), rtol=0, atol=0
            )

    def release(self):
        self.exchange = self.graph = self.result = self.routes = None

    def metadata(self):
        from afdep._build import source_digest

        return {
            "backend": self.backend,
            "config": asdict(self.config),
            "native_source_sha256": source_digest(),
            "backend_binary": str(self.buffer.transport._module.__file__),
            "receive_layout": "expert-major",
            "expert_alignment": self.config.expert_alignment,
            "reply_preparation": "preweighted BF16 ones prepared outside timing",
            "registered_bytes": self.buffer.num_buffer_bytes,
            "transport_sms": self.buffer.transport._resolved_num_sms,
            "host_receive_counts": False,
            "fp8_combine": self.buffer.fp8_combine,
            "fp8_combine_min_bucket": self.buffer.fp8_combine_min_bucket,
        }

    def close(self):
        self.buffer.close()
