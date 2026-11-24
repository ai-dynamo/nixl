# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary-copy benchmark adapter for nixlEP."""

import hashlib
from collections import namedtuple
from pathlib import Path

import torch

from nixl.runtime import ep  # type: ignore[attr-defined]

WorkerReceive = namedtuple(
    "WorkerReceive",
    "recv_x recv_sf recv_topk_weights expert_ends expert_counts host_counts receipt",
)


def source_dispatch(buffer, x, ids, weights, scale, bucket, alignment):
    payload = x if scale is None else (x, scale)
    handle, _ = buffer.dispatch_send(
        payload,
        ids,
        weights,
        num_max_tokens_per_rank=bucket,
        do_expand=True,
        expert_alignment=alignment,
    )
    return handle


def worker_receive(buffer, bucket, alignment, do_cpu_sync=False):
    payload, _, weights, handle, _ = buffer.dispatch_recv(
        with_weights=True,
        num_max_tokens_per_rank=bucket,
        do_expand=True,
        expert_alignment=alignment,
        do_cpu_sync=do_cpu_sync,
    )
    x, sf = payload if isinstance(payload, tuple) else (payload, None)
    return WorkerReceive(
        x,
        sf,
        weights,
        handle.psum_num_recv_tokens_per_expert,
        handle.num_unaligned_recv_tokens_per_expert,
        handle.num_recv_tokens_per_expert_list,
        handle,
    )


def worker_reply(buffer, output, receipt):
    buffer.combine_send(output, receipt)


def source_collect(buffer, handle):
    return buffer.combine_recv(handle)[0]


class Arm:
    def __init__(self, name, config, overrides, args, group):
        self.layout = overrides.pop("receive_layout", "expert-major")
        if self.layout not in ("compact", "expert-major"):
            raise ValueError(f"unsupported receive layout: {self.layout}")
        self.alignment = overrides.pop(
            "expert_alignment", 128 if self.layout == "expert-major" else 1
        )
        if self.layout == "compact" and self.alignment != 1:
            raise ValueError("compact receive requires expert_alignment=1")
        if overrides:
            raise ValueError(f"unsupported nixlEP options: {sorted(overrides)}")
        if args.execution != "eager":
            raise ValueError("nixlEP requires --execution eager")
        if config["ffn_ranks_per_domain"] not in (0, config["ffn_ranks"]):
            raise ValueError("nixlEP requires a single NVLink domain")
        if config["num_sms"]:
            raise ValueError(
                "nixlEP directional operations choose their SM count automatically"
            )
        self.name = name
        self.config = config
        ffn = config["ffn_ranks"]
        self.role = "ffn" if group.rank() < ffn else "attention"
        self.role_rank = group.rank() if self.role == "ffn" else group.rank() - ffn
        self.buffer = ep.ElasticBuffer(
            group=group,
            role=ep.AFDRole.FFN if self.role == "ffn" else ep.AFDRole.ATTENTION,
            num_max_tokens_per_rank=config["capacity_tokens_per_rank"],
            hidden=config["hidden_size"],
            num_topk=config["top_k"],
            num_experts_per_rank=config["num_experts"] // ffn,
            use_fp8_dispatch=config["payload"] == "fp8",
            explicitly_destroy=True,
        )

    def prepare(self, n, x, ids, weights, scale, graph, counts):
        ids = ids.to(ep.topk_idx_t)
        payload = x if scale is None else (x, scale)
        self.result = None

        def compact(verify=False):
            if self.role == "attention":
                source_handle, _ = self.buffer.dispatch_send(
                    payload, ids, weights, num_max_tokens_per_rank=n
                )
                self.result, _, _ = self.buffer.combine_recv(source_handle)
            else:
                rows, local_ids, local_weights, worker_handle, _ = (
                    self.buffer.dispatch_recv(
                        with_weights=True, num_max_tokens_per_rank=n, do_cpu_sync=False
                    )
                )
                # IBM supplies a preweighted one per selected expert. Compact rows
                # coalesce selections; their order can change on every exchange.
                reply = (local_ids >= 0).sum(1, keepdim=True).to(torch.bfloat16)
                self.buffer.combine_send(
                    reply.expand(-1, self.config["hidden_size"]).contiguous(),
                    worker_handle,
                )
                if verify:
                    count = int(worker_handle.num_recv_tokens_device.item())
                    local_ids, local_weights = local_ids[:count], local_weights[:count]
                    valid = local_ids >= 0
                    actual = local_ids[valid].long().bincount(minlength=len(counts))
                    assert actual.tolist() == counts, (actual.tolist(), counts)
                    assert torch.all(
                        local_weights[valid] == 1 / self.config["top_k"]
                    ).item()

        expert = None

        def expanded(verify=False):
            nonlocal expert
            if self.role == "attention":
                handle = source_dispatch(
                    self.buffer, x, ids, weights, scale, n, self.alignment
                )
                self.result = source_collect(self.buffer, handle)
            else:
                received = worker_receive(self.buffer, n, self.alignment)
                if expert is None:
                    expert = torch.ones_like(received.recv_x, dtype=torch.bfloat16)
                worker_reply(self.buffer, expert, received.receipt)
                if verify:
                    assert received.expert_counts.tolist() == counts
                    for count, end in zip(
                        counts, received.expert_ends.tolist(), strict=True
                    ):
                        valid = received.recv_topk_weights[end - count : end]
                        assert torch.all(valid == 1 / self.config["top_k"]).item()

        exchange = expanded if self.layout == "expert-major" else compact
        self.exchange = exchange
        exchange(verify=True)
        exchange()
        torch.cuda.synchronize()
        torch.distributed.barrier()

    def check(self, top_k):
        if self.role == "attention":
            torch.testing.assert_close(
                self.result, torch.full_like(self.result, top_k), rtol=0, atol=0
            )

    def metadata(self):
        native = Path(ep._C.__file__)
        include = Path(ep.__file__).parent / "include"
        sources = sorted(include.rglob("*.cuh"))
        if not sources:
            raise RuntimeError(f"No installed nixlEP JIT headers under {include}")
        sha = hashlib.sha256()
        for path in sources:
            sha.update(path.relative_to(include).as_posix().encode() + b"\0")
            sha.update(path.read_bytes())
        return {
            "backend": "nixlep",
            "config": self.config,
            "native_source_sha256": sha.hexdigest(),
            "backend_binary": str(native),
            "receive_layout": self.layout,
            "reply_preparation": (
                "preweighted BF16 ones prepared outside timing"
                if self.layout == "expert-major"
                else "per-exchange count of local selections and BF16 row materialization inside timing"
            ),
            "registered_bytes": self.buffer.num_bytes,
            "transport_sms": self.buffer.get_topology()[1],
            "transport_mode": self.buffer.get_topology()[0],
            "expert_alignment": self.alignment,
            "host_receive_counts": False,
        }

    def release(self):
        self.exchange = self.result = None

    def close(self):
        self.buffer.destroy()
