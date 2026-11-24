# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time one AFD backend per torchrun process group; screen.py pairs fresh runs.

Adapted from IBM afdEP-comm benchmarks/comm.py at 575d2f9.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from importlib import import_module
from pathlib import Path

import torch
import torch.distributed as dist

METRIC = "per block: max Attention-rank CUDA-event us per exchange"
SCOPE = "role round trip; initialization, correctness checks and capture excluded; graph replay included"


BACKENDS = ("nixlep", "afdep", "fastafd")


def routes(args, source, n):
    """Deterministic strided/skew patterns or seeded random top-k routes."""
    local_experts = args.experts // args.ffn_ranks
    route_experts = local_experts if args.routing == "skew" else args.experts
    stride = 5 if math.gcd(5, route_experts) == 1 else 1
    if args.routing == "random":
        generator = torch.Generator(device="cuda").manual_seed(
            args.routing_seed + source
        )
        return (
            torch.rand((n, args.experts), generator=generator, device="cuda")
            .topk(args.top_k, dim=1)
            .indices
        )
    return (
        torch.arange(n, device="cuda")[:, None] * 3
        + torch.arange(args.top_k, device="cuda")[None, :] * stride
        + source
    ) % route_experts


def logical_work(args, n, ids, weights, role, role_rank):
    """Include every source's ordered routes and weights in the cross-arm comparison."""
    local = None
    if role == "attention":
        cpu_ids = ids.long().cpu()
        owners = cpu_ids // (args.experts // args.ffn_ranks)
        owners = owners.sort(dim=1).values
        local = {
            "rank": dist.get_rank(),
            "source": role_rank,
            "routes_sha256": hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(
                weights.cpu().numpy().tobytes()
            ).hexdigest(),
            "per_expert": cpu_ids.flatten().bincount(minlength=args.experts).tolist(),
            "compact_rows": int(n + (owners[:, 1:] != owners[:, :-1]).sum()),
        }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    sources = sorted((s for s in gathered if s is not None), key=lambda s: s["source"])
    per_expert = list(
        map(sum, zip(*(s.pop("per_expert") for s in sources), strict=True))
    )
    return {
        "attention_sources": sources,
        "source_tokens": n * len(sources),
        "useful_expert_rows": sum(per_expert),
        "compact_rows": sum(s["compact_rows"] for s in sources),
        "per_expert": per_expert,
        "payload": args.payload,
        "hidden": args.hidden,
        "top_k": args.top_k,
        "weight_per_selection": 1 / args.top_k,
        "preweighted_reply_per_selection": 1,
    }


def timed(exchange, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        exchange()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--attention-ranks", type=int, default=2)
    parser.add_argument(
        "--ffn-ranks-per-domain",
        type=int,
        default=0,
        help="FFN ranks are the first N of every NVLink domain (0: ranks [0, F))",
    )
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 128, 2048])
    parser.add_argument("--execution", choices=["eager", "graph"], default="eager")
    parser.add_argument("--hybrid", choices=["on", "off"], default="on")
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument(
        "--routing", choices=["strided", "random", "skew"], default="random"
    )
    parser.add_argument("--routing-seed", type=int, default=49001)
    parser.add_argument(
        "--arms",
        type=json.loads,
        default={"nixlep": {}},
        help='JSON {"name": {Config overrides}} for one backend process',
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens) < 1 or max(args.tokens) > args.capacity:
        parser.error("tokens must be in [1,capacity]")
    if min(args.iterations, args.blocks) < 1 or args.warmup < 0:
        parser.error("iterations and blocks must be positive; warmup nonnegative")
    if not isinstance(args.arms, dict) or len(args.arms) != 1:
        parser.error(
            "select one arm per process, or use benchmarks.afd.screen to compare arms"
        )
    if (
        min(args.ffn_ranks, args.attention_ranks, args.experts, args.top_k, args.hidden)
        < 1
        or args.experts % args.ffn_ranks
    ):
        parser.error(
            "positive dimensions and an equal expert count per FFN are required"
        )
    selectable = (
        args.experts // args.ffn_ranks if args.routing == "skew" else args.experts
    )
    if args.top_k > selectable:
        parser.error(
            "top-k exceeds the distinct experts available under this routing pattern"
        )
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    if world != args.ffn_ranks + args.attention_ranks:
        raise ValueError("torchrun world size must equal FFN plus Attention ranks")
    common = {
        "capacity_tokens_per_rank": args.capacity,
        "hidden_size": args.hidden,
        "num_experts": args.experts,
        "top_k": args.top_k,
        "ffn_ranks": args.ffn_ranks,
        "attention_ranks": args.attention_ranks,
        "ffn_ranks_per_domain": args.ffn_ranks_per_domain,
        "payload": args.payload,
        "allow_hybrid_mode": args.hybrid == "on",
        "num_sms": args.num_sms,
    }
    name, overrides = next(iter(args.arms.items()))
    overrides = dict(overrides)
    backend = overrides.pop("_baseline", name)
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown AFD backend {backend!r}; custom arm names require _baseline"
        )
    if common.keys() & overrides.keys():
        raise ValueError(
            "workload options must be specified on the common command line"
        )
    module = import_module(f".{backend}", package=__package__)
    arm = module.Arm(name, common, overrides, args, dist.group.WORLD)
    attention = arm.role == "attention"
    source = arm.role_rank if attention else 0
    dtype = torch.bfloat16 if args.payload == "bf16" else torch.float8_e4m3fn
    cells = []
    for n in args.tokens:
        x = torch.ones((n, args.hidden), dtype=dtype, device="cuda")
        ids = routes(args, source, n)
        weights = torch.full((n, args.top_k), 1 / args.top_k, device="cuda")
        scale = (
            torch.ones((n, args.hidden // 128), device="cuda")
            if args.payload == "fp8"
            else None
        )
        work = logical_work(args, n, ids, weights, arm.role, arm.role_rank)
        local_experts = args.experts // args.ffn_ranks
        begin = arm.role_rank * local_experts
        counts = (
            work["per_expert"][begin : begin + local_experts] if not attention else []
        )
        arm.prepare(n, x, ids, weights, scale, args.execution == "graph", counts)
        for _ in range(args.warmup):
            arm.exchange()
        torch.cuda.synchronize()
        arm.check(args.top_k)
        dist.barrier()
        blocks = []
        for _ in range(args.blocks):
            dist.barrier()
            us = timed(arm.exchange, args.iterations)
            gathered = [None] * world
            dist.all_gather_object(
                gathered, {"rank": rank, "attention": attention, "us": {name: us}}
            )
            blocks.append({"order": [name], "ranks": gathered})
        arm.check(args.top_k)
        arm.release()
        torch.cuda.synchronize()
        dist.barrier()
        values = [
            max(r["us"][name] for r in b["ranks"] if r["attention"]) for b in blocks
        ]
        summary = {name: {"median_us": statistics.median(values)}}
        cells.append({"tokens": n, "work": work, "blocks": blocks, "summary": summary})
        if rank == 0:
            print(json.dumps({"tokens": n, "arm": name} | summary[name]), flush=True)
        del x, ids, weights, scale
    metadata = arm.metadata()
    with open(metadata["backend_binary"], "rb") as binary:
        binary_sha = hashlib.sha256(binary.read()).hexdigest()
    metadata.update(
        {
            "rank": rank,
            "role": arm.role,
            "role_rank": arm.role_rank,
            "backend_binary_sha256": binary_sha,
            "device": torch.cuda.get_device_name(),
            "device_uuid": str(
                torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
            ),
            "torch": torch.__version__,
        }
    )
    gathered = [None] * world
    dist.all_gather_object(gathered, metadata)
    arm.close()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "args": vars(args) | {"output": str(args.output)},
                    "metadata": gathered,
                    "cells": cells,
                    "metric": METRIC,
                    "scope": SCOPE,
                    "time": time.time(),
                },
                indent=1,
            )
            + "\n"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
