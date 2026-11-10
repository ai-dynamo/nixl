# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check BF16 correctness and JIT reuse across token capacities on one buffer.

Capacity is a launch argument, not a template parameter. Existing CUDA graphs
retain their captured capacity; this test issues fresh launches at each capacity.
Run: python tests/elastic/test_runtime_tokens.py --num-processes 4
"""
import argparse
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from test_adaptive_modes import ModeTestBuffer
from test_ep import test_dispatch_combine

from nixl.runtime.ep.utils.envs import dist_print, init_dist


@torch.inference_mode()
def run(local_rank, num_local_ranks, args):
    _, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert args.num_experts > 0 and args.num_experts % num_ranks == 0
    buffer = ModeTestBuffer(
        group,
        num_experts_per_rank=args.num_experts // num_ranks,
        num_max_tokens_per_rank=max(args.token_counts),
        hidden=args.hidden,
        num_topk=args.num_topk,
        allow_multiple_reduction=bool(args.allow_multiple_reduction),
        explicitly_destroy=True,
    )
    modes = (
        ("direct", "hybrid") if buffer.get_topology()[0] == "direct" else ("hybrid",)
    )
    first_kernels = None
    for capacity in args.token_counts:
        for mode in modes:
            buffer.barrier()
            torch.cuda.synchronize()
            dist.barrier()
            buffer.set_topology_forced(mode)
            for do_cpu_sync in (True, False):
                case = argparse.Namespace(
                    **vars(args), num_tokens=capacity, do_cpu_sync=do_cpu_sync
                )
                test_dispatch_combine(buffer, case, use_fp8_dispatch=False)
        torch.cuda.synchronize()
        dist.barrier()
        kernels = {
            path.name for path in (Path(args.jit_cache_dir) / "cache").glob("kernel.*")
        }
        if first_kernels is None:
            for name in (
                "adaptive_dispatch",
                "adaptive_dispatch_copy_epilogue",
                "adaptive_combine",
                "adaptive_combine_reduce_epilogue",
            ):
                assert any(
                    key.startswith(f"kernel.{name}.") for key in kernels
                ), f"Missing {name} in JIT cache"
            first_kernels = kernels
        assert (
            kernels == first_kernels
        ), f"Capacity {capacity} created JIT entries: {sorted(kernels - first_kernels)}"
        dist_print(
            f"PASS: BF16 capacity {capacity}, {len(kernels)} unchanged JIT entries",
            once_in_node=True,
        )
    buffer.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1024, 17, 16],
        help="Capacities to test; include a non-power-of-two value",
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument(
        "--allow-multiple-reduction", type=int, choices=(0, 1), default=1
    )
    parser.set_defaults(
        test_first_only=True,
        skip_check=False,
        skip_perf_test=True,
        deterministic=False,
        ignore_local_traffic=False,
        dump_profile_traces="",
        unbalanced_ratio=1.0,
        precise_unbalanced_ratio=False,
        masked_ratio=0.0,
    )
    args = parser.parse_args()
    if len(set(args.token_counts)) < 2 or min(args.token_counts) <= 0:
        parser.error("--token-counts needs at least two distinct positive capacities")
    return args


if __name__ == "__main__":
    args = parse_args()
    # Spawned workers import the native module after inheriting this fresh cache path.
    with tempfile.TemporaryDirectory(prefix="nixl_ep_runtime_tokens_") as cache_dir:
        args.jit_cache_dir = cache_dir
        os.environ["NIXL_EP_JIT_CACHE_DIR"] = cache_dir
        torch.multiprocessing.spawn(
            run, args=(args.num_processes, args), nprocs=args.num_processes
        )
