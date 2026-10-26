# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
`num_max_tokens_per_rank` is a run-time kernel argument: one buffer runs `test_ep.py`'s dispatch + combine correctness
checks at several token counts in one process (the default list holds a count that is not a power of two), in both
modes on one NVLink domain (direct, and hybrid forced as in `test_adaptive_modes.py`; hybrid only across domains), and
the JIT cache must hold the same kernel set after every count as after the first (one cubin per kernel serves them all).
The JIT cache is a fresh directory (`NIXL_EP_JIT_CACHE_DIR`), so the first count compiles every kernel it uses.
Usage: python tests/elastic/test_runtime_tokens.py --num-processes 4 --token-counts 4096,1000,128
"""
import argparse
import os
import tempfile

import torch
import torch.distributed as dist
from test_ep import test_dispatch_combine

from nixl.runtime import ep  # type: ignore[attr-defined]
from nixl.runtime.ep.utils.envs import dist_print, init_dist


def list_jit_kernels(cache_dir: str) -> set:
    path = os.path.join(cache_dir, "cache")
    return {name for name in os.listdir(path) if name.startswith("kernel.")}


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert (
        args.num_experts % num_ranks == 0
    ), f"{args.num_experts} experts do not divide by {num_ranks} ranks"
    token_counts = [int(count) for count in args.token_counts.split(",")]
    assert len(set(token_counts)) >= 2, "Needs two different token counts"

    # One buffer sized for the largest count; each dispatch passes its own count, which its handle carries to combine
    buffer = ep.ElasticBuffer(
        group,
        num_experts_per_rank=args.num_experts // num_ranks,
        num_max_tokens_per_rank=max(token_counts),
        hidden=args.hidden,
        allow_multiple_reduction=args.allow_multiple_reduction,
        explicitly_destroy=True,
        num_gpu_timeout_secs=args.num_gpu_timeout_secs,
        num_cpu_timeout_secs=args.num_cpu_timeout_secs,
    )

    # Direct mode is legal on one NVLink domain only; hybrid everywhere
    native_mode = buffer.get_topology()[0]
    modes = ["direct", "hybrid"] if native_mode == "direct" else ["hybrid"]

    first_kernels = None
    for num_tokens in token_counts:
        for mode in modes:
            dist_print(f"Token count {num_tokens}, {mode} mode:", once_in_node=True)
            buffer.set_topology_forced(mode)
            test_dispatch_combine(
                buffer, argparse.Namespace(**vars(args), num_tokens=num_tokens)
            )
        buffer.set_topology_forced(native_mode)
        torch.cuda.synchronize()
        dist.barrier()

        # Every local rank shares the cache directory: after the barrier it holds every kernel built so far
        kernels = list_jit_kernels(args.jit_cache_dir)
        if first_kernels is None:
            first_kernels = kernels
            assert any(
                k.startswith("kernel.adaptive_dispatch.") for k in kernels
            ) and any(
                k.startswith("kernel.adaptive_combine.") for k in kernels
            ), f"No adaptive kernels in the JIT cache: {sorted(kernels)}"
        assert (
            kernels == first_kernels
        ), f"Token count {num_tokens} built new kernels: {sorted(kernels - first_kernels)}"
        dist_print(f" > {len(kernels)} JIT kernels, none new", once_in_node=True)
        dist.barrier()

    dist_print("PASS: one JIT kernel set for every token count", once_in_node=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the run-time token count")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--token-counts",
        type=str,
        default="4096,1000,128",
        help="Comma-separated num_max_tokens_per_rank values, run in order on one buffer (default: 4096,1000,128)",
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-topk", type=int, default=6, help="Number of top-k experts"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Number of expert slots of the world",
    )
    parser.add_argument(
        "--num-sms", type=int, default=0, help="Number of SMs to use (0 means auto)"
    )
    parser.add_argument(
        "--allow-multiple-reduction",
        type=int,
        default=1,
        help="Whether to allow multiple reductions",
    )
    parser.add_argument(
        "--do-cpu-sync", type=int, default=1, help="Whether to do CPU sync"
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Run every test_ep.py case per count, not only the first",
    )
    parser.add_argument(
        "--num-gpu-timeout-secs",
        type=int,
        default=100,
        help="Timeout in seconds (GPU side)",
    )
    parser.add_argument(
        "--num-cpu-timeout-secs",
        type=int,
        default=100,
        help="Timeout in seconds (CPU side)",
    )
    parser.add_argument(
        "--jit-cache-dir",
        type=str,
        default="",
        help="An empty JIT cache directory (default: a new temporary one)",
    )
    args = parser.parse_args()

    # The JIT compiler reads the cache directory at its first build, in each spawned process
    args.jit_cache_dir = args.jit_cache_dir or tempfile.mkdtemp(
        prefix="nixl_ep_v2_jit_"
    )
    assert not os.listdir(args.jit_cache_dir), f"{args.jit_cache_dir} is not empty"
    os.environ["NIXL_EP_JIT_CACHE_DIR"] = args.jit_cache_dir
    print(f"JIT cache: {args.jit_cache_dir}", flush=True)

    # The fields `test_dispatch_combine` reads besides the token count: correctness only, no perf
    test_ep_args = dict(
        test_first_only=not args.all_cases,
        skip_check=False,
        skip_perf_test=True,
        dump_profile_traces="",
        ignore_local_traffic=False,
        unbalanced_ratio=1.0,
        precise_unbalanced_ratio=False,
        masked_ratio=0.0,
    )
    for key, value in test_ep_args.items():
        setattr(args, key, value)

    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
