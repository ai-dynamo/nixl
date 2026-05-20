# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
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

import argparse
import ipaddress
import os
import socket
import sys
import time

# Add elastic subdirectory to path for store_group import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "elastic"))
# noinspection PyUnresolvedReferences
import nixl_ep  # noqa: E402
import store_group  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from utils import (  # noqa: E402
    bench,
    bench_kineto,
    calc_diff,
    create_grouped_scores,
    init_dist,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
)

TCP_STORE_PORT = 9999
FOUR_GPU_SINGLE_NODE_TARGET = "four_gpu_single_node"
FOUR_GPU_SINGLE_NODE_LABEL = "four-GPU single-node HT correctness"


class TargetSelectionError(ValueError):
    pass


class CorrectnessPreflightError(RuntimeError):
    def __init__(self, failure_category: str, message: str):
        super().__init__(message)
        self.failure_category = failure_category


def resolve_correctness_target(args: argparse.Namespace) -> str | None:
    selected = args.correctness_target
    if selected is None:
        return None

    if selected != FOUR_GPU_SINGLE_NODE_TARGET:
        raise TargetSelectionError(
            f"unsupported_target: unsupported correctness target {selected!r}; "
            f"supported target: {FOUR_GPU_SINGLE_NODE_TARGET}"
        )

    if args.num_processes != 4:
        raise TargetSelectionError(
            f"unsupported_target: {FOUR_GPU_SINGLE_NODE_TARGET} requires "
            f"--num-processes 4; got {args.num_processes}"
        )

    world_size = _read_int_env("WORLD_SIZE", 1)
    rank = _read_int_env("RANK", 0)
    if world_size != 1:
        raise TargetSelectionError(
            f"unsupported_target: {FOUR_GPU_SINGLE_NODE_TARGET} requires "
            f"WORLD_SIZE=1; got {world_size}"
        )
    if rank != 0:
        raise TargetSelectionError(
            f"unsupported_target: {FOUR_GPU_SINGLE_NODE_TARGET} requires "
            f"RANK=0; got {rank}"
        )

    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    if master_addr and not _is_local_endpoint(master_addr):
        raise TargetSelectionError(
            f"unsupported_target: {FOUR_GPU_SINGLE_NODE_TARGET} requires a local "
            f"MASTER_ADDR; got {master_addr!r}"
        )

    if args.tcp_server and not _is_local_endpoint(args.tcp_server):
        raise TargetSelectionError(
            f"unsupported_target: {FOUR_GPU_SINGLE_NODE_TARGET} rejects off-node "
            f"TCPStore endpoints; got {args.tcp_server!r}"
        )

    return FOUR_GPU_SINGLE_NODE_TARGET


def preflight_four_gpu_single_node_target(
    args: argparse.Namespace,
    *,
    cuda: object,
    tcp_store_port: int,
    tcp_store_host: str,
) -> None:
    if args.correctness_target != FOUR_GPU_SINGLE_NODE_TARGET:
        return

    if not cuda.is_available():
        raise CorrectnessPreflightError(
            "gpu_unavailable", "gpu_unavailable: CUDA runtime is unavailable"
        )

    visible_cuda_devices = int(cuda.device_count())
    if visible_cuda_devices != 4:
        raise CorrectnessPreflightError(
            "gpu_unavailable",
            f"gpu_unavailable: {FOUR_GPU_SINGLE_NODE_TARGET} requires exactly "
            f"4 visible CUDA devices; got {visible_cuda_devices}",
        )

    current_device = None
    try:
        current_device = int(cuda.current_device())
    except Exception:
        pass

    try:
        for device in range(4):
            try:
                cuda.set_device(device)
                cuda.get_device_properties(device)
            except Exception as exc:
                raise CorrectnessPreflightError(
                    "runtime_not_ready",
                    f"runtime_not_ready: CUDA context readiness failed for device {device}",
                ) from exc
    finally:
        if current_device is not None:
            try:
                cuda.set_device(current_device)
            except Exception:
                pass

    for src in range(4):
        for dst in range(4):
            if src == dst:
                continue
            try:
                can_access = bool(cuda.can_device_access_peer(src, dst))
            except Exception as exc:
                raise CorrectnessPreflightError(
                    "peer_wiring_failed",
                    "peer_wiring_failed: CUDA peer access probe failed",
                ) from exc
            if not can_access:
                raise CorrectnessPreflightError(
                    "peer_wiring_failed",
                    f"peer_wiring_failed: CUDA peer access unavailable between "
                    f"devices {src} and {dst}",
                )

    selected_tcp_store_host = args.tcp_server or tcp_store_host
    if not _is_local_endpoint(selected_tcp_store_host):
        raise CorrectnessPreflightError(
            "unsupported_target",
            f"unsupported_target: {FOUR_GPU_SINGLE_NODE_TARGET} requires a local "
            "TCPStore endpoint",
        )
    if not args.tcp_server:
        _check_local_tcpstore_bind(selected_tcp_store_host, tcp_store_port)


def format_correctness_evidence(
    args: argparse.Namespace,
    *,
    result: str,
    failure_category: str | None = None,
) -> str:
    failure = "none" if result == "pass" else (failure_category or "correctness_failed")
    fields = (
        ("schema", "ep_ht_correctness_evidence_v1"),
        ("evidence_id", "ep_ht_four_gpu_single_node"),
        ("target", FOUR_GPU_SINGLE_NODE_TARGET),
        ("target_label", _evidence_value(FOUR_GPU_SINGLE_NODE_LABEL)),
        ("topology", "single_node"),
        ("world_size", 4),
        ("num_nvl_ranks", 4),
        ("num_rdma_ranks", 1),
        ("workload_tokens", args.num_tokens),
        ("hidden_size", args.hidden),
        ("top_k", args.num_topk),
        ("expert_count", args.num_experts),
        ("correctness_only", "true"),
        ("scope", "single_node_correctness_only_no_multi_node_rdma_no_performance"),
        ("result", result),
        ("failure_category", _evidence_value(failure)),
    )
    return "[evidence] " + " ".join(f"{key}={value}" for key, value in fields)


def classify_correctness_failure(exc: BaseException) -> str:
    if isinstance(exc, CorrectnessPreflightError):
        return exc.failure_category
    if isinstance(exc, TargetSelectionError):
        return "unsupported_target"

    message = str(exc).lower()
    if any(term in message for term in ("cuda ipc", "nvlink", "peer", "ipc")):
        return "peer_wiring_failed"
    if any(term in message for term in ("no cuda", "cuda unavailable", "gpu")):
        return "gpu_unavailable"
    if any(term in message for term in ("nixl", "nccl", "runtime", "c10")):
        return "runtime_not_ready"
    return "correctness_failed"


def _check_local_tcpstore_bind(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
    except Exception as exc:
        raise CorrectnessPreflightError(
            "runtime_not_ready",
            "runtime_not_ready: local TCPStore bind failed for configured endpoint",
        ) from exc


def _is_local_endpoint(endpoint: str) -> bool:
    normalized = endpoint.strip().strip("[]").lower().rstrip(".")
    local_names = {
        "localhost",
        socket.gethostname().lower().rstrip("."),
        socket.getfqdn().lower().rstrip("."),
    }
    if normalized in local_names:
        return True

    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise TargetSelectionError(
            f"unsupported_target: {name} must be an integer; got {value!r}"
        ) from exc


def _evidence_value(value: object) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value)
    ).strip("_")
    return normalized or "unspecified"


# noinspection PyShadowingNames
def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    buffer: nixl_ep.Buffer,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk_groups, num_topk, num_experts = (
        args.num_topk_groups,
        args.num_topk,
        args.num_experts,
    )

    assert num_experts % num_ranks == 0
    if args.ci_correctness_only:
        assert num_local_ranks == 4
    else:
        assert num_local_ranks == 8
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}",
            flush=True,
        )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(
        group_scores, k=num_topk_groups, dim=-1, sorted=False
    ).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[
        1
    ]
    topk_idx = topk_idx.to(nixl_ep.topk_idx_t)
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
    )
    topk_weights_pure_rand = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    )
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    num_tokens_per_rdma_rank = torch.empty((num_nodes,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="cuda"
        )
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    (
        ref_num_tokens_per_rank,
        ref_num_tokens_per_rdma_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    if not args.ci_correctness_only:
        t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
        if local_rank == 0:
            print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
            print("", flush=True)
    group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = (
        nixl_ep.Buffer.get_dispatch_config(num_ranks)
        if args.ci_correctness_only
        else nixl_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)
    )

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (x_pure_rand, x, x_e4m3):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end="",
                        )
                    dispatch_args = {
                        "x": current_x,
                        "num_tokens_per_rank": num_tokens_per_rank,
                        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
                        "is_token_in_rank": is_token_in_rank,
                        "num_tokens_per_expert": num_tokens_per_expert,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        dispatch_args.update(
                            {
                                "topk_idx": topk_idx,
                                "topk_weights": (
                                    topk_weights_pure_rand
                                    if current_x is x_pure_rand
                                    else topk_weights
                                ),
                            }
                        )
                    if previous_mode:
                        dispatch_args.update({"previous_event": buffer.capture()})
                    (
                        recv_x,
                        recv_topk_idx,
                        recv_topk_weights,
                        recv_num_tokens_per_expert_list,
                        handle,
                        event,
                    ) = buffer.ht_dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    recv_x = (
                        per_token_cast_back(*recv_x)
                        if isinstance(recv_x, tuple)
                        else recv_x
                    )

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0
                    ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
                    assert (
                        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                        == recv_num_tokens_per_expert_list
                    )
                    if current_x is not x_pure_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert recv_topk_idx is not None
                        assert recv_topk_weights is not None
                        assert (
                            recv_topk_idx.eq(-1)
                            | (
                                (recv_topk_idx >= 0)
                                & (recv_topk_idx < (num_experts // num_ranks))
                            )
                        ).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = (
                                recv_topk_weights.amax(dim=1, keepdim=True).expand_as(
                                    recv_topk_weights
                                )[recv_topk_idx.eq(-1)]
                            )
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {
                            "x": current_x,
                            "handle": handle,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})
                        recv_x_cached, _, _, _, _, event = buffer.ht_dispatch(
                            **dispatch_args
                        )
                        event.current_stream_wait() if async_mode else ()
                        recv_x_cached = (
                            per_token_cast_back(*recv_x_cached)
                            if isinstance(recv_x_cached, tuple)
                            else recv_x_cached
                        )

                        if current_x is not x_pure_rand:
                            check_data(recv_x_cached, recv_gbl_rank_prefix_sum)

                        # Use cached result for combine
                        recv_x = recv_x_cached

                    # Test combine
                    bias_0 = torch.ones(
                        (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
                    )
                    bias_1 = torch.randn(
                        (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
                    )
                    combine_args = {
                        "x": recv_x,
                        "bias": (bias_0, bias_1),
                        "handle": handle,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        combine_args.update({"topk_weights": recv_topk_weights})
                    if previous_mode:
                        combine_args.update({"previous_event": buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.ht_combine(
                        **combine_args
                    )
                    event.current_stream_wait() if async_mode else ()

                    check_x = (
                        combined_x.float() - bias_0.float() - bias_1.float()
                    ) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = (
                            combined_topk_weights
                            if (current_x is x_pure_rand)
                            else (
                                combined_topk_weights
                                / is_token_in_rank.sum(dim=1).unsqueeze(1)
                            )
                        )
                        ref_topk_weights = (
                            topk_weights_pure_rand
                            if current_x is x_pure_rand
                            else topk_weights
                        )
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    # Sync all ranks before printing passed
                    group.barrier()
                    if local_rank == 0:
                        print(" passed", flush=True)
                    group.barrier()
    if local_rank == 0:
        print("", flush=True)

    if args.ci_correctness_only:
        if local_rank == 0:
            print(
                "[ci] Completed reduced-topology correctness checks; skipping performance tuning.",
                flush=True,
            )
        return

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (
            (dispatch_bf16_rdma_send_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_rdma_send_bytes
        )
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in range(4, 45, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = nixl_ep.Config(
                    num_sms,
                    nvl_chunk_size,
                    nvl_buffer_size,
                    rdma_chunk_size,
                    rdma_buffer_size,
                )
                tune_args = {"x": current_x, "handle": handle, "config": config}
                t, notify_t = bench_kineto(
                    lambda: buffer.ht_dispatch(**tune_args), ("dispatch", "notify")
                )
                if t < best_time:
                    best_time, best_results = t, (
                        num_sms,
                        nvl_chunk_size,
                        rdma_chunk_size,
                        notify_t,
                    )
                if local_rank == 0:
                    print(
                        f"[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, notify: {notify_t * 1e6:.2f} us, BW: {rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ",
                        flush=True,
                    )
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, notify: {best_results[3] * 1e6:.2f} us, BW: {rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)',  # type: ignore[index]
                flush=True,
            )
            print("", flush=True)

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor([best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device="cuda")  # type: ignore[index]
            all_best_fp8_results_list = [
                torch.zeros_like(best_dispatch_results)
                for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(
                all_best_fp8_results_list, best_dispatch_results, group=group
            )
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = nixl_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2], rdma_buffer_size)  # type: ignore[index]

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    recv_x, _, _, _, handle, _ = buffer.ht_dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = nixl_ep.Config(
                num_sms,
                nvl_chunk_size,
                nvl_buffer_size,
                rdma_chunk_size,
                rdma_buffer_size,
            )
            tune_args = {"x": recv_x, "handle": handle, "config": config}
            t, notify_t = bench_kineto(
                lambda: buffer.ht_combine(**tune_args), ("combine", "notify")
            )
            if local_rank == 0:
                print(
                    f"[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, notify: {notify_t * 1e6:.2f} us, BW: {combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ",
                    flush=True,
                )
                if t < best_time:
                    best_time, best_results = t, (
                        num_sms,
                        nvl_chunk_size,
                        rdma_chunk_size,
                        notify_t,
                    )

    if local_rank == 0:
        print(f"[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, notify: {best_results[3] * 1e6:.2f} us, BW: {combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)", flush=True)  # type: ignore[index]
        print("", flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # Pin each process to a distinct GPU so NCCL does not see duplicate devices.
    # Use local_rank so NCCL gets correct device_id; avoid CUDA_VISIBLE_DEVICES
    # so that UCX/DOCA can see all GPUs for GPU-initiated RDMA when needed.
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank % num_local_ranks)

    num_nodes = int(os.getenv("WORLD_SIZE", 1))

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    print(
        f"pid: {os.getpid()}, rank: {rank}, num_ranks: {num_ranks} ,local_rank: {local_rank}",
        flush=True,
    )
    if args.test_ll_compatibility:
        ll_num_experts = 256

    num_sms = 24
    num_qps_per_rank = max(
        num_sms // 2,
        args.num_experts // num_ranks if args.ci_correctness_only else 0,
        ll_num_experts // num_ranks if args.test_ll_compatibility else 0,
    )

    # Create TCPStore client for NIXL metadata exchange
    tcp_server = args.tcp_server if args.tcp_server else "127.0.0.1"
    tcp_store = store_group.create_client_store(
        master_addr=tcp_server,
        port=TCP_STORE_PORT,
    )

    # Initialize NIXL buffer with group (for IPC handles) and TCPStore (for NIXL metadata)
    print(
        f"pid: {os.getpid()}, rank: {rank}, num_ranks: {num_ranks}, initializing buffer",
        flush=True,
    )
    buffer = nixl_ep.Buffer(
        rank=rank,
        low_latency_mode=False,
        explicitly_destroy=True,
        group=group,
        tcp_store_group=tcp_store,
    )
    # HT kernels use RDMA staging metadata even for the four-GPU single-node
    # correctness target.
    num_rdma_bytes = int(1e9)
    buffer.update_memory_buffers(
        num_ranks=num_ranks,
        num_experts_per_rank=num_qps_per_rank,
        num_nvl_bytes=int(2e9),
        num_rdma_bytes=num_rdma_bytes,
    )
    buffer.connect_ranks([i for i in range(num_ranks) if i != rank])

    assert buffer.runtime.get_num_nvl_ranks() == num_local_ranks
    assert buffer.runtime.get_num_rdma_ranks() == num_nodes

    if args.ci_correctness_only:
        assert num_local_ranks == 4 and num_ranks == 4
        if local_rank == 0:
            print(
                "[ci] Reduced 4-GPU HT correctness only; no multi-node RDMA or performance coverage.",
                flush=True,
            )
    else:
        assert num_local_ranks == 8
    torch.manual_seed(rank)

    for i in (num_sms,):
        test_main(
            args,
            i,
            local_rank,
            num_local_ranks,
            num_ranks,
            num_nodes,
            rank,
            buffer,
            group,
        )
        if local_rank == 0:
            print("", flush=True)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


def run_server(bind_host: str = "0.0.0.0"):
    _store = store_group.create_master_store(  # noqa: F841
        port=TCP_STORE_PORT,
        host_name=bind_host,
    )
    # Keep the server process alive while TCPStore serves requests
    while True:
        time.sleep(1)


def _is_global_rank_zero() -> bool:
    try:
        return int(os.getenv("RANK", "0")) == 0
    except ValueError:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test high-throughput EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk-groups",
        type=int,
        default=None,
        help="Number of top-k groups (default: `min(num_nodes, 4)`)",
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--test-ll-compatibility",
        action="store_true",
        help="whether to test compatibility with low-latency kernels",
    )
    parser.add_argument(
        "--tcp-server",
        type=str,
        help="TCP server address (for both TCPStore and rank server). If not set, both will be started locally.",
    )
    parser.add_argument(
        "--correctness-target",
        type=str,
        help=(
            "Named correctness target to run "
            f"(supported: {FOUR_GPU_SINGLE_NODE_TARGET})."
        ),
    )
    args = parser.parse_args()
    args.num_nodes = int(os.getenv("WORLD_SIZE", 1))

    try:
        correctness_target = resolve_correctness_target(args)
    except TargetSelectionError as exc:
        selected_target = getattr(args, "correctness_target", None)
        if selected_target == FOUR_GPU_SINGLE_NODE_TARGET:
            args.correctness_target = FOUR_GPU_SINGLE_NODE_TARGET
            args.correctness_target_label = FOUR_GPU_SINGLE_NODE_LABEL
            args.ci_correctness_only = True
            if _is_global_rank_zero():
                print(
                    format_correctness_evidence(
                        args,
                        result="fail",
                        failure_category=classify_correctness_failure(exc),
                    ),
                    flush=True,
                )
        parser.error(str(exc))

    args.correctness_target = correctness_target
    args.correctness_target_label = (
        FOUR_GPU_SINGLE_NODE_LABEL
        if correctness_target == FOUR_GPU_SINGLE_NODE_TARGET
        else None
    )
    args.ci_correctness_only = correctness_target == FOUR_GPU_SINGLE_NODE_TARGET

    if correctness_target and _is_global_rank_zero():
        print(
            f"[target] Running {args.correctness_target_label} ({correctness_target})",
            flush=True,
        )

    tcp_store_bind_host = "127.0.0.1"
    if correctness_target != FOUR_GPU_SINGLE_NODE_TARGET:
        tcp_store_bind_host = "0.0.0.0"

    try:
        preflight_four_gpu_single_node_target(
            args,
            cuda=torch.cuda,
            tcp_store_port=TCP_STORE_PORT,
            tcp_store_host=tcp_store_bind_host,
        )

        if not args.tcp_server:
            print("Starting TCPStore and rank server locally", flush=True)
            server_process = torch.multiprocessing.Process(
                target=run_server, args=(tcp_store_bind_host,), daemon=True
            )
            server_process.start()
            time.sleep(0.5)

        # Set default `num_topk_groups` if not provided
        if args.num_topk_groups is None:
            num_nodes = int(os.getenv("WORLD_SIZE", 1))
            args.num_topk_groups = min(num_nodes, 4)

        num_processes = args.num_processes
        # 2-node run (WORLD_SIZE=2): run on both nodes with same MASTER_ADDR/MASTER_PORT; node1 needs --tcp-server <node0_ip>.
        # NVL/RDMA timeouts across nodes usually mean RDMA/IB/UCX between nodes is broken or slow (e.g. "accelerated IB support was not found" on one node).
        torch.multiprocessing.spawn(
            test_loop, args=(num_processes, args), nprocs=num_processes
        )
    except Exception as exc:
        if correctness_target == FOUR_GPU_SINGLE_NODE_TARGET:
            if _is_global_rank_zero():
                print(
                    format_correctness_evidence(
                        args,
                        result="fail",
                        failure_category=classify_correctness_failure(exc),
                    ),
                    flush=True,
                )
            sys.exit(1)
        raise

    if correctness_target == FOUR_GPU_SINGLE_NODE_TARGET and _is_global_rank_zero():
        print(format_correctness_evidence(args, result="pass"), flush=True)
