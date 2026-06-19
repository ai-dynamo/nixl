# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import inspect
import os
import time

import torch
import torch.distributed as dist

try:
    import nixl_ep_cu13 as nixl_ep
except ModuleNotFoundError:
    import nixl_ep
from test_ht import TCP_STORE_PORT, run_server, test_main
import store_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Torchrun high-throughput EP test for GB200 workers with fewer than 8 local GPUs."
    )
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk-groups", type=int, default=None)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help=(
            "Total experts. Defaults to the smallest world-size-compatible "
            "multiple at least 256 so odd GB200 group counts, such as 3x4 "
            "and 5x4, exercise HT instead of failing the base test shape assert."
        ),
    )
    parser.add_argument("--test-ll-compatibility", action="store_true")
    parser.add_argument(
        "--tcp-server",
        type=str,
        default=None,
        help="TCPStore server for NIXL metadata. Defaults to MASTER_ADDR and is started by global rank 0.",
    )
    parser.add_argument(
        "--nvl-group-size",
        type=int,
        default=4,
        help="Ranks per CUDA-IPC/NVLink-local EP group. Use 4 for 4-GPU GB200 workers.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run one BF16/no-top-k HT dispatch+combine roundtrip, then exit.",
    )
    parser.add_argument(
        "--debug-smoke-summary",
        action="store_true",
        help="Print recv_x/prefix summaries before smoke data assertions fail.",
    )
    parser.add_argument(
        "--debug-ht-barrier-only",
        action="store_true",
        help="Run only the EP HT device-side inter-RDMA barrier probe, then exit.",
    )
    parser.add_argument(
        "--debug-ht-barrier-channels",
        type=int,
        default=12,
        help="Number of NIXL device channels to exercise in the HT barrier probe.",
    )
    parser.add_argument(
        "--debug-ht-atomic-pairs",
        type=str,
        default=None,
        help="Comma-separated directed remote-atomic pairs, e.g. '0:4,4:0,1:5'. Runs and exits.",
    )
    parser.add_argument(
        "--debug-ht-atomic-repeat",
        type=int,
        default=1,
        help="Repeat each directed atomic pair this many times before failing.",
    )
    parser.add_argument(
        "--debug-ht-put-pairs",
        type=str,
        default=None,
        help="Comma-separated directed remote-put pairs, e.g. '0:4,4:0,1:5'. Runs and exits.",
    )
    parser.add_argument(
        "--debug-ht-put-repeat",
        type=int,
        default=1,
        help="Repeat each directed put pair this many times before failing.",
    )
    return parser.parse_args()


def parse_rank_pairs(pairs: str, num_ranks: int, nvl_group_size: int) -> list[tuple[int, int]]:
    if pairs == "all-cross":
        return [
            (src, dst)
            for src in range(num_ranks)
            for dst in range(num_ranks)
            if src // nvl_group_size != dst // nvl_group_size
        ]
    if pairs == "same-nvl-cross":
        return [
            (src, dst)
            for src in range(num_ranks)
            for dst in range(num_ranks)
            if src // nvl_group_size != dst // nvl_group_size
            and src % nvl_group_size == dst % nvl_group_size
        ]

    parsed = []
    for item in pairs.split(","):
        item = item.strip()
        if not item:
            continue
        src, sep, dst = item.partition(":")
        if sep != ":":
            raise ValueError(f"invalid pair {item!r}; expected SRC:DST")
        parsed.append((int(src), int(dst)))
    if not parsed:
        raise ValueError("--debug-ht-atomic-pairs did not contain any pairs")
    return parsed


def main() -> None:
    args = parse_args()
    assert 0 < args.nvl_group_size <= 8 and 8 % args.nvl_group_size == 0

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", args.nvl_group_size))
    rank = int(os.environ["RANK"])
    num_ranks = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))

    init_kwargs = {"backend": "nccl", "init_method": "env://"}
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**init_kwargs)
    group = dist.new_group(list(range(num_ranks)))

    if rank == 0 and args.tcp_server is None:
        server_process = torch.multiprocessing.Process(target=run_server, daemon=True)
        server_process.start()
        time.sleep(0.5)

    dist.barrier()
    tcp_store = store_group.create_client_store(
        master_addr=args.tcp_server or master_addr,
        port=TCP_STORE_PORT,
    )

    num_rdma_groups = max(1, num_ranks // args.nvl_group_size)
    if args.num_topk_groups is None:
        args.num_topk_groups = min(num_rdma_groups, 4)
    if args.num_experts is None:
        args.num_experts = ((256 + num_ranks - 1) // num_ranks) * num_ranks

    num_sms = 24
    ll_num_experts = 256 if args.test_ll_compatibility else 0
    num_qps_per_rank = max(num_sms // 2, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)

    buffer = nixl_ep.Buffer(
        rank=rank,
        low_latency_mode=False,
        explicitly_destroy=True,
        group=group,
        tcp_store_group=tcp_store,
        nvl_group_size=args.nvl_group_size,
    )
    buffer.update_memory_buffers(
        num_ranks=num_ranks,
        num_experts_per_rank=num_qps_per_rank,
        num_nvl_bytes=int(2e9),
        num_rdma_bytes=int(1e9),
    )
    buffer.connect_ranks([i for i in range(num_ranks) if i != rank])

    if args.debug_ht_put_pairs:
        assert args.debug_ht_put_repeat > 0
        debug_ht_put_pair = getattr(buffer, "debug_ht_put_pair", buffer.runtime.debug_ht_put_pair)
        pair_failures = []
        pair_stats: dict[tuple[int, int], list[int]] = {}
        rank_pairs = parse_rank_pairs(args.debug_ht_put_pairs, num_ranks, args.nvl_group_size)
        if rank == 0:
            print(
                f"[debug-ht-put-pairs] count={len(rank_pairs)} pairs={rank_pairs}",
                flush=True,
            )
        tag_base = 0x9E3779B97F4A7C15
        pair_index = 0
        for repeat_idx in range(args.debug_ht_put_repeat):
            for src_rank, dst_rank in rank_pairs:
                tag = (
                    tag_base
                    ^ ((repeat_idx + 1) << 48)
                    ^ ((pair_index + 1) << 24)
                    ^ (src_rank << 12)
                    ^ dst_rank
                ) & ((1 << 63) - 1)
                if tag == 0:
                    tag = 1
                observed, status = debug_ht_put_pair(src_rank, dst_rank, tag)
                torch.cuda.synchronize()
                gathered_status = [torch.empty_like(status) for _ in range(num_ranks)]
                gathered_observed = [torch.empty_like(observed) for _ in range(num_ranks)]
                dist.all_gather(gathered_status, status)
                dist.all_gather(gathered_observed, observed)
                status_cpu = [int(t.cpu().item()) for t in gathered_status]
                observed_cpu = [[int(v) for v in t.cpu().tolist()] for t in gathered_observed]
                unexpected_hits = [
                    rank_id
                    for rank_id, rank_status in enumerate(status_cpu)
                    if rank_id not in (src_rank, dst_rank) and rank_status == 2
                ]
                ok = int(status_cpu[src_rank] != -2 and status_cpu[dst_rank] == 1 and not unexpected_hits)
                pair_stats.setdefault((src_rank, dst_rank), []).append(ok)
                if rank == 0:
                    print(
                        "[debug-ht-put-pair] "
                        f"repeat={repeat_idx} pair_index={pair_index} src={src_rank} dst={dst_rank} "
                        f"tag={tag} status={status_cpu} "
                        f"dst_observed={observed_cpu[dst_rank]} src_observed={observed_cpu[src_rank]} "
                        f"unexpected_hits={unexpected_hits}",
                        flush=True,
                    )
                if not ok:
                    pair_failures.append(
                        (repeat_idx, src_rank, dst_rank, tag, status_cpu, observed_cpu, unexpected_hits)
                    )
                pair_index += 1
                dist.barrier()
        if rank == 0:
            summary = {
                f"{src}:{dst}": f"{sum(values)}/{len(values)}"
                for (src, dst), values in sorted(pair_stats.items())
            }
            print(f"[debug-ht-put-summary] {summary}", flush=True)
        if pair_failures:
            raise RuntimeError(f"HT debug put pair failures: {pair_failures}")
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.debug_ht_atomic_pairs:
        assert args.debug_ht_atomic_repeat > 0
        debug_ht_atomic_pair = getattr(buffer, "debug_ht_atomic_pair", buffer.runtime.debug_ht_atomic_pair)
        pair_failures = []
        pair_stats: dict[tuple[int, int], list[int]] = {}
        rank_pairs = parse_rank_pairs(args.debug_ht_atomic_pairs, num_ranks, args.nvl_group_size)
        if rank == 0:
            print(
                f"[debug-ht-atomic-pairs] count={len(rank_pairs)} pairs={rank_pairs}",
                flush=True,
            )
        for repeat_idx in range(args.debug_ht_atomic_repeat):
            for src_rank, dst_rank in rank_pairs:
                expected_count = (repeat_idx + 1) * args.debug_ht_barrier_channels
                observed, status = debug_ht_atomic_pair(
                    src_rank,
                    dst_rank,
                    args.debug_ht_barrier_channels,
                    expected_count,
                )
                torch.cuda.synchronize()
                gathered_status = [torch.empty_like(status) for _ in range(num_ranks)]
                gathered_observed = [torch.empty_like(observed) for _ in range(num_ranks)]
                dist.all_gather(gathered_status, status)
                dist.all_gather(gathered_observed, observed)
                status_cpu = [int(t.cpu().item()) for t in gathered_status]
                observed_cpu = [[int(v) for v in t.cpu().tolist()] for t in gathered_observed]
                unexpected_hits = [
                    rank_id
                    for rank_id, rank_status in enumerate(status_cpu)
                    if rank_id != dst_rank and rank_status == 2
                ]
                ok = int(status_cpu[src_rank] != -2 and status_cpu[dst_rank] == 1 and not unexpected_hits)
                pair_stats.setdefault((src_rank, dst_rank), []).append(ok)
                if rank == 0:
                    print(
                        "[debug-ht-atomic-pair] "
                        f"repeat={repeat_idx} src={src_rank} dst={dst_rank} expected={expected_count} status={status_cpu} "
                        f"dst_observed={observed_cpu[dst_rank]} src_observed={observed_cpu[src_rank]} "
                        f"unexpected_hits={unexpected_hits}",
                        flush=True,
                    )
                if not ok:
                    pair_failures.append(
                        (repeat_idx, src_rank, dst_rank, status_cpu, observed_cpu, unexpected_hits)
                    )
                dist.barrier()
        if rank == 0:
            summary = {
                f"{src}:{dst}": f"{sum(values)}/{len(values)}"
                for (src, dst), values in sorted(pair_stats.items())
            }
            print(f"[debug-ht-atomic-summary] {summary}", flush=True)
        if pair_failures:
            raise RuntimeError(f"HT debug atomic pair failures: {pair_failures}")
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.debug_ht_barrier_only:
        debug_ht_barrier = getattr(buffer, "debug_ht_barrier", buffer.runtime.debug_ht_barrier)
        observed, status = debug_ht_barrier(args.debug_ht_barrier_channels)
        torch.cuda.synchronize()
        observed_cpu = [int(v) for v in observed.cpu().tolist()]
        status_cpu = int(status.cpu().item())
        print(
            "[debug-ht-barrier] "
            f"rank={rank} status={status_cpu} observed={observed_cpu[0]} "
            f"expected={observed_cpu[1]} epoch={observed_cpu[2]} "
            f"channels={args.debug_ht_barrier_channels}",
            flush=True,
        )
        gathered = [torch.empty_like(status) for _ in range(num_ranks)]
        gathered_observed = [torch.empty_like(observed) for _ in range(num_ranks)]
        dist.all_gather(gathered, status)
        dist.all_gather(gathered_observed, observed)
        gathered_cpu = [int(t.cpu().item()) for t in gathered]
        gathered_observed_cpu = [[int(v) for v in t.cpu().tolist()] for t in gathered_observed]
        if rank == 0:
            print(
                f"[debug-ht-barrier] gathered_status={gathered_cpu} "
                f"gathered_observed={gathered_observed_cpu}",
                flush=True,
            )
        if any(v != 1 for v in gathered_cpu):
            raise RuntimeError(
                f"HT debug barrier failed: status={gathered_cpu} observed={gathered_observed_cpu}"
            )
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
        return

    if rank == 0 and local_world_size != args.nvl_group_size:
        print(
            f"[warning] LOCAL_WORLD_SIZE={local_world_size} differs from nvl_group_size={args.nvl_group_size}; "
            "rank ordering must still keep each NVL group CUDA-IPC local",
            flush=True,
        )

    torch.manual_seed(rank)
    test_main(
        args,
        num_sms,
        local_rank,
        local_world_size,
        num_ranks,
        num_rdma_groups,
        rank,
        buffer,
        group,
    )

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
