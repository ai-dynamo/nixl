#!/usr/bin/env python3
"""Mini NIXL storage benchmark to isolate descriptor/queue depth behavior.

Tests different block sizes and descriptor counts to find optimal settings.
Run inside the container with NIXL available.
"""

import time
import os
import sys
import struct

# Add kvbench to path
sys.path.insert(0, "/workspace/nixl/benchmark/kvbench")

from nixl._api import nixl_agent, nixl_agent_config
import torch
import numpy as np


def parse_size(s):
    s = s.strip().upper()
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def run_test(agent, backend_name, fd, buf, total_size, block_size, n_iters, warmup):
    """Run a single test configuration."""
    n_blocks = (total_size + block_size - 1) // block_size
    buf_addr = buf.data_ptr()
    dev_id = 0  # CPU

    # Create descriptor tuples
    file_tuples = []
    local_tuples = []
    for i in range(n_blocks):
        off = i * block_size
        chunk = min(block_size, total_size - off)
        file_tuples.append((off, chunk, fd))
        local_tuples.append((buf_addr + off, chunk, dev_id))

    file_descs = agent.get_xfer_descs(file_tuples, "FILE")
    local_descs = agent.get_xfer_descs(local_tuples, "DRAM")

    # Create transfer request ONCE (like nixlbench does)
    xfer = agent.initialize_xfer(
        "READ", local_descs, file_descs, agent.name,
        backends=[backend_name],
    )

    # Warmup
    for _ in range(warmup):
        status = agent.transfer(xfer)
        while status not in ("DONE", "ERR"):
            status = agent.check_xfer_state(xfer)
        if status == "ERR":
            print(f"  ERROR during warmup!")
            return -1

    # Timed iterations
    start = time.time()
    for _ in range(n_iters):
        status = agent.transfer(xfer)
        while status not in ("DONE", "ERR"):
            status = agent.check_xfer_state(xfer)
        if status == "ERR":
            print(f"  ERROR during iteration!")
            return -1
    elapsed = time.time() - start

    bw = (total_size * n_iters) / elapsed / (1024**3)
    avg_lat = elapsed / n_iters * 1000
    print(f"  blocks={n_blocks:5d}  block_size={block_size//1024:6d}K  "
          f"iters={n_iters:4d}  BW={bw:7.2f} GB/s  lat={avg_lat:8.2f} ms")
    return bw


def run_threaded_test(agent, backend_name, fd, buf, total_size, block_size, num_handles, n_iters, warmup):
    """Run test with multiple handles submitted from Python threads (GIL-released)."""
    from concurrent.futures import ThreadPoolExecutor
    buf_addr = buf.data_ptr()
    dev_id = 0
    align = max(block_size, 4096)
    chunk_per_handle = ((total_size // num_handles + align - 1) // align) * align

    xfers = []
    for h in range(num_handles):
        h_offset = h * chunk_per_handle
        h_size = min(chunk_per_handle, total_size - h_offset)
        if h_size <= 0:
            break
        n_blocks = (h_size + block_size - 1) // block_size
        file_tuples = [(h_offset + i * block_size, min(block_size, h_size - i * block_size), fd) for i in range(n_blocks)]
        local_tuples = [(buf_addr + h_offset + i * block_size, min(block_size, h_size - i * block_size), dev_id) for i in range(n_blocks)]
        file_descs = agent.get_xfer_descs(file_tuples, "FILE")
        local_descs = agent.get_xfer_descs(local_tuples, "DRAM")
        xfer = agent.initialize_xfer("READ", local_descs, file_descs, agent.name, backends=[backend_name])
        xfers.append(xfer)

    actual_handles = len(xfers)
    descs_per = (chunk_per_handle + block_size - 1) // block_size

    def submit_and_wait(xfer):
        status = agent.transfer(xfer)
        while status not in ("DONE", "ERR"):
            status = agent.check_xfer_state(xfer)
        return status

    # Warmup
    with ThreadPoolExecutor(max_workers=actual_handles) as pool:
        for _ in range(warmup):
            list(pool.map(submit_and_wait, xfers))

    # Timed
    start = time.time()
    with ThreadPoolExecutor(max_workers=actual_handles) as pool:
        for _ in range(n_iters):
            results = list(pool.map(submit_and_wait, xfers))
            if any(r == "ERR" for r in results):
                print("  ERROR!")
                return -1
    elapsed = time.time() - start

    bw = (total_size * n_iters) / elapsed / (1024**3)
    avg_lat = elapsed / n_iters * 1000
    print(f"  THREADED handles={actual_handles:3d}  descs/h={descs_per:5d}  block={block_size//1024:6d}K  "
          f"iters={n_iters:4d}  BW={bw:7.2f} GB/s  lat={avg_lat:8.2f} ms")
    return bw


def run_multi_handle_test(agent, backend_name, fd, buf, total_size, block_size, num_handles, n_iters, warmup, verbose=False):
    """Run test with multiple concurrent transfer handles (like nixlbench's multi-thread).
    
    With verbose=True, prints detailed per-phase timing for the first iteration.
    """
    buf_addr = buf.data_ptr()
    dev_id = 0

    align = max(block_size, 4096)
    chunk_per_handle = ((total_size // num_handles + align - 1) // align) * align

    # Phase 1: Create handles (measure setup time)
    t_setup_start = time.time()
    xfers = []
    for h in range(num_handles):
        h_offset = h * chunk_per_handle
        h_size = min(chunk_per_handle, total_size - h_offset)
        if h_size <= 0:
            break

        n_blocks = (h_size + block_size - 1) // block_size
        file_tuples = []
        local_tuples = []
        for i in range(n_blocks):
            off = i * block_size
            chunk = min(block_size, h_size - off)
            file_tuples.append((h_offset + off, chunk, fd))
            local_tuples.append((buf_addr + h_offset + off, chunk, dev_id))

        file_descs = agent.get_xfer_descs(file_tuples, "FILE")
        local_descs = agent.get_xfer_descs(local_tuples, "DRAM")
        xfer = agent.initialize_xfer("READ", local_descs, file_descs, agent.name, backends=[backend_name])
        xfers.append(xfer)
    t_setup = (time.time() - t_setup_start) * 1000

    actual_handles = len(xfers)
    descs_per = (chunk_per_handle + block_size - 1) // block_size

    if verbose:
        print(f"    SETUP: {actual_handles} handles x {descs_per} descs = {actual_handles*descs_per} total descs in {t_setup:.1f}ms")

    # Warmup
    for _ in range(warmup):
        pending = []
        for xfer in xfers:
            status = agent.transfer(xfer)
            if status not in ("DONE", "ERR"):
                pending.append(xfer)
        while pending:
            i = 0
            while i < len(pending):
                s = agent.check_xfer_state(pending[i])
                if s == "DONE":
                    pending[i] = pending[-1]
                    pending.pop()
                elif s == "ERR":
                    print("  ERROR during warmup!")
                    return -1
                else:
                    i += 1

    # Detailed timing for first iteration (when verbose)
    if verbose:
        # Single detailed iteration
        t_submit_start = time.time()
        pending = []
        submit_times = []
        for xfer in xfers:
            t0 = time.time()
            status = agent.transfer(xfer)
            submit_times.append((time.time() - t0) * 1000)
            if status not in ("DONE", "ERR"):
                pending.append(xfer)
            elif status == "DONE":
                submit_times[-1] = -submit_times[-1]  # negative = completed immediately
        t_submit = (time.time() - t_submit_start) * 1000

        t_poll_start = time.time()
        poll_count = 0
        first_done_time = None
        last_done_time = None
        done_count = 0
        while pending:
            i = 0
            while i < len(pending):
                s = agent.check_xfer_state(pending[i])
                if s == "DONE":
                    now = time.time()
                    if first_done_time is None:
                        first_done_time = now
                    last_done_time = now
                    done_count += 1
                    pending[i] = pending[-1]
                    pending.pop()
                elif s == "ERR":
                    print("  ERROR!")
                    return -1
                else:
                    i += 1
            poll_count += 1
        t_poll = (time.time() - t_poll_start) * 1000
        t_total_iter = t_submit + t_poll

        # Per-handle submit time stats
        submit_arr = [abs(t) for t in submit_times]
        immediate = sum(1 for t in submit_times if t < 0)
        print(f"    SUBMIT: {len(xfers)} handles in {t_submit:.2f}ms "
              f"(per-handle: min={min(submit_arr):.3f} max={max(submit_arr):.3f} avg={sum(submit_arr)/len(submit_arr):.3f}ms, "
              f"{immediate} completed immediately)")
        if first_done_time and last_done_time and t_poll_start:
            first_ms = (first_done_time - t_poll_start) * 1000
            last_ms = (last_done_time - t_poll_start) * 1000
            spread = last_ms - first_ms
            print(f"    POLL: {poll_count} rounds in {t_poll:.2f}ms "
                  f"(first_done={first_ms:.1f}ms last_done={last_ms:.1f}ms spread={spread:.1f}ms)")
        else:
            print(f"    POLL: {poll_count} rounds in {t_poll:.2f}ms")
        bw_iter = (total_size / t_total_iter * 1000) / (1024**3)
        print(f"    TOTAL ITER: {t_total_iter:.2f}ms = {bw_iter:.2f} GB/s")

    # Timed iterations (all)
    submit_total_us = 0
    poll_total_us = 0
    start = time.time()
    for _ in range(n_iters):
        t0 = time.time()
        pending = []
        for xfer in xfers:
            status = agent.transfer(xfer)
            if status not in ("DONE", "ERR"):
                pending.append(xfer)
        t1 = time.time()
        submit_total_us += (t1 - t0) * 1e6

        while pending:
            i = 0
            while i < len(pending):
                s = agent.check_xfer_state(pending[i])
                if s == "DONE":
                    pending[i] = pending[-1]
                    pending.pop()
                elif s == "ERR":
                    print("  ERROR during iteration!")
                    return -1
                else:
                    i += 1
        poll_total_us += (time.time() - t1) * 1e6
    elapsed = time.time() - start

    bw = (total_size * n_iters) / elapsed / (1024**3)
    avg_lat = elapsed / n_iters * 1000
    avg_submit = submit_total_us / n_iters / 1000
    avg_poll = poll_total_us / n_iters / 1000
    pct_submit = submit_total_us / (elapsed * 1e6) * 100
    pct_poll = poll_total_us / (elapsed * 1e6) * 100

    print(f"  handles={actual_handles:3d}  descs/h={descs_per:5d}  block={block_size//1024:6d}K  "
          f"iters={n_iters:4d}  BW={bw:7.2f} GB/s  lat={avg_lat:8.2f} ms  "
          f"submit={avg_submit:.2f}ms({pct_submit:.0f}%)  poll={avg_poll:.2f}ms({pct_poll:.0f}%)")
    return bw


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-path", required=True)
    parser.add_argument("--total-size", default="1G")
    parser.add_argument("--block-sizes", default="1M,4M,16M,64M,256M,1G")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--posix-api", default="auto", choices=["auto", "aio", "uring"])
    parser.add_argument("--num-handles", type=int, default=0,
                        help="Multi-handle mode: 0=skip, >0=test with N concurrent handles")
    args = parser.parse_args()

    total_size = parse_size(args.total_size)
    block_sizes = [parse_size(s) for s in args.block_sizes.split(",")]

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    print(f"[Rank {rank}] Mini NIXL Storage Benchmark")
    print(f"  total_size={total_size}, direct_io={args.direct_io}, posix_api={args.posix_api}")

    # Create agent
    config = nixl_agent_config(backends=[])
    agent = nixl_agent(f"{rank}", config)

    # Create backend with params
    backend_params = {}
    if args.posix_api == "uring":
        backend_params = {"use_uring": "true", "use_aio": "false", "use_posix_aio": "false"}
    elif args.posix_api == "aio":
        backend_params = {"use_aio": "true", "use_uring": "false", "use_posix_aio": "false"}
    
    agent.create_backend("POSIX", backend_params)

    # Create file
    storage_dir = os.path.join(args.storage_path, f"rank_{rank}")
    os.makedirs(storage_dir, exist_ok=True)
    file_path = os.path.join(storage_dir, "test.bin")

    # Write test file
    if not os.path.exists(file_path) or os.path.getsize(file_path) < total_size:
        print(f"  Creating test file: {file_path} ({total_size} bytes)")
        with open(file_path, "wb") as f:
            chunk = bytes([rank % 256]) * min(8*1024*1024, total_size)
            written = 0
            while written < total_size:
                to_write = min(len(chunk), total_size - written)
                f.write(chunk[:to_write])
                written += to_write
            f.flush()
            os.fsync(f.fileno())

    # Open with O_DIRECT if requested
    flags = os.O_RDWR
    if args.direct_io:
        flags |= os.O_DIRECT
    fd = os.open(file_path, flags)

    # Register file with NIXL
    reg_list = [(0, total_size, fd, file_path)]
    agent.register_memory(reg_list, "FILE", backends=["POSIX"])

    # Allocate aligned CPU buffer
    buf_size = ((total_size + 4095) // 4096) * 4096
    buf = torch.zeros(buf_size, dtype=torch.int8, device="cpu")
    reg_descs = agent.get_reg_descs(buf)
    agent.register_memory(reg_descs)
    agent.register_memory(reg_descs, backends=["POSIX"])

    # Run single-handle block size sweep
    print(f"\n  === Block Size Sweep (total_size={total_size//1024//1024}MB) ===")
    for bs in block_sizes:
        if bs > total_size:
            continue
        run_test(agent, "POSIX", fd, buf, total_size, bs, args.iters, args.warmup)

    # Run multi-handle test if requested
    if args.num_handles > 0:
        print(f"\n  === Multi-Handle Sweep (num_handles sweep, block=1MB) ===")
        for nh in [1, 2, 4, 8, 16]:
            if nh > args.num_handles:
                break
            is_first = (nh == 1)
            run_multi_handle_test(
                agent, "POSIX", fd, buf, total_size,
                block_size=1048576, num_handles=nh,
                n_iters=args.iters, warmup=args.warmup,
                verbose=is_first,
            )

        # Detailed verbose for the 8-handle case
        print(f"\n  === Detailed 8-Handle Analysis ===")
        run_multi_handle_test(
            agent, "POSIX", fd, buf, total_size,
            block_size=1048576, num_handles=min(args.num_handles, 8),
            n_iters=args.iters, warmup=args.warmup,
            verbose=True,
        )

        # Threaded submission (the key experiment)
        print(f"\n  === THREADED Multi-Handle Sweep (block=1MB) ===")
        for nh in [1, 2, 4, 8, 16]:
            if nh > args.num_handles:
                break
            run_threaded_test(
                agent, "POSIX", fd, buf, total_size,
                block_size=1048576, num_handles=nh,
                n_iters=args.iters, warmup=args.warmup,
            )

        # Threaded with different block sizes
        print(f"\n  === THREADED Block Sweep (handles={min(args.num_handles, 8)}) ===")
        nh = min(args.num_handles, 8)
        for bs_str in ["1M", "4M", "16M", "64M", "128M"]:
            bs = parse_size(bs_str)
            run_threaded_test(
                agent, "POSIX", fd, buf, total_size,
                block_size=bs, num_handles=nh,
                n_iters=args.iters, warmup=args.warmup,
            )

    # Cleanup
    os.close(fd)
    print(f"\n[Rank {rank}] Done")


if __name__ == "__main__":
    main()
