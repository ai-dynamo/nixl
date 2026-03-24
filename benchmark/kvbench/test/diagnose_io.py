#!/usr/bin/env python3
"""Diagnostic microbenchmark to isolate storage I/O bottlenecks.

Tests each layer independently:
1. Raw Python read() — baseline OS performance
2. Single NIXL transfer — measures NIXL overhead
3. NIXL postXferReq return time — is it blocking?
4. Multiple NIXL handles serial — current kvbench pattern
5. Multiple NIXL handles via parallel_transfer — new C++ threads
6. nixlbench reference (if available)

Run inside container with NIXL.
"""

import time
import os
import sys
import torch

sys.path.insert(0, "/workspace/nixl/benchmark/kvbench")
from nixl._api import nixl_agent, nixl_agent_config


def parse_size(s):
    s = s.strip().upper()
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def setup_file(path, size):
    """Create test file if needed."""
    if os.path.exists(path) and os.path.getsize(path) >= size:
        return
    print(f"  Creating {path} ({size/(1024**2):.0f}MB)...")
    with open(path, "wb") as f:
        chunk = bytes([42]) * min(8 * 1024 * 1024, size)
        written = 0
        while written < size:
            f.write(chunk[:min(len(chunk), size - written)])
            written += len(chunk)
        f.flush()
        os.fsync(f.fileno())


def test_raw_read(path, size, n_iters):
    """Layer 1: Raw Python read() with O_DIRECT."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    # Aligned buffer for O_DIRECT
    buf = bytearray(size + 4096)
    offset = (4096 - (id(buf) % 4096)) % 4096

    # Warmup
    for _ in range(3):
        os.lseek(fd, 0, os.SEEK_SET)
        os.readv(fd, [memoryview(buf)[offset:offset + size]])

    t0 = time.time()
    for _ in range(n_iters):
        os.lseek(fd, 0, os.SEEK_SET)
        os.readv(fd, [memoryview(buf)[offset:offset + size]])
    elapsed = time.time() - t0
    os.close(fd)

    bw = (size * n_iters) / elapsed / (1024**3)
    lat = elapsed / n_iters * 1000
    print(f"  Layer 1 (raw read):        BW={bw:7.2f} GB/s  lat={lat:.1f}ms")
    return bw


def test_nixl_single(agent, fd, buf, size, n_iters, block_size):
    """Layer 2: Single NIXL transfer handle."""
    buf_addr = buf.data_ptr()
    n_blocks = size // block_size
    file_t = [(i * block_size, block_size, fd) for i in range(n_blocks)]
    local_t = [(buf_addr + i * block_size, block_size, 0) for i in range(n_blocks)]
    f_descs = agent.get_xfer_descs(file_t, "FILE")
    l_descs = agent.get_xfer_descs(local_t, "DRAM")
    xfer = agent.initialize_xfer("READ", l_descs, f_descs, agent.name, backends=["POSIX"])

    # Warmup
    for _ in range(3):
        s = agent.transfer(xfer)
        while s not in ("DONE", "ERR"):
            s = agent.check_xfer_state(xfer)

    # Measure: how long does transfer() take vs check_xfer_state()?
    submit_times = []
    poll_times = []
    for _ in range(n_iters):
        t0 = time.time()
        s = agent.transfer(xfer)
        t1 = time.time()
        while s not in ("DONE", "ERR"):
            s = agent.check_xfer_state(xfer)
        t2 = time.time()
        submit_times.append((t1 - t0) * 1000)
        poll_times.append((t2 - t1) * 1000)

    total_ms = sum(submit_times) + sum(poll_times)
    bw = (size * n_iters) / (total_ms / 1000) / (1024**3)
    avg_submit = sum(submit_times) / len(submit_times)
    avg_poll = sum(poll_times) / len(poll_times)
    pct_submit = sum(submit_times) / total_ms * 100

    print(f"  Layer 2 (NIXL single):     BW={bw:7.2f} GB/s  submit={avg_submit:.1f}ms({pct_submit:.0f}%)  poll={avg_poll:.1f}ms")
    print(f"    → transfer() is {'BLOCKING' if pct_submit > 90 else 'ASYNC'} ({avg_submit:.1f}ms per call)")
    return xfer, bw


def test_nixl_serial_multi(agent, fd, buf, buf_addr, size, n_iters, block_size, num_handles):
    """Layer 3: Multiple handles, serial submission."""
    chunk_per_h = size // num_handles
    xfers = []
    for h in range(num_handles):
        h_off = h * chunk_per_h
        n_blocks = chunk_per_h // block_size
        file_t = [(h_off + i * block_size, block_size, fd) for i in range(n_blocks)]
        local_t = [(buf_addr + h_off + i * block_size, block_size, 0) for i in range(n_blocks)]
        f_descs = agent.get_xfer_descs(file_t, "FILE")
        l_descs = agent.get_xfer_descs(local_t, "DRAM")
        xfer = agent.initialize_xfer("READ", l_descs, f_descs, agent.name, backends=["POSIX"])
        xfers.append(xfer)

    # Warmup
    for _ in range(3):
        for x in xfers:
            s = agent.transfer(x)
            while s not in ("DONE", "ERR"):
                s = agent.check_xfer_state(x)

    # Timed
    t0 = time.time()
    for _ in range(n_iters):
        for x in xfers:
            s = agent.transfer(x)
            while s not in ("DONE", "ERR"):
                s = agent.check_xfer_state(x)
    elapsed = time.time() - t0
    bw = (size * n_iters) / elapsed / (1024**3)
    lat = elapsed / n_iters * 1000
    print(f"  Layer 3 (serial {num_handles:2d}h):      BW={bw:7.2f} GB/s  lat={lat:.1f}ms")
    return xfers, bw


def test_nixl_parallel(agent, xfers, size, n_iters, num_handles):
    """Layer 4: Multiple handles, C++ threaded submission."""
    has_parallel = hasattr(agent, 'parallel_transfer')
    if not has_parallel:
        print(f"  Layer 4 (parallel {num_handles:2d}h):    SKIPPED (parallel_transfer not available)")
        return 0

    # Warmup
    for _ in range(3):
        agent.parallel_transfer(xfers, num_threads=num_handles)

    # Timed
    t0 = time.time()
    for _ in range(n_iters):
        agent.parallel_transfer(xfers, num_threads=num_handles)
    elapsed = time.time() - t0
    bw = (size * n_iters) / elapsed / (1024**3)
    lat = elapsed / n_iters * 1000
    print(f"  Layer 4 (parallel {num_handles:2d}h):    BW={bw:7.2f} GB/s  lat={lat:.1f}ms")
    return bw


def test_nixl_fire_all_then_wait(agent, xfers, size, n_iters, num_handles):
    """Layer 5: Fire all handles (non-blocking), then poll all."""
    # Warmup
    for _ in range(3):
        pending = []
        for x in xfers:
            s = agent.transfer(x)
            if s not in ("DONE", "ERR"):
                pending.append(x)
        while pending:
            i = 0
            while i < len(pending):
                s = agent.check_xfer_state(pending[i])
                if s == "DONE":
                    pending[i] = pending[-1]
                    pending.pop()
                else:
                    i += 1

    # Measure submit phase vs poll phase
    total_submit = 0
    total_poll = 0
    for _ in range(n_iters):
        t0 = time.time()
        pending = []
        for x in xfers:
            s = agent.transfer(x)
            if s not in ("DONE", "ERR"):
                pending.append(x)
        t1 = time.time()
        while pending:
            i = 0
            while i < len(pending):
                s = agent.check_xfer_state(pending[i])
                if s == "DONE":
                    pending[i] = pending[-1]
                    pending.pop()
                else:
                    i += 1
        t2 = time.time()
        total_submit += t1 - t0
        total_poll += t2 - t1

    elapsed = total_submit + total_poll
    bw = (size * n_iters) / elapsed / (1024**3)
    pct_submit = total_submit / elapsed * 100
    print(f"  Layer 5 (fire-all {num_handles:2d}h):   BW={bw:7.2f} GB/s  submit={pct_submit:.0f}%  poll={100-pct_submit:.0f}%")
    return bw


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-path", required=True)
    parser.add_argument("--size", default="1G")
    parser.add_argument("--block-size", default="1M")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--handles", type=int, default=8)
    parser.add_argument("--posix-api", default="uring", choices=["auto", "aio", "uring"])
    args = parser.parse_args()

    size = parse_size(args.size)
    block_size = parse_size(args.block_size)
    n_handles = args.handles

    print("=" * 70)
    print(f"STORAGE I/O DIAGNOSTIC — {size/(1024**3):.0f}GB, {block_size/(1024**2):.0f}MB blocks, {n_handles} handles")
    print("=" * 70)

    # Setup
    os.makedirs(args.storage_path, exist_ok=True)
    fpath = os.path.join(args.storage_path, "diag_test.bin")
    setup_file(fpath, size)

    config = nixl_agent_config(backends=[])
    agent = nixl_agent("0", config)
    params = {}
    if args.posix_api == "uring":
        params = {"use_uring": "true", "use_aio": "false"}
    agent.create_backend("POSIX", params)

    fd = os.open(fpath, os.O_RDWR | os.O_DIRECT)
    agent.register_memory([(0, size, fd, fpath)], "FILE", backends=["POSIX"])

    buf_size = ((size + 4095) // 4096) * 4096
    buf = torch.zeros(buf_size, dtype=torch.int8, device="cpu")
    reg = agent.get_reg_descs(buf)
    agent.register_memory(reg)
    agent.register_memory(reg, backends=["POSIX"])
    buf_addr = buf.data_ptr()

    print(f"\n  Target: ~45 GB/s (nixlbench reference)")
    print()

    # Layer 1: Raw OS read
    try:
        test_raw_read(fpath, size, args.iters)
    except Exception as e:
        print(f"  Layer 1: FAILED ({e})")

    # Layer 2: Single NIXL handle
    _, _ = test_nixl_single(agent, fd, buf, size, args.iters, block_size)

    # Layer 3: Serial multi-handle
    xfers, _ = test_nixl_serial_multi(agent, fd, buf, buf_addr, size, args.iters, block_size, n_handles)

    # Layer 4: C++ parallel
    test_nixl_parallel(agent, xfers, size, args.iters, n_handles)

    # Layer 5: Fire-all-then-wait
    test_nixl_fire_all_then_wait(agent, xfers, size, args.iters, n_handles)

    # Summary
    print()
    print("=" * 70)
    print("DIAGNOSIS:")
    print("  If Layer 1 >> Layer 2: NIXL has overhead")
    print("  If Layer 2 submit is BLOCKING: postXferReq waits for I/O completion")
    print("  If Layer 3 == Layer 4: C++ threads don't help (kernel serializes)")
    print("  If Layer 4 >> Layer 3: C++ threads work! Use parallel_transfer()")
    print("  If Layer 5 submit% > 90%: transfer() itself blocks (not async)")
    print("=" * 70)

    os.close(fd)


if __name__ == "__main__":
    main()
