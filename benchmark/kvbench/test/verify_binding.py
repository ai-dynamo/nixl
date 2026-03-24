#!/usr/bin/env python3
"""Verify parallelStorageTransfer matches nixlbench ~49 GB/s.

Key insight: nixlbench distributes each thread's 256 SQEs round-robin
across all 16 file descriptors. This utilizes all NFS connections.
"""
import os, sys, time

C = "/mnt/nixl/_bindings.cpython-312-x86_64-linux-gnu.so"
if os.path.exists(C):
    import importlib.util
    s = importlib.util.spec_from_file_location("nixl._bindings", C)
    m = importlib.util.module_from_spec(s)
    sys.modules["nixl._bindings"] = m
    s.loader.exec_module(m)
    print("Custom bindings loaded")

sys.path.insert(0, "/workspace/nixl/benchmark/kvbench")
from nixl._api import nixl_agent, nixl_agent_config


def ps(s):
    s = s.strip().upper()
    m = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(s[:-1]) * m[s[-1]]) if s[-1] in m else int(s)


import argparse
p = argparse.ArgumentParser()
p.add_argument("--storage-path", required=True)
p.add_argument("--num-files", type=int, default=16)
p.add_argument("--num-threads", type=int, default=16)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--block-size", type=str, default="1M")
p.add_argument("--file-size", type=str, default="1G")
p.add_argument("--iters", type=int, default=100)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--api", default="uring", choices=["aio", "uring"])
a = p.parse_args()

rank = int(os.environ.get("SLURM_PROCID", "0"))
bs = ps(a.block_size)
fs = ps(a.file_size)
pgsz = os.sysconf("SC_PAGE_SIZE")
fs = ((fs + pgsz - 1) // pgsz) * pgsz
nf = a.num_files
nt = a.num_threads

print(f"[Rank {rank}] files={nf} threads={nt} batch={a.batch_size} block={bs // 1024}K")
print(f"  file={fs // (1024**2)}MB iters={a.iters} warmup={a.warmup} api={a.api}")

ag = nixl_agent(f"v{rank}", nixl_agent_config(backends=[]))
bp = {"use_uring": "true", "use_aio": "false"} if a.api == "uring" else {}
ag.create_backend("POSIX", bp)
bh = ag.backends["POSIX"]
print(f"  backend={bh}")

sd = os.path.join(a.storage_path, f"rank_{rank}")
os.makedirs(sd, exist_ok=True)

fds = []
for i in range(nf):
    fp = os.path.join(sd, f"s{i}.bin")
    if not os.path.exists(fp) or os.path.getsize(fp) < fs:
        print(f"  Creating {fp} ({fs // (1024**2)}MB)...")
        buf = bytes([i % 256]) * min(8 * 1024 * 1024, fs)
        with open(fp, "wb") as f:
            w = 0
            while w < fs:
                f.write(buf[:min(len(buf), fs - w)])
                w += len(buf)
            f.flush()
            os.fsync(f.fileno())
    fd = os.open(fp, os.O_RDWR | os.O_DIRECT)
    fds.append(fd)
    ag.register_memory([(0, fs, fd, fp)], "FILE", backends=["POSIX"])

bl = [bh]


def mkd_single_fd(bsz, num_t=None):
    """OLD pattern: each thread reads from ONE file (35 GB/s)."""
    if num_t is None:
        num_t = nt
    r = []
    for t in range(num_t):
        fd = fds[t % nf]
        r.append([(j * bs % fs, bs, fd) for j in range(bsz)])
    return r


def mkd_round_robin(bsz, num_t=None):
    """NEW pattern matching nixlbench: round-robin across ALL fds (target ~49 GB/s).

    Replicates exchangeIOV logic where fd_idx and file_offset carry across threads.
    """
    if num_t is None:
        num_t = nt
    r = []
    fd_idx = 0
    file_offset = 0
    for t in range(num_t):
        descs = []
        for j in range(bsz):
            descs.append((file_offset, bs, fds[fd_idx]))
            fd_idx += 1
            if fd_idx >= nf:
                file_offset += bs
                fd_idx = 0
                if file_offset + bs > fs:
                    file_offset = 0
        r.append(descs)
        file_offset += bs
        if file_offset + bs > fs:
            file_offset = 0
    return r


print(f"\n=== OLD: single-fd per thread ({nt}t x {a.batch_size} descs) ===")
try:
    bw = ag.agent.parallelStorageTransfer(
        mkd_single_fd(a.batch_size), ag.name, bl,
        a.iters, a.warmup, nt)
    print(f"  single-fd:    {bw:.2f} GB/s")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

print(f"\n=== NEW: round-robin fds per thread ({nt}t x {a.batch_size} descs) ===")
try:
    bw = ag.agent.parallelStorageTransfer(
        mkd_round_robin(a.batch_size), ag.name, bl,
        a.iters, a.warmup, nt)
    print(f"  round-robin:  {bw:.2f} GB/s")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

print(f"\n=== Iodepth sweep - round-robin ({nt}t) ===")
for b in [1, 4, 16, 64, 128, 256]:
    try:
        bw = ag.agent.parallelStorageTransfer(
            mkd_round_robin(b), ag.name, bl,
            max(a.iters, 20), a.warmup, nt)
        print(f"  iodepth={b:4d} xfer/t={b * bs // (1024**2):4d}MB BW={bw:7.2f} GB/s")
    except Exception as e:
        print(f"  iodepth={b:4d} ERROR: {e}")

for fd in fds:
    os.close(fd)
print(f"\n[Rank {rank}] Done")
