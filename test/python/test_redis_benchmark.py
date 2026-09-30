#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL Redis Plugin — end-to-end nixlbench performance benchmark script.

Builds NIXL (with static REDIS plugin) and nixlbench into a persistent
directory, optionally starts a Redis container with host networking and
io-threads enabled, pins both Redis and nixlbench to the same NUMA node,
runs a full WRITE/READ sweep across block sizes and thread counts, and emits
a Markdown report with results tables, server info, and PCIe link speeds.

The build directory is persistent across runs: a second invocation reuses
the existing build and only recompiles changed files.

Usage:
    python3 test/python/test_redis_benchmark.py [OPTIONS]

Options:
    --bench-dir PATH        Persistent build/install root  [default: /tmp/nixl-redis-bench]
    --rebuild               Force a clean rebuild (wipes and recreates build dirs)
    --start-redis           Start a Redis Docker container before benchmarking (default)
    --no-start-redis        Skip container management; assume Redis is already running
    --redis-container NAME  Docker container name  [default: nixl-redis-bench]
    --redis-io-threads N    Redis --io-threads value  [default: auto = physical cores on NUMA node, ≤16]
    --redis-host HOST       Redis host  [default: 127.0.0.1]
    --redis-port PORT       Redis port  [default: 6379]
    --redis-pool-size N     REDIS_POOL_SIZE passed to the plugin  [default: 8]
    --numa-node N           NUMA node to pin Redis and nixlbench to (-1 = auto)  [default: -1]
    --warmup-iter N         Warmup iterations per run  [default: 32]
    --num-iter N            Measured iterations per run  [default: 208]
    --total-buffer-size N   Total buffer size in bytes  [default: 67108864]
    --output PATH           Write Markdown report to file  [default: stdout]
    --skip-build            Skip build phase, use existing binaries
    --help                  Show this message and exit
"""

import argparse
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo layout (resolved relative to this script's location)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # test/python/… → test/ → repo root
NIXLBENCH_SRC = REPO_ROOT / "benchmark" / "nixlbench"

# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

BLOCK_SIZES_KB = [128, 256, 512, 1024, 2048, 4096]
THREAD_COUNTS = [1, 4, 8, 16]
OPERATIONS = ["WRITE", "READ"]

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

# Directories that are likely to contain meson, pybind11-config, and nvcc.
_TOOL_PATH_HINTS = [
    "/images/nixl/venv-vllm-cu/bin",
    "/images/nixl/pr/venv-sglang-nixl/bin",
    "/usr/local/cuda/bin",
    "/usr/local/bin",
    "/usr/bin",
]


def _extended_path() -> str:
    """Return a PATH that includes CUDA and known venv bin dirs."""
    existing = os.environ.get("PATH", "")
    extra = ":".join(p for p in _TOOL_PATH_HINTS if Path(p).is_dir())
    return f"{extra}:{existing}" if extra else existing


def _find_tool(name: str) -> Optional[str]:
    """Find an executable by searching _TOOL_PATH_HINTS then $PATH."""
    for directory in _TOOL_PATH_HINTS:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def _run_build(cmd: List[str], cwd: Optional[Path] = None, label: str = "") -> None:
    """Run a build command, streaming output; raise on failure."""
    env = os.environ.copy()
    env["PATH"] = _extended_path()
    display = label or " ".join(cmd[:3])
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f"\nBuild failed ({display}), exit code {result.returncode}")


def build_nixl(
    meson: str,
    nixl_build_dir: Path,
    nixl_install_dir: Path,
    rebuild: bool,
) -> None:
    if rebuild and nixl_build_dir.exists():
        print(f"  Wiping {nixl_build_dir} ...", file=sys.stderr)
        shutil.rmtree(nixl_build_dir)

    if not nixl_build_dir.exists():
        print(f"  Configuring NIXL in {nixl_build_dir} ...", file=sys.stderr)
        _run_build(
            [
                meson, "setup", str(nixl_build_dir),
                f"--prefix={nixl_install_dir}",
                "-Denable_plugins=REDIS",
                "-Dstatic_plugins=REDIS",
                "--buildtype=release",
            ],
            label="meson setup nixl",
        )
    else:
        print(f"  NIXL build dir exists — skipping setup ({nixl_build_dir})", file=sys.stderr)

    print("  Compiling NIXL ...", file=sys.stderr)
    _run_build([meson, "compile", "-C", str(nixl_build_dir)], label="meson compile nixl")

    print("  Installing NIXL ...", file=sys.stderr)
    _run_build([meson, "install", "-C", str(nixl_build_dir)], label="meson install nixl")


def build_nixlbench(
    meson: str,
    nixlbench_build_dir: Path,
    nixl_install_dir: Path,
    rebuild: bool,
) -> None:
    if rebuild and nixlbench_build_dir.exists():
        print(f"  Wiping {nixlbench_build_dir} ...", file=sys.stderr)
        shutil.rmtree(nixlbench_build_dir)

    if not nixlbench_build_dir.exists():
        print(f"  Configuring nixlbench in {nixlbench_build_dir} ...", file=sys.stderr)
        _run_build(
            [
                meson, "setup", str(nixlbench_build_dir),
                str(NIXLBENCH_SRC),
                f"-Dnixl_path={nixl_install_dir}",
            ],
            label="meson setup nixlbench",
        )
    else:
        print(
            f"  nixlbench build dir exists — skipping setup ({nixlbench_build_dir})",
            file=sys.stderr,
        )

    print("  Compiling nixlbench ...", file=sys.stderr)
    _run_build([meson, "compile", "-C", str(nixlbench_build_dir)], label="meson compile nixlbench")


def ensure_built(bench_dir: Path, rebuild: bool) -> Tuple[Path, Path]:
    """Build NIXL and nixlbench into bench_dir; return (nixlbench_bin, nixl_install_dir)."""
    nixl_build_dir = bench_dir / "nixl-build"
    nixl_install_dir = bench_dir / "nixl-install"
    nixlbench_build_dir = bench_dir / "nixlbench-build"
    nixlbench_bin = nixlbench_build_dir / "nixlbench"

    bench_dir.mkdir(parents=True, exist_ok=True)
    nixl_install_dir.mkdir(parents=True, exist_ok=True)

    meson = _find_tool("meson")
    if not meson:
        sys.exit(
            "ERROR: meson not found. Add the directory containing meson to PATH "
            "or install it (pip install meson)."
        )
    print(f"  meson: {meson}", file=sys.stderr)

    build_nixl(meson, nixl_build_dir, nixl_install_dir, rebuild)
    build_nixlbench(meson, nixlbench_build_dir, nixl_install_dir, rebuild)

    if not nixlbench_bin.is_file():
        sys.exit(f"ERROR: nixlbench binary not found after build: {nixlbench_bin}")

    return nixlbench_bin, nixl_install_dir


# ---------------------------------------------------------------------------
# NUMA helpers
# ---------------------------------------------------------------------------

def _numa_node_dirs() -> List[Path]:
    return sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))


def detect_numa_nodes() -> List[int]:
    return [int(p.name[4:]) for p in _numa_node_dirs()]


def numa_node_cpulist(node: int) -> str:
    try:
        return Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except OSError:
        return ""


def numa_node_free_kib(node: int) -> int:
    try:
        text = Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
        m = re.search(r"MemFree:\s+(\d+)", text)
        return int(m.group(1)) if m else 0
    except OSError:
        return 0


def numa_node_total_kib(node: int) -> int:
    try:
        text = Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
        m = re.search(r"MemTotal:\s+(\d+)", text)
        return int(m.group(1)) if m else 0
    except OSError:
        return 0


def _parse_cpulist(cpulist: str) -> int:
    """Count logical CPUs in a cpulist string (e.g. '0-15,32-47' → 32)."""
    count = 0
    for part in cpulist.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                count += int(hi) - int(lo) + 1
            except ValueError:
                pass
        elif part.isdigit():
            count += 1
    return count


def numa_node_cpu_count(node: int) -> int:
    return _parse_cpulist(numa_node_cpulist(node))


def detect_best_numa_node() -> int:
    """Return the NUMA node with the most total memory.

    Total memory is a proxy for memory bandwidth: more DIMMs populate more
    memory channels, raising the theoretical DRAM bandwidth for that node.
    Ties are broken by free memory so we also avoid a nearly-full node.
    """
    nodes = detect_numa_nodes()
    if not nodes:
        return 0
    return max(nodes, key=lambda n: (numa_node_total_kib(n), numa_node_free_kib(n)))


def numactl_prefix(node: int) -> List[str]:
    """Return a numactl prefix list, or empty if numactl is not available."""
    numactl = shutil.which("numactl")
    if not numactl or node < 0:
        return []
    return [numactl, f"--cpunodebind={node}", f"--membind={node}"]


# ---------------------------------------------------------------------------
# PCIe helpers
# ---------------------------------------------------------------------------

@dataclass
class PcieDevice:
    address: str
    description: str
    numa_node: int          # -1 = unknown
    link_speed: str         # e.g. "16GT/s"
    link_width: int         # number of lanes, 0 = unknown
    max_bw_gbps: float      # theoretical unidirectional max in GB/s


def _pcie_bw_gbps(speed: str, width: int) -> float:
    """Theoretical unidirectional bandwidth in GB/s from link speed and width."""
    rates = {
        "2.5GT/s": 2.5, "5GT/s": 5.0, "8GT/s": 8.0,
        "16GT/s": 16.0, "32GT/s": 32.0, "64GT/s": 64.0,
    }
    gts = rates.get(speed, 0.0)
    if gts == 0.0 or width == 0:
        return 0.0
    # Gen 1/2 use 8b/10b encoding (80% efficiency); Gen 3+ use 128b/130b
    encoding = 0.8 if gts <= 5.0 else (128 / 130)
    return gts * width * encoding / 8  # convert Gb/s → GB/s


def _pcie_link_info(address: str) -> Tuple[str, int]:
    """Parse current link speed and width for a PCIe device via lspci."""
    try:
        r = subprocess.run(
            ["lspci", "-s", address, "-vvv"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "", 0

    cap_speed, cap_width = "", 0
    sta_speed, sta_width = "", 0
    for line in r.stdout.splitlines():
        m = re.search(r"LnkSta:.*?Speed (\S+),\s*Width x(\d+)", line)
        if m:
            sta_speed, sta_width = m.group(1), int(m.group(2))
        m = re.search(r"LnkCap:.*?Speed (\S+),\s*Width x(\d+)", line)
        if m:
            cap_speed, cap_width = m.group(1), int(m.group(2))

    # Prefer LnkSta (actual negotiated speed) over LnkCap (maximum capable)
    return (sta_speed, sta_width) if sta_speed else (cap_speed, cap_width)


def _pcie_numa_node(address: str) -> int:
    try:
        return int(
            Path(f"/sys/bus/pci/devices/0000:{address}/numa_node").read_text().strip()
        )
    except (OSError, ValueError):
        return -1


_PCIE_INTERESTING_CLASSES = [
    "Network", "Ethernet", "InfiniBand", "Non-Volatile", "VGA", "3D", "Display",
]
_PCIE_INTERESTING_VENDORS = ["nvidia", "mellanox", "broadcom"]


def collect_pcie_devices() -> List[PcieDevice]:
    """Return key PCIe devices with NUMA affinity and link speed info."""
    try:
        r = subprocess.run(
            ["lspci", "-vmm"], capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices: List[PcieDevice] = []
    current: Dict[str, str] = {}

    for raw_line in r.stdout.splitlines() + [""]:
        line = raw_line.strip()
        if line:
            key, _, val = line.partition(":")
            current[key.strip()] = val.strip()
            continue

        if not current:
            continue

        addr = current.get("Slot", "")
        class_name = current.get("Class", "")
        vendor = current.get("Vendor", "")
        device_name = current.get("Device", "")
        vendor_lower = vendor.lower()

        interesting = any(kw in class_name for kw in _PCIE_INTERESTING_CLASSES) or \
                      any(v in vendor_lower for v in _PCIE_INTERESTING_VENDORS)

        if interesting and addr:
            speed, width = _pcie_link_info(addr)
            devices.append(PcieDevice(
                address=addr,
                description=f"{vendor} {device_name}".strip(),
                numa_node=_pcie_numa_node(addr),
                link_speed=speed or "—",
                link_width=width,
                max_bw_gbps=_pcie_bw_gbps(speed, width),
            ))

        current = {}

    return devices[:20]


# ---------------------------------------------------------------------------
# Redis container management
# ---------------------------------------------------------------------------

def _redis_container_running(container: str) -> bool:
    r = subprocess.run(
        ["docker", "inspect", "--format={{.State.Running}}", container],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def start_redis_container(
    container: str,
    port: int,
    io_threads: int,
    numa_node: int,
) -> bool:
    """Start a Redis container with host networking, io-threads, and NUMA pinning.

    Returns True if this call started the container (caller should stop it),
    False if it was already running (caller must not stop it).
    """
    if _redis_container_running(container):
        print(
            f"  Redis container '{container}' is already running — skipping start.",
            file=sys.stderr,
        )
        return False

    # --network=host: container shares the host network stack directly,
    # eliminating the Docker bridge veth overhead present with -p port mapping.
    cmd = [
        "docker", "run", "--detach", "--rm",
        "--name", container,
        "--network=host",
    ]

    # Pin container to the chosen NUMA node so Redis memory allocations and
    # socket interrupts stay on the same NUMA domain as nixlbench.
    if numa_node >= 0:
        cpulist = numa_node_cpulist(numa_node)
        if cpulist:
            cmd += ["--cpuset-cpus", cpulist, "--cpuset-mems", str(numa_node)]

    cmd += [
        "redis:7-alpine",
        "--io-threads", str(io_threads),
        "--io-threads-do-reads", "yes",
        # Disable output-buffer soft/hard limits for normal clients so that large
        # GET replies (2MB+) are never truncated or disconnected mid-transfer.
        "--client-output-buffer-limit", "normal 0 0 0",
    ]

    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ERROR: failed to start Redis container:\n{result.stderr.strip()}")

    # Wait up to 15 s for the server to accept connections.
    for _ in range(30):
        ping = subprocess.run(
            ["docker", "exec", container, "redis-cli", "-p", str(port), "PING"],
            capture_output=True, text=True,
        )
        if ping.stdout.strip() == "PONG":
            numa_label = f"node {numa_node}" if numa_node >= 0 else "any node"
            print(
                f"  Redis ready (io-threads={io_threads}, host network, NUMA {numa_label}).",
                file=sys.stderr,
            )
            return True
        time.sleep(0.5)

    sys.exit(f"ERROR: Redis container '{container}' did not become ready after 15 s.")


def stop_redis_container(container: str) -> None:
    print(f"  Stopping Redis container '{container}' ...", file=sys.stderr)
    subprocess.run(["docker", "stop", container], capture_output=True)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    op: str
    block_size_b: int
    num_threads: int
    bw_gbps: float
    avg_lat_us: float
    avg_tx_us: float
    p99_tx_us: float
    avg_prep_us: float
    p99_prep_us: float
    avg_post_us: float
    p99_post_us: float


@dataclass
class ServerInfo:
    hostname: str = ""
    os_name: str = ""
    kernel: str = ""
    cpu_model: str = ""
    cpu_sockets: str = ""
    cpu_cores_per_socket: str = ""
    cpu_threads_per_core: str = ""
    cpu_total_logical: str = ""
    cpu_freq_mhz: str = ""
    mem_total_gib: str = ""
    mem_available_gib: str = ""
    numa_nodes: List[int] = field(default_factory=list)
    pcie_devices: List[PcieDevice] = field(default_factory=list)
    python_version: str = ""


# ---------------------------------------------------------------------------
# System information collection
# ---------------------------------------------------------------------------

def _cmd(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return ""


def collect_server_info() -> ServerInfo:
    info = ServerInfo()
    info.hostname = socket.gethostname()
    info.kernel = platform.release()
    info.python_version = platform.python_version()

    os_release: Dict[str, str] = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                os_release[k] = v.strip('"')
    except OSError:
        pass
    info.os_name = os_release.get("PRETTY_NAME", platform.system())

    lscpu = _cmd("lscpu")

    def _field(key: str) -> str:
        m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", lscpu, re.MULTILINE)
        return m.group(1).strip() if m else ""

    info.cpu_model = _field("Model name")
    info.cpu_sockets = _field("Socket(s)")
    info.cpu_cores_per_socket = _field("Core(s) per socket")
    info.cpu_threads_per_core = _field("Thread(s) per core")
    info.cpu_total_logical = _field("CPU(s)")
    info.cpu_freq_mhz = _field("CPU MHz") or _field("CPU max MHz")
    info.numa_nodes = detect_numa_nodes()

    meminfo: Dict[str, str] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = parts[1]
    except OSError:
        pass

    def _kib_to_gib(v: str) -> str:
        try:
            return f"{int(v) / 1024 / 1024:.1f}"
        except (ValueError, TypeError):
            return ""

    info.mem_total_gib = _kib_to_gib(meminfo.get("MemTotal", ""))
    info.mem_available_gib = _kib_to_gib(meminfo.get("MemAvailable", ""))

    print("  Collecting PCIe device info (may take a few seconds) ...", file=sys.stderr)
    info.pcie_devices = collect_pcie_devices()

    return info


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _bench_env(
    redis_host: str,
    redis_port: int,
    redis_pool_size: int,
    nixl_install_dir: Path,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["REDIS_HOST"] = redis_host
    env["REDIS_PORT"] = str(redis_port)
    env["REDIS_POOL_SIZE"] = str(redis_pool_size)
    arch = platform.machine()
    lib_dir = nixl_install_dir / "lib" / f"{arch}-linux-gnu"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    return env


def _parse_row(stdout: str, block_size_b: int) -> Optional[Tuple[float, ...]]:
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[0] == str(block_size_b):
            try:
                return tuple(float(p) for p in parts[2:10])
            except ValueError:
                continue
    return None


def _run_one(
    nixlbench: Path,
    op: str,
    block_size_b: int,
    num_threads: int,
    env: Dict[str, str],
    warmup_iter: int,
    num_iter: int,
    total_buffer_size: int,
    numa_node: int,
) -> Optional[BenchResult]:
    effective_iter = max(num_threads, (num_iter // num_threads) * num_threads)

    # nixlbench's WRITE consistency check (GET-after-SET) does not work
    # correctly with num_threads > 1: the checker looks for a neighbouring
    # thread's key and spuriously reports it as missing.  Disable for
    # multi-threaded WRITE; correctness is verified by unit tests.
    check = "true" if not (op == "WRITE" and num_threads > 1) else "false"

    bench_cmd = [
        str(nixlbench),
        "--backend=REDIS", "--runtime_type=ASIO",
        f"--op_type={op}",
        f"--start_block_size={block_size_b}",
        f"--max_block_size={block_size_b}",
        "--start_batch_size=1", "--max_batch_size=1",
        "--pipeline_depth=1",
        f"--num_threads={num_threads}",
        f"--total_buffer_size={total_buffer_size}",
        f"--warmup_iter={warmup_iter}",
        f"--num_iter={effective_iter}",
        f"--check_consistency={check}",
    ]

    # Pin nixlbench to the same NUMA node as Redis so both processes share the
    # same memory domain; this eliminates cross-NUMA memory traffic on the
    # hot path (TCP socket buffers, hiredis callback allocations).
    cmd = numactl_prefix(numa_node) + bench_cmd

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=300
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return None

    if proc.returncode != 0:
        err = next(
            (l for l in output.splitlines() if "ERROR" in l or "failed" in l.lower()),
            output.splitlines()[-1] if output.splitlines() else "unknown",
        )
        print(f"  FAIL: {err}", file=sys.stderr)
        return None

    parsed = _parse_row(output, block_size_b)
    if parsed is None:
        print("  WARN: result row not found in output", file=sys.stderr)
        return None

    bw, avg_lat, avg_prep, p99_prep, avg_post, p99_post, avg_tx, p99_tx = parsed
    return BenchResult(
        op=op, block_size_b=block_size_b, num_threads=num_threads,
        bw_gbps=bw, avg_lat_us=avg_lat, avg_tx_us=avg_tx, p99_tx_us=p99_tx,
        avg_prep_us=avg_prep, p99_prep_us=p99_prep,
        avg_post_us=avg_post, p99_post_us=p99_post,
    )


def run_all(
    nixlbench: Path,
    env: Dict[str, str],
    warmup_iter: int,
    num_iter: int,
    total_buffer_size: int,
    numa_node: int,
) -> List[BenchResult]:
    results: List[BenchResult] = []
    total = len(OPERATIONS) * len(BLOCK_SIZES_KB) * len(THREAD_COUNTS)
    done = 0

    for op in OPERATIONS:
        for bs_kb in BLOCK_SIZES_KB:
            bs_b = bs_kb * 1024
            for nt in THREAD_COUNTS:
                done += 1
                print(
                    f"  [{done:2d}/{total}] {op:5s} {bs_kb:4d} KB  {nt:2d}T ... ",
                    end="", flush=True, file=sys.stderr,
                )
                r = _run_one(
                    nixlbench, op, bs_b, nt, env,
                    warmup_iter, num_iter, total_buffer_size, numa_node,
                )
                if r:
                    results.append(r)
                    print(f"{r.bw_gbps:.3f} GB/s  avg {r.avg_tx_us:.0f} µs", file=sys.stderr)
                else:
                    print("SKIPPED", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_Index = Dict[Tuple[str, int, int], BenchResult]


def _bw(r: Optional[BenchResult]) -> str:
    return f"{r.bw_gbps:.3f}" if r else "—"


def _lat(r: Optional[BenchResult], attr: str) -> str:
    return f"{getattr(r, attr):.0f}" if r else "—"


def _bw_table(op: str, idx: _Index) -> str:
    hdr = "| Block Size |" + "".join(f" {t}T (GB/s) |" for t in THREAD_COUNTS)
    sep = "|:----------:|" + "|:---------:|" * len(THREAD_COUNTS)
    rows = [hdr, sep]
    for bs_kb in BLOCK_SIZES_KB:
        cells = [f"{bs_kb} KB"] + [_bw(idx.get((op, bs_kb * 1024, t))) for t in THREAD_COUNTS]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _lat_table(op: str, idx: _Index, attr: str, label: str) -> str:
    hdr = "| Block Size |" + "".join(f" {t}T {label} |" for t in THREAD_COUNTS)
    sep = "|:----------:|" + "|:-----------:|" * len(THREAD_COUNTS)
    rows = [hdr, sep]
    for bs_kb in BLOCK_SIZES_KB:
        cells = [f"{bs_kb} KB"] + [_lat(idx.get((op, bs_kb * 1024, t)), attr) for t in THREAD_COUNTS]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _pcie_table(devices: List[PcieDevice]) -> str:
    if not devices:
        return "_No PCIe devices detected._"
    hdr = "| Address | Description | NUMA Node | Link Speed | Width | Max BW (GB/s) |"
    sep = "|:--------|:------------|:---------:|:----------:|:-----:|:-------------:|"
    rows = [hdr, sep]
    for d in devices:
        numa = str(d.numa_node) if d.numa_node >= 0 else "—"
        width = f"x{d.link_width}" if d.link_width else "—"
        bw = f"{d.max_bw_gbps:.1f}" if d.max_bw_gbps else "—"
        rows.append(
            f"| {d.address} | {d.description} | {numa} | {d.link_speed} | {width} | {bw} |"
        )
    return "\n".join(rows)


def render_report(
    results: List[BenchResult],
    info: ServerInfo,
    args: argparse.Namespace,
    bench_dir: Path,
    timestamp: str,
    numa_node: int,
) -> str:
    idx: _Index = {(r.op, r.block_size_b, r.num_threads): r for r in results}
    L: List[str] = []

    numa_label = f"node {numa_node}" if numa_node >= 0 else "not pinned"
    if numa_node >= 0:
        cpulist = numa_node_cpulist(numa_node)
        total_kib = numa_node_total_kib(numa_node)
        free_kib = numa_node_free_kib(numa_node)
        numa_detail = (
            f"node {numa_node}"
            + (f"  CPUs: {cpulist}" if cpulist else "")
            + (f"  Mem: {free_kib//1024//1024:.1f}/{total_kib//1024//1024:.1f} GiB free" if total_kib else "")
        )
    else:
        numa_detail = "not pinned"

    redis_cfg = (
        f"`{args.redis_host}:{args.redis_port}`"
        f"  pool\\_size={args.redis_pool_size}"
        f"  io\\_threads={args.redis_io_threads}"
        f"  network=host"
    )
    L += [
        "# NIXL Redis Plugin — Performance Benchmark Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Host:** {info.hostname}  ",
        f"**Redis:** {redis_cfg}  ",
        f"**NUMA placement:** {numa_detail}  ",
        f"**Build dir:** `{bench_dir}`  ",
        f"**Warmup / measured iterations:** {args.warmup_iter} / {args.num_iter}  ",
        f"**Total buffer size:** {args.total_buffer_size // 1024 // 1024} MiB  ",
        "",
        "> Redis runs via Docker host networking — the container shares the host network",
        "> stack directly, with no veth bridge overhead. Client and server communicate",
        "> over host loopback TCP (127.0.0.1). Both Redis and nixlbench are pinned to",
        "> the same NUMA node to avoid cross-NUMA memory traffic on the hot path.",
        "> WRITE multi-thread runs disable the built-in consistency check",
        "> (nixlbench limitation); data integrity is covered by unit tests.",
        "",
    ]

    # Server config
    numa_mem_rows = []
    for n in info.numa_nodes:
        total_kib = numa_node_total_kib(n)
        free_kib = numa_node_free_kib(n)
        cpulist = numa_node_cpulist(n)
        marker = " ← benchmark" if n == numa_node else ""
        numa_mem_rows.append(
            f"| {n}{marker} | {cpulist} "
            f"| {total_kib//1024//1024:.1f} GiB "
            f"| {free_kib//1024//1024:.1f} GiB |"
        )

    L += [
        "## Server Configuration",
        "",
        "### Operating System",
        "",
        "| Field | Value |",
        "|:------|:------|",
        f"| Hostname | `{info.hostname}` |",
        f"| OS | {info.os_name} |",
        f"| Kernel | {info.kernel} |",
        f"| Python | {info.python_version} |",
        "",
        "### CPU",
        "",
        "| Field | Value |",
        "|:------|:------|",
        f"| Model | {info.cpu_model} |",
        f"| Sockets | {info.cpu_sockets} |",
        f"| Cores per socket | {info.cpu_cores_per_socket} |",
        f"| Threads per core | {info.cpu_threads_per_core} |",
        f"| Logical CPUs | {info.cpu_total_logical} |",
        f"| Frequency (MHz) | {info.cpu_freq_mhz} |",
        "",
        "### NUMA Topology",
        "",
        "| Node | CPU List | Total Memory | Free Memory |",
        "|:----:|:---------|:------------:|:-----------:|",
    ]
    L += numa_mem_rows
    L += [""]

    L += [
        "### Memory",
        "",
        "| Field | Value |",
        "|:------|:------|",
        f"| Total | {info.mem_total_gib} GiB |",
        f"| Available | {info.mem_available_gib} GiB |",
        "",
    ]

    L += [
        "### PCIe Devices",
        "",
        _pcie_table(info.pcie_devices),
        "",
    ]

    # Results
    L += [
        "## Benchmark Results",
        "",
        "Columns are thread counts (`--num_threads`). Batch size = 1, pipeline depth = 1.",
        "",
    ]

    for op in OPERATIONS:
        L += [
            f"### {op}",
            "",
            "#### Bandwidth (GB/s)",
            "",
            _bw_table(op, idx),
            "",
            "#### Avg Transfer Latency — Avg Tx (µs)",
            "",
            _lat_table(op, idx, "avg_tx_us", "Avg Tx (µs)"),
            "",
            "#### P99 Transfer Latency — P99 Tx (µs)",
            "",
            _lat_table(op, idx, "p99_tx_us", "P99 Tx (µs)"),
            "",
            "#### Avg End-to-End Latency — Avg Lat (µs)",
            "",
            _lat_table(op, idx, "avg_lat_us", "Avg Lat (µs)"),
            "",
        ]

    # Summary
    L += [
        "## Summary: Peak Bandwidth",
        "",
        "Best observed bandwidth per operation across all block sizes and thread counts.",
        "",
        "| Operation | Block Size | Threads | B/W (GB/s) | Avg Tx (µs) | P99 Tx (µs) |",
        "|:----------|:----------:|:-------:|:----------:|:-----------:|:-----------:|",
    ]
    for op in OPERATIONS:
        best = max((r for r in results if r.op == op), key=lambda r: r.bw_gbps, default=None)
        if best:
            L.append(
                f"| {op} | {best.block_size_b // 1024} KB | {best.num_threads}T"
                f" | {best.bw_gbps:.3f} | {best.avg_tx_us:.0f} | {best.p99_tx_us:.0f} |"
            )

    L += ["", "---", "", "_Report generated by `test/python/test_redis_benchmark.py`_", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIXL Redis end-to-end nixlbench sweep → Markdown report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # First run: builds everything, starts Redis, benchmarks, stops Redis
              python3 test/python/test_redis_benchmark.py

              # Subsequent run: reuses build, manages Redis, writes report to file
              python3 test/python/test_redis_benchmark.py --output report.md

              # Use a Redis server that is already running
              python3 test/python/test_redis_benchmark.py --no-start-redis

              # Pin to NUMA node 1 explicitly
              python3 test/python/test_redis_benchmark.py --numa-node 1

              # Force clean rebuild
              python3 test/python/test_redis_benchmark.py --rebuild

              # Skip build entirely (use existing binaries)
              python3 test/python/test_redis_benchmark.py --skip-build
        """),
    )
    p.add_argument(
        "--bench-dir", default="/tmp/nixl-redis-bench",
        help="Persistent root for build and install directories",
    )
    p.add_argument(
        "--rebuild", action="store_true",
        help="Wipe and recreate build directories before building",
    )
    p.add_argument(
        "--skip-build", action="store_true",
        help="Skip the build phase and use existing binaries in --bench-dir",
    )
    p.add_argument(
        "--start-redis", default=True, action=argparse.BooleanOptionalAction,
        help="Start (and stop) a Redis Docker container for the benchmark (default: on)",
    )
    p.add_argument(
        "--redis-container", default="nixl-redis-bench",
        help="Docker container name for the managed Redis instance",
    )
    p.add_argument(
        "--redis-io-threads", type=int, default=0,
        help="Redis --io-threads value (0 = auto: physical cores on the chosen NUMA node, capped at 16)",
    )
    p.add_argument("--redis-host", default="127.0.0.1")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--redis-pool-size", type=int, default=8)
    p.add_argument(
        "--numa-node", type=int, default=-1,
        help="NUMA node to pin Redis container and nixlbench to (-1 = auto-detect best node)",
    )
    p.add_argument("--warmup-iter", type=int, default=32)
    p.add_argument("--num-iter", type=int, default=208)
    p.add_argument(
        "--total-buffer-size", type=int, default=67108864,
        help="Total buffer size in bytes (default: 64 MiB)",
    )
    p.add_argument("--output", default=None, help="Write Markdown report to file (default: stdout)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench_dir = Path(args.bench_dir).resolve()

    nixlbench_bin = bench_dir / "nixlbench-build" / "nixlbench"
    nixl_install_dir = bench_dir / "nixl-install"

    # Build phase
    if args.skip_build:
        print("Build phase skipped (--skip-build).", file=sys.stderr)
        if not nixlbench_bin.is_file():
            sys.exit(f"ERROR: nixlbench not found at {nixlbench_bin}. Run without --skip-build first.")
    else:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Build phase  (bench-dir: {bench_dir})", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        nixlbench_bin, nixl_install_dir = ensure_built(bench_dir, rebuild=args.rebuild)
        print(f"\nBuild complete. nixlbench: {nixlbench_bin}\n", file=sys.stderr)

    # NUMA selection — prefer the node with the most total memory (bandwidth proxy)
    if args.numa_node < 0:
        numa_node = detect_best_numa_node()
        print(
            f"NUMA auto-detect: selected node {numa_node} "
            f"(total {numa_node_total_kib(numa_node) // 1024 // 1024:.1f} GiB, "
            f"{numa_node_free_kib(numa_node) // 1024 // 1024:.1f} GiB free, "
            f"{numa_node_cpu_count(numa_node)} logical CPUs)",
            file=sys.stderr,
        )
    else:
        numa_node = args.numa_node
        print(
            f"NUMA: using node {numa_node} (--numa-node, "
            f"{numa_node_cpu_count(numa_node)} logical CPUs)",
            file=sys.stderr,
        )

    # io-threads auto-detection: physical cores on the NUMA node, capped at 16
    # (Redis io-threads beyond the physical core count add contention, not throughput)
    if args.redis_io_threads <= 0:
        logical = numa_node_cpu_count(numa_node)
        # Assume 2 hyperthreads per core; fall back to logical count if lscpu unavailable
        ht_per_core = 2
        try:
            lscpu_out = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=5
            ).stdout
            m = re.search(r"Thread\(s\) per core:\s*(\d+)", lscpu_out)
            if m:
                ht_per_core = max(int(m.group(1)), 1)
        except Exception:
            pass
        physical = max(logical // ht_per_core, 1)
        args.redis_io_threads = min(physical, 16)
        print(
            f"io-threads auto-detect: {logical} logical CPUs / {ht_per_core} threads-per-core"
            f" → {args.redis_io_threads} io-threads",
            file=sys.stderr,
        )

    numactl_avail = bool(shutil.which("numactl"))
    if not numactl_avail:
        print(
            "  WARNING: numactl not found — NUMA pinning disabled. "
            "Install numactl for best results.",
            file=sys.stderr,
        )

    # Redis phase
    we_started_redis = False
    if args.start_redis:
        print(f"\n{'='*60}", file=sys.stderr)
        print("Redis phase", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        we_started_redis = start_redis_container(
            args.redis_container,
            args.redis_port,
            args.redis_io_threads,
            numa_node if numactl_avail else -1,
        )

    try:
        # Collect server info
        print("\nCollecting server information ...", file=sys.stderr)
        info = collect_server_info()

        # Build runtime environment
        env = _bench_env(args.redis_host, args.redis_port, args.redis_pool_size, nixl_install_dir)

        # Benchmark phase
        total_runs = len(OPERATIONS) * len(BLOCK_SIZES_KB) * len(THREAD_COUNTS)
        print(f"\n{'='*60}", file=sys.stderr)
        print(
            f"Benchmark phase  ({total_runs} runs: "
            f"{len(OPERATIONS)} ops × {len(BLOCK_SIZES_KB)} block sizes × {len(THREAD_COUNTS)} threads)",
            file=sys.stderr,
        )
        print(f"{'='*60}\n", file=sys.stderr)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = run_all(
            nixlbench_bin, env,
            args.warmup_iter, args.num_iter, args.total_buffer_size,
            numa_node if numactl_avail else -1,
        )

        # Report phase
        print(f"\nGenerating report ({len(results)}/{total_runs} runs succeeded) ...\n", file=sys.stderr)
        report = render_report(results, info, args, bench_dir, timestamp, numa_node)

        if args.output:
            Path(args.output).write_text(report)
            print(f"Report written to: {args.output}", file=sys.stderr)
        else:
            print(report)

    finally:
        if we_started_redis:
            print(f"\n{'='*60}", file=sys.stderr)
            print("Teardown phase", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            stop_redis_container(args.redis_container)


if __name__ == "__main__":
    main()
