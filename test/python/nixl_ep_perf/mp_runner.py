# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-process test runner for NIXL EP performance tests.

Spawns worker processes with proper GPU assignment and UCX configuration.
"""

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import store_group
import torch
import torch.multiprocessing as mp
from rank_server import RankClient, start_server

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from a single rank's test execution."""

    rank: int
    test_name: str
    passed: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


# Cached topology (discovered once per process)
_GPU_NIC_TOPOLOGY: Optional[Dict[int, str]] = None
_RANK_SERVER_ADDR: Optional[str] = None
_RANK_SERVER_PORT: int = 9998


def discover_gpu_nic_topology() -> Optional[Dict[int, str]]:
    """Discover GPU-NIC topology using nvidia-smi topo -m."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")

        # Parse NIC legend (e.g., "NIC0: mlx5_0")
        nic_legend = {}
        for line in lines:
            match = re.match(r"\s*(NIC\d+):\s*(\S+)", line)
            if match:
                nic_legend[match.group(1)] = match.group(2)

        if not nic_legend:
            return None

        # Find header line with GPU0 and NIC0
        header_idx = None
        for i, line in enumerate(lines):
            if "GPU0" in line and "NIC0" in line:
                header_idx = i
                break

        if header_idx is None:
            return None

        header = lines[header_idx].split()
        nic_columns = {col: i for i, col in enumerate(header) if col.startswith("NIC")}

        if not nic_columns:
            return None

        # Connection priority (best to worst)
        priority = {"PIX": 0, "PXB": 1, "PHB": 2, "NODE": 3, "SYS": 4, "X": 99}
        gpu_to_nic = {}

        for line in lines[header_idx + 1 :]:
            parts = line.split()
            if not parts or not parts[0].startswith("GPU"):
                continue
            if parts[0].startswith("NIC") or "Legend" in line:
                break

            match = re.match(r"GPU(\d+)", parts[0])
            if not match:
                continue
            gpu_idx = int(match.group(1))

            best_nic, best_priority = None, 100
            for nic_name, col_idx in nic_columns.items():
                data_col_idx = col_idx + 1
                if data_col_idx < len(parts):
                    p = priority.get(parts[data_col_idx], 50)
                    if p < best_priority:
                        best_priority = p
                        best_nic = nic_legend.get(nic_name)

            if best_nic:
                gpu_to_nic[gpu_idx] = best_nic

        return gpu_to_nic if gpu_to_nic else None

    except Exception as e:
        logger.warning("Failed to discover GPU-NIC topology: %s", e)
        return None


def get_gpu_nic_mapping(local_rank: int) -> Optional[str]:
    """Get UCX_NET_DEVICES string for a GPU.

    Format matches elastic.py: RDMA NIC + TCP fallback interfaces
    """
    if _GPU_NIC_TOPOLOGY is None:
        return None  # Topology not set - let UCX auto-select

    if local_rank in _GPU_NIC_TOPOLOGY:
        rdma_nic = f"cuda0-{_GPU_NIC_TOPOLOGY[local_rank]}:1"

        # Add TCP fallback interfaces (like elastic.py) for cross-node communication
        # These are IPoIB (InfiniBand) interfaces used as TCP fallback
        tcp_nics = (
            ",ibp26s0,ibp44s0,ibp64s0,ibp101s0,ibp156s0,ibp173s0,ibp192s0,ibp227s0"
        )

        return rdma_nic + tcp_nics
    return None


def setup_worker_environment(
    local_rank: int,
    etcd_server: str = "http://127.0.0.1:2379",
    use_tcp_store: bool = False,
):
    """Set up GPU, UCX, and NIXL environment for a worker process.

    Args:
        local_rank: Local GPU index on this node (0-7), like elastic.py
        etcd_server: etcd server URL (only used if not use_tcp_store)
        use_tcp_store: If True, use TCPStore instead of etcd
    """
    cuda_device = local_rank % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # Set UCX_NET_DEVICES using local_rank (like elastic.py)
    # Maps to the optimal RDMA NIC for this GPU + TCP fallback interfaces
    ucx_devices = get_gpu_nic_mapping(local_rank)
    if ucx_devices:
        os.environ["UCX_NET_DEVICES"] = ucx_devices

    # Don't set UCX_TLS here - buffer.py will set it to "^cuda_ipc" when nvlink_backend != "nixl"
    # which tells UCX to auto-detect all transports except cuda_ipc (including RDMA)

    # Only set NIXL_ETCD_ENDPOINTS when NOT using TCPStore (copy elastic.py pattern)
    # This prevents C++ code from activating etcd path when we want TCPStore
    if not use_tcp_store:
        os.environ["NIXL_ETCD_ENDPOINTS"] = etcd_server
        logger.info(
            f"Worker local_rank={local_rank}: Set NIXL_ETCD_ENDPOINTS={etcd_server}"
        )

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)


def worker_fn(
    torch_rank: int,
    num_processes: int,
    test_fn: Callable,
    result_queue: mp.Queue,
    etcd_server: str,
    rank_server_addr: str,
    gpu_nic_topology: Dict[int, str],
    extra_kwargs: Optional[Dict[Any, Any]],
    rank_server_port: int,
    use_tcp_store: bool,
    world_size: int = 1,
    node_rank: int = 0,
):
    """Worker function executed by each spawned process."""
    global _GPU_NIC_TOPOLOGY, _RANK_SERVER_ADDR, _RANK_SERVER_PORT

    _GPU_NIC_TOPOLOGY = gpu_nic_topology
    _RANK_SERVER_ADDR = rank_server_addr
    _RANK_SERVER_PORT = rank_server_port

    if extra_kwargs is None:
        extra_kwargs = {}

    # Pass node_rank to test function for logging prefix
    extra_kwargs["node_rank"] = node_rank

    total_ranks = num_processes * world_size

    # Compute ranks deterministically based on node_rank and process index
    # This ensures predictable assignment:
    #   Node 0: global ranks 0-7
    #   Node 1: global ranks 8-15
    #   etc.
    local_rank = torch_rank  # Process index within this node (0-7)
    global_rank = node_rank * num_processes + local_rank

    try:
        # Setup environment using local_rank for GPU/NIC selection
        setup_worker_environment(local_rank, etcd_server, use_tcp_store)

        start_time = time.perf_counter()
        result = test_fn(
            rank=global_rank,  # Global rank for Buffer
            world_size=total_ranks,
            local_rank=local_rank,  # Local rank for GPU index
            **extra_kwargs,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        if isinstance(result, bool):
            test_result = TestResult(
                rank=global_rank,
                test_name=test_fn.__name__,
                passed=result,
                duration_ms=duration_ms,
            )
        elif isinstance(result, dict):
            test_result = TestResult(
                rank=global_rank,
                test_name=test_fn.__name__,
                passed=result.get("passed", True),
                error=result.get("error"),
                metrics=result.get("metrics"),
                duration_ms=duration_ms,
            )
        else:
            test_result = TestResult(
                rank=global_rank,
                test_name=test_fn.__name__,
                passed=True,
                metrics={"result": result},
                duration_ms=duration_ms,
            )

        result_queue.put(test_result)

    except Exception as e:
        import traceback

        result_queue.put(
            TestResult(
                rank=global_rank,
                test_name=test_fn.__name__,
                passed=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
        )


def wait_for_tcp_port(
    host: str,
    port: int,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for a TCP port to accept connections.

    Args:
        host: Hostname or IP to connect to
        port: Port number
        timeout: Maximum time to wait in seconds
        poll_interval: Initial interval between connection attempts

    Returns:
        True if port is ready, raises TimeoutError otherwise
    """
    import socket

    start_time = time.time()
    attempt = 0
    current_interval = poll_interval

    while time.time() - start_time < timeout:
        attempt += 1
        try:
            s = socket.create_connection((host, port), timeout=2.0)
            s.close()
            logger.info(
                f"TCP port {host}:{port} is ready "
                f"(attempt {attempt}, waited {time.time() - start_time:.1f}s)"
            )
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            if attempt == 1:
                logger.info(f"Waiting for TCP port {host}:{port}...")
            elif attempt % 10 == 0:
                logger.info(
                    f"Still waiting for {host}:{port}... "
                    f"(attempt {attempt}, {time.time() - start_time:.1f}s)"
                )
            time.sleep(current_interval)
            current_interval = min(current_interval * 1.2, 2.0)

    raise TimeoutError(f"TCP port {host}:{port} not ready after {timeout}s")


def check_etcd_running(etcd_endpoints: str = "http://127.0.0.1:2379") -> bool:
    """Check if etcd is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-x", "etcd"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass

    try:
        env = os.environ.copy()
        env["ETCDCTL_API"] = "3"
        result = subprocess.run(
            ["etcdctl", "--endpoints", etcd_endpoints, "endpoint", "health"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        if result.returncode == 0 and "is healthy" in result.stdout:
            return True
    except Exception:
        pass

    return False


def clean_etcd_state(etcd_endpoints: str = "http://127.0.0.1:2379"):
    """Clean all keys from etcd."""
    try:
        env = os.environ.copy()
        env["ETCDCTL_API"] = "3"

        # Delete all keys (empty prefix = all keys)
        result = subprocess.run(
            ["etcdctl", "--endpoints", etcd_endpoints, "del", "--prefix", ""],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        if result.returncode == 0:
            time.sleep(1.0)
    except Exception:
        pass


def run_multiprocess_test(
    test_fn: Callable,
    num_processes: int = 8,
    etcd_server: str = "http://127.0.0.1:2379",
    timeout: float = 120.0,
    clean_etcd: bool = True,
    rank_server_port: int = 9998,
    tcp_store_port: int = 9999,
    skip_nic_discovery: bool = True,
    use_tcp_store: bool = True,
    world_size: int = 1,
    rank: int = 0,
    master_addr: str = "127.0.0.1",
    **kwargs,
) -> List[TestResult]:
    """
    Run a test function across multiple GPU processes (single or multi-node).

    Args:
        test_fn: Function receiving (rank, world_size, local_rank, **kwargs)
        num_processes: Number of processes to spawn per node
        timeout: Timeout in seconds
        use_tcp_store: If True (default), use TCPStore; if False, use etcd
        tcp_store_port: Port for TCPStore server (default: 9999)
        world_size: Total number of nodes (env: WORLD_SIZE, default: 1 for single-node)
        rank: This node's rank 0=master (env: RANK, default: 0)
        master_addr: Master node address (env: MASTER_ADDR, for TCPStore and rank server)
        **kwargs: Passed to test_fn

    Returns:
        List of TestResult, one per local rank on this node
    """
    # Always use master_addr for etcd (works for both single-node and multi-node)
    etcd_server = f"http://{master_addr}:2379"

    # Configure logger with node prefix for multi-node debugging
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(f"[Node {rank}] %(message)s"))

    # Calculate total ranks and set master address
    total_ranks = num_processes * world_size
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["WORLD_SIZE"] = str(total_ranks)  # Total ranks, not nodes
    os.environ["RANK"] = str(rank)  # This node's rank
    is_master = rank == 0

    logger.info(
        f"etcd_server={etcd_server}, master_addr={master_addr}, "
        f"world_size={world_size}, num_processes={num_processes}"
    )

    if world_size > 1:
        logger.info(
            "Multi-node mode: This is %s node (RANK=%d/%d, MASTER_ADDR=%s)",
            "MASTER" if is_master else "WORKER",
            rank,
            world_size - 1,
            master_addr,
        )

    # Start TCPStore server if requested (master node only)
    tcp_store_process = None
    if use_tcp_store:
        if is_master:
            logger.info(f"Starting TCPStore server on port {tcp_store_port}")

            def run_tcp_store_server():
                # Keep reference to prevent garbage collection
                _store = store_group.create_master_store(  # noqa: F841
                    port=tcp_store_port
                )
                # Keep server alive
                import signal

                signal.pause()

            tcp_store_process = mp.Process(target=run_tcp_store_server, daemon=True)
            tcp_store_process.start()

        # Wait for TCPStore to be ready (both master and worker nodes)
        logger.info(f"Waiting for TCPStore at {master_addr}:{tcp_store_port}...")
        wait_for_tcp_port(master_addr, tcp_store_port, timeout=60.0)
        logger.info(f"✓ TCPStore ready at {master_addr}:{tcp_store_port}")
        kwargs["tcp_store_port"] = tcp_store_port
    else:
        # Only check/clean etcd on master node when not using TCPStore
        if is_master:
            if not check_etcd_running(etcd_server):
                raise RuntimeError(f"etcd is not running at {etcd_server}")

            if clean_etcd:
                clean_etcd_state(etcd_server)
                logger.info("Cleaned etcd state")
        else:
            logger.info("Skipping etcd check (master handles it)")

    # Pass use_tcp_store to the test function via kwargs
    kwargs["use_tcp_store"] = use_tcp_store

    # Discover topology once (skipped by default unless --discover-nics is set)
    gpu_nic_topology = None
    if skip_nic_discovery:
        logger.info("Skipping GPU-NIC discovery (default), UCX will auto-select")
    else:
        gpu_nic_topology = discover_gpu_nic_topology()
        if gpu_nic_topology is None:
            raise RuntimeError(
                "Failed to discover GPU-NIC topology. "
                "Ensure nvidia-smi is available and GPUs are present. "
                "Or omit --discover-nics to let UCX auto-select (default)."
            )
        logger.info(f"Discovered GPU-NIC topology: {gpu_nic_topology}")

    # Start rank server (master node only)
    server_process = None
    if is_master:
        logger.info(f"Starting rank server on port {rank_server_port}")
        server_process = start_server(port=rank_server_port)
        time.sleep(1.0)

        try:
            client = RankClient(master_addr, rank_server_port)
            client.clear_barriers()
            client.reset()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to rank server: {e}")
    else:
        # Worker node: wait for master's rank server to be ready
        # NOTE: Do NOT call clear_barriers() here - only master should do that
        # to avoid clearing barriers that master's processes are already using
        logger.info(f"Waiting for rank server at {master_addr}:{rank_server_port}...")
        client = RankClient(master_addr, rank_server_port)
        client.wait_for_server(timeout=60.0)
        logger.info(
            f"✓ Master is alive! "
            f"Connected to rank server at {master_addr}:{rank_server_port}"
        )

    spawn_ctx = mp.get_context("spawn")
    result_queue = spawn_ctx.Queue()

    try:
        ctx = mp.spawn(
            worker_fn,
            args=(
                num_processes,
                test_fn,
                result_queue,
                etcd_server,
                master_addr,
                gpu_nic_topology,
                kwargs,
                rank_server_port,
                use_tcp_store,
                world_size,
                rank,
            ),
            nprocs=num_processes,
            join=False,
            daemon=False,
            start_method="spawn",
        )

        deadline = time.time() + timeout
        for p in ctx.processes:
            remaining = max(0.1, deadline - time.time())
            p.join(timeout=remaining)
            if p.is_alive():
                p.terminate()

        results = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except Exception:
                break

        # Calculate expected global rank range for this node
        start_rank = rank * num_processes
        end_rank = start_rank + num_processes
        expected_ranks = set(range(start_rank, end_rank))

        result_ranks = {r.rank for r in results}
        for expected_rank in expected_ranks:
            if expected_rank not in result_ranks:
                results.append(
                    TestResult(
                        rank=expected_rank,
                        test_name=test_fn.__name__,
                        passed=False,
                        error="Timeout or process died",
                    )
                )

        results.sort(key=lambda r: r.rank)
        return results

    finally:
        if server_process and server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=2)
        if tcp_store_process and tcp_store_process.is_alive():
            tcp_store_process.terminate()
            tcp_store_process.join(timeout=2)


# ============================================================================
# Synchronization
# ============================================================================


class DistributedBarrier:
    """TCP-based barrier using rank_server."""

    def __init__(
        self,
        world_size: int,
        barrier_id: str,
        server_addr: str = "127.0.0.1",
        port: int = 9998,
    ):
        self.world_size = world_size
        self.barrier_id = barrier_id
        self.server_addr = server_addr
        self.port = port

    def wait(self, rank: int, timeout: float = 60.0):
        """Wait for all ranks to reach this barrier."""
        client = RankClient(self.server_addr, self.port)
        return client.barrier_wait(self.barrier_id, rank, self.world_size, timeout)


def sync_all_ranks(
    rank: int,
    world_size: int,
    barrier_name: str,
    timeout: float = 60.0,
    server_addr: Optional[str] = None,
    port: Optional[int] = None,
):
    """Synchronize all ranks at a named barrier point."""
    if server_addr is None:
        server_addr = _RANK_SERVER_ADDR or os.environ.get("MASTER_ADDR", "127.0.0.1")
    if port is None:
        port = _RANK_SERVER_PORT

    assert server_addr is not None  # Ensured by default above
    barrier = DistributedBarrier(world_size, barrier_name, server_addr, port)
    barrier.wait(rank, timeout)
