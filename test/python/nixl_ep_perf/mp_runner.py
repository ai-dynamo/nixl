# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    """Get UCX_NET_DEVICES string for a GPU."""
    if _GPU_NIC_TOPOLOGY is None:
        return None  # Topology not set - let UCX auto-select

    if local_rank in _GPU_NIC_TOPOLOGY:
        return f"cuda0-{_GPU_NIC_TOPOLOGY[local_rank]}:1"
    return None


def setup_worker_environment(
    torch_rank: int,
    etcd_server: str = "http://127.0.0.1:2379",
    use_tcp_store: bool = False,
):
    """Set up GPU, UCX, and NIXL environment for a worker process."""
    cuda_device = torch_rank % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # Set UCX_NET_DEVICES if topology was discovered
    # If --skip-nic-discovery was used, ucx_devices is None and UCX auto-selects
    ucx_devices = get_gpu_nic_mapping(cuda_device)
    if ucx_devices:
        os.environ["UCX_NET_DEVICES"] = ucx_devices

    # Set UCX_TLS for RDMA support (inherit from parent or use default)
    # This is critical for multi-node communication
    # Note: buffer.py may append "^cuda_ipc" if nvlink_backend != "nixl"
    if "UCX_TLS" not in os.environ:
        # Default: try RDMA first, fall back to TCP, exclude cuda_ipc
        os.environ["UCX_TLS"] = "rc_mlx5,dc_mlx5,tcp,^cuda_ipc"

    # Only set NIXL_ETCD_ENDPOINTS when NOT using TCPStore
    # This prevents C++ code from activating etcd path when we want TCPStore
    if not use_tcp_store:
        os.environ["NIXL_ETCD_ENDPOINTS"] = etcd_server

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
    rank: int = 0,
):
    """Worker function executed by each spawned process."""
    global _GPU_NIC_TOPOLOGY, _RANK_SERVER_ADDR, _RANK_SERVER_PORT

    _GPU_NIC_TOPOLOGY = gpu_nic_topology
    _RANK_SERVER_ADDR = rank_server_addr
    _RANK_SERVER_PORT = rank_server_port

    if extra_kwargs is None:
        extra_kwargs = {}

    rank_client = None
    global_rank = None
    local_rank = torch_rank
    total_ranks = num_processes * world_size

    try:
        rank_client = RankClient(rank_server_addr, rank_server_port)
        
        # Always use rank server's assignment (like elastic.py)
        # This ensures unique global ranks across all nodes
        local_rank_from_server, global_rank = rank_client.get_rank()

        setup_worker_environment(torch_rank, etcd_server, use_tcp_store)
        
        # Debug: log UCX_TLS setting
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Rank {global_rank}: torch_rank={torch_rank}, local_rank_from_server={local_rank_from_server}, UCX_TLS={os.environ.get('UCX_TLS', 'NOT SET')}")

        start_time = time.perf_counter()
        result = test_fn(
            rank=global_rank,
            world_size=total_ranks,
            local_rank=torch_rank,  # Use torch_rank (0-7 per node) for CUDA device
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
        if rank_client:
            rank_client.release_rank()

    except Exception as e:
        import traceback

        report_rank = global_rank if global_rank is not None else torch_rank
        result_queue.put(
            TestResult(
                rank=report_rank,
                test_name=test_fn.__name__,
                passed=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
        )
        if rank_client:
            try:
                rank_client.release_rank()
            except Exception:
                pass


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
    """Clean NIXL-related keys from etcd."""
    try:
        env = os.environ.copy()
        env["ETCDCTL_API"] = "3"

        for prefix in ["/nixl", "nixl"]:
            result = subprocess.run(
                ["etcdctl", "--endpoints", etcd_endpoints, "del", "--prefix", prefix],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )
            if result.returncode == 0 and result.stdout.strip() not in ["", "0"]:
                time.sleep(1.0)
                break
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
    skip_nic_discovery: bool = False,
    use_tcp_store: bool = False,
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
        use_tcp_store: If True, skip etcd check (using TCPStore instead)
        tcp_store_port: Port for TCPStore server (default: 9999)
        world_size: Total number of nodes (env: WORLD_SIZE, default: 1 for single-node)
        rank: This node's rank 0=master (env: RANK, default: 0)
        master_addr: Master node address (env: MASTER_ADDR, for TCPStore and rank server)
        **kwargs: Passed to test_fn

    Returns:
        List of TestResult, one per local rank on this node
    """
    # Calculate total ranks and set master address
    total_ranks = num_processes * world_size
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["WORLD_SIZE"] = str(total_ranks)  # Total ranks, not nodes
    os.environ["RANK"] = str(rank)  # This node's rank
    is_master = (rank == 0)
    
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
                _store = store_group.create_master_store(port=tcp_store_port)
                # Keep server alive
                import signal
                signal.pause()

            tcp_store_process = mp.Process(target=run_tcp_store_server, daemon=True)
            tcp_store_process.start()
            time.sleep(1.0)
        else:
            logger.info(f"Connecting to TCPStore server at {master_addr}:{tcp_store_port}")
            time.sleep(2.0)  # Give master node time to start
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
            logger.info("Worker node: skipping etcd check (master handles it)")

    # Pass use_tcp_store to the test function via kwargs
    kwargs["use_tcp_store"] = use_tcp_store

    # Discover topology once (skip if --skip-nic-discovery is set)
    gpu_nic_topology = None
    if skip_nic_discovery:
        logger.info(
            "Skipping GPU-NIC discovery (--skip-nic-discovery), UCX will auto-select"
        )
    else:
        gpu_nic_topology = discover_gpu_nic_topology()
        if gpu_nic_topology is None:
            raise RuntimeError(
                "Failed to discover GPU-NIC topology. "
                "Ensure nvidia-smi is available and GPUs are present. "
                "Or use --skip-nic-discovery to let UCX auto-select."
            )

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
        logger.info(f"Connecting to rank server at {master_addr}:{rank_server_port}")
        time.sleep(2.0)  # Give master node time to start

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

        result_ranks = {r.rank for r in results}
        for i in range(num_processes):
            if i not in result_ranks:
                results.append(
                    TestResult(
                        rank=i,
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
