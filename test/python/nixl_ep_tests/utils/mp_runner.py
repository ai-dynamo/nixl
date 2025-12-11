# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-process test runner for NIXL EP functional tests.

Spawns worker processes with proper GPU assignment and UCX configuration.
Uses test_rank_server for coordination across nodes.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

# Add parent directories to path
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(UTILS_DIR)
ELASTIC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "elastic")
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, ELASTIC_DIR)

# Use our own test_rank_server for tests (has barrier support)
# Falls back to elastic/rank_server for compatibility
try:
    from utils.test_rank_server import RankClient, RankServer, start_test_server

    _USE_TEST_RANK_SERVER = True
except ImportError:
    try:
        from test_rank_server import RankClient, RankServer, start_test_server

        _USE_TEST_RANK_SERVER = True
    except ImportError:
        import rank_server

        _USE_TEST_RANK_SERVER = False


@dataclass
class TestResult:
    """Result from a single rank's test execution."""

    rank: int
    test_name: str
    passed: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


# Cached GPU-NIC topology (discovered once per process)
_GPU_NIC_TOPOLOGY = None
_TCP_NICS = None


def discover_gpu_nic_topology():
    """
    Discover GPU-NIC topology using nvidia-smi topo -m.
    Returns: {gpu_id: nic_name} mapping, or None if discovery fails.
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None

        output = result.stdout
        lines = output.strip().split("\n")

        # Step 1: Parse NIC Legend at the bottom (e.g., "NIC0: mlx5_0")
        nic_legend = {}
        for line in lines:
            match = re.match(r"\s*(NIC\d+):\s*(\S+)", line)
            if match:
                nic_legend[match.group(1)] = match.group(2)

        if not nic_legend:
            return None

        # Step 2: Find the header line (contains GPU0 and NIC0)
        header_idx = None
        for i, line in enumerate(lines):
            if "GPU0" in line and "NIC0" in line:
                header_idx = i
                break

        if header_idx is None:
            return None

        header = lines[header_idx].split()

        # Step 3: Find NIC column indices in header (NIC0, NIC1, ...)
        nic_columns = {}
        for col_idx, col_name in enumerate(header):
            if col_name.startswith("NIC"):
                nic_columns[col_name] = col_idx

        if not nic_columns:
            return None

        # Step 4: Parse GPU rows to find best NIC for each GPU
        # Connection types (best to worst): PIX, PXB, PHB, NODE, SYS
        connection_priority = {
            "PIX": 0,
            "PXB": 1,
            "PHB": 2,
            "NODE": 3,
            "SYS": 4,
            "X": 99,
        }

        gpu_to_nic = {}

        for line in lines[header_idx + 1 :]:
            parts = line.split()
            if not parts or not parts[0].startswith("GPU"):
                continue

            # Stop if we hit the NIC rows or legend section
            if parts[0].startswith("NIC") or "Legend" in line or ":" in parts[0]:
                break

            # Extract GPU number
            match = re.match(r"GPU(\d+)", parts[0])
            if not match:
                continue
            gpu_idx = int(match.group(1))

            # Find best NIC for this GPU (lowest priority = best connection)
            best_nic_name = None
            best_nic_device = None
            best_priority = 100

            # NOTE: Data rows have row label (GPU0) as first column, but header doesn't.
            # So we need col_idx + 1 to get the correct column in data rows.
            for nic_name, col_idx in nic_columns.items():
                data_col_idx = col_idx + 1  # Adjust for row label column
                if data_col_idx < len(parts):
                    conn_type = parts[data_col_idx]
                    priority = connection_priority.get(conn_type, 50)
                    if priority < best_priority:
                        best_priority = priority
                        best_nic_name = nic_name
                        best_nic_device = nic_legend.get(nic_name)

            if best_nic_device:
                gpu_to_nic[gpu_idx] = best_nic_device

        return gpu_to_nic if gpu_to_nic else None

    except Exception as e:
        logger.warning("Failed to discover GPU-NIC topology: %s", e)
        return None


def discover_tcp_nics():
    """Discover available TCP/Ethernet NICs. Returns comma-separated list."""
    import subprocess

    try:
        # Get list of InfiniBand interfaces
        result = subprocess.run(
            ["ip", "link", "show"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return ""

        nics = []
        for line in result.stdout.split("\n"):
            # Look for interfaces like ibp*, ib*, eth*
            if ": ibp" in line or ": ib" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    nic_name = parts[1].strip().split("@")[0]
                    if nic_name and not nic_name.startswith("lo"):
                        nics.append(nic_name)

        return "," + ",".join(nics) if nics else ""

    except Exception:
        return ""


def get_gpu_nic_mapping(local_rank: int):
    """Get UCX_NET_DEVICES string for a GPU. Uses pre-discovered topology."""
    global _GPU_NIC_TOPOLOGY, _TCP_NICS

    # Topology should be pre-set by worker_fn; fallback just in case
    if _GPU_NIC_TOPOLOGY is None:
        _GPU_NIC_TOPOLOGY = {
            0: "mlx5_0",
            1: "mlx5_1",
            2: "mlx5_2",
            3: "mlx5_4",
            4: "mlx5_5",
            5: "mlx5_6",
            6: "mlx5_7",
            7: "mlx5_8",
        }

    # Discover TCP NICs once per process (lightweight)
    if _TCP_NICS is None:
        _TCP_NICS = discover_tcp_nics()

    # Build UCX_NET_DEVICES string
    if local_rank in _GPU_NIC_TOPOLOGY:
        rdma_nic = _GPU_NIC_TOPOLOGY[local_rank]
        return f"cuda0-{rdma_nic}:1{_TCP_NICS}"

    return None


def setup_worker_environment(
    torch_rank: int,
    local_rank: int,
    global_rank: int,
    etcd_server: str = "http://127.0.0.1:2379",
):
    """Set up GPU, UCX, and NIXL environment for a worker process."""
    # Use torch_rank for GPU (matches elastic.py pattern)
    cuda_device = torch_rank % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # Set UCX network devices based on GPU device (matches elastic.py)
    ucx_devices = get_gpu_nic_mapping(cuda_device)
    if ucx_devices:
        os.environ["UCX_NET_DEVICES"] = ucx_devices

    os.environ["NIXL_ETCD_ENDPOINTS"] = etcd_server

    # Initialize torch
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)


def create_buffer(
    rank: int,
    world_size: int,
    num_experts_per_rank: int = 8,
    num_rdma_bytes: int = None,
    nvlink_backend: str = "ipc",
    enable_shrink: bool = True,
    hidden: int = 4096,
    num_tokens: int = 512,
):
    """Create and initialize a NIXL EP Buffer instance."""
    import nixl_ep

    # Calculate RDMA buffer size if not provided
    if num_rdma_bytes is None:
        num_experts = num_experts_per_rank * world_size
        num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
            num_tokens, hidden, world_size, num_experts
        )

    buffer = nixl_ep.Buffer(
        rank=rank,
        nvlink_backend=nvlink_backend,
        explicitly_destroy=True,
        enable_shrink=enable_shrink,
    )
    buffer.update_memory_buffers(
        num_ranks=world_size,
        num_experts_per_rank=num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
    )
    return buffer


def worker_fn(
    torch_rank: int,
    num_processes: int,
    test_fn: Callable,
    result_queue: mp.Queue,
    etcd_server: str,
    rank_server_addr: str,
    gpu_nic_topology: dict,
    extra_kwargs: dict = None,
    rank_server_port: int = 9998,
):
    """Worker function executed by each spawned process."""
    global _GPU_NIC_TOPOLOGY, _RANK_SERVER_ADDR, _RANK_SERVER_PORT

    _GPU_NIC_TOPOLOGY = gpu_nic_topology  # Use pre-discovered topology
    _RANK_SERVER_ADDR = rank_server_addr  # Store for DistributedBarrier
    _RANK_SERVER_PORT = rank_server_port  # Store for DistributedBarrier

    if extra_kwargs is None:
        extra_kwargs = {}

    rank_client = None
    global_rank = None  # Track if global_rank was assigned
    try:
        # Get rank from server
        if _USE_TEST_RANK_SERVER:
            rank_client = RankClient(rank_server_addr, rank_server_port)
            local_rank, global_rank = rank_client.get_rank()
        else:
            rank_client = rank_server.RankClient(rank_server_addr)
            local_rank, global_rank, _ = rank_client.get_rank()

        # Setup environment (uses cached topology, no discovery needed)
        # Pass torch_rank for GPU assignment (guaranteed unique per process)
        setup_worker_environment(torch_rank, local_rank, global_rank, etcd_server)

        # Run test
        start_time = time.perf_counter()
        result = test_fn(
            rank=global_rank,
            world_size=num_processes,
            local_rank=local_rank,
            **extra_kwargs,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Package result
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

        # Release rank
        if rank_client:
            rank_client.release_rank()

    except Exception as e:
        import traceback

        # Use global_rank if assigned, otherwise fall back to torch_rank
        report_rank = global_rank if global_rank is not None else torch_rank
        result_queue.put(
            TestResult(
                rank=report_rank,
                test_name=test_fn.__name__ if test_fn else "unknown",
                passed=False,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            )
        )
        if rank_client:
            try:
                rank_client.release_rank()
            except:
                pass


def check_etcd_running(etcd_endpoints: str = "http://127.0.0.1:2379") -> bool:
    """Check if etcd is running and accessible."""
    import subprocess

    # Method 1: Check if etcd process is running
    try:
        result = subprocess.run(
            ["pgrep", "-x", "etcd"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # Method 2: Try etcdctl endpoint health
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
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return False


def clean_etcd_state(etcd_endpoints: str = "http://127.0.0.1:2379"):
    """Clean NIXL-related keys from etcd before test run."""
    import subprocess

    cleaned_any = False

    try:
        # Set ETCDCTL_API=3 for v3 API
        env = os.environ.copy()
        env["ETCDCTL_API"] = "3"

        # Delete all keys with '/nixl' prefix (the actual key format)
        result = subprocess.run(
            ["etcdctl", "--endpoints", etcd_endpoints, "del", "--prefix", "/nixl"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        if result.returncode == 0:
            deleted = result.stdout.strip()
            if deleted and deleted != "0":
                logger.info("Cleaned %d etcd keys with '/nixl' prefix", deleted)
                cleaned_any = True

        # Also try without leading slash (in case of different key formats)
        result = subprocess.run(
            ["etcdctl", "--endpoints", etcd_endpoints, "del", "--prefix", "nixl"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        if result.returncode == 0:
            deleted = result.stdout.strip()
            if deleted and deleted != "0":
                logger.info("Cleaned %d etcd keys with 'nixl' prefix", deleted)
                cleaned_any = True

        # Add delay after cleanup to ensure etcd has propagated deletions
        # This prevents race conditions where new processes register before
        # stale keys from previous tests are fully removed
        if cleaned_any:
            time.sleep(1.0)  # 1 second delay after cleanup

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # etcdctl not available or timeout - not critical, continue
        pass


def run_multiprocess_test(
    test_fn: Callable,
    num_processes: int = 8,
    etcd_server: str = "http://127.0.0.1:2379",
    timeout: float = 120.0,
    clean_etcd: bool = True,
    rank_server_port: int = 9998,
    **kwargs,
) -> List[TestResult]:
    """
    Run a test function across multiple GPU processes.

    Args:
        test_fn: Function receiving (rank, world_size, local_rank, **kwargs)
        num_processes: Number of processes to spawn
        timeout: Timeout in seconds
        **kwargs: Passed to test_fn

    Returns:
        List of TestResult, one per rank
    """

    # Check if etcd is running
    if not check_etcd_running(etcd_server):
        logger.error("etcd is not running or not accessible at %s", etcd_server)
        logger.error("Please run 'source setup_env.sh' to start etcd")
        raise RuntimeError(f"etcd is not running at {etcd_server}")

    # Clean etcd state from previous runs
    if clean_etcd:
        clean_etcd_state(etcd_server)

    # Discover GPU-NIC topology ONCE in main process
    gpu_nic_topology = discover_gpu_nic_topology()
    if gpu_nic_topology is None:
        # Fallback to hardcoded DFW cluster mapping
        gpu_nic_topology = {
            0: "mlx5_0",
            1: "mlx5_1",
            2: "mlx5_2",
            3: "mlx5_4",
            4: "mlx5_5",
            5: "mlx5_6",
            6: "mlx5_7",
            7: "mlx5_8",
        }
        logger.info("Using hardcoded DFW GPU-NIC mapping")
    else:
        logger.info("Discovered GPU-NIC topology: %s", gpu_nic_topology)

    # Determine rank server address
    # Use MASTER_ADDR if set (multi-node SLURM/torchrun), otherwise localhost
    rank_server_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    if rank_server_addr != "127.0.0.1":
        logger.info("Multi-node mode: using MASTER_ADDR=%s", rank_server_addr)

    # Start rank server (use test_rank_server if available, else fall back to elastic one)
    if _USE_TEST_RANK_SERVER:
        logger.info("Starting test rank server on port %d...", rank_server_port)
        server_process = start_test_server(port=rank_server_port)
        time.sleep(1.0)  # Give server more time to start
        # Clean stale barriers from previous runs
        try:
            logger.info(
                "Connecting to rank server at %s:%d to clear state...",
                rank_server_addr,
                rank_server_port,
            )
            client = RankClient(rank_server_addr, rank_server_port)
            cleared = client.clear_barriers()
            if cleared > 0:
                logger.info("Cleared %d stale barriers from rank server", cleared)
            # Also reset rank assignments for clean state
            client.reset()
            logger.info("Rank server ready")
        except Exception as e:
            logger.error("Could not connect to rank server: %s", e)
            raise RuntimeError(
                f"Failed to connect to rank server at {rank_server_addr}:{rank_server_port}: {e}"
            )
    else:
        server_process = rank_server.start_server_process()
        time.sleep(0.5)  # Give server time to start

    # Use spawn context for Queue to match mp.spawn
    spawn_ctx = mp.get_context("spawn")
    result_queue = spawn_ctx.Queue()

    try:
        # Spawn workers using torch.multiprocessing.spawn pattern
        # Note: mp.spawn doesn't support kwargs, so we pass them in args tuple
        # gpu_nic_topology is discovered once and passed to all workers
        ctx = mp.spawn(
            worker_fn,
            args=(
                num_processes,
                test_fn,
                result_queue,
                etcd_server,
                rank_server_addr,
                gpu_nic_topology,
                kwargs,
                rank_server_port,
            ),
            nprocs=num_processes,
            join=False,
            daemon=False,
            start_method="spawn",
        )

        # Wait for all processes with timeout
        deadline = time.time() + timeout
        for p in ctx.processes:
            remaining = max(0.1, deadline - time.time())
            p.join(timeout=remaining)

            if p.is_alive():
                p.terminate()

        # Collect results
        results = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except:
                break

        # Add timeout results for missing ranks
        result_ranks = {r.rank for r in results}
        for i in range(num_processes):
            if i not in result_ranks and -1 not in result_ranks:
                results.append(
                    TestResult(
                        rank=i,
                        test_name=test_fn.__name__,
                        passed=False,
                        error="Timeout or process died",
                    )
                )

        # Sort by rank
        results.sort(key=lambda r: r.rank)

        return results

    finally:
        # Cleanup server
        if server_process and server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=2)


def all_passed(results: List[TestResult]) -> bool:
    """Check if all test results passed."""
    return all(r.passed for r in results)


def print_results(results: List[TestResult], verbose: bool = True):
    """Print test results summary."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    sys.stderr.write(f"\n{'='*60}\n")
    sys.stderr.write(f"Test: {results[0].test_name if results else 'unknown'}\n")
    sys.stderr.write(f"Result: {passed}/{total} ranks passed\n")
    sys.stderr.write(f"{'='*60}\n")

    if verbose:
        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            sys.stderr.write(f"  Rank {r.rank}: {status} ({r.duration_ms:.1f}ms)\n")
            if r.error and not r.passed:
                # Show full error with traceback for debugging
                sys.stderr.write(f"    Error: {r.error}\n")
            if r.metrics:
                for k, v in list(r.metrics.items())[:5]:  # Limit metrics shown
                    sys.stderr.write(f"    {k}: {v}\n")


class MultiProcessTestCase:
    """
    Base class for multi-process functional tests.

    Usage:
        class TestConnection(MultiProcessTestCase):
            num_processes = 8

            def test_connect_ranks(self, rank, world_size, local_rank):
                buffer = create_buffer(rank, world_size)
                other_ranks = [r for r in range(world_size) if r != rank]
                buffer.connect_ranks(other_ranks)
                # ... assertions ...
                buffer.destroy()
                return True
    """

    num_processes = 8
    etcd_server = "http://127.0.0.1:2379"
    timeout = 120.0

    def run_test(self, test_method: Callable, **kwargs) -> List[TestResult]:
        """Run a test method across all ranks."""
        return run_multiprocess_test(
            test_method,
            num_processes=self.num_processes,
            etcd_server=self.etcd_server,
            timeout=self.timeout,
            **kwargs,
        )


# ============================================================================
# Synchronization primitives for multi-process tests
# ============================================================================

# Global rank server address for distributed barriers (set by worker_fn)
_RANK_SERVER_ADDR = None
_RANK_SERVER_PORT = 9998


class DistributedBarrier:
    """TCP-based barrier using test_rank_server for multi-node sync."""

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
        if _USE_TEST_RANK_SERVER:
            client = RankClient(self.server_addr, self.port)
            return client.barrier_wait(self.barrier_id, rank, self.world_size, timeout)
        else:
            raise RuntimeError("DistributedBarrier requires test_rank_server")

    def cleanup(self):
        """No-op for distributed barrier (server handles cleanup)."""
        pass


def sync_all_ranks(
    rank: int,
    world_size: int,
    barrier_name: str,
    timeout: float = 60.0,
    server_addr: str = None,
    port: int = None,
):
    """Synchronize all ranks at a named barrier point."""
    global _RANK_SERVER_ADDR, _RANK_SERVER_PORT

    # Always use TCP-based barrier for multi-node readiness
    master_addr = os.environ.get("MASTER_ADDR", "")
    if server_addr is None:
        server_addr = _RANK_SERVER_ADDR or master_addr or "127.0.0.1"
    if port is None:
        port = _RANK_SERVER_PORT

    barrier = DistributedBarrier(world_size, barrier_name, server_addr, port)
    barrier.wait(rank, timeout)
