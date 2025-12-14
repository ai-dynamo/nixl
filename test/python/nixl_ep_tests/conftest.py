# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared pytest fixtures for NIXL EP Buffer tests.
"""

import os
import sys
from typing import Any, Dict, Optional

import pytest
import torch

# Add nixl_ep to path
# nixl_ep module is in examples/device/ep/
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(TESTS_DIR))
)  # Go up to repo root
NIXL_EP_DIR = os.path.join(REPO_ROOT, "examples", "device", "ep")
sys.path.insert(0, NIXL_EP_DIR)

# Import results reporter (init_reporter used in results_reporter fixture)
from utils.results_reporter import init_reporter  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

# Cluster configuration
GPUS_PER_NODE = 8  # Fixed: 8 GPUs per node

# Test scale configurations
# Format: (num_nodes, total_ranks, description)
TEST_SCALES = [
    (1, 8, "single-node"),  # 1 node,  8 ranks  - NVLink only
    (2, 16, "two-node"),  # 2 nodes, 16 ranks - NVLink + RDMA
    (4, 32, "four-node"),  # 4 nodes, 32 ranks - NVLink + RDMA
]

# Default buffer parameters
DEFAULT_NUM_EXPERTS_PER_RANK = 8
DEFAULT_HIDDEN_DIM = 4096
DEFAULT_NUM_TOKENS = 512
DEFAULT_RDMA_BYTES = 64 * 1024 * 1024  # 64MB


# =============================================================================
# Session-scoped fixtures (initialized once per test session)
# =============================================================================


@pytest.fixture(scope="session")
def cuda_available():
    """Check CUDA availability."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture(scope="session")
def device_count():
    """Get number of available GPUs."""
    return torch.cuda.device_count()


# =============================================================================
# Function-scoped fixtures (initialized per test)
# =============================================================================


@pytest.fixture
def buffer_params():
    """Default buffer parameters."""
    return {
        "num_experts_per_rank": DEFAULT_NUM_EXPERTS_PER_RANK,
        "hidden": DEFAULT_HIDDEN_DIM,
        "num_tokens": DEFAULT_NUM_TOKENS,
        "rdma_bytes": DEFAULT_RDMA_BYTES,
    }


@pytest.fixture
def sample_tensors(buffer_params):
    """Create sample input tensors for dispatch/combine."""
    num_tokens = buffer_params["num_tokens"]
    hidden = buffer_params["hidden"]
    num_experts = buffer_params["num_experts_per_rank"]  # For single rank tests

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, 2), dtype=torch.int64, device="cuda"
    )
    topk_weights = torch.rand(num_tokens, 2, dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)  # Normalize

    return {
        "x": x,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
    }


# =============================================================================
# Buffer fixtures
# =============================================================================


@pytest.fixture
def buffer_single_rank(cuda_available, buffer_params):
    """
    Create a Buffer instance for single-rank tests.
    Automatically cleans up after test.
    """
    from nixl_ep import Buffer

    buf = Buffer(rank=0, explicitly_destroy=True)
    buf.update_memory_buffers(
        num_ranks=1,
        num_experts_per_rank=buffer_params["num_experts_per_rank"],
        num_rdma_bytes=buffer_params["rdma_bytes"],
    )

    yield buf

    buf.destroy()


# =============================================================================
# Multi-rank fixtures (for distributed tests)
# =============================================================================


@pytest.fixture(scope="session")
def dist_info():
    """
    Get distributed training info.
    Returns None if not running in distributed mode.
    """
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return {
                "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
                "backend": dist.get_backend(),
            }
    except Exception:
        pass
    return None


@pytest.fixture
def buffer_multi_rank(cuda_available, buffer_params, dist_info):
    """
    Create a Buffer instance for multi-rank tests.
    Skip if not running in distributed mode.
    """
    if dist_info is None:
        pytest.skip("Not running in distributed mode (use mpirun)")

    import torch.distributed as dist

    from nixl_ep import Buffer

    rank = dist_info["rank"]
    world_size = dist_info["world_size"]

    buf = Buffer(rank=rank, explicitly_destroy=True)
    buf.update_memory_buffers(
        num_ranks=world_size,
        num_experts_per_rank=buffer_params["num_experts_per_rank"],
        num_rdma_bytes=buffer_params["rdma_bytes"],
    )

    # Connect to all other ranks
    other_ranks = [r for r in range(world_size) if r != rank]
    if other_ranks:
        buf.connect_ranks(other_ranks)

    yield buf

    # Cleanup
    if other_ranks:
        buf.disconnect_ranks(other_ranks)
    buf.destroy()
    dist.barrier()


# =============================================================================
# Performance test fixtures
# =============================================================================


@pytest.fixture
def perf_params():
    """Parameters for performance tests."""
    return {
        "warmup_iterations": 10,
        "benchmark_iterations": 100,
        "gpus_per_node": GPUS_PER_NODE,
    }


# =============================================================================
# Rank Mapping Utility
# =============================================================================


class RankMapper:
    """
    Utility for mapping ranks to nodes and categorizing by transport type.

    Assumes:
    - 8 GPUs per node
    - Intra-node communication uses NVLink
    - Inter-node communication uses RDMA

    Example for 16 ranks (2 nodes):
        Node 0: ranks 0-7   (NVLink between them)
        Node 1: ranks 8-15  (NVLink between them)
        RDMA: between any rank on Node 0 and any rank on Node 1
    """

    def __init__(self, world_size: int, gpus_per_node: int = GPUS_PER_NODE):
        self.world_size = world_size
        self.gpus_per_node = gpus_per_node
        self.num_nodes = (world_size + gpus_per_node - 1) // gpus_per_node

    def get_node_id(self, rank: int) -> int:
        """Map rank to node ID."""
        return rank // self.gpus_per_node

    def get_local_rank(self, rank: int) -> int:
        """Get rank's position within its node (0-7)."""
        return rank % self.gpus_per_node

    def get_ranks_on_node(self, node_id: int) -> list:
        """Get all ranks on a specific node."""
        start = node_id * self.gpus_per_node
        end = min(start + self.gpus_per_node, self.world_size)
        return list(range(start, end))

    def is_same_node(self, rank1: int, rank2: int) -> bool:
        """Check if two ranks are on the same node (NVLink path)."""
        return self.get_node_id(rank1) == self.get_node_id(rank2)

    def get_transport(self, rank1: int, rank2: int) -> str:
        """
        Get transport type between two ranks.

        Returns:
            'nvlink' if same node
            'rdma' if different nodes
        """
        if self.is_same_node(rank1, rank2):
            return "nvlink"
        return "rdma"

    def get_intra_node_ranks(self, rank: int) -> list:
        """Get ranks on the same node (excluding self). These use NVLink."""
        my_node = self.get_node_id(rank)
        return [r for r in self.get_ranks_on_node(my_node) if r != rank]

    def get_inter_node_ranks(self, rank: int) -> list:
        """Get ranks on different nodes. These use RDMA."""
        my_node = self.get_node_id(rank)
        return [
            r
            for r in range(self.world_size)
            if r != rank and self.get_node_id(r) != my_node
        ]

    def get_all_other_ranks(self, rank: int) -> list:
        """Get all ranks except self."""
        return [r for r in range(self.world_size) if r != rank]

    def categorize_ranks(self, rank: int) -> dict:
        """
        Categorize all other ranks by transport type.

        Returns:
            {
                'nvlink': [ranks using NVLink],
                'rdma': [ranks using RDMA],
                'all': [all other ranks]
            }
        """
        intra = self.get_intra_node_ranks(rank)
        inter = self.get_inter_node_ranks(rank)
        return {
            "nvlink": intra,
            "rdma": inter,
            "all": intra + inter,
        }

    def get_scale_info(self) -> dict:
        """Get information about the current test scale."""
        return {
            "world_size": self.world_size,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "has_nvlink": True,  # Always true (intra-node)
            "has_rdma": self.num_nodes > 1,  # Only if multi-node
        }

    def __repr__(self):
        return f"RankMapper(world_size={self.world_size}, nodes={self.num_nodes}, gpus_per_node={self.gpus_per_node})"


# =============================================================================
# Helpers available to all tests (legacy compatibility)
# =============================================================================


def get_node_id(rank: int, gpus_per_node: int = GPUS_PER_NODE) -> int:
    """Map rank to node ID."""
    return rank // gpus_per_node


def is_intra_node(rank1: int, rank2: int, gpus_per_node: int = GPUS_PER_NODE) -> bool:
    """Check if two ranks are on the same node."""
    return get_node_id(rank1, gpus_per_node) == get_node_id(rank2, gpus_per_node)


# =============================================================================
# Rank Mapper Fixture
# =============================================================================


@pytest.fixture
def rank_mapper(dist_info):
    """
    Get a RankMapper for the current distributed environment.
    """
    if dist_info is None:
        # Single rank fallback
        return RankMapper(world_size=1)
    return RankMapper(world_size=dist_info["world_size"])


# =============================================================================
# Results Reporter Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def results_reporter():
    """
    Get the results reporter for saving test metrics.

    Usage in tests:
        def test_something(results_reporter):
            # ... run test ...
            results_reporter.add_result(
                test_name='P-CONN-01',
                category='connection',
                metric='latency_ms',
                value=123.45,
                params={'num_ranks': 8}
            )
    """
    results_dir = os.environ.get("NIXL_TEST_RESULTS_DIR", "./results")
    reporter = init_reporter(results_dir=results_dir)

    yield reporter

    # Save results at end of session
    if reporter.results:
        reporter.save()
        reporter.save_summary()


@pytest.fixture
def record_result(results_reporter):
    """
    Convenience fixture for recording results.

    Usage:
        def test_connect_latency(record_result):
            latency = measure_connect()
            record_result('P-CONN-01', 'connection', 'latency_ms', latency)
    """

    def _record(
        test_name: str,
        category: str,
        metric: str,
        value: float,
        unit: str = "",
        params: Optional[Dict[str, Any]] = None,
    ):
        results_reporter.add_result(
            test_name=test_name,
            category=category,
            metric=metric,
            value=value,
            unit=unit,
            params=params,
        )

    return _record
