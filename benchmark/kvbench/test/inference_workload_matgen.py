# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils to generate matrices that represent the communication patterns of an inference workload.

Generates rank-to-rank RDMA communication matrices for disaggregated prefill/decode.
Each matrix represents one LLM forward pass with compute time estimation.

Disclaimers:
- For now there is only support for TP and CP
- The compute time estimation is naive (configurable via --flops-per-gpu, default 1000 TFLOPS for H100)
- The batching is very naive and just aggregates requests into batches until it exceeds max_batch_mem or batch_size

Example usage:
python inference_workload_matgen.py generate \
    --num-user-requests 10 \
    --batch-size 1 \
    --num-prefill-nodes 54 \
    --num-decode-nodes 54 \
    --prefill-tp 8 \
    --prefill-pp 1 \
    --prefill-cp 1 \
    --decode-tp 8 \
    --decode-pp 1 \
    --decode-cp 1 \
    --results-dir /tmp/matrices \
    --isl-mean 16000 \
    --isl-scale 10000 \
    --min-isl 1000 \
    --max-isl 128000 \
    --max-batch-mem 100000000000 \
    --model llama-405b \
    --prefix-hit-rate 0.75

With --prefix-hit-rate specified, the metadata.yaml includes storage configuration.
Storage files are created by kvbench at runtime (not by matgen).
"""

from dataclasses import dataclass
from itertools import cycle
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


from nixl.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a language model."""

    hidden_size: int  # Model's hidden dimension (H)
    num_layers: int  # Number of layers (L)
    num_heads: int = 1  # Number of attention heads (N_heads)
    num_kv_heads: Optional[int] = None  # Number of key/value heads (for MQA/GQA)
    dtype_size: float = 2  # Size in bytes (2 for FP16, 4 for FP32)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    def bytes_per_token(self):
        if self.num_kv_heads is not None:
            return (
                2
                * self.head_dim
                * self.num_kv_heads
                * self.num_layers
                * self.dtype_size
            )
        else:
            return 2 * self.hidden_size * self.num_layers * self.dtype_size

    def kv_cache_size(self, isl):
        """KV cache size in bytes"""
        return isl * self.bytes_per_token()


@dataclass
class TaskConfig:
    """Configuration for an inference task."""

    isl_mean: int = 0  # Context/sequence length (S)
    isl_scale: int = 10000  # Context/sequence length scale
    min_isl: int = 1000  # Minimum context/sequence length
    max_isl: int = 128000  # Maximum context/sequence length
    batch_size: int = 1  # Batch size (B)
    max_batch_mem: float = 100e9  # Maximum batch memory (100GB)


@dataclass
class WorkerConfig:
    """Configuration for the GPU cluster."""

    tp: int
    pp: int = 1
    cp: int = 1
    ep: int = 1

    def __post_init__(self):
        assert self.pp == 1, "PP is not supported yet"
        assert self.ep == 1, "EP is not supported yet"


@dataclass
class UserRequest:
    isl: int

    @classmethod
    def rand(
        cls,
        mean: int = 16000,
        scale: int = 10000,
        min_isl: int = 1000,
        max_isl: int = 128000,
    ):
        # Sample and clip to ensure we stay within bounds
        isl = int(np.random.normal(mean, scale))
        # Ensure ISL is positive
        isl = min(max(min_isl, isl), max_isl)
        return cls(isl=isl)


@dataclass
class Batch:
    user_requests: List[UserRequest]

    def kv_cache_size(self, model_config: ModelConfig):
        """KV cache size in bytes"""
        return sum(model_config.kv_cache_size(req.isl) for req in self.user_requests)

    @property
    def size(self):
        return len(self.user_requests)

    @property
    def total_isl(self):
        return sum(req.isl for req in self.user_requests)


@dataclass
class TransferMatrix:
    matrix: np.ndarray
    compute_time: float
    isl: int
    sender_ranks: List[int]  # Which ranks are senders in this matrix


def gen_batches(
    num_user_requests: int,
    task_config: TaskConfig,
    model_config: ModelConfig,
    max_batch_mem: float = 100e9,  # 100GB - capacity of a gpu
):
    """
    For now very naive, aggregate requests into batches until it exceeds max_batch_mem or batch_size
    Args:
        - num_user_requests: Number of user requests
        - task_config: Task configuration
    """
    batches = []
    curr = []
    curr_mem = 0
    for _ in range(num_user_requests):
        req = UserRequest.rand(
            task_config.isl_mean,
            task_config.isl_scale,
            task_config.min_isl,
            task_config.max_isl,
        )
        curr_mem += model_config.kv_cache_size(req.isl)
        curr.append(req)
        if curr_mem > task_config.max_batch_mem or len(curr) >= task_config.batch_size:
            batches.append(Batch(user_requests=curr))
            curr = []
            curr_mem = 0
    if curr:
        batches.append(Batch(user_requests=curr))
        logger.warning("Last batch is incomplete, with size %d", len(curr))

    return batches


def gen_matrices_and_compute_time(
    batches: List[Batch],
    prefill_workers: List[List[int]],
    decode_workers: List[List[int]],
    model_config: ModelConfig,
    prefill_worker_config: WorkerConfig,
    decode_worker_config: WorkerConfig,
    flops_per_gpu: float = 1000 * 1e12,
) -> List[TransferMatrix]:
    """Generate communication matrices for all batches.

    Args:
        batches: List of batches
        prefill_workers: List of prefill worker rank groups
        decode_workers: List of decode worker rank groups
        model_config: Model configuration
        prefill_worker_config: Prefill worker configuration
        decode_worker_config: Decode worker configuration

    Returns:
        List of TransferMatrix objects
    """
    # Handle storage-only mode with no decode workers
    storage_only_no_decode = len(decode_workers) == 0

    workers_coupling: list[tuple[list[int], list[int] | None]]
    if not storage_only_no_decode:
        # Support N:1 ratio (multiple prefill workers per decode worker)
        assert len(prefill_workers) >= len(
            decode_workers
        ), f"Prefill workers ({len(prefill_workers)}) must be >= decode workers ({len(decode_workers)})"
        assert (
            len(prefill_workers) % len(decode_workers) == 0
        ), f"Prefill workers ({len(prefill_workers)}) must be divisible by decode workers ({len(decode_workers)})"

        # Assertions
        all_ranks = list(
            r for worker in prefill_workers + decode_workers for r in worker
        )
        world_size = max(all_ranks) + 1
        assert set(all_ranks) == set(range(world_size)), "Ranks are missing"

        # Pair prefill workers with decode workers (cycling decode if N:1)
        ratio = len(prefill_workers) // len(decode_workers)
        workers_coupling = [
            (pw, decode_workers[i // ratio]) for i, pw in enumerate(prefill_workers)
        ]
    else:
        # Storage-only: only prefill workers, no decode
        all_ranks = list(r for worker in prefill_workers for r in worker)
        world_size = max(all_ranks) + 1
        assert set(all_ranks) == set(range(world_size)), "Ranks are missing"
        # Pair each prefill worker with None for decode
        workers_coupling = [(pw, None) for pw in prefill_workers]

    workers_pool = cycle(workers_coupling)
    matrices = []

    for batch in tqdm(batches, desc="Generating matrices"):
        prefill_worker, decode_worker = next(workers_pool)

        if decode_worker is not None:
            # Normal mode: generate RDMA matrix
            mat = gen_matrix(
                batch,
                world_size,
                prefill_worker,
                decode_worker,
                model_config,
                prefill_worker_config,
                decode_worker_config,
            )
        else:
            # Storage-only with no decode: create empty matrix placeholder
            mat = np.zeros((world_size, world_size), dtype=np.int64)

        compute_time = estimate_compute_time(
            batch, model_config, prefill_worker_config, flops_per_gpu
        )
        matrix_obj = TransferMatrix(
            matrix=mat,
            compute_time=compute_time,
            isl=batch.total_isl,
            sender_ranks=prefill_worker,
        )

        matrices.append(matrix_obj)

    return matrices


def gen_matrix(
    batch: Batch,
    world_size: int,
    prefill_worker: List[int],
    decode_worker: List[int],
    model_config: ModelConfig,
    prefill_worker_config: WorkerConfig,
    decode_worker_config: WorkerConfig,
):
    """Generate rank-to-rank RDMA communication matrix for a batch.

    Matrix layout: world_size x world_size
    Each entry [i,j] = bytes sent from rank i to rank j
    """
    kv_size = batch.kv_cache_size(model_config)
    kv_slice_size = (
        kv_size
        / prefill_worker_config.tp
        / prefill_worker_config.pp
        / prefill_worker_config.cp
    )

    num_peers = (
        decode_worker_config.tp / prefill_worker_config.tp / prefill_worker_config.cp
    )
    if num_peers % 1 != 0:
        raise ValueError("Prefill TP*Prefill CP must be a divisor of decode TP")
    num_peers = int(num_peers)
    buf_size = kv_slice_size / num_peers

    mat = np.zeros((world_size, world_size))

    # Prefill → Decode RDMA transfers
    dst_iter = iter(decode_worker)
    for rank in prefill_worker:
        for _ in range(num_peers):
            dst = next(dst_iter)
            mat[rank, dst] = buf_size

    return mat


def estimate_compute_time(
    batch: Batch,
    model_config: ModelConfig,
    worker_config: WorkerConfig,
    flops_per_gpu: float = 1000 * 1e12,  # 1000 TFlops (H100 FP16, conservative)
):
    """Estimate the compute time of a batch, in seconds.

    Accounts for:
    - Attention FLOPs: O(S²) per layer
    - MLP FLOPs: O(S × H) per layer (assumes intermediate_size = 4*H)
    - TP parallelism: divides compute across TP GPUs

    Formula based on:
    https://medium.com/@plienhar/llm-inference-series-3-kv-caching-unveiled-048152e461c8
    """
    B = batch.size
    S = batch.total_isl
    H = model_config.hidden_size
    L = model_config.num_layers

    # Attention score computation: 4 * B * S² * H per layer (Q·Kᵀ and Attn·V)
    # Note: excludes QKV and output projections (8 * B * S * H² per layer).
    # These are linear ops like MLP but omitted here for simplicity since
    # they are relatively small compared to MLP (8H² vs 16H²).
    attn_flop = 4 * B * S**2 * H * L

    # MLP: 16 * B * S * H² per layer (H→4H and 4H→H projections)
    mlp_flop = 16 * B * S * H**2 * L

    total_flop = attn_flop + mlp_flop

    # TP divides compute across GPUs
    effective_flops = flops_per_gpu * worker_config.tp

    return total_flop / effective_flops


def format_size(nbytes: float, precision=2) -> str:
    if nbytes == 0:
        return "0"

    units = ["B", "K", "M", "G"]
    units_ix = 0
    while nbytes / 1024 >= 1 and units_ix < len(units) - 1:
        nbytes /= 1024
        units_ix += 1

    nbytes = round(nbytes, precision)
    return f"{nbytes:g}{units[units_ix]}"


def main(
    num_user_requests: int,
    task_config: TaskConfig,
    num_prefill_gpus: int,
    num_decode_gpus: int,
    prefill_worker_config: WorkerConfig,
    decode_worker_config: WorkerConfig,
    model_config: ModelConfig,
    results_dir: Optional[PathLike] = None,
    rail_optimized: bool = False,
    prefix_hit_rate: Optional[float] = None,
    flops_per_gpu: float = 1000 * 1e12,
    storage_only: bool = False,
    read_only: bool = False,
    all_nodes_per_pattern: bool = False,
    mem_type: str = "cuda",
    iters: int = 10,
    isolation_iters: int = 5,
):
    """Generate communication matrices for inference workload.

    Args:
        num_user_requests: Number of user requests to simulate
        task_config: Task configuration
        num_prefill_gpus: Number of GPUs for prefill
        num_decode_gpus: Number of GPUs for decode
        prefill_worker_config: Prefill worker configuration
        decode_worker_config: Decode worker configuration
        model_config: Model configuration
        results_dir: Directory to save results
        rail_optimized: Whether to reorder decode workers for rail-optimized communication
        prefix_hit_rate: Fraction of KV cache to read from storage (0.0-1.0)
        storage_only: If True, generate storage-only config (no RDMA matrices)
        read_only: If True, skip write operations in storage patterns
        all_nodes_per_pattern: If True, all prefill nodes are active in each pattern
        mem_type: Memory type for storage operations (cuda, cpu)
        iters: Number of iterations per traffic pattern
        isolation_iters: Number of isolation iterations

    Returns:
        matrices
    """
    # Rules of thumb - only apply when there are decode workers
    if num_decode_gpus > 0:
        assert (
            prefill_worker_config.tp <= decode_worker_config.tp
        ), "Prefill TP must be less than or equal to decode TP"
        assert (
            prefill_worker_config.pp >= decode_worker_config.pp
        ), "Prefill PP must be more or equal to decode PP"
        assert (
            prefill_worker_config.cp >= decode_worker_config.cp
        ), "Prefill CP must be more or equal to decode CP"
        assert decode_worker_config.cp == 1, "Decode CP must be 1"
        assert (
            prefill_worker_config.ep <= decode_worker_config.ep
        ), "Prefill EP must be less or equal to decode EP"
    if rail_optimized:
        assert (
            decode_worker_config.tp == 8
        ), "Rail optimized communication is only supported when decode worker is a full node (8 GPUs)"
        assert (
            prefill_worker_config.tp == 4
        ), "Rail optimized communication is only supported when prefill worker is half a node (4 GPUs)"

    # Create workers - group of gpus that do prefill/decode
    prefill_worker_size = (
        prefill_worker_config.tp * prefill_worker_config.pp * prefill_worker_config.cp
    )
    decode_worker_size = (
        decode_worker_config.tp * decode_worker_config.pp * decode_worker_config.cp
    )
    world_size = num_prefill_gpus + num_decode_gpus

    # Create list of all GPU ranks
    prefill_ranks = list(range(num_prefill_gpus))
    decode_ranks = list(range(num_prefill_gpus, num_prefill_gpus + num_decode_gpus))

    # Chunk the ranks into worker groups
    prefill_workers = [
        prefill_ranks[i : i + prefill_worker_size]
        for i in range(0, len(prefill_ranks), prefill_worker_size)
    ]

    decode_workers = [
        decode_ranks[i : i + decode_worker_size]
        for i in range(0, len(decode_ranks), decode_worker_size)
    ]
    if rail_optimized:
        # Reorder the decode workers to match rail-optimized communication
        reordered = []
        order = [0, 4, 1, 5, 2, 6, 3, 7]
        for worker in decode_workers:
            new_worker = [worker[ix] for ix in order]
            reordered.append(new_worker)

        decode_workers = reordered

    logger.info("Prefill workers: %s", prefill_workers)
    logger.info("Decode workers: %s", decode_workers)

    batches = gen_batches(num_user_requests, task_config, model_config)
    logger.info("Generated %d batches", len(batches))
    matrices = gen_matrices_and_compute_time(
        batches,
        prefill_workers,
        decode_workers,
        model_config,
        prefill_worker_config,
        decode_worker_config,
        flops_per_gpu,
    )

    # Save matrices and metadata to files
    results_dir = results_dir or Path(f"matrices_{world_size}ranks")
    results_dir = Path(results_dir)
    logger.info("Saving %d matrices to %s", len(matrices), results_dir)

    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    tps_dir = results_dir / "tps"
    tps_dir.mkdir(parents=True, exist_ok=True)
    if not storage_only:
        matrices_dir = results_dir / "matrices"
        matrices_dir.mkdir(parents=True, exist_ok=True)

    storage_enabled = prefix_hit_rate is not None or storage_only
    hit_rate = (
        prefix_hit_rate if prefix_hit_rate is not None else 1.0
    )  # 100% read for storage_only

    if storage_enabled:
        logger.info(
            "Storage enabled with prefix_hit_rate=%.1f%%, mem_type=%s",
            hit_rate * 100,
            mem_type,
        )
    if storage_only:
        logger.info("Storage-only mode: skipping RDMA matrix files")
    if all_nodes_per_pattern:
        logger.info(
            "All-nodes-per-pattern mode: all prefill nodes active in each pattern"
        )

    # Build metadata
    metadata: dict[str, Any] = {
        "traffic_patterns": [],
        "iters": iters,
        "isolation_iters": isolation_iters,
    }

    for idx, matrix in enumerate(tqdm(matrices, desc="Saving traffic patterns")):
        # Compute per-rank storage sizes (if storage enabled)
        has_read = False
        has_write = False
        read_sizes: List[str] = []
        write_sizes: List[str] = []

        if storage_enabled:
            world_size = matrix.matrix.shape[0]
            read_sizes = ["0"] * world_size
            write_sizes = ["0"] * world_size

            if all_nodes_per_pattern:
                kv_size = model_config.kv_cache_size(matrix.isl)
                per_rank_size = int(
                    kv_size
                    / prefill_worker_config.tp
                    / prefill_worker_config.pp
                    / prefill_worker_config.cp
                )
                read_size = int(per_rank_size * hit_rate)
                write_size = 0 if read_only else int(per_rank_size * (1 - hit_rate))

                for rank in range(num_prefill_gpus):
                    if read_size > 0:
                        read_sizes[rank] = format_size(read_size)
                    if write_size > 0:
                        write_sizes[rank] = format_size(write_size)
            elif storage_only:
                kv_size = model_config.kv_cache_size(matrix.isl)
                per_rank_size = int(
                    kv_size
                    / prefill_worker_config.tp
                    / prefill_worker_config.pp
                    / prefill_worker_config.cp
                )
                read_size = int(per_rank_size * hit_rate)
                write_size = 0 if read_only else int(per_rank_size * (1 - hit_rate))

                num_prefill_workers = num_prefill_gpus // (
                    prefill_worker_config.tp
                    * prefill_worker_config.pp
                    * prefill_worker_config.cp
                )
                worker_idx = idx % num_prefill_workers
                worker_size = (
                    prefill_worker_config.tp
                    * prefill_worker_config.pp
                    * prefill_worker_config.cp
                )
                start_rank = worker_idx * worker_size

                for rank in range(start_rank, start_rank + worker_size):
                    if read_size > 0:
                        read_sizes[rank] = format_size(read_size)
                    if write_size > 0:
                        write_sizes[rank] = format_size(write_size)
            else:
                for rank in matrix.sender_ranks:
                    transfer_size = int(matrix.matrix[rank].sum())
                    if transfer_size > 0:
                        read_size = int(transfer_size * hit_rate)
                        write_size = (
                            0 if read_only else int(transfer_size * (1 - hit_rate))
                        )
                        if read_size > 0:
                            read_sizes[rank] = format_size(read_size)
                        if write_size > 0:
                            write_sizes[rank] = format_size(write_size)

            has_read = any(s != "0" for s in read_sizes)
            has_write = any(s != "0" for s in write_sizes)

        # Write unified .tp file with [rdma], [read], [write] sections
        tp_path = tps_dir / f"tp_{idx}.tp"
        with open(tp_path, "w") as f:
            if not storage_only:
                f.write("[rdma]\n")
                for row in matrix.matrix:
                    f.write(" ".join(format_size(val) for val in row) + "\n")
            if has_read:
                f.write("\n[read]\n")
                f.write(" ".join(read_sizes) + "\n")
            if has_write:
                f.write("\n[write]\n")
                f.write(" ".join(write_sizes) + "\n")

        # Also write legacy matrix file for backward compatibility
        if not storage_only:
            matrix_path = matrices_dir / f"matrix_{idx}.txt"
            with open(matrix_path, "w") as f:
                for row in matrix.matrix:
                    f.write(" ".join(format_size(val) for val in row) + "\n")

        # Build YAML entry with BOTH tp_file (new) and matrix_file+storage (legacy)
        # New main.py checks tp_file first; old main.py ignores tp_file, uses matrix_file
        tp_entry: Dict[str, Any] = {
            "tp_file": f"tps/tp_{idx}.tp",
            "sleep_before_launch_sec": matrix.compute_time * (1 - hit_rate),
            "metadata": {
                "isl": matrix.isl,
            },
        }
        # Legacy keys for backward compatibility with old main.py
        if not storage_only:
            tp_entry["matrix_file"] = f"matrices/matrix_{idx}.txt"
        if storage_only or storage_enabled:
            tp_entry["mem_type"] = mem_type
        if has_read or has_write:
            storage_entry: Dict[str, List[str]] = {}
            if has_read:
                storage_entry["read"] = read_sizes
            if has_write:
                storage_entry["write"] = write_sizes
            tp_entry["storage"] = storage_entry

        metadata["traffic_patterns"].append(tp_entry)

    # Save metadata to YAML
    metadata_path = results_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=None, width=1000, sort_keys=False)
        logger.info("Saved metadata to %s", metadata_path)

    # Print summary
    total_patterns = len(metadata["traffic_patterns"])
    logger.info("Generated %d traffic patterns", total_patterns)
    if storage_only:
        logger.info(
            "Model: %d layers, %d KV heads, head_dim=%d",
            model_config.num_layers,
            model_config.num_kv_heads or model_config.num_heads,
            model_config.head_dim,
        )
        logger.info("Bytes per token: %s", format_size(model_config.bytes_per_token()))


if __name__ == "__main__":
    import click

    PREDEFINED_MODELS = {
        "llama-8b": ModelConfig(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            dtype_size=2,
        ),
        "llama-70b": ModelConfig(
            hidden_size=8192,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,
            dtype_size=2,
        ),
        "llama-405b": ModelConfig(
            hidden_size=16384,
            num_layers=126,
            num_heads=128,
            num_kv_heads=8,
            dtype_size=2,
        ),
        "qwen3-30B": ModelConfig(
            hidden_size=32768, num_layers=48, num_heads=32, num_kv_heads=4, dtype_size=2
        ),
        "deepseek-r1-distill-llama-70b": ModelConfig(
            hidden_size=8192, num_layers=80, num_heads=64, num_kv_heads=8, dtype_size=2
        ),
        "deepseek-r1": ModelConfig(
            hidden_size=12288,
            num_layers=100,
            num_heads=96,
            num_kv_heads=12,
            dtype_size=2,
        ),
    }

    @click.group()
    def cli():
        """Generate communication matrices for inference workloads"""
        pass

    @cli.command()
    @click.option(
        "--num-user-requests",
        type=int,
        default=1000,
        help="Number of user requests to simulate",
    )
    @click.option("--batch-size", type=int, default=1, help="Batch size for requests")
    @click.option(
        "--num-prefill-nodes",
        type=int,
        required=True,
        help="Number of nodes for prefill",
    )
    @click.option(
        "--num-decode-nodes", type=int, required=True, help="Number of nodes for decode"
    )
    @click.option(
        "--prefill-tp", type=int, default=1, help="Tensor parallelism for prefill"
    )
    @click.option(
        "--prefill-pp", type=int, default=1, help="Pipeline parallelism for prefill"
    )
    @click.option(
        "--prefill-cp",
        type=int,
        default=1,
        help="Communication parallelism for prefill",
    )
    @click.option(
        "--decode-tp", type=int, default=1, help="Tensor parallelism for decode"
    )
    @click.option(
        "--decode-pp", type=int, default=1, help="Pipeline parallelism for decode"
    )
    @click.option(
        "--decode-cp", type=int, default=1, help="Communication parallelism for decode"
    )
    @click.option("--model", type=str, help="Name of predefined model")
    @click.option(
        "--hidden-size", type=int, help="Model hidden size (for custom model)"
    )
    @click.option(
        "--num-layers", type=int, help="Number of model layers (for custom model)"
    )
    @click.option(
        "--num-heads", type=int, help="Number of attention heads (for custom model)"
    )
    @click.option(
        "--num-kv-heads",
        type=int,
        help="Number of KV attention heads (for custom model)",
    )
    @click.option(
        "--dtype-size", type=int, help="Size of model dtype in bytes (for custom model)"
    )
    @click.option("--results-dir", type=str, help="Directory to save results")
    @click.option(
        "--isl-mean", default=16000, type=int, help="Mean context/sequence length"
    )
    @click.option(
        "--isl-scale", default=10000, type=int, help="Scale context/sequence length"
    )
    @click.option(
        "--min-isl", default=1000, type=int, help="Minimum context/sequence length"
    )
    @click.option(
        "--max-isl", default=128000, type=int, help="Maximum context/sequence length"
    )
    @click.option(
        "--max-batch-mem", default=100e9, type=float, help="Maximum batch memory"
    )
    @click.option(
        "--rail-optimized/--no-rail-optimized",
        default=False,
        help="Whether to use rail optimization",
    )
    @click.option("--ppn", default=8, type=int, help="Number of GPUs per node")
    @click.option(
        "--flops-per-gpu",
        type=float,
        default=1000e12,
        help="GPU FLOPS for compute time estimation (default: 1000 TFLOPS for H100 FP16)",
    )
    @click.option(
        "--prefix-hit-rate",
        type=click.FloatRange(0.0, 1.0),
        default=None,
        help="Prefix hit rate (0.0-1.0). Enables storage when specified. 0.0=write-only, 1.0=read-only.",
    )
    @click.option(
        "--storage-only/--no-storage-only",
        default=False,
        help="Generate storage-only config (no RDMA matrices). Uses 100%% read if prefix-hit-rate not set.",
    )
    @click.option(
        "--read-only/--no-read-only",
        default=False,
        help="Generate read-only storage patterns (no writes). Default: False",
    )
    @click.option(
        "--all-nodes-per-pattern/--no-all-nodes-per-pattern",
        default=False,
        help="Make all prefill nodes active in each traffic pattern (for storage benchmarks).",
    )
    @click.option(
        "--mem-type",
        type=str,
        default="cuda",
        help="Memory type for storage operations (cuda, cpu). Default: cuda",
    )
    @click.option(
        "--iters",
        type=int,
        default=10,
        help="Number of iterations per traffic pattern. Default: 10",
    )
    @click.option(
        "--isolation-iters",
        type=int,
        default=5,
        help="Number of isolation iterations. Default: 5",
    )
    def generate(
        num_user_requests,
        batch_size,
        num_prefill_nodes,
        num_decode_nodes,
        prefill_tp,
        prefill_pp,
        prefill_cp,
        decode_tp,
        decode_pp,
        decode_cp,
        model,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        dtype_size,
        results_dir,
        isl_mean,
        isl_scale,
        min_isl,
        max_isl,
        max_batch_mem,
        rail_optimized,
        ppn,
        flops_per_gpu,
        prefix_hit_rate,
        storage_only,
        read_only,
        all_nodes_per_pattern,
        mem_type,
        iters,
        isolation_iters,
    ):
        """Generate communication matrices for given configuration"""

        if model:
            if model not in PREDEFINED_MODELS:
                raise click.BadParameter(
                    f"Unknown model {model}. Available models: {list(PREDEFINED_MODELS.keys())}"
                )
            model_config = PREDEFINED_MODELS[model]
        else:
            if not all([hidden_size, num_layers, num_heads, num_kv_heads, dtype_size]):
                raise click.BadParameter(
                    "Must specify either --model or all custom model parameters"
                )
            model_config = ModelConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dtype_size=dtype_size,
            )

        task_config = TaskConfig(
            isl_mean=isl_mean,
            isl_scale=isl_scale,
            min_isl=min_isl,
            max_isl=max_isl,
            batch_size=batch_size,
            max_batch_mem=max_batch_mem,
        )

        main(
            num_user_requests=num_user_requests,
            task_config=task_config,
            num_prefill_gpus=num_prefill_nodes * ppn,
            num_decode_gpus=num_decode_nodes * ppn,
            prefill_worker_config=WorkerConfig(
                tp=prefill_tp, pp=prefill_pp, cp=prefill_cp
            ),
            decode_worker_config=WorkerConfig(tp=decode_tp, pp=decode_pp, cp=decode_cp),
            model_config=model_config,
            results_dir=results_dir,
            rail_optimized=rail_optimized,
            prefix_hit_rate=prefix_hit_rate,
            flops_per_gpu=flops_per_gpu,
            storage_only=storage_only,
            read_only=read_only,
            all_nodes_per_pattern=all_nodes_per_pattern,
            mem_type=mem_type,
            iters=iters,
            isolation_iters=isolation_iters,
        )

    cli()
