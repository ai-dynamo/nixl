"""Utils to generate matrices that represent the communication patterns of an inference workload

Q:
    - the batch size is fixed?
"""

from tqdm import tqdm
from pathlib import Path
import os
from os import PathLike
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
import json
import logging
from itertools import cycle
import yaml

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    hidden_size: int        # Model's hidden dimension (H)
    num_layers: int         # Number of layers (L)
    num_heads: int   =1       # Number of attention heads (N_heads)
    num_kv_heads: int = None  # Number of key/value heads (for MQA/GQA)
    dtype_size: float = 2   # Size in bytes (2 for FP16, 4 for FP32)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    def bytes_per_token(self):
        #return 2 * self.hidden_size * self.num_layers * self.dtype_size * self.num_heads
        if self.num_kv_heads is not None:
            return 2 * self.head_dim * self.num_kv_heads * self.num_layers * self.dtype_size
        else:
            return 2 * self.hidden_size * self.num_layers * self.dtype_size
    
    def kv_cache_size(self, isl):
        """KV cache size in bytes"""
        return isl * self.bytes_per_token()

@dataclass
class TaskConfig:
    """Configuration for an inference task."""
    base_size: float = 0.0      # Base size in GB (deprecated when using model_config)
    context_size: int = 0       # Context/sequence length (S)
    batch_size: int = 1         # Batch size (B)
    model_config: Optional[ModelConfig] = None  # Model configuration
    computation_overhead: float = 1.0  # Computation overhead factor (1.0 = standard)
    fixed_duration: Optional[float] = None  # Optional fixed duration (overrides calculation)
    probability_distribution: str = "uniform"  # Distribution type for task arrival


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
        assert self.cp == 1, "CP is not supported yet"


@dataclass
class UserRequest:
    isl: int

    @classmethod
    def rand(cls, mean: int = 32000, scale: int = 10000):
        # Sample and clip to ensure we stay within bounds
        isl = int(np.random.normal(mean, scale))
        # Ensure ISL is positive
        isl = min(max(8000, isl), 128000)
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



def gen_batches(
    num_user_requests: int,
    batch_size: int,
    model_config: ModelConfig,
    max_batch_mem: int = 100E9, # 100GB - capacity of a gpu
):
    """
    For now very naive, aggregate requests into batches until it exceeds max_batch_mem or batch_size
    Args:
        - num_user_requests: Number of user requests
        - batch_size: Batch size
    """
    batches = []
    curr = []
    curr_mem = 0
    for req in range(num_user_requests):
        req = UserRequest.rand()
        curr_mem += model_config.kv_cache_size(req.isl)
        curr.append(req)
        if curr_mem > max_batch_mem or len(curr) >= batch_size:
            batches.append(Batch(user_requests=curr))
            curr = []
            curr_mem = 0
    if curr:
        batches.append(Batch(user_requests=curr))
        print(f"Last batch is incomplete, his size is {len(curr)}")

    return batches

def gen_matrices_and_compute_time(
    batches: List[Batch],
    prefill_workers: List[List[int]],
    decode_workers: List[List[int]],
    model_config: ModelConfig,
    prefill_worker_config: WorkerConfig,
    decode_worker_config: WorkerConfig,
) -> List[TransferMatrix]:
    """
    Args:
        - batches: List of batches
    """
    # For now, every prefill worker is bound to a single decode worker
    assert len(prefill_workers) == len(decode_workers)

    # Assertions
    all_ranks = list(r for worker in prefill_workers + decode_workers for r in worker)
    world_size = max(all_ranks) + 1
    assert set(all_ranks) == set(range(world_size)), "Ranks are missing"


    workers_coupling = list(zip(prefill_workers, decode_workers))


    workers_pool = cycle(workers_coupling)
    matrices = []

    sorted_batches = sorted(batches, key=lambda x: x.kv_cache_size(model_config))


    for batch in tqdm(sorted_batches, desc="Generating matrices"): # Sorting the batches so that the load is equally distributed on the workers

        prefill_worker, decode_worker = next(workers_pool)
        mat = gen_matrix(batch, world_size, prefill_worker, decode_worker, model_config, prefill_worker_config, decode_worker_config)

        compute_time = estimate_compute_time(batch, model_config)
        matrix_obj = TransferMatrix(matrix=mat, compute_time=compute_time, isl=batch.total_isl)
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
    """TODO check logic """
    kv_size = batch.kv_cache_size(model_config)
    kv_slice_size = kv_size / prefill_worker_config.tp / prefill_worker_config.pp

    buf_size = kv_slice_size / decode_worker_config.tp
    #print(f"kv_size: {format_size(kv_size)}, kv_slice_size: {format_size(kv_slice_size)}, buf_size: {format_size(buf_size)}, num_peers: {num_peers}")

    num_peers = decode_worker_config.tp // prefill_worker_config.tp

    mat = np.zeros((world_size, world_size))

    dst_iter = iter(decode_worker)
    for rank in prefill_worker:
        for _ in range(num_peers):
            dst = next(dst_iter)
            mat[rank, dst] = buf_size
    return mat

def estimate_compute_time(
    batch: Batch,
    model_config: ModelConfig,
    flops = 36*1E12, # 36 TFlops (h100)
):
    """Estimate the compute time of a batch, in seconds"""
    flop = 2 * batch.size * model_config.num_layers * model_config.hidden_size * batch.total_isl**2
    return flop / flops


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
    batch_size: int,
    num_prefill_gpus: int,
    num_decode_gpus: int,
    prefill_worker_config: WorkerConfig,
    decode_worker_config: WorkerConfig,
    model_config: ModelConfig,
    results_dir: PathLike = None
):
    """
    Args:
        - prefill_gpus: List of GPUs ranks that are used for prefill
    
    Returns:
        matrices
    """
    # TODO Add assertions of dheeva rules of thumbs
    assert prefill_worker_config.tp <= decode_worker_config.tp, "Prefill TP must be less than or equal to decode TP"
    assert prefill_worker_config.pp >= decode_worker_config.pp, "Prefill PP must be more or equal to decode PP"
    assert prefill_worker_config.cp >= decode_worker_config.cp, "Prefill CP must be more or equal to decode CP"
    assert prefill_worker_config.ep <= decode_worker_config.ep, "Prefill EP must be less or equal to decode EP"

    # Create workers - group of gpus that do prefill/decode
    prefill_worker_size = prefill_worker_config.tp * prefill_worker_config.pp * prefill_worker_config.cp
    decode_worker_size = decode_worker_config.tp * decode_worker_config.pp * decode_worker_config.cp
    world_size = num_prefill_gpus + num_decode_gpus

    # Create list of all GPU ranks
    prefill_ranks = list(range(num_prefill_gpus))
    decode_ranks = list(range(num_prefill_gpus, num_prefill_gpus + num_decode_gpus))
    
    # Chunk the ranks into worker groups
    prefill_workers = [
        prefill_ranks[i:i + prefill_worker_size] 
        for i in range(0, len(prefill_ranks), prefill_worker_size)
    ]
    
    decode_workers = [
        decode_ranks[i:i + decode_worker_size]
        for i in range(0, len(decode_ranks), decode_worker_size) 
    ]

    print(f"Prefill workers: {prefill_workers}")
    print(f"Decode workers: {decode_workers}")

    batches = gen_batches(num_user_requests, batch_size, model_config)
    print(f"Generated {len(batches)} batches")
    matrices = gen_matrices_and_compute_time(batches, prefill_workers, decode_workers, model_config, prefill_worker_config, decode_worker_config)

    # Save matrices and metadata to files
    results_dir = results_dir or f"matrices_{world_size}ranks"
    results_dir = Path(results_dir)
    print(f"Saving {len(matrices)} matrices to {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "traffic_patterns": [],
    }
    
    for idx, matrix in enumerate(tqdm(matrices, desc="Saving matrices")):
        # Save matrix to npy file
        matrix_path = results_dir / f"matrix_{idx}.txt"
        with open(matrix_path, "w") as f:
            for row in matrix.matrix:
                f.write(" ".join(f"{format_size(val)}" for val in row) + "\n")
        
        # Add metadata
        metadata["traffic_patterns"].append({
            "matrix_file": matrix_path.absolute().as_posix(),
            "sleep_before_launch_sec": matrix.compute_time,
            "metadata": {
                "isl": matrix.isl,
            }
        })
    
    # Save metadata to YAML
    metadata_path = results_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)
        print(f"Saved metadata to {metadata_path}")

    


if __name__ == "__main__":
    # LLaMA-3B config
    model_config = ModelConfig(hidden_size=2048, num_layers=32, num_heads=32, num_kv_heads=32, dtype_size=2)
    num_prefill_nodes = 4
    num_decode_nodes = 8
    print("World size: ", num_prefill_nodes*8 + num_decode_nodes*8)
    main(
        num_user_requests=100,
        batch_size=1,
        num_prefill_gpus=num_prefill_nodes*8,
        num_decode_gpus=num_decode_nodes*8,
        prefill_worker_config=WorkerConfig(tp=4, pp=1, cp=1),
        decode_worker_config=WorkerConfig(tp=8, pp=1, cp=1),
        model_config=model_config,
        results_dir=f"/swgwork/eshukrun/nixl/tools/perf/matrices_folders/llama-3b-8k-to-64k-{num_prefill_nodes}Np_tp4-{num_decode_nodes}Nd_tp8"
    )