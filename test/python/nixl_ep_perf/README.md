# NIXL EP Data Plane Performance Tests

Measures throughput and latency of NIXL EP Buffer dispatch/combine operations.

## Prerequisites

- etcd running locally (`etcd &` or `source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh`)
- CUDA device with RDMA support

## Environment Setup

```bash
# For RDMA performance (recommended)
export UCX_TLS=rc_mlx5,dc_mlx5,tcp
export UCX_IB_AR_ENABLE=no  # Disable Adaptive Routing for consistent performance
```

## Usage

```bash
cd test/python/nixl_ep_perf

# IPC/NVLink backend (default)
python3 test_data_plane.py --num-processes=8 --mode=e2e

# RDMA-only (disable NVLink)
UCX_TLS=rc_mlx5,dc_mlx5,tcp UCX_IB_AR_ENABLE=no \
  python3 test_data_plane.py --num-processes=8 --mode=e2e --nvlink-backend none

# Dispatch only (measures dispatch throughput)
python3 test_data_plane.py --num-processes=8 --mode=dispatch

# Combine only (one dispatch, many combines)
python3 test_data_plane.py --num-processes=8 --mode=combine
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-processes` | 8 | Number of ranks/GPUs |
| `--mode` | e2e | Test mode: dispatch, combine, e2e |
| `--tokens` | 512 | Number of tokens |
| `--hidden` | 4096 | Hidden dimension |
| `--experts-per-rank` | 8 | Experts per rank |
| `--topk` | 2 | TopK value |
| `--nvlink-backend` | ipc | Backend: ipc, nixl, none (RDMA only) |
| `--warmup` | 10 | Warmup iterations |
| `--iters` | 100 | Measurement iterations |
| `--skip-nic-discovery` | false | Skip GPU-NIC discovery (let UCX auto-select) |

## Example Output

```
======================================================================
NIXL EP Data Plane Performance Test
======================================================================
Mode: e2e
Ranks: 8, Tokens: 128, Hidden: 7168
Experts: 36/rank (288 total), TopK: 8
Backend: none (RDMA forced)
Warmup: 10, Measure: 100 iterations
======================================================================

======================================================================
Data Plane (e2e): 8/8 ranks passed
======================================================================
Bandwidth (GB/s): avg=42.88, min=42.86, max=42.89
Latency (μs):     avg=519.3, min=519.1, max=519.5
```

## Expected Performance (DFW cluster, RDMA, AR=no)

| Mode | Bandwidth | Latency |
|------|-----------|---------|
| E2E | ~42.8 GB/s | ~520 μs |
| Dispatch | ~42.1 GB/s | ~180 μs |
| Combine | ~43.3 GB/s | ~340 μs |

*Config: 128 tokens, 7168 hidden, topk=8, 288 experts (36/rank), 8 GPUs*

## Files

| File | Description |
|------|-------------|
| `test_data_plane.py` | Main test with dispatch/combine/e2e modes |
| `mp_runner.py` | Multi-process test runner |
| `rank_server.py` | Coordination server for distributed tests |


