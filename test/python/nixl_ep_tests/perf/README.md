# Performance Tests

Performance benchmarks for NIXL EP Buffer operations.

## Overview

These tests measure latency and throughput of control plane (init/connect/disconnect/destroy) and data plane (dispatch/combine) operations.

## Test Files

| File | Description | Metrics |
|------|-------------|---------|
| `test_control_plane.py` | Init, connect, disconnect, destroy latency | Latency (ms) |
| `test_data_plane.py` | Dispatch, combine, e2e throughput | Tokens/sec, GB/s, μs |
| `results_collector.py` | CI/CD results management | JSON/CSV export |

## Running Tests

### Prerequisites

```bash
# Start etcd
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh

# Set PYTHONPATH
export PYTHONPATH=/workspace/nixl/build/examples/device/ep:$PYTHONPATH
export PYTHONPATH=/workspace/nixl/test/python/nixl_ep_tests:$PYTHONPATH
```

### Control Plane Tests

```bash
cd /workspace/nixl/test/python/nixl_ep_tests/perf

# Basic run (8 experts/rank, single measurement)
python test_control_plane.py --num-processes=8 --experts-per-rank=8 --warmup=0 --rounds=1

# Full cycle test
python test_control_plane.py --num-processes=8 --experts-per-rank=8 --test=cycle --timeout=300

# Available tests: init, connect, disconnect, destroy, cycle, all
```

**⚠️ Important**: Use `--warmup=0 --rounds=1` to avoid BUG-01 (segfault on repeated buffer creation). This bug only affects control plane tests.

### Data Plane Tests

```bash
# Basic run (warmup is safe for data plane - no buffer recreation)
python test_data_plane.py --num-processes=8 --test=all

# Full configuration
python test_data_plane.py \
    --num-processes=8 \
    --test=all \
    --tokens=512 \
    --hidden=4096 \
    --experts-per-rank=8 \
    --topk=2 \
    --timeout=300

# Available tests: dispatch, combine, e2e, all
```

## Command Line Arguments

### test_control_plane.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-processes` | 8 | Number of GPU ranks |
| `--experts-per-rank` | 8 | Experts per rank (comma-separated for sweep). Total = experts × ranks |
| `--test` | cycle | Test type: init, connect, disconnect, destroy, cycle, all |
| `--timeout` | 300 | Barrier timeout (seconds) |
| `--warmup` | 2 | Warmup rounds (use 0 to avoid BUG-01, control plane only) |
| `--rounds` | 5 | Measurement rounds |
| `--results-dir` | results/ | Output directory |

### test_data_plane.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-processes` | 8 | Number of GPU ranks |
| `--test` | all | Test type: dispatch, combine, e2e, all |
| `--tokens` | 512 | Token count (comma-separated for sweep) |
| `--hidden` | 4096 | Hidden dimension |
| `--experts-per-rank` | 8 | Experts per rank. Total = experts × ranks |
| `--topk` | 2 | TopK value |
| `--nvlink-backend` | ipc | Backend: ipc, nixl, none |
| `--warmup` | 5 | Warmup iterations |
| `--iters` | 100 | Benchmark iterations |
| `--timeout` | 300 | Timeout (seconds) |
| `--results-dir` | results/ | Output directory |

## Reference Results

### Single Node (8x H100, NVLink)

**Control Plane (8 experts/rank = 64 total)**:
| Operation | Latency |
|-----------|---------|
| init | ~1200 ms |
| connect | ~2800 ms |
| disconnect | ~40 ms |
| reconnect | ~2800 ms |
| destroy | ~900 ms |
| **TOTAL** | ~8000 ms |

**Data Plane (512 tokens, 4096 hidden, 8 experts/rank = 64 total)**:
| Test | Throughput | Bandwidth | Latency |
|------|------------|-----------|---------|
| dispatch | 12.2M tok/s | 103 GB/s | 42 μs |
| combine | 9.5M tok/s | 156 GB/s | 54 μs |
| e2e | 5.5M tok/s | 138 GB/s | 92 μs |

## Results Collection

Results are saved automatically to `results/raw/`:

```bash
# List saved results
ls results/raw/*.json

# Export to CSV
python results_collector.py export --format csv --output all_results.csv

# Summary of latest run
python results_collector.py summary

# Check for regressions (CI/CD)
python results_collector.py check --threshold 10
```

### Output Format

```
results/
├── raw/                           # Individual JSON results
│   ├── 20251211_093109_abc123.json
│   └── 20251211_093133_def456.json
├── history.csv                    # Cumulative history
└── progress_log_001.txt           # Real-time progress
```

## Known Issues

| Issue | Description | Workaround |
|-------|-------------|------------|
| BUG-01 | Segfault on repeated buffer creation (control plane only) | Use `--warmup=0 --rounds=1` |
| BUG-02 | rcache assertion with 16 experts | Use `--experts-per-rank=2,4,8,32` |
| Variance | Results vary ~5-10% between runs | Run multiple times, report average |
| etcd race | "Failed to fetch key" warnings | Cosmetic, tests still pass |

## Adding New Benchmarks

1. Add new test function in appropriate file
2. Use `run_multiprocess_test()` from `utils.mp_runner`
3. Return results with timing metrics in `extra` dict
4. Update argument parser for new options
5. Add results formatting in output section

