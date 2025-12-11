# NIXL EP Test Suite

Comprehensive test suite for the NIXL EP (Expert-Parallel) communication buffer.

## Prerequisites

### Required Software
- **Python 3.10+** with `pytest` and `pytest-benchmark`
- **PyTorch** with CUDA support
- **etcd** - distributed key-value store for metadata exchange
- **NIXL** - compiled with NIXL EP enabled (`--build-nixl-ep`)

### Hardware Requirements
- NVIDIA GPUs with NVLink (H100 recommended)
- InfiniBand NICs for RDMA (multi-node tests)
- 8 GPUs per node (typical HPC configuration)

### Environment Setup

```bash
# 1. Activate virtual environment (if using container)
source /workspace/.venv/bin/activate

# 2. Set PYTHONPATH to include nixl_ep module and test utilities
export PYTHONPATH=/workspace/nixl/build/examples/device/ep:$PYTHONPATH
export PYTHONPATH=/workspace/nixl/test/python/nixl_ep_tests:$PYTHONPATH

# 3. Set NIXL plugin directory
export NIXL_PLUGIN_DIR=/usr/local/nixl/lib/x86_64-linux-gnu/plugins

# 4. Start etcd (required for all multi-process tests)
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh
```

### Installing Test Dependencies

```bash
# Using uv (recommended in container)
uv pip install pytest pytest-benchmark

# Or using pip
pip install pytest pytest-benchmark
```

---

## Directory Structure

```
test/python/nixl_ep_tests/
├── README.md              # This file
├── conftest.py            # Shared pytest fixtures
├── TEST_PLAN.md           # Full test plan with test IDs
├── SETUP.md               # Detailed setup instructions
│
├── functional/            # Multi-GPU functional tests
│   ├── test_connection.py # Connect/disconnect/reconnect tests
│   ├── test_dispatch.py   # Token dispatch tests
│   ├── test_combine.py    # Token combine tests
│   ├── test_masking.py    # Rank masking tests
│   └── test_e2e.py        # End-to-end dispatch→combine tests
│
├── perf/                  # Performance benchmarks
│   ├── test_control_plane.py  # Init/connect/disconnect latency
│   ├── test_data_plane.py     # Dispatch/combine throughput
│   └── results_collector.py   # CI/CD results management
│
├── bugs/                  # Bug reproduction tests
│   ├── README.md          # Bug documentation
│   ├── test_bug_01_segfault.py    # Repeated buffer creation
│   ├── test_bug_02_rcache.py      # UCX rcache assertion
│   ├── test_bug_03_gdr_copy.py    # GDR copy warnings
│   ├── test_bug_04_invalidate.py  # Metadata invalidation
│   └── test_bug_05_connect_cuda_error.py  # Intermittent CUDA errors
│
├── unit/                  # Single-GPU unit tests
│   ├── test_init.py           # Buffer initialization
│   ├── test_static_methods.py # Static method validation
│   └── test_buffer_access.py  # Buffer tensor access
│
└── utils/                 # Shared test utilities
    ├── mp_runner.py       # Multi-process test runner
    ├── test_rank_server.py # TCP rank coordination server
    ├── results_reporter.py # Results formatting
    └── helpers.py         # General utilities
```

---

## Quick Start

### 1. Start Container

```bash
# Example: Slurm cluster with Pyxis/Enroot
srun -A <account> -N 1 --gpus=8 -p <partition> \
     --time=02:00:00 --container-image=./nixl_ep.sqsh --pty bash

# Or run container directly (local machine with GPUs)
docker run --gpus all -it nixl_ep:latest bash
```

### 2. Setup Environment

```bash
cd /workspace/nixl
source /workspace/.venv/bin/activate
export PYTHONPATH=/workspace/nixl/build/examples/device/ep:$PYTHONPATH
export PYTHONPATH=/workspace/nixl/test/python/nixl_ep_tests:$PYTHONPATH

# Start etcd
source examples/device/ep/scripts/reset_etcd.sh
```

### 3. Run Tests

```bash
cd /workspace/nixl/test/python/nixl_ep_tests

# Unit tests (fast, single GPU)
python -m pytest unit/ -v

# Functional tests (8 GPUs)
python functional/test_connection.py --num-processes=8 --test=all

# Performance tests
python perf/test_control_plane.py --num-processes=8 --experts-per-rank=8 --warmup=0 --rounds=1
python perf/test_data_plane.py --num-processes=8 --experts-per-rank=8 --test=all
```

---

## Running Tests

### Unit Tests

```bash
# All unit tests
python -m pytest unit/ -v

# Specific test file
python -m pytest unit/test_static_methods.py -v
```

### Functional Tests

```bash
# Connection tests
python functional/test_connection.py --num-processes=8 --test=all

# Available tests: connect, disconnect, reconnect, incremental, barrier, all
python functional/test_connection.py --num-processes=8 --test=connect
```

### Performance Tests

```bash
# Control plane (init/connect/disconnect/destroy latency)
python perf/test_control_plane.py --num-processes=8 --experts-per-rank=8 --warmup=0 --rounds=1 --timeout=300

# Data plane (dispatch/combine throughput)
python perf/test_data_plane.py --num-processes=8 --test=all \
    --tokens=512 --hidden=4096 --experts-per-rank=8
```

### Bug Reproduction Tests

```bash
# Run specific bug test
python bugs/test_bug_01_segfault.py

# Run all bug tests
python -m pytest bugs/ -v --timeout=300
```

---

## Known Issues

| Bug ID | Severity | Description | Workaround |
|--------|----------|-------------|------------|
| BUG-01 | 🔴 Critical | Segfault on repeated buffer creation | Use `--warmup=0 --rounds=1` |
| BUG-02 | 🔴 Critical | UCX rcache assertion (16 experts) | Skip 16-expert tests |
| BUG-03 | 🟡 Low | GDR copy memory warnings | Cosmetic, ignore |
| BUG-04 | 🟡 Low | invalidateRemoteMD warnings | Cosmetic, ignore |
| BUG-05 | 🟠 Medium | Intermittent CUDA errors | Reset etcd, retry |

See `bugs/README.md` for detailed bug documentation.

---

## Troubleshooting

### etcd Not Running

```bash
# Check if etcd is running
pgrep -a etcd

# Start/restart etcd
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh
```

### Stale State Between Tests

```bash
# Clean etcd keys
etcdctl del /nixl --prefix

# Full reset
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh
```

### Tests Hang at Barrier

```bash
# Increase timeout
python perf/test_control_plane.py --timeout=600

# Or kill and restart
pkill -9 -f python
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh
```

### Module Not Found: nixl_ep

```bash
# Verify PYTHONPATH includes the built module
export PYTHONPATH=/workspace/nixl/build/examples/device/ep:$PYTHONPATH

# Verify the module exists
ls /workspace/nixl/build/examples/device/ep/nixl_ep/
```

---

## Multi-Node Testing

For 2+ node tests:

```bash
# On Node 1 (master)
export MASTER_ADDR=$(hostname -i)
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh

# On Node 2+ (workers)  
export MASTER_ADDR=<node1-ip>
export NIXL_ETCD_ENDPOINTS=http://$MASTER_ADDR:2379
# Don't start local etcd - use Node 1's
```

---

## CI/CD Integration

### Results Collection

Tests save results to `perf/results/`:

```bash
# Results are saved automatically to JSON
ls perf/results/raw/*.json

# Export to CSV
python perf/results_collector.py export --format csv --output results.csv
```

### Expected Marks

Tests use pytest marks for categorization:

```bash
# Run only functional tests
python -m pytest -m functional

# Run only performance tests
python -m pytest -m perf
```

Available marks: `unit`, `functional`, `perf`, `bugs`, `connection`, `dispatch`, `combine`, `e2e`, `masking`, `control_plane`, `data_plane`

---

## Contributing

1. Follow test naming: `test_<category>_<description>.py`
2. Use `mp_runner.py` helpers for multi-process tests
3. Add pytest marks for test categorization
4. Document new bugs in `bugs/README.md`
5. Update this README when adding new test categories
