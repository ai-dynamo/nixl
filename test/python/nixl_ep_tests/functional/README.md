# Functional Tests

Multi-GPU functional tests for NIXL EP Buffer operations.

## Overview

These tests verify correct behavior of the Buffer API across multiple GPU ranks. Each test spawns 8 worker processes (one per GPU) and coordinates them via a TCP rank server.

## Test Files

| File | Description | Test IDs |
|------|-------------|----------|
| `test_connection.py` | Connect/disconnect/reconnect operations | F-CONN-01 to F-CONN-06 |
| `test_dispatch.py` | Token dispatch to experts | F-DISP-01 to F-DISP-09 |
| `test_combine.py` | Token combine from experts | F-COMB-01 to F-COMB-07 |
| `test_masking.py` | Rank masking operations | F-MASK-01 to F-MASK-04 |
| `test_e2e.py` | End-to-end dispatch→combine flows | F-E2E-01 to F-E2E-04 |

## Running Tests

### Prerequisites

```bash
# Start etcd
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh

# Set PYTHONPATH
export PYTHONPATH=/workspace/nixl/build/examples/device/ep:$PYTHONPATH
export PYTHONPATH=/workspace/nixl/test/python/nixl_ep_tests:$PYTHONPATH
```

### Running Individual Tests

```bash
cd /workspace/nixl/test/python/nixl_ep_tests

# Connection tests
python functional/test_connection.py --num-processes=8 --test=connect
python functional/test_connection.py --num-processes=8 --test=disconnect
python functional/test_connection.py --num-processes=8 --test=reconnect
python functional/test_connection.py --num-processes=8 --test=incremental
python functional/test_connection.py --num-processes=8 --test=barrier

# All connection tests
python functional/test_connection.py --num-processes=8 --test=all
```

### Running with pytest

```bash
python -m pytest functional/ -v -m connection
python -m pytest functional/ -v -m dispatch
python -m pytest functional/ -v -m e2e
```

## Test Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Process                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ RankServer  │  │ mp.spawn()  │  │ Results     │          │
│  │ (TCP:9998)  │  │ 8 workers   │  │ Collection  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
         │                │                    │
         ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Worker Processes                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐        │
│  │ Rank 0  │ │ Rank 1  │ │ Rank 2  │ ... │ Rank 7  │        │
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │     │ GPU 7   │        │
│  └─────────┘ └─────────┘ └─────────┘     └─────────┘        │
│       │           │           │               │              │
│       └───────────┴───────────┴───────────────┘              │
│                    etcd (metadata)                           │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### mp_runner.py (in utils/)

The multi-process test runner handles:
- Spawning 8 worker processes via `torch.multiprocessing`
- GPU assignment (one GPU per worker)
- UCX network device configuration
- Rank coordination via TCP server
- Result collection and reporting

### sync_all_ranks()

Synchronization barrier for multi-process tests:
```python
from utils.mp_runner import sync_all_ranks

# Wait for all ranks at this point
sync_all_ranks(barrier_name="my_barrier", timeout=60)
```

### create_buffer()

Helper to create and initialize a Buffer:
```python
from utils.mp_runner import create_buffer

buffer = create_buffer(
    global_rank=rank,
    world_size=8,
    num_experts_per_rank=8,
    hidden_dim=4096,
    num_tokens=512
)
```

## Known Issues

| Issue | Description | Workaround |
|-------|-------------|------------|
| Race conditions | Ranks may try to connect before others are registered in etcd | Add `sync_all_ranks()` after `create_buffer()` |
| UCX warnings | `rcache gdr_copy` warnings during cleanup | Cosmetic, ignore |
| Barrier timeout | Tests hang if a rank crashes | Increase `--timeout`, check etcd |

## Adding New Tests

1. Create `test_<category>.py` in this directory
2. Use `run_multiprocess_test()` from `utils.mp_runner`
3. Add pytest marks: `@pytest.mark.functional`, `@pytest.mark.<category>`
4. Define a `_test_<name>_fn(rank, world_size, **kwargs)` worker function
5. Return `TestResult` from each worker

