# Test Utilities

Shared utilities for NIXL EP tests.

## Overview

This package provides reusable components for multi-process testing, rank coordination, and results reporting.

## Modules

| Module | Description |
|--------|-------------|
| `mp_runner.py` | Multi-process test runner with GPU/UCX coordination |
| `test_rank_server.py` | TCP-based rank server for distributed synchronization |
| `results_reporter.py` | Test results formatting and CSV export |
| `helpers.py` | General utility functions |

## mp_runner.py

The core multi-process test runner. Handles:
- Spawning worker processes via `torch.multiprocessing`
- GPU assignment (CUDA_VISIBLE_DEVICES)
- UCX network device configuration
- Rank coordination
- Result collection

### Key Functions

```python
from utils.mp_runner import (
    run_multiprocess_test,
    create_buffer,
    sync_all_ranks,
    all_passed,
    print_results,
)
```

#### run_multiprocess_test()

Run a test function across multiple GPU ranks:

```python
results = run_multiprocess_test(
    test_fn=my_test_function,
    num_processes=8,
    timeout=300,
    test_name="My Test",
    test_kwargs={'param1': value1}
)
```

#### create_buffer()

Create and initialize a Buffer instance:

```python
buffer = create_buffer(
    global_rank=rank,
    world_size=8,
    num_experts_per_rank=8,
    hidden_dim=4096,
    num_tokens=512
)
```

#### sync_all_ranks()

Synchronization barrier across all ranks:

```python
sync_all_ranks(barrier_name="after_connect", timeout=60)
```

### TestResult Class

```python
@dataclass
class TestResult:
    rank: int
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    test_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
```

## test_rank_server.py

TCP-based rank coordination server for multi-node tests.

### Classes

```python
from utils.test_rank_server import (
    RankServer,
    RankClient,
    start_test_server,
)
```

#### RankServer

Server that assigns global ranks to workers:

```python
server = RankServer(port=9998, world_size=8)
server.start()
# Workers connect and receive ranks
server.stop()
```

#### RankClient

Client for workers to get their rank:

```python
client = RankClient(host="127.0.0.1", port=9998)
global_rank, local_rank, node_id = client.register()
```

## results_reporter.py

Test results collection and reporting.

### Classes

```python
from utils.results_reporter import (
    ResultsReporter,
    get_reporter,
    init_reporter,
)
```

#### ResultsReporter

```python
reporter = ResultsReporter(results_dir="./results")

reporter.add_result(
    test_name="P-CONN-01",
    category="connection",
    metric="latency_ms",
    value=123.45,
    unit="ms",
    params={'num_ranks': 8}
)

reporter.save()  # Saves to CSV
reporter.save_summary()  # Human-readable summary
```

## helpers.py

General utility functions.

### GPU-NIC Topology Discovery

```python
from utils.helpers import discover_gpu_nic_topology

# Returns {gpu_id: nic_name}
topology = discover_gpu_nic_topology()
# {0: 'mlx5_0', 1: 'mlx5_1', ...}
```

## Usage Example

```python
from utils.mp_runner import (
    run_multiprocess_test,
    create_buffer,
    sync_all_ranks,
    TestResult,
)
import time

def my_test_fn(rank, world_size, num_experts=8, **kwargs):
    """Worker function run on each GPU."""
    start = time.time()

    try:
        # Create buffer
        buffer = create_buffer(
            global_rank=rank,
            world_size=world_size,
            num_experts_per_rank=num_experts
        )

        # Wait for all ranks
        sync_all_ranks("buffer_created")

        # Do test operations...
        other_ranks = [r for r in range(world_size) if r != rank]
        buffer.connect_ranks(other_ranks)

        sync_all_ranks("connected")

        # Cleanup
        buffer.destroy()

        return TestResult(
            rank=rank,
            passed=True,
            duration_ms=(time.time() - start) * 1000
        )

    except Exception as e:
        return TestResult(
            rank=rank,
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            error=str(e)
        )

# Run the test
if __name__ == "__main__":
    results = run_multiprocess_test(
        test_fn=my_test_fn,
        num_processes=8,
        timeout=300,
        test_name="My Test",
        test_kwargs={'num_experts': 8}
    )

    print(f"Passed: {sum(1 for r in results if r.passed)}/8")
```

## Known Issues

| Issue | Description |
|-------|-------------|
| Single-node barriers | File-based barriers don't work multi-node |
| TCP port conflicts | RankServer uses fixed port 9998 |
| UCX cache warnings | Normal during cleanup, not errors |

