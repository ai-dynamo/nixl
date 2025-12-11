# Unit Tests

Single-GPU unit tests for NIXL EP Buffer.

## Overview

These tests verify individual Buffer methods and properties without multi-GPU coordination. They're fast to run and good for quick validation.

## Test Files

| File | Description | Test IDs |
|------|-------------|----------|
| `test_init.py` | Buffer initialization and nvlink_backend options | U-INIT-01 to U-INIT-06 |
| `test_static_methods.py` | Static methods: `is_sm90_compiled`, `set_num_sms`, etc. | U-STAT-01 to U-STAT-05 |
| `test_buffer_access.py` | Buffer tensor access: `get_comm_stream`, `get_local_buffer_tensor` | U-BUF-01 to U-BUF-04 |

## Running Tests

### Prerequisites

```bash
# Set PYTHONPATH
export PYTHONPATH=/workspace/nixl/build/examples/device/ep:$PYTHONPATH

# For buffer access tests, start etcd
source /workspace/nixl/examples/device/ep/scripts/reset_etcd.sh
```

### Running All Unit Tests

```bash
cd /workspace/nixl/test/python/nixl_ep_tests
python -m pytest unit/ -v
```

### Running Individual Test Files

```bash
# Static methods (no etcd required)
python -m pytest unit/test_static_methods.py -v

# Initialization tests
python -m pytest unit/test_init.py -v

# Buffer access tests (requires etcd)
python -m pytest unit/test_buffer_access.py -v
```

## Test Coverage

### Initialization (test_init.py)

- `nvlink_backend='nixl'` - Sets `NIXL_EP_NVLINK_BACKEND_IPC=0`
- `nvlink_backend='ipc'` - Sets `NIXL_EP_NVLINK_BACKEND_IPC=1`
- `nvlink_backend='none'` - Sets `UCX_TLS=^cuda_ipc`
- `explicitly_destroy=True/False` - Controls destroy behavior

### Static Methods (test_static_methods.py)

- `Buffer.is_sm90_compiled()` - Check SM90 compilation
- `Buffer.set_num_sms(n)` - Set number of SMs (must be even)
- `Buffer.capture()` - Get EventOverlap object
- `Buffer.get_rdma_size_hint()` - Calculate RDMA buffer size

### Buffer Access (test_buffer_access.py)

- `buffer.get_comm_stream()` - Get CUDA communication stream
- `buffer.get_local_buffer_tensor()` - Get local buffer as tensor
- `buffer.get_next_combine_buffer()` - Get combine buffer

## Known Issues

| Issue | Description | Workaround |
|-------|-------------|------------|
| etcd required | Some tests need etcd running | Start etcd before tests |
| Single GPU | Tests use only one GPU | N/A - by design |

## Adding New Tests

1. Create `test_<category>.py` in this directory
2. Add `@pytest.mark.unit` mark
3. Keep tests fast and isolated
4. Don't require multi-GPU coordination

