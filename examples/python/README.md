# NIXL Python Examples

This directory contains Python examples demonstrating NIXL usage patterns.

## Directory Structure

```
python/
├── 2proc/                    # Basic two-process example
│   ├── nixl_api_2proc.py     # Target-initiator pattern
│   └── README.md             # Documentation
│
├── send_recv/                # High-throughput streaming example
│   ├── nixl_sender_receiver.py  # Backpressure + circular buffers
│   └── README.md             # Documentation
│
├── utils/                    # Shared utilities
│   ├── README.md             # Utilities documentation
│   ├── __init__.py           # Package exports
│   ├── tcp_server.py         # TCP metadata server
│   ├── memory_utils.py       # Memory read/write helpers
│   ├── metadata_utils.py     # Metadata exchange (TCP + etcd)
│   └── NIXL_PYTHON_GUIDE.md  # API patterns guide
│
└── (other examples)          # Additional standalone examples
```

## Quick Start

### Two-Process Example (Basic)

```bash
cd 2proc
python3 nixl_api_2proc.py
```

Demonstrates:
- Both transfer modes (`initialize_xfer` and `make_prepped_xfer`)
- Target-initiator pattern
- Synchronous completion checking

### Sender-Receiver Example (Streaming)

```bash
cd send_recv
python3 nixl_sender_receiver.py
```

Demonstrates:
- High-throughput streaming (~10-25 GB/s on RDMA)
- Notification-based backpressure
- Circular buffer management
- Sequence number verification

## Other Examples

- `basic_two_peers.py`: Simple two-peer example showing registration, metadata exchange, a single READ, notification, verification, and teardown.
- `expanded_two_peers.py`: Comprehensive example with parallel READs (prepped), parallel WRITEs (initialized), reposting READs, notification patterns, and minimal layout sharing.
- `nixl_gds_example.py`: GPU Direct Storage integration—buffer-to-file write and read-back verification.
- `partial_md_example.py`: Shows handling of partial metadata updates and retrying initialize_xfer.
- `query_mem_example.py`: Demonstrates querying backend memory/storage details for registered regions.
- `telemetry_reader.py`: Example for fetching and validating transfer telemetry.
- `remote_storage_example/`: Client-server peer-to-peer storage pipeline with supporting utilities and diagrams.

## Documentation

- **Utilities Guide**: `utils/README.md`
  - Memory utilities
  - Metadata exchange (TCP + etcd)
  - Quick function reference

- **API Patterns Guide**: `utils/NIXL_PYTHON_GUIDE.md`
  - Transfer modes comparison
  - Polling patterns
  - Notifications
  - Backpressure implementation

- **Example READMEs**: Each example folder has its own README with detailed explanations.

## Metadata Exchange Options

The examples support two metadata exchange methods:

1. **TCP Server** (default): Simple local key-value store
2. **NIXL Built-in (etcd)**: Set `NIXL_ETCD_ENDPOINTS` environment variable

See `utils/NIXL_PYTHON_GUIDE.md` for details.

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
