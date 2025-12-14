# NIXL Python Examples

This directory contains Python examples demonstrating NIXL usage patterns.

## Directory Structure

```
python/
├── 2proc/                      # Two-process example (target-initiator pattern)
│   ├── nixl_api_2proc.py
│   └── README.md
│
├── send_recv/                  # High-throughput streaming with backpressure
│   ├── nixl_sender_receiver.py
│   └── README.md
│
├── nixl_utils/                      # Shared utilities for examples
│   ├── __init__.py
│   ├── tcp_server.py
│   ├── memory_utils.py
│   ├── metadata_utils.py
│   ├── README.md
│   └── NIXL_PYTHON_GUIDE.md
│
├── remote_storage_example/     # Client-server storage pipeline
│
├── basic_two_peers.py          # Simple two-peer READ + notification
├── expanded_two_peers.py       # Parallel READs/WRITEs, reposting, notifications
├── nixl_gds_example.py         # GPU Direct Storage integration
├── partial_md_example.py       # Partial metadata updates
├── query_mem_example.py        # Query backend memory details
└── telemetry_reader.py         # Transfer telemetry example
```

## Quick Start

### Two-Process Example

```bash
cd 2proc
python3 nixl_api_2proc.py
```

### Sender-Receiver (Streaming)

```bash
cd send_recv
python3 nixl_sender_receiver.py
```

### Basic Two Peers

```bash
# Terminal 1: Start target
python3 basic_two_peers.py --mode target --ip 127.0.0.1 --port 5555

# Terminal 2: Start initiator
python3 basic_two_peers.py --mode initiator --ip 127.0.0.1 --port 5555
```

## Documentation

- **API Patterns Guide**: `nixl_utils/NIXL_PYTHON_GUIDE.md`
- **Utilities Reference**: `nixl_utils/README.md`
- **Example READMEs**: Each folder has its own README

## Metadata Exchange

Examples support two methods:
1. **TCP Server** (default): Simple local key-value store
2. **NIXL Built-in (etcd)**: Use `--use-etcd` flag (requires `NIXL_ETCD_ENDPOINTS`)

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
