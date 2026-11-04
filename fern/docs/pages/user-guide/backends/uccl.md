---
title: UCCL-P2P
description: UCCL-P2P network backend for VRAM and DRAM transfers.
---

## Overview

The UCCL-P2P backend provides peer-to-peer network transport for VRAM and DRAM transfers across multiple GPUs. Use it for multi-GPU communication workloads.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ VRAM; VRAM ↔ DRAM; DRAM ↔ DRAM |
| **Protocol** | Peer-to-peer transport |
| **Best For** | Peer-to-peer multi-GPU communication |

## Installation

The UCCL-P2P backend requires the UCCL P2P engine to be installed before building NIXL.

### Build from Source

```bash
git clone https://github.com/uccl-project/uccl.git
cd uccl/p2p
make -j
sudo make install
```

For the latest installation instructions, see the [UCCL P2P repository](https://github.com/uccl-project/uccl/tree/main/p2p).

## Configuration

The UCCL-P2P backend has no backend-specific environment variables or build options.

## When to Use

- Peer-to-peer multi-GPU data transfers across nodes
- GPU clusters using direct point-to-point communication patterns
