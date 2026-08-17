---
title: GUSLI
description: Block-level I/O backend for transfers between block storage devices and DRAM.
---

## Overview

The GUSLI backend provides block-level I/O for transfers between block storage devices and DRAM, bypassing filesystem overhead for high-throughput workloads.

| Property | Value |
|----------|-------|
| **Transfer Type** | DRAM ↔ Block |
| **Protocol** | Block I/O |
| **Best For** | Raw block device access for high-throughput storage |

## Installation

The GUSLI backend requires the GUSLI client library and headers to be installed before building NIXL.

### Build from Source

```bash
git clone https://github.com/nvidia/gusli.git
cd gusli
make all BUILD_RELEASE=1 BUILD_FOR_UNITEST=0 VERBOSE=1 ALLOW_USE_URING=0
```

Ensure that `libgusli_clnt.so` is installed under `/usr/lib/` and GUSLI headers are installed under `/usr/include/gusli_*.hpp`.

<Warning>
GUSLI must be built before NIXL. The NIXL build system detects the GUSLI library at configure time.
</Warning>

## Configuration

The GUSLI backend has no backend-specific environment variables or build options.

## When to Use

- **Direct block I/O** -- Raw block device access without filesystem overhead.
- **High-throughput storage** -- Workloads requiring maximum throughput through block-level access.
