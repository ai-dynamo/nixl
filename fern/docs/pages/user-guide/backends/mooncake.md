---
title: Mooncake
description: Mooncake Transfer Engine backend for VRAM and DRAM transfers in Mooncake-enabled deployments.
---

## Overview

The Mooncake backend uses the Mooncake Transfer Engine for VRAM and DRAM transfers in Mooncake-enabled deployments. Use it in environments running the Mooncake distributed infrastructure.

| Property | Value |
|----------|-------|
| **Transfer Type** | VRAM ↔ VRAM; VRAM ↔ DRAM; DRAM ↔ DRAM |
| **Protocol** | Mooncake Transfer Engine |
| **Best For** | Mooncake distributed infrastructure deployments |

## Installation

The Mooncake backend requires the Mooncake Transfer Engine shared library to be installed before building NIXL.

### Build from Source

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j
sudo make install
```

<Warning>
You must build with `-DBUILD_SHARED_LIBS=ON` to produce the shared library required by the NIXL Mooncake backend.
</Warning>

For the latest installation instructions, see the [Mooncake repository](https://github.com/kvcache-ai/Mooncake).

See [Configuration](#configuration) for build options.

## Configuration

### Environment Variables

<Markdown src="/snippets/env-vars-mooncake.mdx" />

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `disable_mooncake_backend` | `false` | Disable Mooncake backend. |

## When to Use

- Deployments running the Mooncake distributed infrastructure
- Workloads where the Mooncake Transfer Engine is the preferred transport over UCX
