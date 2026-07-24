---
title: Building KVBench
description: Install KVBench using the NIXLBench Docker container or a Python virtual environment.
---

KVBench requires Python 3.12 or later. For GPU-accelerated benchmarks, PyTorch is also required.

## Installation

<Tabs>
<Tab title="Docker">

KVBench is included in the NIXLBench Docker container. See [Building NIXLBench](../nixlbench/build.md) for Docker build and setup instructions.

After building the container, KVBench is available at `/workspace/benchmark/kvbench/` inside the container.

</Tab>
<Tab title="Python venv">

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl/benchmark/kvbench
python3 -m venv venv
source venv/bin/activate
pip install uv
uv sync --active
```

Verify the installation:

```bash
python main.py --help
```

</Tab>
</Tabs>
