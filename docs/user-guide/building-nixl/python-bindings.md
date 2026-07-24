---
title: Python Bindings
description: Build and install NIXL Python bindings from source.
---

## Prerequisites

- **uv** (required):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:${PATH}"
```

- **tomlkit**: installed below via uv or pip
- **PyTorch**: [pytorch.org](https://pytorch.org/get-started/locally/)

## Virtual Environment Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install tomlkit
```

Then install PyTorch from the [PyTorch website](https://pytorch.org/get-started/locally/).

## Build and Install

**For CUDA 12:**

```bash
pip install .
meson setup build
ninja -C build install
pip install build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
```

**For CUDA 13:**

```bash
pip install .
./contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml
meson setup build
ninja -C build install
pip install build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
```

## Verify Installation

```bash
python3 -c "import nixl; agent = nixl.nixl_agent('agent1')"
```

Expected output:

```
NIXL INFO    _api.py:363 Backend UCX was instantiated
NIXL INFO    _api.py:253 Initialized NIXL agent: agent1
```

<Note>
For backend-specific build instructions, see [NIXL Backends](/docs/user-guide/backend-selection).
</Note>
