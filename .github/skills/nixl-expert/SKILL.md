---
name: NIXL - NVIDIA Inference Xfer Library
description: NIXL (NVIDIA Inference Xfer Library) is a high-performance library for accelerating point-to-point communications in AI inference frameworks. It provides abstraction over various memory types (CPU, GPU) and storage systems (file, block, object store) through a modular plugin architecture.
---

## When to Use This Skill

Use NIXL when you need to:
- Optimize data transfer between CPU and GPU memory in AI inference pipelines
- Implement high-performance distributed AI inference systems
- Work with heterogeneous storage backends (file, block, object store)
- Benchmark and optimize memory transfer operations
- Build distributed inference frameworks with ETCD coordination
- Leverage NVIDIA's UCX and GDRCopy for maximum performance

## Prerequisites

**Platform**: Linux only (tested on Ubuntu 22.04/24.04 and Fedora)

**Required Dependencies**:
- NVIDIA CUDA (12.x or 13.x)
- UCX (v1.20.x)
- Python 3.8+ (for Python bindings)

**Optional Dependencies**:
- GDRCopy (for maximum performance)
- ETCD (for distributed metadata coordination)
- Docker (for containerized deployment)

## Installation

### PyPI Installation (Recommended)

```bash
# For CUDA 12
pip install nixl[cu12]

# For CUDA 13
pip install nixl[cu13]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/ai-dynamo/nixl.git
cd nixl

# Build from source (requires meson, ninja, CMake)
# See repository documentation for detailed build instructions
```

## Usage Patterns

### Basic Memory Transfer

```python
import nixl

# Initialize NIXL with GPU memory support
# Transfer data between CPU and GPU
# Leverage plugin architecture for custom storage backends
```

### Benchmarking

```bash
# Use NIXLBench for performance evaluation
nixlbench --help

# Run KVBench for key-value store benchmarks
kvbench --config config.yaml
```

### Distributed Setup

```bash
# Configure ETCD for distributed coordination
# Set up multiple nodes with shared metadata
# Run distributed inference workloads
```

## Key Features

- **Memory Abstraction**: Unified interface for CPU and GPU memory
- **Storage Flexibility**: Plugin-based support for file, block, and object storage
- **Multi-Language Support**: C++, Python, and Rust bindings
- **Performance Optimized**: UCX and GDRCopy integration
- **Distributed Ready**: ETCD support for cluster coordination
- **Telemetry Built-in**: Observability and monitoring capabilities

## Use Cases

1. **AI Model Serving**: Accelerate inference pipelines with optimized memory transfers
2. **Distributed Inference**: Coordinate multiple GPU nodes for large-scale inference
3. **Hybrid Storage Systems**: Seamlessly work across different storage backends
4. **Performance Benchmarking**: Evaluate and optimize transfer performance
5. **Custom Inference Frameworks**: Build on NIXL's modular architecture

## Examples

### Example 1: Basic Setup

```python
# Install NIXL
# pip install nixl[cu12]

import nixl

# Initialize with default configuration
# Perform GPU memory operations
# Clean up resources
```

### Example 2: Performance Benchmarking

```bash
# Run point-to-point transfer benchmark
nixlbench --mode p2p --size 1GB --iterations 100

# Benchmark with different memory types
nixlbench --src-type cpu --dst-type gpu --size 512MB
```

### Example 3: Distributed Inference

```python
# Configure ETCD endpoints
# Initialize NIXL in distributed mode
# Coordinate inference across multiple nodes
# Share metadata and synchronize operations
```

## Resources

- **Repository**: https://github.com/ai-dynamo/nixl
- **License**: Apache 2.0
- **Documentation**: See repository for detailed docs
- **Docker Images**: Pre-configured containers available

## Notes

- Linux-only support (no Windows/macOS)
- Requires NVIDIA hardware for GPU features
- GDRCopy recommended for production deployments
- ETCD required for distributed setups
- Check CUDA version compatibility before installation

## Related Technologies

- NVIDIA UCX (Unified Communication X)
- GDRCopy (GPUDirect RDMA)
- ETCD (Distributed key-value store)
- NVIDIA Dynamo (AI inference framework)
