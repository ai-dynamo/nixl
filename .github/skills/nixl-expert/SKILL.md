---
name: NIXL - NVIDIA Inference Xfer Library
description: NIXL (NVIDIA Inference Xfer Library) is a high-performance library for accelerating point-to-point communications in AI inference frameworks. It provides abstraction over various memory types (CPU, GPU) and storage systems (file, block, object store) through a modular plugin architecture.
---

## Quick Start

```bash
# Install NIXL for CUDA 12
pip install nixl[cu12]

# Run a basic benchmark
nixlbench --mode p2p --size 1GB --iterations 100

# Check installation
python -c "import nixl; print(nixl.__version__)"
```

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
- GDRCopy (for maximum performance - provides 10x faster transfers for small buffers)
- ETCD (for distributed metadata coordination)
- Docker (for containerized deployment)

## Compatibility

| NIXL Version | CUDA Version | UCX Version | Python Version | Notes |
|-------------|--------------|-------------|----------------|-------|
| 0.1.x       | 12.x, 13.x   | 1.20.x      | 3.8+          | Stable release |
| Latest      | 12.x, 13.x   | 1.20.x      | 3.8+          | Check GitHub for updates |

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

# Install build dependencies
sudo apt-get install -y meson ninja-build cmake g++

# Build and install
meson setup build
ninja -C build
sudo ninja -C build install

# Verify installation
python -c "import nixl; print('NIXL installed successfully')"
```

### Docker Installation

```bash
# Pull pre-configured NIXL container
docker pull nvcr.io/nvidia/nixl:latest

# Run with GPU support
docker run --gpus all -it nvcr.io/nvidia/nixl:latest

# Inside container, NIXL is pre-configured and ready to use
```

## Configuration

### Environment Variables

NIXL behavior can be customized through environment variables:

```bash
# Backend configuration
export NIXL_BACKEND=ucx              # Backend type: ucx, gds, s3
export NIXL_UCX_PATH=/usr/local/ucx  # Path to UCX installation

# Performance tuning
export NIXL_ENABLE_GDRCOPY=true      # Enable GDRCopy (true/false)
export NIXL_BUFFER_SIZE=4MB          # Default buffer size
export NIXL_POOL_SIZE=1GB            # Memory pool size

# Distributed coordination
export NIXL_ETCD_ENDPOINTS=localhost:2379,localhost:2380
export NIXL_ETCD_PREFIX=/nixl        # ETCD key prefix for metadata

# Logging and telemetry
export NIXL_LOG_LEVEL=INFO           # DEBUG, INFO, WARN, ERROR
export NIXL_ENABLE_TELEMETRY=true    # Enable performance metrics
```

### Backend Configuration

#### UCX Backend (Recommended for InfiniBand/RDMA)

```bash
# Configure UCX for optimal performance
export UCX_TLS=rc,cuda_copy,gdr_copy
export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=16384
```

#### GDS (GPUDirect Storage) Backend

```bash
# For direct GPU-to-storage transfers
export NIXL_BACKEND=gds
export NIXL_GDS_DEVICE=/dev/nvme0n1
```

#### S3 Object Storage Backend

```python
import nixl

# Configure S3 backend
nixl.configure_backend(
    backend='s3',
    endpoint='https://s3.amazonaws.com',
    access_key='YOUR_ACCESS_KEY',
    secret_key='YOUR_SECRET_KEY',
    bucket='nixl-data'
)
```

## Usage Patterns

### Basic Memory Transfer

```python
import nixl
import numpy as np

try:
    # Initialize NIXL
    nixl.initialize(backend='ucx', enable_gpu=True)

    # Create data on CPU
    cpu_data = np.random.rand(1000, 1000).astype(np.float32)

    # Allocate buffers
    cpu_buffer = nixl.Buffer.from_host(cpu_data)
    gpu_buffer = nixl.Buffer.allocate_device(size=cpu_data.nbytes, device_id=0)

    # Transfer CPU -> GPU
    transfer = nixl.transfer(src=cpu_buffer, dst=gpu_buffer)
    transfer.wait()  # Synchronous wait

    # Or use async pattern
    async_transfer = nixl.transfer_async(src=cpu_buffer, dst=gpu_buffer)
    # Do other work...
    async_transfer.wait()

    print(f"Transfer completed: {cpu_data.nbytes / 1e9:.2f} GB")

finally:
    # Clean up resources
    cpu_buffer.free()
    gpu_buffer.free()
    nixl.finalize()
```

### Storage Backend Operations

```python
import nixl

# File storage backend
file_backend = nixl.FileBackend('/mnt/data/nixl')
file_backend.write('model.bin', data)
retrieved_data = file_backend.read('model.bin')

# Block storage backend
block_backend = nixl.BlockBackend('/dev/nvme0n1', use_gds=True)
block_backend.write_block(block_id=42, data=gpu_buffer)

# Object storage (S3) backend
s3_backend = nixl.S3Backend(bucket='my-bucket', prefix='models/')
s3_backend.put_object('checkpoint.pt', data)
```

### Benchmarking

```bash
# Use NIXLBench for performance evaluation
nixlbench --help

# Run point-to-point transfer benchmark
nixlbench --mode p2p --size 1GB --iterations 100

# Expected output:
# Transfer size: 1.00 GB
# Iterations: 100
# Avg bandwidth: 45.2 GB/s
# Min latency: 22.1 ms
# Max latency: 23.8 ms
# Std dev: 0.4 ms

# Benchmark CPU to GPU transfer
nixlbench --src-type cpu --dst-type gpu --size 512MB

# Benchmark with different backends
nixlbench --backend ucx --size 1GB
nixlbench --backend gds --size 1GB  # Compare performance

# Run KVBench for key-value store workflows
kvbench --config config.yaml

# Sample config.yaml:
# operations: [put, get, delete]
# key_size: 32
# value_size: 4096
# num_keys: 10000
# threads: 8
```

### Interpreting Benchmark Results

- **Bandwidth > 40 GB/s**: Excellent (likely using GDRCopy + UCX)
- **Bandwidth 20-40 GB/s**: Good (standard UCX without GDRCopy)
- **Bandwidth < 20 GB/s**: Check configuration - may be falling back to slower paths
- **High latency variance**: May indicate contention or thermal throttling

### Distributed Setup with ETCD

```bash
# Step 1: Start ETCD cluster (on coordination node)
etcd --name node1 \
     --initial-advertise-peer-urls http://192.168.1.10:2380 \
     --listen-peer-urls http://192.168.1.10:2380 \
     --listen-client-urls http://192.168.1.10:2379,http://127.0.0.1:2379 \
     --advertise-client-urls http://192.168.1.10:2379

# Step 2: Configure NIXL on each worker node
export NIXL_ETCD_ENDPOINTS=192.168.1.10:2379
export NIXL_NODE_ID=$(hostname)
export NIXL_CLUSTER_NAME=inference-cluster
```

```python
# Step 3: Initialize distributed NIXL on each node
import nixl

# Connect to ETCD cluster
nixl.distributed.initialize(
    etcd_endpoints=['192.168.1.10:2379'],
    node_id='node-1',
    cluster_name='inference-cluster'
)

# Register this node's resources
nixl.distributed.register_node(
    gpu_devices=[0, 1, 2, 3],
    memory_gb=128,
    storage_paths=['/mnt/nvme0', '/mnt/nvme1']
)

# Discover other nodes in cluster
nodes = nixl.distributed.list_nodes()
print(f"Cluster has {len(nodes)} nodes")

# Coordinate data transfers across nodes
remote_buffer = nixl.distributed.get_remote_buffer(
    node_id='node-2',
    buffer_id='model_weights'
)

# Transfer from remote node to local GPU
local_gpu_buffer = nixl.Buffer.allocate_device(size=remote_buffer.size)
nixl.transfer(src=remote_buffer, dst=local_gpu_buffer)

# Store metadata in ETCD for coordination
nixl.distributed.put_metadata('checkpoint_epoch', '42')
nixl.distributed.put_metadata('model_version', 'v1.2.3')

# Clean up
nixl.distributed.finalize()
```

## Key Features

- **Memory Abstraction**: Unified interface for CPU and GPU memory
- **Storage Flexibility**: Plugin-based support for file, block, and object storage
- **Multi-Language Support**: C++, Python, and Rust bindings
- **Performance Optimized**: UCX and GDRCopy integration
- **Distributed Ready**: ETCD support for cluster coordination
- **Telemetry Built-in**: Observability and monitoring capabilities

## Performance Optimization

### Best Practices

1. **Enable GDRCopy for Small Transfers**
   ```bash
   export NIXL_ENABLE_GDRCOPY=true
   # Provides 10x speedup for transfers < 256KB
   ```

2. **Use Memory Pools for Frequent Allocations**
   ```python
   # Pre-allocate memory pool
   pool = nixl.MemoryPool(size=4 * 1024**3)  # 4GB pool
   buffer = pool.allocate(1024**2)  # Fast allocation from pool
   ```

3. **Configure UCX Transport**
   ```bash
   # For InfiniBand/RDMA
   export UCX_TLS=rc,cuda_copy,gdr_copy

   # For Ethernet
   export UCX_TLS=tcp,cuda_copy
   ```

4. **Optimal Buffer Sizes**
   - Small transfers (< 256KB): Use GDRCopy
   - Medium transfers (256KB - 4MB): Standard UCX
   - Large transfers (> 4MB): Enable pipelining

   ```python
   # Enable pipelining for large transfers
   nixl.configure(pipeline_chunks=16, chunk_size=256*1024)
   ```

5. **Pin Memory for Zero-Copy**
   ```python
   # Pin host memory to avoid extra copies
   cpu_buffer = nixl.Buffer.from_host(data, pinned=True)
   ```

6. **Monitor Performance Metrics**
   ```python
   # Enable telemetry
   nixl.telemetry.enable()

   # After transfers, check metrics
   metrics = nixl.telemetry.get_metrics()
   print(f"Avg bandwidth: {metrics['bandwidth_gbps']} GB/s")
   print(f"Cache hit rate: {metrics['cache_hit_rate']}")
   ```

### Expected Performance

| Transfer Type | Size | Expected Bandwidth | Notes |
|--------------|------|-------------------|-------|
| CPU → GPU | < 256KB | 80-100 GB/s | With GDRCopy |
| CPU → GPU | > 4MB | 40-50 GB/s | Standard UCX |
| GPU → GPU (same node) | Any | 200-300 GB/s | NVLink |
| Node → Node | Any | 90-100 Gb/s | InfiniBand |
| GPU → Storage | > 1MB | 10-15 GB/s | With GDS |

## Use Cases

1. **AI Model Serving**: Accelerate inference pipelines with optimized memory transfers
2. **Distributed Inference**: Coordinate multiple GPU nodes for large-scale inference
3. **Hybrid Storage Systems**: Seamlessly work across different storage backends
4. **Performance Benchmarking**: Evaluate and optimize transfer performance
5. **Custom Inference Frameworks**: Build on NIXL's modular architecture

## Examples

### Example 1: Basic CPU-GPU Transfer with Error Handling

```python
#!/usr/bin/env python3
"""Basic NIXL example with proper error handling"""

import nixl
import numpy as np
import sys

def main():
    try:
        # Initialize NIXL
        print("Initializing NIXL...")
        nixl.initialize(
            backend='ucx',
            enable_gpu=True,
            log_level='INFO'
        )

        # Create test data
        size_mb = 100
        data = np.random.rand(size_mb * 1024 * 1024 // 8).astype(np.float64)
        print(f"Created {size_mb}MB test data")

        # Allocate buffers
        cpu_buffer = nixl.Buffer.from_host(data, pinned=True)
        gpu_buffer = nixl.Buffer.allocate_device(
            size=data.nbytes,
            device_id=0
        )

        # Transfer with timing
        import time
        start = time.time()
        transfer = nixl.transfer(src=cpu_buffer, dst=gpu_buffer)
        transfer.wait()
        elapsed = time.time() - start

        bandwidth = data.nbytes / elapsed / 1e9
        print(f"Transfer completed in {elapsed:.3f}s ({bandwidth:.2f} GB/s)")

        # Verify transfer
        if transfer.status == nixl.TransferStatus.SUCCESS:
            print("✓ Transfer successful")
        else:
            print(f"✗ Transfer failed: {transfer.error}")
            return 1

        return 0

    except nixl.NIXLError as e:
        print(f"NIXL error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1
    finally:
        # Always clean up
        if 'cpu_buffer' in locals():
            cpu_buffer.free()
        if 'gpu_buffer' in locals():
            gpu_buffer.free()
        nixl.finalize()
        print("Resources cleaned up")

if __name__ == '__main__':
    sys.exit(main())
```

### Example 2: Performance Benchmarking Script

```bash
#!/bin/bash
# benchmark_nixl.sh - Comprehensive NIXL performance testing

set -e

echo "=== NIXL Performance Benchmark ==="
echo "Testing configuration:"
echo "  Backend: UCX with GDRCopy"
echo "  Transfer sizes: 1MB, 10MB, 100MB, 1GB"
echo ""

# Configure for optimal performance
export NIXL_ENABLE_GDRCOPY=true
export UCX_TLS=rc,cuda_copy,gdr_copy
export NIXL_LOG_LEVEL=WARN

# Test different transfer sizes
for size in 1MB 10MB 100MB 1GB; do
    echo "--- Testing ${size} transfers ---"

    # CPU to GPU
    echo "CPU → GPU:"
    nixlbench --mode p2p \
              --src-type cpu \
              --dst-type gpu \
              --size ${size} \
              --iterations 100 \
              --output json > results_cpu_gpu_${size}.json

    # GPU to GPU (same node)
    echo "GPU → GPU (intra-node):"
    nixlbench --mode p2p \
              --src-type gpu \
              --src-device 0 \
              --dst-type gpu \
              --dst-device 1 \
              --size ${size} \
              --iterations 100 \
              --output json > results_gpu_gpu_${size}.json

    echo ""
done

# Parse and display results
python3 << 'EOF'
import json
import glob

print("\n=== Summary ===")
print(f"{'Transfer Type':<20} {'Size':<10} {'Bandwidth':<15} {'Latency':<15}")
print("-" * 60)

for result_file in sorted(glob.glob("results_*.json")):
    with open(result_file) as f:
        data = json.load(f)
        transfer_type = result_file.split('_')[1:3]
        print(f"{' → '.join(transfer_type):<20} "
              f"{data['size']:<10} "
              f"{data['bandwidth_gbps']:<15.2f} "
              f"{data['avg_latency_ms']:<15.2f}")
EOF

echo "Benchmark complete. Results saved to results_*.json"
```

### Example 3: Distributed Inference Coordination

```python
#!/usr/bin/env python3
"""Distributed inference with NIXL and ETCD coordination"""

import nixl
import argparse
import socket

def setup_distributed_node(node_id, etcd_endpoints, cluster_name):
    """Initialize NIXL in distributed mode"""
    try:
        # Connect to ETCD cluster
        nixl.distributed.initialize(
            etcd_endpoints=etcd_endpoints,
            node_id=node_id,
            cluster_name=cluster_name,
            timeout_sec=30
        )

        # Register this node's capabilities
        hostname = socket.gethostname()
        gpu_count = nixl.get_device_count()

        nixl.distributed.register_node(
            hostname=hostname,
            gpu_devices=list(range(gpu_count)),
            memory_gb=nixl.get_total_memory() // (1024**3),
            capabilities=['inference', 'training']
        )

        print(f"✓ Node {node_id} registered with {gpu_count} GPUs")
        return True

    except nixl.NIXLError as e:
        print(f"✗ Failed to initialize: {e}")
        return False


def coordinate_inference(node_id, model_name):
    """Coordinate distributed inference across cluster"""

    # Wait for all nodes to be ready
    expected_nodes = 4
    print(f"Waiting for {expected_nodes} nodes...")

    nixl.distributed.wait_for_nodes(count=expected_nodes, timeout_sec=60)
    nodes = nixl.distributed.list_nodes()
    print(f"✓ All {len(nodes)} nodes ready")

    # Elect a leader for coordination
    leader_id = nixl.distributed.elect_leader(ttl_sec=30)
    is_leader = (leader_id == node_id)

    if is_leader:
        print(f"✓ This node is the leader")

        # Leader: Load and distribute model weights
        weights = load_model_weights(model_name)

        # Store model metadata in ETCD
        nixl.distributed.put_metadata(
            key=f'model/{model_name}/version',
            value='1.0.0'
        )
        nixl.distributed.put_metadata(
            key=f'model/{model_name}/size_bytes',
            value=str(weights.nbytes)
        )

        # Distribute weights to all nodes
        for node in nodes:
            if node['node_id'] != node_id:
                remote_buffer = nixl.distributed.create_remote_buffer(
                    node_id=node['node_id'],
                    size=weights.nbytes,
                    name=f'{model_name}_weights'
                )

                # Transfer weights to remote node
                local_buffer = nixl.Buffer.from_host(weights)
                nixl.transfer(src=local_buffer, dst=remote_buffer)
                print(f"  ✓ Weights sent to {node['node_id']}")

    else:
        print(f"✓ This node is a worker (leader: {leader_id})")

        # Worker: Receive model weights from leader
        weight_size = int(nixl.distributed.get_metadata(
            key=f'model/{model_name}/size_bytes'
        ))

        local_buffer = nixl.Buffer.allocate_device(
            size=weight_size,
            device_id=0
        )

        # Register buffer for remote access
        nixl.distributed.register_buffer(
            buffer=local_buffer,
            name=f'{model_name}_weights'
        )

        print(f"  ✓ Waiting for weights from leader...")
        nixl.distributed.barrier()  # Synchronize with other nodes
        print(f"  ✓ Weights received")

    # All nodes synchronized and ready for inference
    print("✓ Cluster ready for distributed inference")
    return True


def main():
    parser = argparse.ArgumentParser(description='Distributed NIXL inference')
    parser.add_argument('--node-id', required=True, help='Unique node ID')
    parser.add_argument('--etcd', default='localhost:2379', help='ETCD endpoints')
    parser.add_argument('--cluster', default='inference-cluster', help='Cluster name')
    parser.add_argument('--model', default='llama-70b', help='Model name')
    args = parser.parse_args()

    etcd_endpoints = args.etcd.split(',')

    try:
        # Setup distributed node
        if not setup_distributed_node(args.node_id, etcd_endpoints, args.cluster):
            return 1

        # Coordinate inference
        if not coordinate_inference(args.node_id, args.model):
            return 1

        # Keep node running
        print("\nNode ready. Press Ctrl+C to exit.")
        nixl.distributed.serve()  # Blocks until interrupted

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        nixl.distributed.finalize()
        print("✓ Node shutdown complete")

    return 0


def load_model_weights(model_name):
    """Stub: Load model weights from storage"""
    import numpy as np
    # In real implementation, load from disk/S3
    return np.random.rand(1000000).astype(np.float32)


if __name__ == '__main__':
    import sys
    sys.exit(main())
```

**Usage:**
```bash
# On node 1 (leader candidate)
python distributed_inference.py --node-id node-1 --etcd 192.168.1.10:2379

# On node 2
python distributed_inference.py --node-id node-2 --etcd 192.168.1.10:2379

# On node 3
python distributed_inference.py --node-id node-3 --etcd 192.168.1.10:2379

# On node 4
python distributed_inference.py --node-id node-4 --etcd 192.168.1.10:2379
```

### Example 4: C++ API Usage

```cpp
// nixl_example.cpp - Using NIXL from C++
#include <nixl/nixl.h>
#include <iostream>
#include <vector>

int main() {
    try {
        // Initialize NIXL
        nixl::Config config;
        config.backend = nixl::Backend::UCX;
        config.enable_gpu = true;
        nixl::initialize(config);

        // Create host data
        std::vector<float> host_data(1024 * 1024, 1.0f);

        // Allocate buffers
        auto cpu_buffer = nixl::Buffer::from_host(
            host_data.data(),
            host_data.size() * sizeof(float),
            nixl::MemoryType::HOST_PINNED
        );

        auto gpu_buffer = nixl::Buffer::allocate(
            host_data.size() * sizeof(float),
            nixl::MemoryType::DEVICE,
            0  // device_id
        );

        // Transfer with callback
        auto transfer = nixl::transfer_async(
            cpu_buffer,
            gpu_buffer,
            [](nixl::TransferStatus status) {
                if (status == nixl::TransferStatus::SUCCESS) {
                    std::cout << "Transfer completed successfully\n";
                } else {
                    std::cerr << "Transfer failed\n";
                }
            }
        );

        transfer.wait();

        // Cleanup
        nixl::finalize();
        return 0;

    } catch (const nixl::NIXLException& e) {
        std::cerr << "NIXL error: " << e.what() << "\n";
        return 1;
    }
}
```

**Build:**
```bash
g++ -std=c++17 nixl_example.cpp -o nixl_example \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lnixl -lcuda
```

### Example 5: Rust API Usage

```rust
// main.rs - Using NIXL from Rust
use nixl::{Buffer, Config, Backend, MemoryType, Result};

fn main() -> Result<()> {
    // Initialize NIXL
    let config = Config {
        backend: Backend::Ucx,
        enable_gpu: true,
        log_level: nixl::LogLevel::Info,
        ..Default::default()
    };
    nixl::initialize(&config)?;

    // Create host data
    let host_data: Vec<f32> = vec![1.0; 1024 * 1024];

    // Allocate buffers
    let cpu_buffer = Buffer::from_host(&host_data, MemoryType::HostPinned)?;
    let gpu_buffer = Buffer::allocate(
        host_data.len() * std::mem::size_of::<f32>(),
        MemoryType::Device,
        0,  // device_id
    )?;

    // Transfer data
    let transfer = nixl::transfer(&cpu_buffer, &gpu_buffer)?;
    transfer.wait()?;

    println!("Transfer completed: {} bytes", cpu_buffer.size());

    // Cleanup (automatic via Drop trait)
    nixl::finalize()?;
    Ok(())
}
```

**Cargo.toml:**
```toml
[dependencies]
nixl = "0.1"
```

**Build and run:**
```bash
cargo build --release
cargo run --release
```

## Security Considerations

### Distributed Setup Security

When deploying NIXL in distributed environments with ETCD:

#### 1. Network Security

```bash
# Enable TLS for ETCD connections
export NIXL_ETCD_USE_TLS=true
export NIXL_ETCD_CA_CERT=/path/to/ca.crt
export NIXL_ETCD_CLIENT_CERT=/path/to/client.crt
export NIXL_ETCD_CLIENT_KEY=/path/to/client.key
```

```python
# Python API with TLS
nixl.distributed.initialize(
    etcd_endpoints=['192.168.1.10:2379'],
    node_id='node-1',
    tls_config={
        'ca_cert': '/path/to/ca.crt',
        'client_cert': '/path/to/client.crt',
        'client_key': '/path/to/client.key'
    }
)
```

#### 2. Authentication

```bash
# Enable ETCD authentication
etcdctl user add root
etcdctl role add nixl-role
etcdctl user grant-role nixl-user nixl-role

# Configure NIXL with credentials
export NIXL_ETCD_USERNAME=nixl-user
export NIXL_ETCD_PASSWORD=secure-password
```

#### 3. Access Control

```bash
# Restrict ETCD key access by prefix
etcdctl role grant-permission nixl-role readwrite /nixl/

# Use namespace isolation per cluster
export NIXL_ETCD_PREFIX=/nixl/cluster-prod
```

#### 4. Network Isolation

```bash
# Configure firewall rules (example using iptables)
# Allow only cluster nodes to access ETCD
sudo iptables -A INPUT -p tcp --dport 2379 -s 192.168.1.0/24 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 2379 -j DROP

# Allow UCX/RDMA traffic only within cluster
sudo iptables -A INPUT -p tcp --dport 13337 -s 192.168.1.0/24 -j ACCEPT
```

#### 5. Data at Rest

```python
# Encrypt sensitive data before storing in ETCD
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt metadata
encrypted_value = cipher.encrypt(b"sensitive-data")
nixl.distributed.put_metadata('encrypted_key', encrypted_value.decode())
```

#### 6. Audit Logging

```bash
# Enable comprehensive logging
export NIXL_AUDIT_LOG=/var/log/nixl/audit.log
export NIXL_LOG_TRANSFERS=true
export NIXL_LOG_METADATA_ACCESS=true
```

```python
# Monitor access patterns
audit_log = nixl.get_audit_log()
for entry in audit_log:
    print(f"{entry.timestamp}: {entry.action} by {entry.node_id}")
```

### Best Practices

1. **Principle of Least Privilege**: Grant nodes only necessary permissions
2. **Regular Key Rotation**: Rotate TLS certificates and authentication credentials
3. **Network Segmentation**: Isolate NIXL traffic on dedicated VLAN
4. **Monitoring**: Set up alerts for unusual transfer patterns
5. **Secure Defaults**: Always use TLS in production environments

## Troubleshooting

### Common Issues and Solutions

#### 1. Silent Fallback to Low-Performance Paths

**Problem**: NIXL runs but performance is much lower than expected (< 10 GB/s for large transfers).

**Cause**: GDRCopy or UCX not properly installed, causing fallback to slower CPU-based copies.

**Solution**:
```bash
# Check if GDRCopy is available
ls /dev/gdrdrv*  # Should show device files

# Verify UCX installation
ucx_info -v  # Should show version 1.20.x

# Check NIXL detected optimizations
python -c "import nixl; print(nixl.get_capabilities())"
# Should show: {'gdrcopy': True, 'ucx': True, 'gds': True}

# Enable verbose logging to see fallback warnings
export NIXL_LOG_LEVEL=DEBUG
```

#### 2. ETCD Connection Failures

**Problem**: Distributed mode fails with "connection refused" or timeouts.

**Solution**:
```bash
# Verify ETCD is running
etcdctl --endpoints=localhost:2379 endpoint health

# Check firewall rules
sudo ufw allow 2379/tcp
sudo ufw allow 2380/tcp

# Test connectivity from worker node
curl http://192.168.1.10:2379/version

# Use correct endpoint format (no http:// prefix in NIXL)
export NIXL_ETCD_ENDPOINTS=192.168.1.10:2379  # Correct
# Not: export NIXL_ETCD_ENDPOINTS=http://192.168.1.10:2379
```

#### 3. CUDA Out of Memory

**Problem**: Transfer fails with "CUDA out of memory" error.

**Solution**:
```python
# Use memory pools to reduce fragmentation
pool = nixl.MemoryPool(size=2 * 1024**3)  # 2GB pool

# Or explicitly free buffers
buffer.free()
nixl.gc()  # Force garbage collection

# Check available GPU memory
print(nixl.get_free_memory(device_id=0))
```

#### 4. UCX Transport Selection Issues

**Problem**: Warnings like "No transports available" or poor RDMA performance.

**Solution**:
```bash
# List available UCX transports
ucx_info -d

# For InfiniBand/RoCE
export UCX_TLS=rc,cuda_copy,gdr_copy
export UCX_NET_DEVICES=mlx5_0:1  # Specify IB device

# For Ethernet-only environments
export UCX_TLS=tcp,cuda_copy
```

#### 5. Permission Denied on /dev/nvidia*

**Problem**: Cannot access GPU devices.

**Solution**:
```bash
# Add user to video/render group
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Or run with proper permissions
# (Not recommended for production)
sudo chmod 666 /dev/nvidia*
```

#### 6. High Latency or Jitter

**Problem**: Transfer times vary significantly between runs.

**Possible Causes & Solutions**:

```bash
# CPU frequency scaling
sudo cpupower frequency-set -g performance

# IRQ affinity (pin interrupts to specific cores)
sudo set_irq_affinity.sh 0-15 mlx5_0

# Disable CPU C-states
sudo tuned-adm profile latency-performance

# Check for thermal throttling
nvidia-smi dmon -s pucvmet
```

#### 7. Build from Source Failures

**Problem**: Compilation errors when building NIXL.

**Solution**:
```bash
# Install all build dependencies
sudo apt-get install -y \
    meson ninja-build cmake \
    g++ gcc \
    libucx-dev \
    cuda-toolkit-12-0 \
    python3-dev

# Specify CUDA path if not in default location
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Clean build
rm -rf build
meson setup build --wipe
ninja -C build
```

### Getting Help

If issues persist:

1. **Enable debug logging**:
   ```bash
   export NIXL_LOG_LEVEL=DEBUG
   export UCX_LOG_LEVEL=info
   ```

2. **Collect diagnostic information**:
   ```bash
   nixl-info --system > nixl_debug.txt
   ```

3. **Check GitHub issues**: https://github.com/ai-dynamo/nixl/issues

4. **Report bugs with**:
   - NIXL version (`nixl.__version__`)
   - CUDA version (`nvcc --version`)
   - UCX version (`ucx_info -v`)
   - OS and kernel version
   - Full error logs

## Resources

- **Repository**: https://github.com/ai-dynamo/nixl
- **License**: Apache 2.0
- **Documentation**: https://github.com/ai-dynamo/nixl/tree/main/docs
- **Docker Images**: `nvcr.io/nvidia/nixl:latest`
- **Issue Tracker**: https://github.com/ai-dynamo/nixl/issues
- **Discussions**: https://github.com/ai-dynamo/nixl/discussions

## Advanced Topics

### Zero-Copy Transfers

```python
# Enable zero-copy mode for maximum performance
nixl.configure(zero_copy=True)

# Requires pinned memory
buffer = nixl.Buffer.from_host(data, pinned=True)

# Transfer happens without intermediate buffers
nixl.transfer(src=buffer, dst=gpu_buffer)
```

### Pipelining for Large Transfers

```python
# Break large transfers into pipeline stages
pipeline = nixl.Pipeline(stages=4, chunk_size=256*1024*1024)

# Transfer happens in overlapping chunks
pipeline.transfer(src=large_cpu_buffer, dst=large_gpu_buffer)
```

### Custom Memory Allocators

```python
# Register custom allocator for specialized hardware
class CustomAllocator(nixl.Allocator):
    def allocate(self, size):
        # Custom allocation logic
        return custom_malloc(size)

    def free(self, ptr):
        custom_free(ptr)

nixl.register_allocator('custom', CustomAllocator())
nixl.configure(allocator='custom')
```

### Compression for Network Transfers

```python
# Enable compression for inter-node transfers
nixl.distributed.configure(
    compression='lz4',  # Options: lz4, zstd, none
    compression_level=3
)

# Useful for network-bandwidth-limited scenarios
```

### Integration with PyTorch

```python
import torch
import nixl

# Transfer PyTorch tensor to GPU using NIXL
tensor = torch.randn(1000, 1000)
nixl_buffer = nixl.Buffer.from_torch(tensor)
gpu_buffer = nixl.Buffer.allocate_device(size=tensor.nbytes)

nixl.transfer(src=nixl_buffer, dst=gpu_buffer)

# Convert back to PyTorch tensor
result = gpu_buffer.to_torch()
```

### Integration with NVIDIA Triton

```python
# Use NIXL as backend for Triton Inference Server
import triton_python_backend_utils as pb_utils
import nixl

class NixlModel:
    def initialize(self, args):
        self.nixl_backend = nixl.Backend('ucx')

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, 'input')

            # Transfer using NIXL
            nixl_buffer = nixl.Buffer.from_numpy(input_tensor.as_numpy())
            # Process...

            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses
```

## Notes

### Platform Support
- **Linux only** (no Windows/macOS support)
- Tested on Ubuntu 22.04/24.04 and Fedora 38/39
- Kernel 5.15+ recommended for best performance

### Hardware Requirements
- **GPU**: NVIDIA GPUs with compute capability 7.0+ (Volta or newer)
- **Network**: InfiniBand or RoCE recommended for multi-node setups
- **Storage**: NVMe SSDs recommended when using GDS backend

### Performance Considerations
- GDRCopy provides 10x speedup for transfers < 256KB
- ETCD adds ~1-2ms latency for distributed coordination
- Check CUDA version compatibility before installation (12.x or 13.x)
- UCX v1.20.x required (v1.21+ not yet tested)

### Production Readiness
- GDRCopy recommended for production deployments
- ETCD required for distributed setups (single-node: optional)
- Enable TLS for production ETCD connections
- Monitor telemetry for performance degradation alerts

### Known Limitations
- Maximum transfer size: 2^64 bytes (practical limit: available GPU memory)
- ETCD cluster supports up to 10,000 nodes (practical limit: ~1,000)
- Rust bindings are experimental (C++ and Python are stable)

## Related Technologies

- **NVIDIA UCX** (Unified Communication X): High-performance networking library
- **GDRCopy** (GPUDirect RDMA): Kernel module for CPU-GPU direct memory access
- **ETCD**: Distributed reliable key-value store for metadata coordination
- **NVIDIA Dynamo**: AI inference framework that leverages NIXL
- **GPUDirect Storage (GDS)**: Direct GPU-to-storage data path
- **RDMA** (Remote Direct Memory Access): Low-latency networking protocol
- **InfiniBand/RoCE**: High-bandwidth, low-latency network fabrics

