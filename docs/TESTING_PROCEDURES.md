# NIXL Performance Optimization Testing Procedures

This document provides step-by-step procedures for testing the RDMA performance optimizations.

## Prerequisites

1. **AWS EKS Cluster** with p5.48xlarge instances (EFA-enabled)
2. **kubectl** configured with cluster access
3. **Docker** for building test images
4. **ECR access** for pushing images

## Test Images

| Optimization | Image Tag |
|--------------|-----------|
| FI_INJECT | `public.ecr.aws/v9l4g5s4/nixl-efa-dev:fi-inject-2026-01-16` |
| Doorbell Batching | `public.ecr.aws/v9l4g5s4/nixl-efa-dev:doorbell-batch-2026-01-16` |
| Transfer Coalescing | `public.ecr.aws/v9l4g5s4/nixl-efa-dev:transfer-coalesce-2026-01-16` |
| Baseline (MRRC) | `public.ecr.aws/v9l4g5s4/nixl-efa-dev:mrrc-2026-01-15` |

---

## Test 1: Baseline Performance (Control)

### Purpose
Establish baseline performance metrics without new optimizations.

### Steps

1. **Deploy baseline pods**
```bash
# Create test namespace
kubectl create namespace nixl-perf-test

# Deploy baseline test pod
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: nixl-baseline
  namespace: nixl-perf-test
spec:
  containers:
  - name: nixl
    image: public.ecr.aws/v9l4g5s4/nixl-efa-dev:mrrc-2026-01-15
    resources:
      limits:
        nvidia.com/gpu: 1
        vpc.amazonaws.com/efa: 1
    command: ["/bin/bash", "-c", "sleep infinity"]
  nodeSelector:
    node.kubernetes.io/instance-type: p5.48xlarge
EOF
```

2. **Run benchmark**
```bash
kubectl exec -it nixl-baseline -n nixl-perf-test -- /bin/bash

# Inside pod
cd /workspace/nixl
python3 -c "
from nixl_cu13 import nixl_agent
import time

# Initialize agent
agent = nixl_agent('test-agent')

# Run transfer benchmark
# ... benchmark code
"
```

3. **Record metrics**
- Throughput (GB/s)
- Latency (microseconds)
- CPU utilization

---

## Test 2: FI_INJECT Optimization

### Purpose
Measure impact of inline send optimization for small messages.

### Expected Improvement
- 10-20% latency reduction for messages < 256 bytes

### Steps

1. **Deploy FI_INJECT pod**
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: nixl-fi-inject
  namespace: nixl-perf-test
spec:
  containers:
  - name: nixl
    image: public.ecr.aws/v9l4g5s4/nixl-efa-dev:fi-inject-2026-01-16
    resources:
      limits:
        nvidia.com/gpu: 1
        vpc.amazonaws.com/efa: 1
    command: ["/bin/bash", "-c", "sleep infinity"]
  nodeSelector:
    node.kubernetes.io/instance-type: p5.48xlarge
EOF
```

2. **Run small message benchmark**
```bash
kubectl exec -it nixl-fi-inject -n nixl-perf-test -- /bin/bash

# Focus on small message sizes
for size in 64 128 256 512 1024; do
    echo "Testing message size: $size bytes"
    # Run benchmark with this size
done
```

3. **Compare with baseline**
```
| Message Size | Baseline Latency | FI_INJECT Latency | Improvement |
|--------------|------------------|-------------------|-------------|
| 64 bytes     | X us             | Y us              | Z%          |
| 128 bytes    | X us             | Y us              | Z%          |
| 256 bytes    | X us             | Y us              | Z%          |
```

---

## Test 3: Doorbell Batching Optimization

### Purpose
Measure impact of batching doorbell rings for multi-rail striping.

### Expected Improvement
- Up to 2x throughput for large transfers striped across 4+ rails

### Steps

1. **Deploy Doorbell Batching pod**
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: nixl-doorbell
  namespace: nixl-perf-test
spec:
  containers:
  - name: nixl
    image: public.ecr.aws/v9l4g5s4/nixl-efa-dev:doorbell-batch-2026-01-16
    resources:
      limits:
        nvidia.com/gpu: 4
        vpc.amazonaws.com/efa: 4
    command: ["/bin/bash", "-c", "sleep infinity"]
  nodeSelector:
    node.kubernetes.io/instance-type: p5.48xlarge
EOF
```

2. **Run multi-rail benchmark**
```bash
kubectl exec -it nixl-doorbell -n nixl-perf-test -- /bin/bash

# Test with large transfers that trigger striping
for size in 128K 256K 512K 1M 4M 16M; do
    echo "Testing transfer size: $size"
    # Run benchmark with multi-rail striping
done
```

3. **Compare with baseline**
```
| Transfer Size | Baseline Throughput | Doorbell Batch | Improvement |
|--------------|---------------------|----------------|-------------|
| 128KB        | X GB/s              | Y GB/s         | Z%          |
| 1MB          | X GB/s              | Y GB/s         | Z%          |
| 16MB         | X GB/s              | Y GB/s         | Z%          |
```

---

## Test 4: Transfer Coalescing Optimization

### Purpose
Measure impact of adaptive rail selection for mid-size transfers.

### Expected Improvement
- Better efficiency for transfers 16KB-128KB

### Steps

1. **Deploy Transfer Coalescing pod**
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: nixl-coalesce
  namespace: nixl-perf-test
spec:
  containers:
  - name: nixl
    image: public.ecr.aws/v9l4g5s4/nixl-efa-dev:transfer-coalesce-2026-01-16
    resources:
      limits:
        nvidia.com/gpu: 4
        vpc.amazonaws.com/efa: 4
    command: ["/bin/bash", "-c", "sleep infinity"]
  nodeSelector:
    node.kubernetes.io/instance-type: p5.48xlarge
EOF
```

2. **Run mid-size transfer benchmark**
```bash
kubectl exec -it nixl-coalesce -n nixl-perf-test -- /bin/bash

# Focus on sizes where coalescing kicks in
# With 4 rails, coalescing activates when chunk_size < 16KB
# So transfers < 64KB will be coalesced
for size in 16K 32K 48K 64K 80K 96K 128K; do
    echo "Testing transfer size: $size"
    # Run benchmark
done
```

3. **Verify coalescing is active**
```bash
# Enable debug logging
export NIXL_LOG_LEVEL=DEBUG

# Look for log messages like:
# "Transfer coalescing: reducing rails from 4 to 2 for transfer_size=32768"
```

4. **Compare with baseline**
```
| Transfer Size | Rails (Baseline) | Rails (Coalesced) | Throughput Change |
|--------------|------------------|-------------------|-------------------|
| 32KB         | 4 (8KB chunks)   | 2 (16KB chunks)   | +X%               |
| 48KB         | 4 (12KB chunks)  | 3 (16KB chunks)   | +X%               |
```

---

## Test 5: Combined Optimizations

### Purpose
Test all optimizations together for production readiness.

### Steps

1. **Build combined image**
```bash
cd /home/ubuntu/nixl-pr

# Checkout branch with all optimizations
git checkout perf/doorbell-batching

# Apply transfer coalescing changes
git cherry-pick perf/transfer-coalescing
git cherry-pick perf/fi-inject-small-messages

# Build combined image
docker build -f Dockerfile.combined -t nixl-all-opts:test .
```

2. **Run full benchmark suite**
- Small messages (64-1024 bytes)
- Mid-size transfers (16KB-128KB)
- Large transfers (256KB-16MB)

3. **Compare with baseline**

---

## Metrics Collection

### Key Metrics
1. **Throughput** - GB/s for data transfers
2. **Latency** - p50, p95, p99 in microseconds
3. **CPU Utilization** - % during transfers
4. **Completion Rate** - Completions per second

### Collection Methods

```python
import time
import statistics

def benchmark_transfer(agent, size, iterations=1000):
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        # Perform transfer
        agent.transfer(...)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000)  # Convert to microseconds

    return {
        'p50': statistics.median(latencies),
        'p95': statistics.quantiles(latencies, n=20)[18],
        'p99': statistics.quantiles(latencies, n=100)[98],
        'throughput_gbps': (size * iterations) / (sum(latencies) * 1000) * 8
    }
```

---

## Cleanup

```bash
# Delete test pods
kubectl delete pod nixl-baseline nixl-fi-inject nixl-doorbell nixl-coalesce -n nixl-perf-test

# Delete namespace
kubectl delete namespace nixl-perf-test
```

---

## Troubleshooting

### Common Issues

1. **EFA not available**
   ```bash
   # Check EFA devices
   kubectl exec -it <pod> -- fi_info -p efa
   ```

2. **OOM errors**
   - Reduce batch size
   - Check GPU memory utilization

3. **Slow transfers**
   - Verify EFA is being used (not TCP fallback)
   - Check for network congestion
   - Verify multi-rail is active

### Debug Logging

```bash
# Enable debug output
export NIXL_LOG_LEVEL=DEBUG
export FI_LOG_LEVEL=debug

# Watch for optimization-specific messages
# FI_INJECT: "Using fi_injectdata for small message"
# Doorbell: "RDMA write posted with FI_MORE (batched)"
# Coalescing: "Transfer coalescing: reducing rails from X to Y"
```
