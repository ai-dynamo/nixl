---
title: NIXLBench
description: A benchmarking tool for measuring NIXL data transfer performance across network and storage backends with etcd-based coordination.
---

NIXLBench is a benchmarking tool for the NVIDIA Inference Xfer Library (NIXL) that measures data transfer performance across distributed computing environments. It enables developers to evaluate throughput and latency for point-to-point communication, multi-node transfers, and storage I/O by exercising the full range of NIXL backends. Workers coordinate through [etcd](../../etcd-metadata-exchange.md), making NIXLBench well suited for containerized and cloud-native deployments.

## Features

- **Network backends** -- [UCX](../../backends/ucx.md), [Libfabric](../../backends/libfabric.md), [Mooncake](../../backends/mooncake.md), and [DOCA GPUNetIO](../../backends/gpunetio.md) for high-speed network communication
- **Storage backends** -- [GPUDirect Storage](../../backends/gds.md), [GPUDirect Storage MT](../../backends/gds-mt.md), [POSIX](../../backends/posix.md), [HF3FS](../../backends/hf3fs.md), [OBJ](../../backends/obj.md), [Azure Blob](../../backends/azure-blob.md), and [GUSLI](../../backends/gusli.md) for storage operations
- **Communication patterns** -- Pairwise, many-to-one, one-to-many, and TP (tensor parallel)
- **Memory types** -- CPU (DRAM) and GPU (VRAM) transfers
- **Worker types** -- NIXL worker with full backend support, and NVSHMEM worker for GPU-focused VRAM-only transfers
- **Coordination** -- etcd-based worker coordination for multi-node and containerized environments
- **Performance metrics** -- Multi-threading support, VMM memory allocation, latency percentiles, and data consistency validation

## Next Steps

- **[Building NIXLBench](./build.md)** -- Docker and native build instructions
- **[Usage and Troubleshooting](./usage.md)** -- Running benchmarks and resolving common issues
