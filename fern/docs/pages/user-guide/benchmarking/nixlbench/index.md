---
title: NIXLBench
description: A benchmarking tool for measuring NIXL data transfer performance across network and storage backends with etcd-based coordination.
---

NIXLBench is a benchmarking tool for the NVIDIA Inference Xfer Library (NIXL) that measures data transfer performance across distributed computing environments. It enables developers to evaluate throughput and latency for point-to-point communication, multi-node transfers, and storage I/O by exercising the full range of NIXL backends. Workers coordinate through [etcd](/nixl/user-guide/metadata-exchange-with-etcd), making NIXLBench well suited for containerized and cloud-native deployments.

## Features

- **Network backends** -- [UCX](/nixl/user-guide/backend-selection/ucx), [Libfabric](/nixl/user-guide/backend-selection/libfabric), [Mooncake](/nixl/user-guide/backend-selection/mooncake), and [DOCA GPUNetIO](/nixl/user-guide/backend-selection/gpunetio) for high-speed network communication
- **Storage backends** -- [GPUDirect Storage](/nixl/user-guide/backend-selection/gds), [GPUDirect Storage MT](/nixl/user-guide/backend-selection/gds-mt), [POSIX](/nixl/user-guide/backend-selection/posix), [HF3FS](/nixl/user-guide/backend-selection/hf3fs), [OBJ](/nixl/user-guide/backend-selection/obj), [Azure Blob](/nixl/user-guide/backend-selection/azure-blob), and [GUSLI](/nixl/user-guide/backend-selection/gusli) for storage operations
- **Communication patterns** -- Pairwise, many-to-one, one-to-many, and TP (tensor parallel)
- **Memory types** -- CPU (DRAM) and GPU (VRAM) transfers
- **Worker types** -- NIXL worker with full backend support, and NVSHMEM worker for GPU-focused VRAM-only transfers
- **Coordination** -- etcd-based worker coordination for multi-node and containerized environments
- **Performance metrics** -- Multi-threading support, VMM memory allocation, latency percentiles, and data consistency validation

## Next Steps

- **[Building NIXLBench](/nixl/user-guide/benchmarking-nixl/nixl-bench/building-nixl-bench)** -- Docker and native build instructions
- **[Usage and Troubleshooting](/nixl/user-guide/benchmarking-nixl/nixl-bench/using-nixl-bench)** -- Running benchmarks and resolving common issues
