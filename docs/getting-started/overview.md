---
title: Overview
description: NIXL is a high-performance point-to-point data transfer library for AI inference frameworks.
---

## What is NIXL?

NIXL (NVIDIA Inference Xfer Library) accelerates point-to-point data transfers in AI inference frameworks such as [NVIDIA Dynamo](https://github.com/ai-dynamo). It abstracts heterogeneous memory types -- VRAM, DRAM, file, block, and object storage -- through a modular plug-in architecture.

Distributed inference demands high-bandwidth networking across heterogeneous data paths and dynamic scaling. NIXL addresses these requirements with a unified transfer API.

NIXL delivers high-bandwidth, low-latency point-to-point data transfers across VRAM (HBM), DRAM, local and remote SSDs, and distributed storage. Multiple backend plug-ins -- UCX, Libfabric, GPUDirect Storage, S3 over RDMA, and others -- handle the underlying transports. NIXL abstracts connection management, addressing schemes, and memory characteristics so inference frameworks integrate through a single API.

## Key Capabilities

- **High-performance point-to-point data transfers** with low latency and high bandwidth
- **Unified abstraction across heterogeneous memory types** -- VRAM (HBM), DRAM, block, file, and object storage
- **Multiple backend plug-ins** -- UCX, Libfabric, GPUDirect Storage, S3 over RDMA, and others
- **Dynamic scaling** -- agents can be added or removed at runtime without disrupting existing transfers
- **Asynchronous transfer model** with non-blocking status checking for overlapping computation with data movement
- **Modular plug-in architecture** for extensibility, allowing new backends to be added without modifying the core library

<Tip>
NIXL automatically selects the optimal backend based on the source and destination
memory types and the backends available on both agents. You do not need to manually
specify which transport to use. See [Backend Selection](/docs/user-guide/backend-selection) for the full support matrix.
</Tip>

<div className="nixl-stack-diagram">
<div className="nixl-stack-layer"><div className="nixl-stack-layer-title">Framework and Application Layer</div></div>
<div className="nixl-stack-agent-box">
<div className="nixl-stack-agent-layer nixl-stack-agent-nb"><div className="nixl-stack-agent-layer-title">Northbound API</div><span className="nixl-stack-agent-layer-sub"><a href="/nixl/api-reference/c-api">C++</a> / <a href="/nixl/api-reference/python-api">Python</a> / <a href="/nixl/api-reference/rust-api">Rust</a></span></div>
<div className="nixl-stack-agent-layer nixl-stack-agent-core"><div className="nixl-stack-agent-layer-title">NIXL Core</div><div className="nixl-stack-agent-chips"><span className="nixl-stack-chip">Metadata</span><span className="nixl-stack-chip">Cost Model</span><span className="nixl-stack-chip">Resiliency</span><span className="nixl-stack-chip">Telemetry</span><span className="nixl-stack-chip">Batching</span><span className="nixl-stack-chip">Multi-threading</span><span className="nixl-stack-chip">Path Optimization</span></div></div>
<div className="nixl-stack-agent-layer nixl-stack-agent-sb"><div className="nixl-stack-agent-layer-title">Southbound API</div><span className="nixl-stack-agent-layer-sub"><a href="/nixl/api-reference/plugin-southbound-api">C++</a></span></div>
</div>
<div className="nixl-stack-grid">
<div className="nixl-stack-category">
<div className="nixl-stack-category-header">Network</div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-vram="" data-backend-compute-nvidia="" tabIndex="0">UCX<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">UCX</span><span className="nixl-stack-tooltip-desc">High-performance network transport via RDMA and TCP</span></span></div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-vram="" data-backend-compute-nvidia="" data-backend-compute-aws="" tabIndex="0">Libfabric<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">Libfabric</span><span className="nixl-stack-tooltip-desc">High-performance fabric communication via OFI</span></span></div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-vram="" data-backend-compute-nvidia="" tabIndex="0">Mooncake<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">Mooncake</span><span className="nixl-stack-tooltip-desc">Mooncake transfer engine for distributed memory</span></span></div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-vram="" data-backend-compute-nvidia="" tabIndex="0">UCCL-P2P<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">UCCL-P2P</span><span className="nixl-stack-tooltip-desc">Peer-to-peer transport for multi-GPU communication</span></span></div>
</div>
<div className="nixl-stack-category">
<div className="nixl-stack-category-header">GPU Initiated Communication</div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-vram="" data-backend-compute-nvidia="" tabIndex="0">UCX-device<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">UCX-device</span><span className="nixl-stack-tooltip-desc">GPU-initiated UCX transport via device API</span></span></div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-vram="" data-backend-compute-nvidia="" tabIndex="0">DOCA GPUNetIO<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">DOCA GPUNetIO</span><span className="nixl-stack-tooltip-desc">GPUDirect Async via NVIDIA DOCA framework</span></span></div>
</div>
<div className="nixl-stack-category">
<div className="nixl-stack-category-header">File</div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-vram="" data-mem-file="" data-backend-compute-nvidia="" tabIndex="0">GPUDirect Storage<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">GPUDirect Storage</span><span className="nixl-stack-tooltip-desc">GPUDirect Storage for direct GPU-to-storage transfers</span></span></div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-vram="" data-mem-file="" data-backend-compute-nvidia="" tabIndex="0">GPUDirect Storage MT<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">GPUDirect Storage MT</span><span className="nixl-stack-tooltip-desc">Multi-threaded GPUDirect Storage for parallel I/O</span></span></div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-file="" tabIndex="0">POSIX<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">POSIX</span><span className="nixl-stack-tooltip-desc">Standard POSIX file I/O for local and NFS storage</span></span></div>
<div className="nixl-stack-plugin" data-mem-file="" data-mem-dram="" tabIndex="0">HF3FS<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">HF3FS</span><span className="nixl-stack-tooltip-desc">FUSE-based 3FS distributed file system access</span></span></div>
</div>
<div className="nixl-stack-category">
<div className="nixl-stack-category-header">Object</div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-obj="" tabIndex="0">Object<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">Object</span><span className="nixl-stack-tooltip-desc">S3-compatible object storage (S3, S3_CRT, S3/RDMA)</span></span></div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-obj="" tabIndex="0">Object/RDMA<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">Object/RDMA</span><span className="nixl-stack-tooltip-desc">S3/RDMA accelerated object storage</span></span></div>
<div className="nixl-stack-plugin" data-mem-dram="" data-mem-obj="" tabIndex="0">Azure Blob<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">Azure Blob</span><span className="nixl-stack-tooltip-desc">Azure Blob Storage via REST API</span></span></div>
</div>
<div className="nixl-stack-category">
<div className="nixl-stack-category-header">Block</div>
<div className="nixl-stack-plugin" data-mem-blk="" data-mem-dram="" tabIndex="0">GUSLI<span className="nixl-stack-tooltip" role="tooltip"><span className="nixl-stack-tooltip-title">GUSLI</span><span className="nixl-stack-tooltip-desc">Block-level I/O for GUSLI storage devices</span></span></div>
</div>
</div>
<div className="nixl-stack-bottom-row">
<div className="nixl-stack-bottom-section">
<div className="nixl-stack-section-label">Memory Types</div>
<div className="nixl-stack-memrow">
<div className="nixl-stack-memtype" data-type="VRAM" tabIndex="0">VRAM<span className="nixl-stack-mem-tooltip" role="tooltip">VRAM (HBM) — GPU High Bandwidth Memory</span></div>
<div className="nixl-stack-memtype" data-type="DRAM" tabIndex="0">DRAM<span className="nixl-stack-mem-tooltip" role="tooltip">DRAM — Host Memory</span></div>
<div className="nixl-stack-memtype" data-type="FILE" tabIndex="0">FILE<span className="nixl-stack-mem-tooltip" role="tooltip">File — Local and remote file systems</span></div>
<div className="nixl-stack-memtype" data-type="OBJ" tabIndex="0">OBJ<span className="nixl-stack-mem-tooltip" role="tooltip">Object — Distributed object store</span></div>
<div className="nixl-stack-memtype" data-type="BLK" tabIndex="0">BLK<span className="nixl-stack-mem-tooltip" role="tooltip">Block Storage — Raw block storage device</span></div>
</div>
</div>
<div className="nixl-stack-bottom-section">
<div className="nixl-stack-section-label">Compute Types</div>
<div className="nixl-stack-compute-row">
<div className="nixl-stack-compute-type" data-compute-nvidia="" tabIndex="0">NVIDIA</div>
<div className="nixl-stack-compute-type" data-compute-aws="" tabIndex="0">AWS Trainium / Inferentia</div>
</div>
</div>
</div>
</div>
