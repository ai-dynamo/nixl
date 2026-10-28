---
title: Backend Selection
description: How to choose the right NIXL transfer backend for your memory types and use case.
slug: user-guide/backend-selection
---

## How NIXL Selects Backends

NIXL automatically selects the optimal backend based on the source and destination memory types and the backends available on both the local and remote agents. When multiple backends support a given transfer, NIXL chooses the most efficient one.

To override automatic selection, specify the desired backend when creating a transfer request. In most cases, automatic selection is the recommended approach.

<Tip>
In Python, backends listed in `nixl_agent_config.backends` (default: `['UCX']`) are
auto-initialized when the agent is created. In C++ and Rust, you must call
`createBackend()` / `create_backend()` explicitly for each backend you want to use.
</Tip>

For a detailed explanation of how backends interact with the Transfer Agent, Memory Sections, and Metadata Handler, see [Architecture and Concepts](/nixl/getting-started/architecture).

## Memory Types

NIXL provides a unified interface for registering and transferring data across five memory and storage types. Each backend supports a specific subset of these types.

| Memory Type | API Enum | Python String | Description |
|-------------|----------|---------------|-------------|
| VRAM (HBM) | `VRAM_SEG` | `"VRAM"` or `"cuda"` | GPU high-bandwidth memory |
| DRAM | `DRAM_SEG` | `"DRAM"` or `"cpu"` | Standard host memory |
| File | `FILE_SEG` | `"FILE"` | Local and remote file systems |
| Object Storage | `OBJ_SEG` | `"OBJ"` | Distributed object stores (S3, Azure Blob, DDN INFINIA) |
| Block Storage | `BLK_SEG` | `"BLOCK"` | Block-level storage devices |

For more on how memory types fit into the NIXL architecture, see the [Overview](/nixl/getting-started/overview).

## Backend Support Matrix

NIXL loads each backend as a plug-in, and each plug-in supports specific memory types and transport protocols.

<table>
  <thead>
    <tr>
      <th>Backend</th>
      <th style="border-left: 1px solid var(--border, var(--grayscale-a5));">Transfer Types</th>
      <th style="border-left: 1px solid var(--border, var(--grayscale-a5));">Protocol / Technology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/ucx">UCX</a></td>
      <td rowspan="5" style="border-left: 1px solid var(--border, var(--grayscale-a5)); line-height: 2;">VRAM ↔ VRAM<br/>VRAM ↔ DRAM<br/>DRAM ↔ DRAM</td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">RoCE, InfiniBand, TCP</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/libfabric">Libfabric</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">OFI (verbs, EFA, TCP providers)</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/mooncake">Mooncake</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">Mooncake Transfer Engine</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/uccl">UCCL-P2P</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">Peer-to-peer transport</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/gpunetio">DOCA GPUNetIO</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">DOCA GPUDirect Async</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/gds">GDS</a></td>
      <td rowspan="2" style="border-left: 1px solid var(--border, var(--grayscale-a5)); border-top: 1px solid var(--border, var(--grayscale-a5)); line-height: 2;">VRAM ↔ File<br/>DRAM ↔ File</td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">GPUDirect Storage (cuFile API)</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/gds-mt">GDS_MT</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">GPUDirect Storage (cuFile API, multi-threaded)</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/posix">POSIX</a></td>
      <td rowspan="2" style="border-left: 1px solid var(--border, var(--grayscale-a5)); border-top: 1px solid var(--border, var(--grayscale-a5));">DRAM ↔ File</td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">io_uring / linux_aio / posix_aio</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/hf3fs">HF3FS</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">3FS fuse filesystem</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/obj">OBJ</a></td>
      <td rowspan="2" style="border-left: 1px solid var(--border, var(--grayscale-a5)); border-top: 1px solid var(--border, var(--grayscale-a5));">DRAM ↔ Object</td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">S3, S3_CRT, S3/RDMA</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/azure-blob">Azure Blob</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">Azure Blob REST API</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/infinia">INFINIA</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5)); border-top: 1px solid var(--border, var(--grayscale-a5)); line-height: 2;">DRAM ↔ Object<br/>VRAM ↔ Object</td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5)); border-top: 1px solid var(--border, var(--grayscale-a5));">DDN INFINIA Client and Async libraries</td>
    </tr>
    <tr>
      <td><a href="/nixl/user-guide/backend-selection/gusli">GUSLI</a></td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5)); border-top: 1px solid var(--border, var(--grayscale-a5));">DRAM ↔ Block</td>
      <td style="border-left: 1px solid var(--border, var(--grayscale-a5));">Block I/O</td>
    </tr>
  </tbody>
</table>

## Common Scenarios

The table below maps common scenarios to recommended backends. NIXL selects the right backend automatically if it is initialized on both agents.

| Scenario | Source | Destination | Recommended Backend | Why |
|----------|--------|-------------|--------------------|----|
| GPU-to-GPU (same or remote node) | VRAM | VRAM | UCX | RDMA-capable, supports RoCE, InfiniBand, and TCP |
| CPU-to-CPU (remote node) | DRAM | DRAM | UCX | Standard high-performance network transport |
| GPU-to-file (local NVMe) | VRAM | File | GDS / GDS_MT | GPUDirect Storage bypasses CPU bounce buffer |
| CPU-to-file | DRAM | File | POSIX | Standard filesystem I/O |
| CPU-to-S3 | DRAM | Object Storage | OBJ | S3, S3_CRT, S3/RDMA protocol support |
| GPU-to-GPU (Mooncake network) | VRAM | VRAM | Mooncake | Mooncake Transfer Engine for specialized deployments |
| GPU-to-CPU (DOCA/GPUDirect Async) | VRAM | DRAM | DOCA GPUNetIO | DOCA GPUDirect Async path on SmartNIC-equipped systems |
| Block storage I/O | Block Storage | DRAM | GUSLI | Block-level I/O operations |
| CPU-to-Azure Blob | DRAM | Object Storage | Azure Blob | Azure Blob REST API |
| CPU/GPU to DDN INFINIA | DRAM / VRAM | Object Storage | INFINIA | Asynchronous object I/O from registered host or GPU memory |
| File I/O (3FS) | File | DRAM | HF3FS | 3FS fuse-based filesystem |

<Note>
When multiple backends can handle a transfer (e.g., both UCX and Libfabric support DRAM-to-VRAM),
NIXL automatically selects the most efficient one based on the available backends on both agents.
You only need to ensure the desired backends are installed and initialized.
</Note>

Backend plug-ins use the Southbound API exposed by the NIXL core.
