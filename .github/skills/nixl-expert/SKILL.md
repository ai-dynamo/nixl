---
name: nixl-expert
description: Specialized in NVIDIA Inference Xfer Library (NIXL) for high-performance distributed data transfers, backend plugin configuration (UCX, GDS, S3), and integration with NVIDIA Dynamo.
---

# NIXL Expert Skill

Use this skill when you need to configure, debug, or integrate NIXL for distributed AI inference workloads.

## Core Tasks
- **Backend Configuration**: Set up and optimize transfer backends like UCX, GDS (GPUDirect Storage), or Amazon S3 for point-to-point data movement.
- **Distributed Coordination**: Configure ETCD endpoints (`NIXL_ETCD_ENDPOINTS`) and namespaces for metadata distribution across nodes.
- **Memory Management**: Register local/remote memory regions (HBM, DRAM, or SSD) using the `nixl_connect.Connector` and `Descriptor` classes.
- **Integration**: Assist in integrating NIXL with inference engines such as vLLM, SGLang, and TRT-LLM to accelerate KV cache offloading.

## Guidelines
- **Prefer Source Builds**: When encountering `nixlBackendError`, advise users to install NIXL from source rather than via `pip` for the latest stability fixes.
- **Asynchronous Operations**: Always leverage NIXL's asynchronous API to minimize TTFT (Time-to-First-Token) and maximize throughput.
- **Environment Variables**:
  - `NIXL_ETCD_ENDPOINTS`: Required for distributed mode (e.g., `http://localhost:2379`).
  - `NIXL_NO_STUBS_FALLBACK`: Set to `1` to prevent silent fallback to non-functional stubs if the library build fails.

## Example Usage
- **Initializing an Agent with Mooncake**:
  ```python
  from nixl._api import nixl_config
  from nixl._wrapper import NixlWrapper
  agent = NixlWrapper(str(uuid.uuid4()), nixl_agent_config(backends=['Mooncake']))
