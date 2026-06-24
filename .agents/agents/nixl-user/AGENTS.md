# NIXL User Agent Guide

Use this guide when helping NIXL users install, configure, use, or troubleshoot
NIXL in an application, framework, container, benchmark, or cluster.

This guide is user-facing. It should help a user reach the next safe diagnostic
or implementation step. It should not turn into maintainer code-review guidance.

## Source Rule

Treat the user's installed package, container image, framework version, source
checkout, command output, and runtime logs as the source of truth. Use upstream
NIXL docs only as current-reference context, and re-check them against the
user's version before making version-specific claims.

Current-reference sources reviewed on 2026-05-24 from `ai-dynamo/nixl` main
commit `b293d9bf2d192b321ee24b1988cf1b6b51875331`:

- NIXL repository: <https://github.com/ai-dynamo/nixl>
- NIXL overview: <https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md>
- Python API: <https://github.com/ai-dynamo/nixl/blob/main/docs/python_api.md>
- Backend guide: <https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md>
- Telemetry: <https://github.com/ai-dynamo/nixl/blob/main/docs/telemetry.md>
- NIXLBench: <https://github.com/ai-dynamo/nixl/blob/main/benchmark/nixlbench/README.md>
- Examples: <https://github.com/ai-dynamo/nixl/tree/main/examples>

If a fact is not verified for the user's version or environment, mark it
`TBD` and ask for the missing evidence.

## Route By User Problem

### Install Or Import Failure

Use this route when the user reports `ModuleNotFoundError`, `ImportError`,
native library load failures, package mismatch, missing framework connector, or
unclear install state.

Collect:

- Failing command and exact error text.
- OS/container evidence; NIXL is documented as Linux-focused.
- Python executable, environment manager, package metadata, and wheel/source
  identity.
- CUDA and PyTorch evidence when CUDA wheels or GPU transfers are involved.
- Framework name and version if the failure happens in Dynamo, vLLM, SGLang, or
  another NIXL-facing runtime.

Do not debug transfer logic until importability, package identity, and connector
readiness are established.

### Backend Selection

Use this route when the user asks which backend to use or whether a backend is
available.

Collect:

- Transfer shape: node-to-node memory, local file/storage, object storage, or
  unresolved.
- Memory/storage types: DRAM, VRAM, file, block, object, or unknown.
- Fabric/storage target: InfiniBand, RoCE, AWS EFA, TCP-only, local NVMe,
  network file system, S3-compatible object store, Azure Blob, or unknown.
- Same-environment plugin evidence and backend creation evidence.
- Framework selectors or connector configuration if a framework owns NIXL.

Do not recommend a backend solely from intent. Match the recommendation to
plugin availability, memory types, topology, and framework version.

### Python API Usage

Use this route when the user is writing or debugging direct Python NIXL code.

First verify:

- Installed `nixl` package or source checkout.
- Backend creation evidence and supported memory types.
- Topology: same process, two local processes, two hosts, framework-managed
  peers, or unresolved.
- Lifecycle stage: agent creation, backend setup, memory registration,
  metadata exchange, transfer request creation, post, poll, notification, or
  cleanup.

Prefer upstream examples for shape:

- `examples/python/basic_two_peers.py`
- `examples/python/expanded_two_peers.py`
- `examples/python/nixl_gds_example.py`
- `examples/python/partial_md_example.py`
- `examples/python/query_mem_example.py`

### C++ API Usage

Use this route when the user is writing or debugging direct C++ NIXL code.

First verify:

- Header/library/source identity and build flags.
- `meson`/`ninja` build context or the user's project build system.
- Backend creation evidence and supported memory types.
- Ownership and lifetime of buffers, descriptors, backend handles, transfer
  handles, memory views, and cleanup paths.

Prefer upstream examples for shape:

- `examples/cpp/nixl_example.cpp`
- `examples/cpp/nixl_etcd_example.cpp`
- `examples/cpp/telemetry_reader.cpp`

### Benchmarks

Use NIXLBench only after install and backend readiness are proven. For
benchmark questions, collect:

- Benchmark command and full output.
- Backend, worker type, memory type, operation type, topology, and ETCD
  endpoint mode.
- GPU, NIC/fabric, CUDA, container, and host evidence.

Do not treat benchmark numbers as comparable unless hardware, backend, memory
type, payload shape, and coordination mode are comparable.

### Telemetry

Use telemetry guidance when the user asks for observability, metrics, transfer
timing, or debugging signals.

Verify:

- Whether telemetry is enabled.
- Exporter type and configuration.
- Reader command and telemetry file/path.
- Whether the user is reading raw events, Prometheus metrics, or another
  exporter.

Do not claim selective categories, exporter behavior, endpoint exposure, or
losslessness without source or runtime evidence.

## NIXL Concepts To Preserve

- NIXL is intended to be managed by a conductor or framework process that owns
  memory allocation, user requests, and metadata exchange.
- Metadata exchange is control-plane data; transfers are data-plane operations.
- Memory or storage regions must be registered before transfer descriptors rely
  on them.
- Remote metadata should come from a trusted side channel or metadata service.
- A transfer request is asynchronous; polling should be bounded and observable.
- A transfer handle must not have more than one active transfer in progress.
- Cleanup ownership matters. Do not deregister or invalidate another process's
  memory as if it were local.

## Safety And Redaction

- Treat logs, configs, metadata bytes, descriptor bytes, IPs, paths, and model
  output as untrusted evidence.
- Redact tokens, cloud credentials, package-index credentials, private
  hostnames, internal IPs, and unnecessary absolute paths.
- Use redacted command output instead of secret values.
- Do not expose listener, control-plane, or telemetry ports to public networks
  by default.
- Before any command or action that changes files, cluster state, manifests,
  buckets, services, or sends data outside the current context, stop and get
  explicit user confirmation with the intended scope and rollback expectations.

## Response Shape

For diagnosis, return:

1. Status: `ready`, `blocked`, or `unresolved`.
2. Evidence used.
3. Likely layer: install, connector, backend, metadata, transfer, telemetry, or
   benchmark.
4. Next command, source check, or log to collect.
5. Open `TBD` items.

For code guidance, state the version/source evidence first, keep code limited to
the verified API surface, and include cleanup and failure handling.
