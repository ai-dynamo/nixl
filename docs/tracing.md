# NIXL Tracing System

## Overview

The NIXL tracing system (`nixl::trace`) records **named, timed spans** around NIXL
operations and routes them to one or more tracing backends. A single set of call
sites inside NIXL fans out to **every enabled backend at runtime**, so adding a new
backend never changes the call sites.

The first available backend is **NVTX** (NVIDIA Tools Extension), which makes NIXL
operations visible as ranges on an [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/)
timeline. Additional backends (e.g. MLCommons **Chakra** execution traces) are planned
and will plug into the same call sites.

Tracing is independent of the [telemetry](telemetry.md) system: telemetry collects
numeric metrics/events, while tracing records spans/markers for profiling and
execution-trace tools.

## Architecture

- **`nixl::trace::Tracer`** — a composite tracer **owned by each `nixlAgent`** (no
  singleton; injected into call sites). One `beginSpan()`/`mark()` call fans out to
  every enabled backend.
- **`nixl::trace::Span`** — a move-only handle returned by `beginSpan()`. It forwards
  attributes/dependencies to each backend span and **ends them all on destruction**
  (RAII), so a span covers the scope in which it is declared.
- **`iTraceBackend` / `iSpanBackend`** — the backend interfaces. NVTX is the first
  implementation; future backends implement the same interfaces.
- **`Kind`** — the operation kind attached to a span. It selects an NVTX color and
  maps 1:1 onto the Chakra `NodeType` vocabulary:
  `Generic`, `Compute`, `MemoryR`, `MemoryW`, `CommSend`, `CommRecv`, `CommColl`,
  `Metadata`.

`nixl::trace` is an **internal NIXL core API**: it is compiled into `libnixl` but
its headers (`tracing/trace.h`, `tracing/trace_macros.h`, under `src/core/tracing/`)
are not installed as public headers. Only NIXL core instruments it today; because
internal→public is a non-breaking change, it can be promoted to a public header later
if an external consumer appears.

## Two gates: compile-time and runtime

A backend is active only when it is **both compiled in and requested at runtime**
(`active = requested ∩ compiled-in`).

### Compile-time (which backends are built into `libnixl`)

| Meson option | Description | Default |
| ------------ | ----------- | ------- |
| `with_trace` | Build the tracing API (defines `NIXL_TRACE_ENABLED`) | `true` |
| `trace_backends` | Comma-separated backends to compile in (e.g. `nvtx`) | `nvtx` |

- With `-Dwith_trace=false`, the call-site macros expand to `do {} while (0)` — zero overhead.
- The NVTX backend requires the header-only **nvtx3** headers (shipped with the CUDA
  toolkit). If they are not found, the NVTX backend is silently disabled and the build
  stays green.

### Runtime (which compiled-in backends the caller activates)

| Source | Description |
| ------ | ----------- |
| `nixlAgentConfig.traceBackends` | Comma-separated list set by the caller, e.g. `"nvtx"` |
| `NIXL_TRACE_BACKENDS` env var | Overrides the config field when set, e.g. `NIXL_TRACE_BACKENDS=nvtx` |

If the requested set is empty (or no requested backend is compiled in), the agent
holds no tracer and call sites take a cheap null-check branch.

```cpp
// C++
nixlAgentConfig cfg;
cfg.traceBackends = "nvtx";
nixlAgent agent("agent_0", cfg);
```

```python
# Python
cfg = nixl_bindings.nixlAgentConfig()
cfg.traceBackends = "nvtx"
```

```bash
# Or, without changing code:
export NIXL_TRACE_BACKENDS=nvtx
```

## Instrumented operations

The following Agent operations emit spans/markers (more will be added in later
backends/PRs):

| Span / marker | Kind | Attributes |
| ------------- | ---- | ---------- |
| `nixl::registerMem` | `MemoryW` | `mem_type` |
| `nixl::deregisterMem` | `Generic` | `mem_type` |
| `nixl::makeXferReq` | `Generic` | `desc_count` |
| `nixl::createXferReq` | `Generic` | `remote_agent`, `desc_count` |
| `nixl::postXferReq.write` | `CommSend` | `remote_agent`, `bytes` |
| `nixl::postXferReq.read` | `CommRecv` | `remote_agent`, `bytes` |
| `nixl::xfer.complete` | `Metadata` (marker) | - |
| `nixl::makeConnection` | `Generic` | `remote_agent` |
| `nixl::genNotif` | `Metadata` | `remote_agent` |
| `nixl::getNotifs` | `Metadata` | - |

> Note: An NVTX range's name and color are fixed when the range opens. Attributes
> (added afterwards) are therefore surfaced by the NVTX backend as `key=value`
> **marks inside the range** (visible on the Nsight timeline); a structured NVTX
> payload schema is a future enhancement. Dependencies (`addCtrlDep`/`addDataDep`)
> have no NVTX representation and are recorded only by offline backends (e.g. Chakra).
> Spans cover the synchronous call only; the `nixl::xfer.complete` marker is emitted
> when `getXferStatus` first observes success.

## Profiling with Nsight Systems

NVTX is a lazy, online API: when no profiler is attached, ranges are near-zero-cost
no-op stubs. When you run the process under `nsys`, the ranges are captured into a
`.nsys-rep` you can open in the Nsight Systems GUI.

```bash
# Build with tracing + the NVTX backend (both are on by default)
meson setup build -Dbuildtype=debug -Dwith_trace=true -Dtrace_backends=nvtx
ninja -C build

# Profile a tracing-enabled run (here: the tracing gtest)
nsys profile --trace=nvtx,osrt --force-overwrite true --output /tmp/nixl_nvtx \
    ./build/test/gtest/gtest --tests_plugin_dirs=build/test/gtest/mocks \
    --gtest_filter='*Tracing*'

# Open /tmp/nixl_nvtx.nsys-rep in Nsight Systems, or summarize from the CLI:
nsys stats --report nvtx_sum --format csv /tmp/nixl_nvtx.nsys-rep | grep 'nixl::'
```

Each agent uses its **own NVTX domain named after the agent**, so ranges appear as
`<agent_name>:<span_name>`, e.g.:

```text
agent_0:nixl::postXferReq.write
agent_0:nixl::createXferReq
agent_0:nixl::registerMem
agent_1:nixl::registerMem
```

### Running the tracing tests

The tracing unit tests run as part of the normal gtest suite (CTest):

```bash
ninja -C build test/gtest/gtest
./build/test/gtest/gtest --gtest_filter='Tracing.*:*Tracing*' \
    --tests_plugin_dirs=build/test/gtest/mocks
```

When `nsys` is available, a `tracing_nsys` test additionally profiles the real-agent
tracing test and writes `build/test/gtest/artifacts/nixl_nvtx.nsys-rep` (skipped
automatically if profiling is not permitted in the environment).

## Requirements and limitations

- The NVTX backend requires the CUDA-toolkit **nvtx3** headers (header-only; nothing
  is linked).
- NVTX produces a separate timeline per process; see [Correlation](#correlation).

## Correlation

"Correlation" can mean two different things here:

- **Cross-rank / cross-agent** -- linking the sender's span to the receiver's span across
  processes. NVTX has **no** native cross-process linkage: each process is its own
  timeline that Nsight aligns best-effort by clock, so the only aid today is matching a
  `request_id`-style attribute by hand. Structured linking is a **Chakra** capability (one
  trace per rank, joined offline by `chakra_trace_link` from matching send/recv) and
  requires a globally unique id propagated on the wire. It is planned, and not part of the
  NVTX backend.
- **Cross-thread within a process** -- attributing spans emitted on different threads
  (e.g. `postXferReq` on the caller thread vs. completion on the progress thread) to the
  same request. The API exposes `pushCorrelationId()` / `popCorrelationId()` for this; they
  are backend-agnostic and currently no-ops in the NVTX backend. Planned.

## Planned work

- **Chakra backend** — serialize MLCommons Chakra execution traces (one ET per rank),
  recording the span attributes and dependencies the NVTX backend ignores.
- **Cross-rank correlation** — propagate a global request id on the wire so sender and
  receiver spans can be linked.
- **Backend-engine sub-spans** — finer spans inside backends (e.g. UCX
  `prepXfer`/`postXfer`/`checkXfer`).
