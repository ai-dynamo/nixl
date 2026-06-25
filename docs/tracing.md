# NIXL Tracing System

## Overview

The NIXL tracing system (`nixl::trace`) records **named, timed spans** around NIXL
operations and routes them to one or more tracing backends. A single set of call
sites inside NIXL fans out to **every active backend at runtime**, so adding a new
backend never changes the call sites.

Tracing **backends ship as separate, on-demand `.so` plugins** — loaded by
`nixlPluginManager` exactly like NIXL's data backends and telemetry exporters. Only
the tracing **facade** is compiled into `libnixl`; a backend's (often heavy) optional
dependency stays out of `libnixl` and is `dlopen`'d only when the backend is actually
requested.

The first backend is **NVTX** (NVIDIA Tools Extension), which makes NIXL operations
visible as ranges on an [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/)
timeline; it ships as `libtrace_backend_nvtx.so` in a follow-up change. Additional
backends (e.g. MLCommons **Chakra** execution traces) are planned and plug into the
same call sites.

> Scope: this page describes the tracing facade and the backend-plugin mechanism.
> No backend ships in the base; backend-specific usage (NVTX profiling with Nsight
> Systems, etc.) is documented alongside each backend plugin.

Tracing is independent of the [telemetry](telemetry.md) system: telemetry collects
numeric metrics/events, while tracing records spans/markers for profiling and
execution-trace tools.

## Architecture

- **`nixl::trace::Tracer`** — a composite tracer **owned by each `nixlAgent`** (no
  singleton; injected into call sites). One `beginSpan()`/`mark()` call fans out to
  every active backend.
- **`nixl::trace::Span`** — a move-only handle returned by `beginSpan()`. It forwards
  attributes/dependencies to each backend span and **ends them all on destruction**
  (RAII), so a span covers the scope in which it is declared.
- **`TraceBackend` / `SpanBackend`** — the backend interfaces. Each backend
  implementation provides them.
- **Backend plugins** — each backend is a `libtrace_backend_<name>.so` that exports
  `nixl_trace_plugin_init()` returning a `nixlTracePlugin` (a `create_backend`
  factory + api version). `nixlPluginManager::loadTracePlugin()` discovers and
  `dlopen`s them on demand by the `libtrace_backend_` prefix — the same machinery used
  for data backends and telemetry exporters. The plugin contract lives in
  `src/core/tracing/trace_plugin.h`.
- **`Kind`** — the operation kind attached to a span. It selects an NVTX color and
  maps 1:1 onto the Chakra `NodeType` vocabulary:
  `Generic`, `Compute`, `MemoryR`, `MemoryW`, `CommSend`, `CommRecv`, `CommColl`,
  `Metadata`.

`nixl::trace` is an **internal NIXL core API**: it is compiled into `libnixl` but its
headers (`tracing/trace.h`, `tracing/trace_macros.h`, `tracing/trace_plugin.h`, under
`src/core/tracing/`) are not installed as public headers. Only NIXL core instruments
it today; because internal→public is a non-breaking change, it can be promoted to a
public header later if an external consumer appears.

## Two gates: build/packaging and runtime

A backend is active only when its plugin is **both built (packaged) and requested at
runtime**.

### Build / packaging (which backend plugins are built)

| Meson option | Description | Default |
| ------------ | ----------- | ------- |
| `with_trace` | Build the tracing facade (defines `NIXL_TRACE_ENABLED`) | `true` |
| `trace_backends` | Comma-separated backend plugins to build, e.g. `nvtx` | `nvtx` |

- With `-Dwith_trace=false`, the call-site macros expand to `do {} while (0)` — zero overhead.
- Each requested backend builds a `libtrace_backend_<name>.so` under
  `src/plugins/tracing/<name>/`, installed alongside the other NIXL plugins. A
  backend with an unmet dependency (e.g. NVTX without the CUDA-toolkit `nvtx3`
  headers) is silently skipped so the build stays green.

### Runtime (which installed backends the caller activates)

Runtime selection is **environment-only**, via the `NIXL_TRACE_BACKENDS` variable.
There is intentionally **no** `nixlAgentConfig` field for it, so adding tracing does
not change the public config struct's size/layout (ABI safety).

| Source | Description |
| ------ | ----------- |
| `NIXL_TRACE_BACKENDS` env var | Comma-separated backends to activate, e.g. `NIXL_TRACE_BACKENDS=nvtx` (empty/unset = tracing off) |

```bash
# Activate the NVTX backend (affects every nixlAgent created afterwards):
export NIXL_TRACE_BACKENDS=nvtx
```

The variable is read when the agent is constructed, so set it before creating the
`nixlAgent`. Each requested name is loaded as `libtrace_backend_<name>.so` through
`nixlPluginManager`. If the requested set is empty (or no requested plugin is found),
the agent holds no tracer and call sites take a cheap null-check branch — no `dlopen`,
no allocation.

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

Spans cover the synchronous call only; the `nixl::xfer.complete` marker is emitted
when `getXferStatus` first observes success. How a backend renders attributes and
dependencies (`addCtrlDep`/`addDataDep`) is backend-specific and documented with each
backend (e.g. NVTX surfaces attributes as `key=value` marks inside the range and
ignores dependencies; offline backends such as Chakra record them).

## Running the tracing tests

The facade unit tests run as part of the normal gtest suite (CTest); they exercise
the fan-out, attribute, and inert/no-backend paths against mock backends, so they
need no backend plugin:

```bash
ninja -C build test/gtest/unit/unit
./build/test/gtest/unit/unit --gtest_filter='Tracing.*'
```

Real-agent tracing tests and an `nsys` timeline-capture test ship with the NVTX
backend.

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

- **NVTX backend plugin** (`libtrace_backend_nvtx.so`) — ranges/marks on the Nsight
  Systems timeline, plus an `nsys` capture test (immediate follow-up).
- **Chakra backend** — serialize MLCommons Chakra execution traces (one ET per rank),
  recording the span attributes and dependencies the NVTX backend ignores.
- **Cross-rank correlation** — propagate a global request id on the wire so sender and
  receiver spans can be linked.
- **Backend-engine sub-spans** — finer spans inside backends (e.g. UCX
  `prepXfer`/`postXfer`/`checkXfer`).
