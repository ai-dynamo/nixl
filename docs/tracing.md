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
timeline; it ships as `libtrace_backend_nvtx.so`. Additional backends (e.g. MLCommons
**Chakra** execution traces) are planned and plug into the same call sites.

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

#### Auto-enable under Nsight Systems

When `NIXL_TRACE_BACKENDS` is **unset**, NIXL auto-enables the **NVTX** backend if it
detects that the process is running under Nsight Systems — so a profiled run produces a
timeline without anyone setting the variable by hand. Detection keys off
`NVTX_INJECTION64_PATH`, which `nsys` injects into the environment of the process it
profiles (it is **not** set merely because `nsys` is installed). Outside an `nsys` run
the agent stays inert (no plugin load, no NVTX domain).

Auto-enabling NVTX under nsys is **additive**: it never suppresses a backend you
requested explicitly. The only hard override is a set-but-empty value, which
forces tracing off. Selection when the agent is constructed:

| `NIXL_TRACE_BACKENDS` | Running under nsys | Active backends |
| --------------------- | ------------------ | --------------- |
| set, non-empty (e.g. `chakra`) | no | as listed (`chakra`) |
| set, non-empty (e.g. `chakra`) | yes | as listed **plus** `nvtx` (`chakra,nvtx`; deduplicated) |
| set but empty (`NIXL_TRACE_BACKENDS=`) | either | none (explicit "off" beats auto-enable) |
| unset | yes | `nvtx` |
| unset | no | none |

So `nsys profile ... ./app` needs no NIXL-specific flags to get an NVTX timeline
(and still records any other backend you asked for), while `NIXL_TRACE_BACKENDS=`
force-disables tracing even under a profiler.

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

## Profiling with NVTX / Nsight Systems

NVTX is a lazy, online API: when no profiler is attached, ranges are near-zero-cost
no-op stubs. Run the process under `nsys` to capture the ranges into a `.nsys-rep`
you can open in the Nsight Systems GUI.

The NVTX backend plugin is **optional at build time**: it is built only when the CUDA
toolkit's header-only `nvtx3` headers are present, and is silently skipped otherwise
(there is no separate non-CUDA NVTX build). When it isn't built, requesting it at
runtime (`NIXL_TRACE_BACKENDS=nvtx`) just logs a warning and tracing stays off — the
rest of NIXL is unaffected.

```bash
# Build with tracing + the NVTX backend plugin (both on by default; the NVTX
# plugin is built when the CUDA-toolkit nvtx3 headers are found, else skipped)
meson setup build -Dbuildtype=debug -Dwith_trace=true -Dtrace_backends=nvtx
ninja -C build

# Profile a tracing-enabled run (here: the tracing gtest). Running under nsys
# auto-enables NVTX (see "Auto-enable under Nsight Systems"), so the explicit
# NIXL_TRACE_BACKENDS=nvtx below is optional — kept here to be explicit.
NIXL_TRACE_BACKENDS=nvtx nsys profile --trace=nvtx,cuda,osrt --force-overwrite true \
    --output /tmp/nixl_nvtx \
    ./build/test/gtest/gtest --tests_plugin_dirs=build/test/gtest/mocks \
    --gtest_filter='*Tracing*'

# Summarize from the CLI, or open /tmp/nixl_nvtx.nsys-rep in Nsight Systems:
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

For GPU transfers the host-side NVTX ranges line up on the Nsight timeline with the
GPU activity Nsight captures via CUDA tracing (`--trace=cuda,nvtx`), i.e. a GPU-aware
view by correlation. In-kernel / device-side tracing is out of scope (NVTX cannot run
in `__device__` code).

## Running the tracing tests

The facade unit tests run against mock backends (no plugin needed); the real-agent
`TestTransferTracing` tests load the NVTX plugin and exercise live UCX transfers with
NVTX active:

```bash
ninja -C build test/gtest/unit/unit test/gtest/gtest
# facade unit suite (no backend plugin needed):
./build/test/gtest/unit/unit --gtest_filter='Tracing.*'
# real-agent NVTX suite (loads libtrace_backend_nvtx.so from the build tree):
./build/test/gtest/gtest --tests_plugin_dirs=build/test/gtest/mocks \
    --gtest_filter='*TestTransferTracing*'
```

When `nsys` is available, a `tracing_nsys` CTest additionally profiles the real-agent
test and writes `build/test/gtest/artifacts/nixl_nvtx.nsys-rep` (skipped automatically
if profiling is not permitted in the environment).

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

- **NVTX completeness** — structured, typed NVTX payload attributes (vs the current
  `key=value` marks), hot-path registered-string labels, and broader call-site coverage.
- **Chakra backend** — serialize MLCommons Chakra execution traces (one ET per rank),
  recording the span attributes and dependencies the NVTX backend ignores.
- **Cross-rank correlation** — propagate a global request id on the wire so sender and
  receiver spans can be linked.
- **Backend-engine sub-spans** — finer spans inside backends (e.g. UCX
  `prepXfer`/`postXfer`/`checkXfer`).
