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
  singleton; injected into call sites). One `beginSpan()`, `mark()`, `emitSpan()` or
  `emitEvent()` call fans out to every active backend.
- **`nixl::trace::Span`** — a move-only handle returned by `beginSpan()`. It forwards
  attributes/dependencies to each backend span and **ends them all on destruction**
  (RAII), so a span covers the scope in which it is declared.
- **`emitSpan()` / `emitEvent()`** — record a span or a point event after the fact,
  with its times, trace context, kind and typed attributes passed explicitly instead
  of taken from the clock and the active correlation scope. They are for records
  timed elsewhere or reported from another thread, such as the backend phases below.
  Times are `nixlTime` microseconds on `std::chrono::steady_clock`
  (`nixlTime::getUs()`), not a wall clock; a backend that needs wall-clock time
  converts at export. NVTX can draw only at the moment of the call, so it draws both
  as zero-length ranges there, and an emitted span carries its times as the
  `nixl.start_us` and `nixl.end_us` attributes.
- **`TraceBackend` / `SpanBackend`** — the backend interfaces. Each backend
  implementation provides them.
- **Backend plugins** — each backend is a `libtrace_backend_<name>.so` that exports
  `nixl_trace_plugin_init()` returning a `nixlTracePlugin` (a `create_backend`
  factory + api version). `nixlPluginManager::loadTracePlugin()` discovers and
  `dlopen`s them on demand by the `libtrace_backend_` prefix — the same machinery used
  for data backends and telemetry exporters. The plugin contract lives in
  `src/core/tracing/trace_plugin.h`.
- **`Kind`** — the operation kind attached to a span or event. It selects an NVTX
  color and maps 1:1 onto the Chakra `NodeType` vocabulary:
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

#### Sampling

| Source | Description |
| ------ | ----------- |
| `NIXL_TRACE_SAMPLE_RATIO` env var | Fraction of requests marked as sampled: `0` (off) or a number in `[2^-56, 1]`. Empty is an explicit off; when unset, the variable below decides |
| `OTEL_TRACES_SAMPLE_RATIO` env var | Fallback used only when the NIXL variable is unset; Dynamo's variable, read so a NIXL inside Dynamo inherits its ratio. Unset means the opposite in each: Dynamo samples everything, NIXL samples nothing |

The ratio applies to the trace contexts the agent generates, so it does nothing
on its own: with no backend active — no `NIXL_TRACE_BACKENDS`, no nsys — the
agent creates no contexts, and setting only the ratio produces nothing to see.

The decision is head-based: it is taken once, when the agent creates the trace
context for a request, and derives from that context's own trace id, so it is
stable per request and every peer inspecting the same context agrees with it.
A request is sampled when the random part of its trace id falls in the top
`ratio` of the range, which is the OpenTelemetry consistent-probability rule: a
later phase sampling at a lower ratio thins what NIXL kept instead of discarding
all of it. The sampled flag is also what decides whether a context crosses the
wire: UCX carries only sampled contexts to the remote agent.

A value that is neither `0` nor a number in `[2^-56, 1]` makes agent
construction fail, whether or not tracing is otherwise enabled, rather than
silently sampling nothing. `2^-56` is OpenTelemetry's minimum probability and
means one trace in every value 56 bits can express; anything smaller and
nonzero cannot be represented, so it is rejected instead of quietly behaving
like `0`. The
same is not true of `OTEL_TRACES_SAMPLE_RATIO`, which NIXL does not own: a bad
value there is warned about and ignored. Note that in a `.nixl.cfg` file the
value must be quoted (`NIXL_TRACE_SAMPLE_RATIO = "0.25"`), since it is read as a
string.

## Instrumented operations

The following Agent operations emit spans or events (more will be added in later
backends/PRs):

| Span / event | Kind | Attributes |
| ------------- | ---- | ---------- |
| `nixl::registerMem` | `MemoryW` | `mem_type` |
| `nixl::deregisterMem` | `Generic` | `mem_type` |
| `nixl::makeXferReq` | `Generic` | `desc_count`, `nixl.xfer_request_id` |
| `nixl::createXferReq` | `Generic` | `remote_agent`, `desc_count`, `nixl.xfer_request_id` |
| `nixl::postXferReq.write` | `CommSend` | `remote_agent`, `bytes`, `nixl.xfer_request_id`, `nixl.xfer_execution` |
| `nixl::postXferReq.read` | `CommRecv` | `remote_agent`, `bytes`, `nixl.xfer_request_id`, `nixl.xfer_execution` |
| `nixl::xfer.complete` | `Metadata` (event) | `nixl.xfer_request_id`, `nixl.xfer_execution` |
| `nixl::makeConnection` | `Generic` | `remote_agent` |
| `nixl::genNotif` | `Metadata` | `remote_agent` |
| `nixl::getNotifs` | `Metadata` | - |
| `nixl::loadRemoteMD` | `Metadata` | - |
| `nixl::fetchRemoteMD` | `Metadata` | `remote_agent` |
| `nixl::prepMemView` | `MemoryR` | `mem_type`, `desc_count` |
| `nixl::releaseMemView` | `Generic` | - |

Spans cover the synchronous call only; the `nixl::xfer.complete` event is emitted
once per completed post, by the first call that sees the transfer complete:
`postXferReq` when the backend completes it at once, and otherwise `getXferStatus`,
a repost of the same request, or `releaseXferReq`. How a backend renders attributes and
dependencies (`addCtrlDep`/`addDataDep`) is backend-specific and documented with each
backend (e.g. NVTX attaches attributes as typed payloads on the range via
`nvtxRangePopPayload` and ignores dependencies; offline backends such as Chakra
record them).

### Request id and execution number

While tracing is active, each request gets a request id when `createXferReq` or
`makeXferReq` creates it: 64 random bits, fixed for the request's life and recorded
as `nixl.xfer_request_id`. Each post that passes `postXferReq`'s checks and reaches
the backend takes the request's next execution number, `nixl.xfer_execution`,
starting at 1, whether or not the backend then fails it. A post rejected before
that, such as a repost refused with `NIXL_ERR_REPOST_ACTIVE`, takes none, so its
span carries the request id but no execution number. A post's `nixl::xfer.complete`
carries the same pair as its span. The id is recorded as an unsigned decimal string,
since an `int64_t` attribute would show half of all ids as negative; the execution
number is an `int64_t`. With no tracing backend active, a request has neither. A
drawn id is never 0: if the agent's random source fails while drawing it, the
request's records carry `nixl.xfer_request_id=0` and `nixl.xfer_execution=0`, meaning
none. Both are attributes only: the wire and the NVTX payload carry the trace context
(see "Correlation").

### Backend phases

A backend plugin can record a phase of its own work through
the `nixlBackendTraceSink` it receives in `nixlBackendInitParams`. Core records
each phase as a point event through `emitEvent()`, at the plugin's timestamp and
under the context the plugin passed, so the event keeps its time whichever thread
reports it. The event carries `nixl.backend` (the recording backend's type),
`nixl.phase.timestamp_us` (the plugin's `nixlTime` reading),
`nixl.phase.label` when the plugin supplied one, and any plugin-supplied
attributes. A phase recorded under a trace context also carries the context as
`nixl.traceparent` and is correlated on it, like the agent's own spans. The
phase vocabulary is closed so that timelines stay comparable across backends,
and each phase has a fixed kind. A notification's two ends are `CommSend` and
`CommRecv`, so a backend that builds a graph can pair them; the other phases are
`Metadata`, since whether a wire phase sends or receives depends on the
operation:

| Name | Phase enumerator | Kind | Meaning |
| ---- | ---------------- | ---- | ------- |
| `nixl::submit` | `SUBMIT` | `Metadata` | the backend accepted the operation |
| `nixl::wire.submitted` | `WIRE_SUBMITTED` | `Metadata` | handed to the transport |
| `nixl::wire.completed` | `WIRE_COMPLETED` | `Metadata` | transport signalled completion |
| `nixl::notif.sent` | `NOTIF_SENT` | `CommSend` | notification posted to the wire |
| `nixl::notif.received` | `NOTIF_RECEIVED` | `CommRecv` | notification observed locally |
| `nixl::remote.observed` | `REMOTE_OBSERVED` | `Metadata` | the peer's side was observed |
| `nixl::phase` | `OTHER` | `Metadata` | generic escape hatch; the plugin's label becomes the event name |

The sink interface is internal (`src/core/tracing/backend_trace.h`, not an
installed header), so only in-tree plugins can emit phases. UCX emits
`nixl::submit`, `nixl::wire.submitted`, `nixl::notif.sent` and
`nixl::notif.received`, described below; libfabric's phases come in later PRs.

Recording a phase never fails the transfer: if the trace backend throws, the
phase is dropped and the first drop per backend is logged. With telemetry
enabled, drops are also exported as `agent_trace_phases_dropped_total`, subject
to the same per-metric filter as every other metric (see
[telemetry](telemetry.md)).

A backend also sees the trace context of the request it is executing, as
`traceContext` in the `nixl_opt_b_args_t` that `prepXfer` and `postXfer` receive.
It is the same context at prep and at every post of that request, reposts
included, and the zeroed, invalid context when tracing is off. It is null in
`prepMemView`, which has no request, and `opt_args` itself may be null when an
engine is called directly. The pointer is valid only for the call: a plugin that
needs the context later, at completion or when building a notification, copies
the value onto its own request handle. A carrier that puts it on the wire calls
`encodeTraceContext` itself. `TraceContext` is internal like the sink, so only
in-tree plugins can read it.

UCX is the first such carrier. For a sampled request it appends the 26-byte
record as a `tctx` tag after the notification's `name` and `msg`, so an older
receiver, which stops reading after `msg`, is unaffected; unsampled requests and
`genNotif` add nothing. When the notification arrives at an agent with tracing
on, UCX records `nixl::notif.received` under the sender's context, with
`nixl.remote_agent` naming the sender; an agent with tracing off skips the record
without reading it. The event thus carries the request's trace
id in `nixl.traceparent`, and the same correlation id as the sender's
`nixl::postXferReq.write` or `.read` span. Writes and reads without a
notification give the receiver nothing to observe, so they are traced on the
sender only.

On the sender, UCX records up to three phases for every post of a sampled request,
each under the request's context: `nixl::submit` when `postXfer` accepts the
post, `nixl::wire.submitted` once all of its puts or gets and the flushes behind
them are posted to UCX, and `nixl::notif.sent` when the notification is posted.
If handing a validated post to UCX fails, at a put, a get or the flush, for
example because the peer disconnected, only `nixl::submit` is recorded.
The notification goes out from `postXfer` if the transfer has already completed,
and otherwise from the first status check that finds it complete, so
`notif.sent` appears once per notifying post, from either call. UCX records no
`nixl::wire.completed`: a transfer as a whole is seen to complete only when the
application checks its status, even on the thread-pool engine, whose dedicated
threads see each chunk finish, so the stamp would time the poll rather than the
transport.

A backend that carries the context to its peer says so by overriding
`supportsTraceContext()`, which defaults to `false`; core reads it once, when
the backend is created. Tracing never fails a transfer over it. A sampled
request on a backend that supports remote transfers but not trace contexts runs
as usual, its trace just does not reach the remote agent, and core logs a
warning once per backend in each agent; backends without remote transfers are
not reported. Likewise, if generating a request's context fails, the request
runs with the zeroed context and no correlation id, and each agent logs its
first such failure as an error.

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
ninja -C build test/gtest/unit/unit test/gtest/gtest test/gtest/ucx_trace_receiver \
    test/gtest/mocks/libtrace_backend_recorder.so
# facade unit suite (no backend plugin needed):
./build/test/gtest/unit/unit --gtest_filter='Tracing.*'
# real-agent NVTX suite (loads libtrace_backend_nvtx.so from the build tree):
./build/test/gtest/gtest --tests_plugin_dirs=build/test/gtest/mocks \
    --gtest_filter='*TestTransferTracing*'
# UCX tests: one trace id on two agents, and the sender's phases:
./build/test/gtest/gtest --tests_plugin_dirs=build/test/gtest/mocks \
    --gtest_filter='ucxTraceContextPropagation.*'
```

The two-process test starts its receiver agent as a separate executable,
`ucx_trace_receiver`, which is built and installed next to `gtest`, and checks the
spans both agents record against the sender's request; with `-Dwith_trace=false`,
which compiles the sender's `postXferReq` span out, it checks only the phases, which
go through the sink either way. Four single-process tests in the same suite check
the sender's phases. Transfers to the agent itself, on the default and the
thread-pool engine, record `nixl::submit` and `nixl::wire.submitted` once per post
and `nixl::notif.sent` once per notifying post. A transfer to a second agent that
nothing progresses cannot finish inside `postXfer`, so its `notif.sent` must come
from the status check. An unsampled request, or an agent with tracing off, records
none of them. The agents trace through `recorder`, a test-only backend
(`libtrace_backend_recorder.so` in the mocks directory) that appends one line per
finished or emitted span, per emitted event and per mark to the file named by
`NIXL_TEST_TRACE_FILE`; the tests set both variables themselves.

When `nsys` is available, a `tracing_nsys` CTest additionally profiles the real-agent
test and writes `build/test/gtest/artifacts/nixl_nvtx.nsys-rep` (skipped automatically
if profiling is not permitted in the environment).

## Correlation

In NIXL tracing, "correlation" can mean two different things:

- **Cross-rank / cross-agent** -- linking the sender's span to the receiver's span across
  processes. NVTX has **no** native cross-process linkage: each process is its own
  timeline that Nsight aligns best-effort by clock. What NIXL adds is the id: over UCX
  notifications the receiver's `nixl::notif.received` carries the sender's correlation id
  and `nixl.traceparent` (see "Backend phases"), so the two ranges can be matched
  across timelines. Structured linking is a **Chakra** capability (one trace per rank,
  joined offline by `chakra_trace_link` from matching send/recv); it is planned, and not
  part of the NVTX backend.
- **Cross-thread within a process** -- attributing spans emitted on different threads
  (e.g. `postXferReq` on the caller thread vs. the completion polled on another) to the
  same request. Implemented via the backend-agnostic `pushContext()` / `popContext()`
  API: NIXL wraps `postXferReq` in a correlation scope that hands every backend the
  request's whole trace context, sampled or not, and passes the same context to the
  `xfer.complete` event explicitly. The NVTX backend records the context's span id as
  the event's `uint64` payload -- so a post and its completion carry the same id on the
  timeline regardless of which thread emitted each. A request without a valid context
  pushes nothing.

## Planned work

- **Chakra backend** — serialize MLCommons Chakra execution traces (one ET per rank),
  recording the span attributes and dependencies the NVTX backend ignores.
- **Cross-rank correlation** — propagate a globally unique request id on the wire so
  sender and receiver spans can be linked across processes (distributed tracing; a
  separate NIXL-architecture effort). UCX carries it in notifications (see "Backend
  phases"); the libfabric, gpunetio, mooncake and uccl carriers come next.

- **Backend phase emission** — UCX records `nixl::submit`, `nixl::wire.submitted`,
  `nixl::notif.sent` and `nixl::notif.received` through the sink described under
  "Backend phases". Not recorded yet: an accurate `nixl::wire.completed`,
  which needs a completion point that does not depend on how often the
  application polls, and the other backends' phases.
