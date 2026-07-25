<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL multi-process Prometheus Telemetry exporter plug-in (`prometheus_mp`)

`prometheus_mp` exposes the telemetry of **all** processes of a multi-process
NIXL run (e.g. tensor/data parallelism) behind a **single** Prometheus scrape
endpoint, without any DOCA/DTS dependency.

It complements the single-process [`prometheus`](../prometheus/README.md) exporter
(which binds one port per process, so only one rank's metrics are scraped) and the
DOCA/CollectX exporter (which aggregates via an external DTS service). Use
`prometheus_mp` when you want all ranks aggregated natively with no extra
infrastructure. General NIXL telemetry background: [docs/telemetry.md](../../../../docs/telemetry.md).

## Dependencies

Same as the `prometheus` plug-in: the bundled prometheus-cpp subproject and
`libcurl` (`libcurl4-openssl-dev` / `libcurl-devel`).

## How it works

- **Every process writes its own metric state** to a per-process memory-mapped
  file in a shared directory (`NIXL_TELEMETRY_MULTIPROC_DIR`). Updates are
  lock-free; there is no serialization.
- **Locked owner election.** On startup each process races for an `flock` on
  `nixl-owner.lock` in the shared directory. The one that wins ("owner") binds the
  scrape port and runs the HTTP endpoint plus a collector that, on each scrape,
  reads every live process's file and republishes them as one exposition. The
  processes that lose run in **writer-only** mode and never bind. Losing is
  therefore benign -- every process gets a valid telemetry sink; no rank is dropped
  and no scary error is logged. The lock, not the bind, is what elects: two ranks
  binding concurrently cannot tell which of them got there first, so gating the
  bind on an exclusive lock is what makes exactly one process serve. The kernel
  releases it when the holder dies, so it needs no cleanup.
  That guarantee lasts as long as the lock is usable. If the lock file cannot be
  opened, is not a regular file owned by the run's user, or sits on a filesystem
  without `flock`, every process considers itself elected and falls back to the
  port bind deciding -- one owner still, unless the ranks also disagree on the
  port, in which case each binds its own. Every such process warns first, so the
  fallback is never silent.
- **Two misconfigurations are reported.** The owner records its endpoint in the
  lock file it holds, which lets both silent failure modes be named:
  - The **owner cannot bind** -- since no sibling can be serving, the port belongs
    to something outside the run (a foreign service, or a rank pointed at a
    different `NIXL_TELEMETRY_MULTIPROC_DIR`). Nothing aggregates the directory, so
    it is a warning. Every rank reports it, because the election is conceded on a
    failed bind: a process starting once the port frees -- a conflict as short as a
    previous run still shutting down -- then takes the endpoint over.
  - A **loser was configured for a different endpoint** than the one recorded, so
    the ranks disagree on `NIXL_TELEMETRY_PROMETHEUS_PORT` (or
    `NIXL_TELEMETRY_PROMETHEUS_LOCAL`). Only the owner's endpoint is scrapeable;
    the rank is still exported behind it, so this is a warning, not a failure.

  Ranks split across *directories* are only detected from the abandoned side. The
  directory that did elect an owner cannot tell that ranks it never saw went
  elsewhere: it aggregates a subset and looks healthy.
- **Per-process series.** Each process is exported as its own series (cumulative
  counters, last-operation gauges and duration histograms), never summed across
  processes, so per-process values stay correct and monotonic.
- **Stale handling.** On clean shutdown a process removes its own store file. If
  it instead crashes or is killed -- and so cannot clean up after itself -- the
  owner keeps publishing its last values until *both* the process is gone
  (verified by pid + `/proc` start time) and its last update has aged past the
  TTL; only then are the series dropped and the file reaped. A live process is
  therefore never dropped for being idle, and a dead one lingers for at most the
  TTL.
- **The owner is elected once, and there is no failover.** If the owner process
  exits, no surviving writer promotes itself: the endpoint stays down for the rest
  of the run while every remaining process keeps updating its store file. Reaping
  stops with it, since the owner was the reaper -- a killed owner leaves its own
  file behind. Scraping resumes only when some process starts and wins the election
  (a restarted rank), or the process family is relaunched. Because this looks to
  Prometheus like the target going down rather than the ranks going idle, alert on
  the target's `up` metric, not on absent series. Automatic failover is deliberately
  out of scope for now.

## Configuration

```bash
export NIXL_TELEMETRY_ENABLE="y"
export NIXL_TELEMETRY_EXPORTER="prometheus_mp" # selects libtelemetry_exporter_prometheus_mp.so
export NIXL_TELEMETRY_MULTIPROC_DIR="/run/nixl_metrics" # REQUIRED: shared by all ranks in the pod
```

This mirrors Dynamo's `PROMETHEUS_MULTIPROC_DIR` convention (a shared folder that
every related process writes into, one leader exports): all ranks that should be
aggregated together must point `NIXL_TELEMETRY_MULTIPROC_DIR` at the **same**
directory. Unlike Dynamo -- which auto-creates a temp dir in the parent and lets
child engine processes inherit it -- NIXL is a library loaded independently in each
rank, so there is no parent to propagate the path; the launcher/operator must set
the same directory for every rank (hence it is required, not auto-defaulted, so a
per-process temp dir can never silently break aggregation).

Recommended, following Dynamo's model: a shared **local** folder, one per pod /
process-family, treated as ephemeral (e.g. a per-pod Kubernetes `emptyDir`, or a
temp dir cleaned between runs). It must be a local filesystem -- **not** a network
filesystem (NFS/CIFS), where mmap `MAP_SHARED` cross-process visibility is not
guaranteed (the same restriction Dynamo's multiprocess dir has). tmpfs (e.g. a
Memory-medium `emptyDir` or `/dev/shm`) works and avoids any disk writeback, but is
optional -- a plain local dir is fine, since updates hit the page cache and the
per-process store files are ~one page each.

Use a **private** directory (mode `0700`, owned by the run's user) rather than a
world-writable location like `/tmp`. On a shared host a world-writable directory
lets another user pre-plant paths the owner would truncate or unlink. The plugin
already hardens the files themselves (opened with `O_NOFOLLOW`, created `0600`,
and skipped at scrape time -- with a warning -- when the file's owner is not the
reader's effective uid, so a co-tenant cannot inject series). The same check
guards the election: a `nixl-owner.lock` that is not a regular file owned by the
run's user is ignored rather than contended for, so a co-tenant holding a planted
lock cannot demote every rank to writer-only and leave the run unscrapeable.
The directory's permissions remain the deployment's responsibility.

All aggregated ranks must also share a **PID namespace** (and time namespace):
staleness/liveness uses `kill(pid, 0)` + `/proc/<pid>/stat` and a host-wide
`CLOCK_MONOTONIC`, so ranks must run in one process family / container (or a pod
with `shareProcessNamespace: true`). Ranks in separate PID namespaces sharing only
the directory would misidentify each other's liveness -- keeping dead series or
reaping live ones.

### Optional configuration

```bash
# Scrape port (default 9090) and bind scope -- shared with the prometheus plug-in.
export NIXL_TELEMETRY_PROMETHEUS_PORT="<port_num>"
export NIXL_TELEMETRY_PROMETHEUS_LOCAL="y" # bind 127.0.0.1 instead of 0.0.0.0

# Optional local_rank label: names the env var that holds the rank (default LOCAL_RANK).
# If that env var is unset, no local_rank label is emitted (series stay unique via pid).
export NIXL_TELEMETRY_RANK_ENV="LOCAL_RANK"

# Seconds after a dead process's last update before its store is considered stale
# and reaped (default 30). A live process is always published regardless of age.
export NIXL_TELEMETRY_MP_STALE_TTL="30"

# Histogram bucket upper bounds in microseconds -- shared with the prometheus and
# DOCA exporters. This exporter keeps its buckets in the fixed-layout store, so at
# most 32 bounds are accepted; a longer list fails agent construction rather than
# being silently truncated.
export NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US="10,100,1000"
```

## Metric labels

Every series is labeled by:

- `hostname` -- host where the agent runs.
- `agent_name` -- the agent name given at initialization.
- `pid` -- the producing process id. This guarantees each process is a distinct
  series even if agent names collide; it is deliberately **not** named `instance`
  (a reserved Prometheus target label).
- `agent_instance` -- a per-process counter distinguishing multiple agents created
  in the same process (which share `pid`, `hostname`, and `agent_name`), so their
  series never collide. `0` for the common single-agent-per-process case.
- `local_rank` -- **optional**, present only when a rank env var (see
  `NIXL_TELEMETRY_RANK_ENV`) is set. This is the local/per-GPU (TP) rank, distinct
  from Dynamo's data-parallel `dp_rank`.
- `status` -- only on `agent_errors_total`, bounded by the fixed `AGENT_ERR_*` set.

The metric names, types, semantics, and events are identical to the single-process
[`prometheus`](../prometheus/README.md) exporter (same shared descriptor). That
includes the transfer-duration histograms `agent_xfer_time_us` and
`agent_xfer_post_time_us`, exposed as the usual `_bucket{le="..."}` / `_sum` /
`_count` series per process.

## Design scope & limitations

This exporter is **purpose-built for NIXL's telemetry model, not a generic
Prometheus multiprocess store** (in particular it is not compatible with, and does
not reuse, Python `prometheus_client`'s multiprocess format):

- The metric set is fixed at compile time; slots are positional, so metric names
  are never stored in the files. Histogram bucket bounds are the one exception --
  they are stored per file, because each process resolves them from its own
  environment. Ranks configured with different bounds therefore contribute series
  with different `le` sets to the same family; give every rank the same
  `NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US`.
- Per-process label values (`hostname`, `agent_name`, `pid`, `local_rank`) are
  captured once at startup and never change. Events carry only a numeric value --
  there are **no per-observation labels**.
- Consequently the store **cannot represent a metric with a dynamic /
  high-cardinality label** whose value varies per observation. No NIXL metric has
  such a label today; if one is ever added, this exporter would need a different
  (keyed) store.
- **Process churn creates new series.** `pid` and `agent_instance` are what keep
  each process's counters monotonic, but they also mean a restarted rank is a
  fresh series rather than a continuation: it gets a new `pid`, and the instance
  counter restarts at `0`. The exposition only ever contains live processes, so
  scrape size is bounded by the current process count -- but the TSDB accumulates
  one series set per process seen within the retention window, so a crash-looping
  deployment grows cardinality at the restart rate. Aggregate the churning labels
  away (`sum without (pid, agent_instance) (...)`) for stable per-rank or
  per-host views.

This is the native, dependency-free path. For aggregation through an external
telemetry service, use the DOCA/CollectX exporter (IPC to DTS) instead.
