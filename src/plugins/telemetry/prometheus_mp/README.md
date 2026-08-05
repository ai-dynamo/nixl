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
- **Locked owner election, one per address.** On startup -- and, for a process
  that is not serving, again as it exports -- each process races for an `flock` on
  `nixl-owner.<address:port>.lock` in the shared directory. The one that wins
  ("owner") binds the scrape port and runs the HTTP endpoint plus a collector that,
  on each scrape, reads every live process's file and republishes them as one
  exposition. The processes that lose run in **writer-only** mode and never bind.
  Losing is therefore benign -- every process gets a valid telemetry sink; no rank
  is dropped and no scary error is logged. The lock, not the bind, is what elects:
  two ranks binding concurrently cannot tell which of them got there first, so
  gating the bind on an exclusive lock is what makes exactly one process serve. The
  address is part of the lock file's *name*, so the file never needs contents, and
  ranks contend only with the ranks they would actually collide with. The kernel
  releases the lock when the holder dies, so a crash needs no cleanup; the lock file
  itself stays in the directory and is reused by the next run.
  That guarantee lasts as long as the lock is usable. If the lock file cannot be
  opened, is not a regular file owned by the run's user, or sits on a filesystem
  without `flock`, every process considers itself elected and falls back to the
  port bind deciding -- one owner still, unless the ranks also disagree on the
  port, in which case each binds its own. Every such process warns first, so the
  fallback is never silent.
- **Two misconfigurations are reported.**
  - The **owner cannot bind** -- no rank asking for the same address can be
    serving, so the port belongs to something outside the run (a foreign service, a
    rank pointed at a different `NIXL_TELEMETRY_MULTIPROC_DIR`, or a rank that asked
    for the same port on a different address). Nothing aggregates the directory on
    that address, so it is a warning. Every rank reports it, because the election
    is conceded on a failed bind: once the port frees, the next rank to win the
    election takes the address over.
  - The **directory is served on more than one address**, because the ranks
    disagree on `NIXL_TELEMETRY_PROMETHEUS_PORT` (or
    `NIXL_TELEMETRY_PROMETHEUS_LOCAL`). Each such rank wins its own election and
    serves what it was configured with, which is what the operator asked for, but
    every one of them exports *every* rank -- so a Prometheus that scrapes more
    than one sees the same series on each target. An owner detects this by trying the
    other lock files in the directory: one it can lock is a leftover from an earlier
    run, one it cannot is a live second owner.

  Ranks split across *directories* are only detected from the abandoned side. The
  directory that did elect an owner cannot tell that ranks it never saw went
  elsewhere: it aggregates a subset and looks healthy.
- **Per-process series.** Each process is exported as its own series (cumulative
  counters, last-operation gauges and duration histograms), never summed across
  processes, so per-process values stay correct and monotonic.
- **Liveness is a lock, not a pid.** Each process `flock`s its store *before* the
  file has a name, keeps it locked for as long as it lives, and only then links it
  into the directory. So a reader that manages to take the lock knows the writer
  is gone for good -- no other process ever locks a store it did not create -- and
  a store it cannot lock belongs to a live writer. The kernel releases the lock
  however the process died, so nothing needs cleaning up, and none of this needs
  pids, `/proc`, or a PID namespace shared with the writers. Creating the store
  nameless (`O_TMPFILE`) is what closes the last window: a reader can never find a
  store that is initialized but not yet locked. A filesystem without `O_TMPFILE`
  (NFS) gets a `.staging` name renamed into place instead, and a process killed
  inside that window leaks one staging file, which no reader ever looks at.

  The store descriptor is `O_CLOEXEC`, so a child that `exec`s does not hold its
  parent's store. A child that only `fork`s does inherit the lock, and keeps the
  parent's store looking live until it too exits -- correct, in that such a child
  can still write to it.
- **Stale handling.** A departing process leaves its store file behind, whether it
  exits cleanly or is killed: its last values are usually not scraped yet, and
  unlinking on exit would drop everything produced since the previous scrape. The
  owner keeps publishing them until *both* the writer has released the store and
  its last update has aged past the TTL; only then are the series dropped and the
  file reaped. Both are evaluated during a scrape, so a live process is never
  dropped for being idle, and a dead one keeps being published until the first
  scrape after its TTL expires. Keep the TTL at or above the Prometheus scrape
  interval, or a rank that exits between two scrapes is reaped before its final
  values are ever read.

  Being kept while idle lasts as long as the store's lock is usable, the same
  qualifier the election carries. A writer whose `flock` failed (it warns) looks
  abandoned to every reader that *can* lock the store, so going quiet for longer
  than the TTL gets its series dropped and its file reaped while it still runs --
  after which it writes to an unlinked inode and never reappears. Where no process
  can lock at all, the readers' probes fail too and nothing is reaped, so departed
  ranks stay published indefinitely instead.

  This leaves at most one file per run on disk -- the last process to exit has no
  owner left to reap it. It is harmless: the next run reaps it on its first scrape,
  since it is both dead and stale.
- **The owner's death is survived, not fatal.** The kernel releases the lock when
  the owner dies, so a writer re-running the election wins it, binds the port and
  starts aggregating -- reaping the dead owner's own store included. Writers
  re-elect from their export path, a few times a second at most, which makes the
  endpoint unreachable for that gap plus up to one scrape interval rather than for
  the rest of the run. Two consequences worth knowing: a process that exports
  nothing never re-elects, so a run that goes fully idle at the wrong moment stays
  down until any rank produces telemetry again; and when the port is held from
  outside the run, the re-election backs off to every few seconds instead of
  hammering a bind that cannot succeed. Alert on the scrape target's `up` metric
  rather than on absent series -- a gap looks to Prometheus like the target going
  down, not like the ranks going idle.

## Configuration

```bash
export NIXL_TELEMETRY_ENABLE="y"
export NIXL_TELEMETRY_EXPORTER="prometheus_mp" # selects libtelemetry_exporter_prometheus_mp.so
export NIXL_TELEMETRY_MULTIPROC_DIR="/run/nixl_metrics" # REQUIRED: shared by all ranks in the pod
```

Configuration errors here are **fatal, unlike a bind collision**: a missing (or
uncreatable) `NIXL_TELEMETRY_MULTIPROC_DIR`, and a
`NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US` list longer than 32 bounds, throw out of the
`nixlAgent` constructor rather than leaving the rank running without telemetry.
Selecting this exporter and forgetting the directory therefore fails every rank at
startup, loudly, instead of exporting nothing.

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
and skipped at scrape time when the file's owner is not the reader's effective
uid -- warned about once per file, since such a file is never reaped -- so a
co-tenant cannot inject series). The same check guards the election: a
`nixl-owner.<address:port>.lock` that is not a regular file owned by the run's user is ignored
rather than contended for, so a co-tenant holding a planted lock cannot demote
every rank to writer-only and leave the run unscrapeable.
A missing directory is created `0700`; an existing one is left as it is, with a
warning when it is group- or world-writable.

All aggregated ranks must also share a **time namespace**: staleness compares a
host-wide `CLOCK_MONOTONIC`, so the ranks have to agree on it. No PID namespace
requirement -- liveness is the store's lock, which the kernel resolves regardless
of which namespace the writer's pid lives in. The `pid` label is then only a label
(and part of the file name), never something a reader interprets.

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
