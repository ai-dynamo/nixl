<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
- **Bind-race owner election.** On startup each process tries to bind the scrape
  port. The one that wins ("owner") runs the HTTP endpoint plus a collector that,
  on each scrape, reads every live process's file and republishes them as one
  exposition. The processes that lose the race run in **writer-only** mode (no
  HTTP server). A bind collision is therefore benign -- every process gets a valid
  telemetry sink; no rank is dropped and no scary error is logged.
- **Per-process series.** Each process is exported as its own series (cumulative
  counters and last-operation gauges), never summed across processes, so
  per-process values stay correct and monotonic.
- **Stale handling.** When a process exits, the owner drops its series once the
  process is gone (verified by pid + `/proc` start time) or its data ages past the
  TTL, and reaps the file. Cleanup is performed by the owner, since a killed
  process cannot clean up after itself.

## Configuration

```bash
export NIXL_TELEMETRY_ENABLE="y"
export NIXL_TELEMETRY_EXPORTER="prometheus_mp" # selects libtelemetry_exporter_prometheus_mp.so
export NIXL_TELEMETRY_MULTIPROC_DIR="/tmp/nixl_metrics" # REQUIRED: shared by all ranks in the pod
```

All ranks that should be aggregated together must point `NIXL_TELEMETRY_MULTIPROC_DIR`
at the **same** directory (e.g. a per-pod `emptyDir` under Kubernetes).

### Optional configuration

```bash
# Scrape port (default 9090) and bind scope -- shared with the prometheus plug-in.
export NIXL_TELEMETRY_PROMETHEUS_PORT="<port_num>"
export NIXL_TELEMETRY_PROMETHEUS_LOCAL="y" # bind 127.0.0.1 instead of 0.0.0.0

# Optional dp_rank label: names the env var that holds the rank (default LOCAL_RANK).
# If that env var is unset, no dp_rank label is emitted (series stay unique via pid).
export NIXL_TELEMETRY_RANK_ENV="LOCAL_RANK"

# Seconds after which a dead process's store is considered stale and reaped (default 30).
export NIXL_TELEMETRY_MP_STALE_TTL="30"
```

## Metric labels

Every series is labeled by:

- `hostname` -- host where the agent runs.
- `agent_name` -- the agent name given at initialization.
- `pid` -- the producing process id. This guarantees each process is a distinct
  series even if agent names collide; it is deliberately **not** named `instance`
  (a reserved Prometheus target label).
- `dp_rank` -- **optional**, present only when a rank env var (see
  `NIXL_TELEMETRY_RANK_ENV`) is set.
- `status` -- only on `agent_errors_total`, bounded by the fixed `AGENT_ERR_*` set.

The metric names, types, counter/gauge semantics, and events are identical to the
single-process [`prometheus`](../prometheus/README.md) exporter (same shared
descriptor).

## Design scope & limitations

This exporter is **purpose-built for NIXL's telemetry model, not a generic
Prometheus multiprocess store** (in particular it is not compatible with, and does
not reuse, Python `prometheus_client`'s multiprocess format):

- The metric set is fixed at compile time; slots are positional, so metric names
  are never stored in the files.
- Per-process label values (`hostname`, `agent_name`, `pid`, `dp_rank`) are
  captured once at startup and never change. Events carry only a numeric value --
  there are **no per-observation labels**.
- Consequently the store **cannot represent a metric with a dynamic /
  high-cardinality label** whose value varies per observation. No NIXL metric has
  such a label today; if one is ever added, this exporter would need a different
  (keyed) store.

This is the native, dependency-free path. For aggregation through an external
telemetry service, use the DOCA/CollectX exporter (IPC to DTS) instead.
