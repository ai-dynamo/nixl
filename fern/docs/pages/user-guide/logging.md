---
title: Logging Guide
description: Configuring NIXL's logging -- verbosity, per-process log files, filename templates, size limits, and failure behavior.
---

## Overview

NIXL logs to standard error by default. `NIXL_LOG_LEVEL` selects how much is emitted, and `NIXL_LOG_FILE` additionally mirrors those records into a file. Both are read once, during library initialization, so they must be set before the process starts.

The file is a supplement rather than a redirect: stderr keeps receiving exactly what it received before, in the same format, so existing tooling that scrapes a process's console is unaffected.

## Verbosity

`NIXL_LOG_LEVEL` accepts `ERROR`, `WARN`, `INFO`, `DEBUG` or `TRACE`, defaulting to `WARN`. The level gates a record before any destination is consulted, so it governs the file and stderr identically and the two cannot drift apart.

## Writing to a file

Set `NIXL_LOG_FILE` to a path:

```bash
export NIXL_LOG_LEVEL=INFO
export NIXL_LOG_FILE=/var/log/nixl/agent.log
```

The file is appended to rather than truncated, so a restarted process adds to the record instead of erasing it. Each record is flushed as it is written, which means the log is complete up to the moment a process crashed or hung -- the case the file exists for.

<Note>
Leaving `NIXL_LOG_FILE` unset, or setting it to an empty value, disables file logging entirely. No file is created.
</Note>

## One file per process

A file is written by exactly one process. Two processes given the same path will interleave their records into it, and rotation assumes a single writer. Rather than requiring a different setting per worker, the path may contain escapes that expand at startup:

| Escape | Expands to |
|--------|------------|
| `%h` | Host name |
| `%p` | Process id |
| `%t` | A per-process run marker, in nanoseconds since the Unix epoch, sampled once at startup |
| `%%` | A literal `%` |

An unrecognized escape is left as written, so a path that legitimately contains a percent still works.

This lets one setting serve every worker of a run:

```bash
export NIXL_LOG_FILE=/var/log/nixl/run_%h_%p_%t.log
```

Include `%t` if the same command may be run more than once. Process ids are recycled, and because the file is appended to, a restart handed an earlier run's id would otherwise continue that run's file as though the two were one process. `%t` is at nanosecond resolution because a rapid restart inside a PID namespace can be handed the same id within the same second.

### Retention

`%p` and `%t` mean every process of every run leaves its own file behind, which is unbounded across a restart loop. Two ways to keep that in hand:

- Give each run a directory of its own, and delete it when the run is done.
- Omit `%t` and set `NIXL_LOG_FILE_SIZE`. The set of filenames is then bounded by the hosts and process ids in play, and each is capped, at the cost of a restart continuing an earlier file.

### Processes that fork

The path is expanded once, when logging is initialized during library load. A process that then calls `fork()` without `exec()` does not get a separate file for the child: the child inherits the parent's already-open file and keeps writing to it, under the parent's `%p`. Workers started through `exec`, or by a launcher, each get their own file as expected.

## Bounding the size

Without a limit the log file grows indefinitely. `NIXL_LOG_FILE_SIZE` caps it, in bytes, optionally suffixed with `K`, `M` or `G` for powers of 1024:

```bash
export NIXL_LOG_FILE=/var/log/nixl/agent.log
export NIXL_LOG_FILE_SIZE=64M
```

On reaching the limit the file is renamed with a `.1` suffix, replacing any previous one, and a new file is started. The live file therefore holds the most recent records, which are the ones that answer what a process did just before it failed, and the generation before them sits alongside it. Exactly one rotated generation is kept, so the total on disk stays under roughly twice the limit.

If `NIXL_LOG_FILE_SIZE` is not set, the file grows without limit and must be managed externally, for example with `logrotate` using `copytruncate`, or by giving each run a fresh path.

## When logging itself fails

Losing the log file never stops the process it was meant to describe. Each failure is reported and then logging carries on as best it can:

| Failure | Behavior |
|---------|----------|
| The file cannot be opened | Reported at error severity, so it is visible even at `NIXL_LOG_LEVEL=ERROR`, and NIXL continues without the file. |
| A later write fails | Reported once on stderr, and further records are dropped rather than holding up the process. |
| A rotation cannot be done | Reported on stderr, and the file is left as it is rather than exceeding the limit. Logging to it stops, so the records written up to that point survive. |
| `NIXL_LOG_FILE_SIZE` cannot be parsed | Reported at error severity, and file logging is disabled rather than ignoring the requested limit. |

<Tip>
Reports about the log file are written straight to stderr, so they arrive even when the file itself is the thing that failed.
</Tip>

## Reference

See [Environment Variables](/nixl/resources/environment-variables) for the full list of variables and their defaults.
