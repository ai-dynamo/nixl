---
title: Logging Guide
description: Configuring NIXL's logging -- verbosity, per-process log files, filename templates, size limits, and failure behavior.
---

## Overview

NIXL logs to standard error by default. `NIXL_LOG_LEVEL` controls which records are emitted, and `NIXL_LOG_FILE` additionally mirrors them into a file without changing stderr output.

Logging settings are read during library initialization and must be set before the process starts.

## Verbosity

`NIXL_LOG_LEVEL` accepts `ERROR`, `WARN`, `INFO`, `DEBUG` or `TRACE`, defaulting to `WARN`. The log level applies to both stderr and, when it is enabled, the log file.

## Writing to a file

Set `NIXL_LOG_FILE` to a path:

```bash
export NIXL_LOG_LEVEL=INFO
export NIXL_LOG_FILE=/var/log/nixl/agent.log
```

The file is opened in append mode, and each record is written immediately.

<Note>
Leaving `NIXL_LOG_FILE` unset, or setting it to an empty value, disables file logging.
</Note>

## One file per process

Each process should write to a distinct file. Sharing a path may interleave records and makes rotation unsafe. Use these escapes to generate per-process paths:

| Escape | Expands to |
|--------|------------|
| `%h` | Host name |
| `%p` | Process id |
| `%t` | Process startup time in nanoseconds since the Unix epoch. |
| `%%` | A literal `%` |

Unknown escapes and a trailing `%` are rejected. Use `%%` for a literal percent.

```bash
export NIXL_LOG_FILE=/var/log/nixl/run_%h_%p_%t.log
```

Use `%t` to distinguish restarts because process ids may be reused. `%t` uses nanosecond resolution.

### Retention

Paths containing `%p` and `%t` create a file for every process and run. Manage retention by:

- Deleting each run's log directory when it is no longer needed.
- Omitting `%t` and setting `NIXL_LOG_FILE_SIZE`, at the cost of a restarted process potentially continuing an earlier file.

## Bounding the size

Without a limit the log file grows indefinitely. `NIXL_LOG_FILE_SIZE` caps it at 4 KiB or greater, in bytes, optionally suffixed with `K`, `M` or `G`:

```bash
export NIXL_LOG_FILE=/var/log/nixl/agent.log
export NIXL_LOG_FILE_SIZE=64M
```

On reaching the limit, the file is renamed with a `.1` suffix, replacing the previous generation, and a new file is started. Files created under the limit occupy at most twice the configured size in total.

Existing files are not truncated when a limit is enabled or reduced. An oversized active file or `.1` backup can exceed the total bound until subsequent rotations replace it. Use a previously unused path if the bound must hold immediately.

A record larger than the entire limit cannot fit in either generation. It is still written to stderr, but is omitted from the log file.

Without `NIXL_LOG_FILE_SIZE`, manage file growth externally or give each run a fresh path.

## When logging itself fails

File logging failures do not terminate the process by default:

| Failure | Behavior |
|---------|----------|
| The file cannot be opened | Reported at error severity; file logging is disabled. |
| A later write fails | Reported once on stderr; further file records are dropped. |
| A rotation cannot be done | Reported on stderr; file logging stops. |
| `NIXL_LOG_FILE_SIZE` is below 4 KiB or cannot be parsed | Reported at error severity; file logging is disabled. |

Set `NIXL_LOG_FILE_ERROR_IS_FATAL` to make a setup failure fatal instead: if the log file cannot be initialized -- an unopenable path or an invalid setting -- the process terminates with the report rather than running without its log. Failures after setup, such as a failed write or rotation, remain non-fatal.

## Reference

See [Environment Variables](/nixl/resources/environment-variables) for the full list of variables and their defaults.
