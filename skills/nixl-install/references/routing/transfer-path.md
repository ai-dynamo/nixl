# Transfer-path diagnosis

## Scope

### When to use

Use this route when a running workload should perform a NIXL transfer and the
transfer fails, produces incorrect or incomplete data, uses an unexpected path,
or is itself unexpectedly slow. Also use it when the workload falls back,
recomputes, or otherwise skips the expected transfer, or when the user asks to
verify the effective transfer path.

A skipped transfer belongs here only when no earlier installation, loading,
backend creation, or integration initialization failure explains it.

### When not to use

Do not use this route when installation, import, native-library or plugin
loading, backend creation, or NIXL integration initialization fails before a
transfer can be attempted, including when that failure causes the workload to
skip NIXL. Also exclude general workload slowness or ordinary performance
variation without evidence about the NIXL transfer.

## How to check if uncertain

1. Use the user request and existing configuration, logs, metrics, and traces
   to determine whether a NIXL transfer was expected or path verification was
   requested.
2. Determine whether a transfer symptom, fallback, recomputation, or skipped
   transfer is already visible, and whether existing startup evidence instead
   shows an earlier setup or initialization failure.
3. Request success, aggregate throughput, and general workload latency do not
   prove that NIXL moved the expected data, used a particular path, or was the
   slow layer.

The required routing facts are an expected NIXL transfer or an explicit path
verification request, a transfer-related symptom or that verification request,
and no earlier setup or initialization failure that blocked the transfer. The
effective path and cause are investigation data, not routing prerequisites.
