---
title: Telemetry Reader
description: Read and process NIXL telemetry events programmatically using the shared memory telemetry buffer.
---

The examples below are taken from the `examples/` directory in the [NIXL repository](https://github.com/ai-dynamo/nixl), annotated with inline explanations.

**What you'll learn:** How to read and process NIXL telemetry events programmatically using the shared memory telemetry buffer.

NIXL writes telemetry events to a shared memory ring buffer. The telemetry reader examples show how to open this buffer, read events as they arrive, and format them for display or processing. This is useful for monitoring transfers, debugging performance issues, and building custom telemetry dashboards.

<CodeBlocks>
<Markdown src="/snippets/generated/examples/telemetry-reader-py.mdx" />

<Markdown src="/snippets/generated/examples/telemetry-reader-cpp.mdx" />
</CodeBlocks>

**Representative Python output** (event values vary by workload):

```text
=== NIXL Telemetry Event ===
Event: agent_xfer_post_time
Value: 42
===========================

=== NIXL Telemetry Event ===
Event: agent_xfer_time
Value: 128
===========================

Total events read: 2
Final buffer size: 0 events
```

**Representative C++ output** (event values vary by workload):

```text
=== NIXL Telemetry Event ===
Event name: agent_xfer_post_time
Value: 42
===========================

=== NIXL Telemetry Event ===
Event name: agent_xfer_time
Value: 128
===========================

Total events read: 2
Final buffer size: 0 events
```

<Tip>
For the full telemetry architecture, event categories, Prometheus integration, and configuration details, see the [Telemetry Guide](/nixl/user-guide/telemetry-guide).
</Tip>
