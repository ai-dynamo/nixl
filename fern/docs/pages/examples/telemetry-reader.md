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

**Expected output:**

```text
=== NIXL Telemetry Event ===
Timestamp: 2025-06-15 14:30:22.123456
Category: TRANSFER
Event name: xfer_posted
Value: 1024
===========================

=== NIXL Telemetry Event ===
Timestamp: 2025-06-15 14:30:22.124789
Category: TRANSFER
Event name: xfer_completed
Value: 1024
===========================

Total events read: 2
```

<Tip>
For the full telemetry architecture, event categories, Prometheus integration, and configuration details, see the [Telemetry Guide](/nixl/user-guide/telemetry-guide).
</Tip>
