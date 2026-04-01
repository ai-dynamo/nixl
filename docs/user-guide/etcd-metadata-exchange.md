---
title: "Metadata Exchange with etcd"
description: Configure etcd-based distributed metadata exchange for NIXL agents -- key prefix scheme, watcher lifecycle, and Prometheus integration.
---

NIXL supports three metadata exchange approaches: side-channel (TCP), programmatic, and etcd. This page covers the etcd approach, where agents publish metadata (connection information and registered memory descriptors) to a distributed key-value store and fetch remote agent metadata by name. Each approach suits different deployment patterns -- etcd centralizes coordination, side-channel exchanges metadata peer-to-peer, and programmatic exchange gives the application full control.

When etcd mode is enabled, agents store and retrieve metadata using a structured key prefix scheme. NIXL sets up watchers on remote agents so that when a remote agent goes down or invalidates its metadata, the local agent is notified and discards stale metadata. For the API calls that trigger these operations, see [Quick Start -- Metadata Exchange](../getting-started/quick-start#metadata-exchange).

<Note>
etcd support requires building NIXL with etcd enabled (`HAVE_ETCD` compile flag). If NIXL was not built with etcd support, setting `NIXL_ETCD_ENDPOINTS` will have no effect. The etcd integration depends on the `etcd-cpp-apiv3` library, which is included as a Meson subproject.
</Note>

## Prerequisites

Before using etcd-based metadata exchange:

- **etcd server v3.x** running and accessible from all NIXL agents
- **NIXL built with etcd support** -- the `etcd-cpp-apiv3` library must be available at build time (enabled via the `HAVE_ETCD` compile flag)
- **Network connectivity** from every NIXL agent to the etcd endpoint(s)

<Tip>
A single-node etcd server works for development. For high-availability deployments, see the [etcd documentation](https://etcd.io/docs/) for clustering, TLS, and authentication.
</Tip>

## Configuration

etcd mode is controlled by environment variables and agent configuration.

### NIXL_ETCD_ENDPOINTS

| Property | Value |
|----------|-------|
| **Type** | String (URL) |
| **Default** | None (etcd mode disabled when unset) |
| **Required** | Yes, to enable etcd mode |

The etcd server endpoint URL. Setting this variable activates etcd mode. The agent's communication worker thread creates an etcd client connected to the specified endpoint.

```bash
export NIXL_ETCD_ENDPOINTS="http://localhost:2379"
```

When `NIXL_ETCD_ENDPOINTS` is not set or empty, etcd mode is disabled and agents use side-channel (TCP) or programmatic metadata exchange instead.

### NIXL_ETCD_NAMESPACE

| Property | Value |
|----------|-------|
| **Type** | String (path prefix) |
| **Default** | `/nixl/agents/` |
| **Required** | No |

The key prefix namespace under which all agent metadata is stored in etcd. All agent keys are created under this prefix, allowing multiple NIXL deployments to share the same etcd cluster without key collisions.

```bash
export NIXL_ETCD_NAMESPACE="/myapp/nixl/agents/"
```

If not set, the default namespace `/nixl/agents/` is used (defined as `NIXL_ETCD_NAMESPACE_DEFAULT` in the source).

### etcdWatchTimeout

| Property | Value |
|----------|-------|
| **Type** | `std::chrono::microseconds` (C++), integer microseconds |
| **Default** | 5,000,000 (5 seconds) |
| **Required** | No |

Timeout for etcd watch operations when waiting for remote agent metadata to appear. When an agent fetches metadata that does not yet exist in etcd, it sets up a watcher and waits up to this duration for the key to be created. If the timeout expires, the fetch operation returns an error.

This is set via the `nixlAgentConfig` struct at agent creation time:

<CodeBlocks>
```cpp title="C++"
nixlAgentConfig cfg;
cfg.useProgThread = true;
cfg.etcdWatchTimeout = std::chrono::microseconds(10000000); // 10 seconds
nixlAgent agent("my_agent", cfg);
```

```python title="Python"
config = nixl_agent_config(
    enable_prog_thread=True,
    enable_listen_thread=True
)
# etcdWatchTimeout is configured at the C++ level;
# Python uses the default (5 seconds) unless set via the C++ bindings
agent = nixl_agent("my_agent", config)
```
</CodeBlocks>

## Key Prefix Scheme

NIXL stores metadata in etcd using a structured key hierarchy. Each agent's metadata is stored under a key composed of the namespace, agent name, and metadata type.

**Key format:** `{namespace}/{agent_name}/{metadata_type}`

The components are:

| Component | Source | Example |
|-----------|--------|---------|
| `namespace` | `NIXL_ETCD_NAMESPACE` env var (default `/nixl/agents/`) | `/nixl/agents/` |
| `agent_name` | The agent's unique name, set at construction | `agent_a` |
| `metadata_type` | Metadata label (default `"metadata"`) | `metadata` |

The default metadata label is `"metadata"` (the `default_metadata_label` constant). This means a default key for an agent named `agent_a` would be `/nixl/agents/agent_a/metadata`.

In addition to the metadata key, each agent stores a **prefix key** at `{namespace}/{agent_name}/` as a presence marker. This empty key is used for watch triggers -- when it is deleted, watchers on that agent are notified.

**Example key layout in etcd:**

```
/nixl/agents/
+-- agent_a/
|   +-- (prefix marker)
|   +-- metadata     <- agent_a's serialized metadata
+-- agent_b/
|   +-- (prefix marker)
|   +-- metadata     <- agent_b's serialized metadata
+-- agent_c/
    +-- (prefix marker)
    +-- metadata     <- agent_c's serialized metadata
```

When using a custom namespace (e.g., `NIXL_ETCD_NAMESPACE="/myapp/nixl/"`), the keys shift accordingly:

```
/myapp/nixl/
+-- agent_a/
|   +-- metadata
+-- agent_b/
    +-- metadata
```

## Metadata Exchange Operations

NIXL provides three etcd operations for metadata lifecycle management: publishing, fetching, and invalidating.

### Publishing Metadata (sendLocalMD)

When an agent publishes its metadata, it serializes all registered memory descriptors and backend connection information into a binary blob, then stores it at `{namespace}/{agent_name}/{metadata_label}` via an etcd `put` operation.

Publishing is triggered by calling `sendLocalMD` (C++), `send_local_metadata` (Python), or `send_local_md` (Rust) without specifying an IP address. When no IP address is provided, NIXL routes the operation through the etcd path.

<CodeBlocks>
```python title="Python"
# Publish this agent's metadata to ETCD
# (no IP/port arguments = ETCD mode)
agent.send_local_metadata()
```

```cpp title="C++"
// Publish metadata to ETCD
nixl_status_t status = agent.sendLocalMD();
```

```rust title="Rust"
// Publish metadata to ETCD
agent.send_local_md(None)?;
```
</CodeBlocks>

<Note>
Memory must be registered before publishing metadata. The serialized blob includes all registered memory descriptors and their associated backend connection info. Any memory registered after publishing will not be visible to remote agents until metadata is re-published.
</Note>

### Fetching Remote Metadata (fetchRemoteMD)

Fetching retrieves a remote agent's metadata from etcd by looking up `{namespace}/{remote_agent}/{metadata_label}`. If the key exists, the metadata is loaded immediately. If the key does not yet exist, NIXL sets up an `etcd::Watcher` on the key and waits for it to appear, with a configurable timeout (`etcdWatchTimeout`).

After successfully fetching metadata, NIXL sets up a persistent watcher on the remote agent's prefix key for invalidation notifications (see [Watcher Component](#watcher-component) below).

<CodeBlocks>
```python title="Python"
# Fetch a remote agent's metadata from ETCD
# (no IP/port arguments = ETCD mode)
agent.fetch_remote_metadata("remote_agent_name")

# Verify the metadata is loaded
while not agent.check_remote_metadata("remote_agent_name"):
    pass
```

```cpp title="C++"
// Fetch remote agent's metadata from ETCD
nixl_status_t status = agent.fetchRemoteMD("remote_agent_name");
```

```rust title="Rust"
// Fetch remote agent's metadata from ETCD
agent.fetch_remote_md("remote_agent_name", None)?;
```
</CodeBlocks>

The fetch operation internally follows this flow:

1. **Direct get** -- attempt to read the key from etcd immediately
2. **Watch and wait** -- if the key is not found, set up a watcher and block until the key appears or the timeout expires
3. **Load metadata** -- deserialize the fetched blob and load it into the agent's internal structures
4. **Setup watcher** -- create a persistent watcher on the remote agent's prefix key for invalidation events

### Invalidating Metadata

Invalidation removes all of an agent's keys from etcd using an `rmdir` operation on `{namespace}/{agent_name}/`. This deletes the metadata key, the prefix marker key, and any partial metadata labels stored under that agent's prefix. The deletion triggers `DELETE` events on watchers, notifying all remote agents that have previously fetched this agent's metadata.

Invalidation happens automatically when an agent is destroyed, or can be triggered explicitly:

<CodeBlocks>
```python title="Python"
# Invalidate this agent's metadata in ETCD
agent.invalidate_local_metadata()
```

```cpp title="C++"
// Invalidate this agent's metadata in ETCD
nixl_status_t status = agent.invalidateLocalMD();
```

```rust title="Rust"
// Agent metadata is invalidated automatically on drop.
// Explicit invalidation can be done via invalidate_local_md if needed.
```
</CodeBlocks>

<Warning>
After invalidation, remote agents that have fetched this agent's metadata will be notified to discard it. The agent must re-publish metadata (via `sendLocalMD` / `send_local_metadata` / `send_local_md`) before remote agents can fetch it again.
</Warning>

## Watcher Component

The watcher component provides automatic metadata invalidation across agents. When Agent A fetches Agent B's metadata, NIXL sets up an `etcd::Watcher` on Agent B's prefix key (`{namespace}/agent_b/`). This watcher runs asynchronously and monitors for `DELETE` events on that key.

**How it works:**

1. After a successful `fetchRemoteMD`, the etcd client calls `setupAgentWatcher(remote_agent)` to create a persistent watcher on the remote agent's prefix key
2. The watcher callback monitors for `DELETE` events. When a delete is detected, the remote agent's name is added to an invalidation queue
3. The agent's communication worker thread periodically processes the invalidation queue, calling `invalidateRemoteMD()` for each agent in the queue
4. This removes the cached metadata and disconnects from that remote agent's backends

**The result:** when a remote agent goes down (or explicitly invalidates its metadata), all agents that have fetched its metadata are automatically notified and discard the stale data. There is no need to poll for changes.

<Tip>
The watcher mechanism means you don't need to poll for metadata changes. When a remote agent re-publishes its metadata after a restart, agents that previously connected to it will need to re-fetch the metadata to re-establish the connection.
</Tip>

**Watcher lifecycle:**

- A watcher is created per remote agent when its metadata is first fetched
- Only one watcher exists per remote agent (duplicate calls to `setupAgentWatcher` are no-ops)
- When an invalidation event fires, the watcher for that agent is removed along with the cached metadata
- If the same remote agent's metadata is fetched again later, a new watcher is created

## Agent Configuration

The full set of etcd-related configuration options available when creating a NIXL agent:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NIXL_ETCD_ENDPOINTS` | Environment variable (string) | Unset (etcd disabled) | etcd server URL to enable etcd mode |
| `NIXL_ETCD_NAMESPACE` | Environment variable (string) | `/nixl/agents/` | Key prefix namespace for all agent metadata |
| `etcdWatchTimeout` | `nixlAgentConfig` field (`std::chrono::microseconds`) | 5,000,000 us (5 s) | Timeout for watch operations waiting for metadata |

<CodeBlocks>
```cpp title="C++"
#include "nixl.h"

// Configure agent with custom ETCD watch timeout
nixlAgentConfig cfg;
cfg.useProgThread = true;
cfg.useListenThread = true;
cfg.etcdWatchTimeout = std::chrono::microseconds(10000000); // 10 seconds

// Ensure NIXL_ETCD_ENDPOINTS is set in the environment
// before creating the agent
nixlAgent agent("my_agent", cfg);

// Publish metadata to ETCD
agent.sendLocalMD();

// Fetch another agent's metadata from ETCD
agent.fetchRemoteMD("other_agent");
```

```python title="Python"
from nixl._api import nixl_agent, nixl_agent_config
import os

# Set ETCD endpoint before creating the agent
os.environ["NIXL_ETCD_ENDPOINTS"] = "http://localhost:2379"

config = nixl_agent_config(
    enable_prog_thread=True,
    enable_listen_thread=True
)
agent = nixl_agent("my_agent", config)

# Publish metadata to ETCD (no IP = ETCD mode)
agent.send_local_metadata()

# Fetch another agent's metadata from ETCD
agent.fetch_remote_metadata("other_agent")
```

```rust title="Rust"
use nixl_sys::Agent;
use std::env;

// Set ETCD endpoint before creating the agent
env::set_var("NIXL_ETCD_ENDPOINTS", "http://localhost:2379");

let agent = Agent::new("my_agent")?;

// Publish metadata to ETCD
agent.send_local_md(None)?;

// Fetch another agent's metadata from ETCD
agent.fetch_remote_md("other_agent", None)?;
```
</CodeBlocks>
