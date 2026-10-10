# Metadata And Connection

Use this reference for side-channel metadata, direct peer metadata, metadata
server exchange, partial metadata, and proactive connection.

## Source Anchors

Fallback snapshot `b293d9bf2d192b321ee24b1988cf1b6b51875331`:

- `src/api/cpp/nixl.h`: `getLocalMD()`, `getLocalPartialMD()`,
  `loadRemoteMD()`, `invalidateRemoteMD()`, `sendLocalMD()`,
  `sendLocalPartialMD()`, `fetchRemoteMD()`, `invalidateLocalMD()`,
  `checkRemoteMD()`, `makeConnection()`.
- `src/api/cpp/nixl_types.h`: `nixl_opt_args_t.ipAddr`, `port`,
  `metadataLabel`, and connection-info fields.
- `docs/nixl.md`: metadata handler, side-channel exchange, central metadata,
  and teardown.
- `examples/cpp/nixl_example.cpp`: local side-channel exchange.
- `examples/cpp/nixl_etcd_example.cpp`: etcd metadata exchange example.

Confirm against the user's installed headers before copying code.

## Side-Channel Metadata

Use this when the application already has a trusted authenticated control plane
for moving opaque metadata bytes:

```cpp
nixl_blob_t local_md;
nixl_status_t st = local_agent.getLocalMD(local_md);
if (st != NIXL_SUCCESS) {
    return st;
}

// Send local_md through a trusted control plane to the peer.
// Receive remote_md through the same trusted control plane.

std::string remote_name;
st = local_agent.loadRemoteMD(remote_md, remote_name);
if (st != NIXL_SUCCESS) {
    return st;
}
```

After loading metadata, `makeConnection(remote_name, &args)` is optional and
source/backend-sensitive. If omitted, a backend may connect during the first
transfer path.

## Direct Peer Or Metadata Server

The fallback C++ API uses `nixl_opt_args_t` for `sendLocalMD()` and
`fetchRemoteMD()`:

- If `ipAddr` is set, the call uses peer-to-peer metadata transfer.
- If `ipAddr` is not set, the call uses the metadata server path when supported
  by the environment/source.
- `port` defaults to `default_comm_port` in the fallback headers.
- `metadataLabel` is used for partial metadata with a central metadata server.

Do not expose direct listener endpoints publicly. Require trusted topology,
firewall, and authentication context before suggesting direct listener use.

## Partial Metadata

The fallback header exposes `getLocalPartialMD()` and `sendLocalPartialMD()`.
Use them only when the user's installed source and backend support the partial
metadata path. Verify descriptor inclusion, connection-info inclusion, and
labels against installed source.

## Readiness Checks

Before transfer creation, confirm:

- The remote metadata has been loaded on the initiator.
- The extracted `remote_name` matches the intended agent name.
- Both sides created compatible backend instances.
- Registered memory was included in the metadata or partial metadata needed by
  the transfer.
- Framework-managed integrations are not being bypassed without source/config
  evidence.

## Cleanup

- `invalidateRemoteMD(remote_agent)` removes locally cached remote metadata and
  disconnects as supported by the backend/source.
- `invalidateLocalMD(&args)` is for invalidating this agent's metadata through
  peer or metadata-server paths. Use it only when that metadata path was used
  and trusted endpoint details are known.
