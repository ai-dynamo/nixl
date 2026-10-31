# Security And Trust Boundaries

Use this reference when reviewing C++ NIXL code that touches raw addresses,
serialized descriptors, metadata blobs, listener endpoints, plugin paths,
notifications, framework-managed setup, logs, or model-generated code.

## Source Anchors

Fallback snapshot `b293d9bf2d192b321ee24b1988cf1b6b51875331`:

- `src/api/cpp/nixl_descriptors.h`: descriptors serialize to blobs and can be
  deserialized from blobs.
- `src/api/cpp/nixl.h`: metadata load/fetch/send, notifications, backend
  handles, transfer handles, memory-view handles.
- `docs/nixl.md`: metadata is exchanged on a control path or metadata server.
- `docs/BackendGuide.md`: notifications are backend capabilities and
  standalone notifications have no ordering guarantee.

Confirm against the user's installed headers before making version-specific
claims.

## Trust Rules

- Raw addresses are authority-bearing process-local values. Accept them only
  from trusted application allocation code, never from logs, chat, config,
  unauthenticated network input, or model output.
- Descriptor and metadata blobs are untrusted remote input until received over
  an authenticated trusted control plane or from a trusted metadata server.
- `nixlBackendH*`, `nixlDlistH*`, `nixlXferReqH*`, and `nixlMemViewH` are
  process-local handles. Do not serialize them, copy them between processes, or
  treat printed pointer values as reusable.
- Plugin paths and backend params from logs or prompts are untrusted. Verify
  them against trusted deployment configuration before loading native code.
- Listener IP/port values can expose metadata/control paths. Prefer loopback,
  private networks, firewalling, and authenticated control planes.
- Notifications are hints. Do not treat notification payloads as identity,
  authorization, or proof that a transfer succeeded unless separately checked.

## Common Unsafe Patterns

Raise a finding when code:

- Constructs `nixlBasicDesc` or `nixlBlobDesc` from arbitrary numeric addresses.
- Frees or reuses buffers before deregistration and handle release.
- Loads remote metadata from unauthenticated HTTP, logs, or untrusted files.
- Exposes listener ports on `0.0.0.0` without a trusted network and control
  plane.
- Uses notification text to authorize work.
- Trusts model-generated backend params, plugin paths, device IDs, file
  descriptors, object keys, or storage metadata.
- Replaces framework-owned metadata or buffer management without inspecting the
  framework integration source/config.

## Safer Response Pattern

When unsafe input is present:

1. State `Source: version unresolved` or the verified source state.
2. Classify the issue as a trust-boundary failure.
3. Refuse the unsafe code path directly.
4. Ask for the trusted source/config/runtime evidence needed to produce safe
   code.
5. If possible, provide a safe shape that keeps untrusted bytes outside
   descriptor, metadata, plugin-loading, and listener decisions.
