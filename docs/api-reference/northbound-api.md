---
title: Northbound API Semantics
description: Binding-independent rules for descriptor matching, registration, transfers, and cleanup.
---

This page covers behavior shared by the C++, Python, and Rust bindings. For the
overall data-transfer lifecycle, see [Architecture](../getting-started/architecture).
For a working end-to-end sequence, see [Quick Start](../getting-started/quick-start).
Exact type and method names are documented in the
[C++](./cpp-api), [Python](./python-api), and [Rust](./rust-api) API references.

## Descriptor matching

NIXL identifies a registration by this tuple:

```text
(memory type, device ID, address or offset, length)
```

The metadata attached to a registration is not part of its identity. Registration
descriptors carry metadata because a backend may need configuration such as a file
path or object key. Transfer descriptors contain only the address or offset, length,
and device ID.

A descriptor list has one memory type. The client must set that type correctly for
every descriptor in the list.

The client may select a subrange of a registered region in a transfer descriptor.
The client must use the same device ID as the registration, and the requested
address range must be fully covered by it.

For file, object, and block segments, the client may use a zero-length registration
to describe an unbounded range starting at the supplied offset. For DRAM and VRAM,
the client must provide the actual region length.

## Registration behavior

Registrations belong to one agent. The client must register each local region with
the agent that owns it. An agent cannot use another local agent's registration as
its own.

A descriptor list is registered transactionally within each backend. If one
descriptor fails, that backend rolls back the descriptors from the same list that it
already registered.

Registration across multiple backends is best-effort:

- The call succeeds when at least one selected or compatible backend registers the
  complete descriptor list.
- A successful call does not mean that every compatible backend accepted the list.
- If all backends fail, the call returns an error and the per-backend attempts are
  rolled back.

The client must not assume that registration validates every external resource or
reports every error immediately. For example, an object-store registration can
succeed even when its object key does not exist; the missing key is reported only
when a transfer tries to access it. The client must assume the same deferred-error
behavior for file path mode, even when a backend currently performs some checks
during registration, and must handle these failures when the transfer is executed.

Duplicate-registration behavior is backend-specific. The client must not assume
that the same descriptor can be registered more than once. If a backend accepts a
duplicate, the client must balance every successful registration with a separate
deregistration.

## Device IDs

A device ID is a memory-type-specific lookup key, not always a hardware device
number.

| Memory type | Client requirement |
|---|---|
| DRAM | The client normally uses `0`; compatible backends may ignore the value. |
| VRAM | The client must use the CUDA ordinal of the GPU that owns the allocation. |
| Block | The client must use the resource identifier expected by the backend. |
| File, fd mode | The client must provide an open file descriptor as the device-ID value. |
| File, path mode | The client must assign a file identity that is unique among active path-mode registrations. |
| Object | The client must assign a lookup identity that is unique among active object registrations. |

The client must not reuse a device ID while the resource it identifies remains
registered. File path-mode backends reject a duplicate active ID. Current object
backends keep one object-key mapping per ID, so reuse can replace the existing
mapping.

### File path mode

File path mode lets the backend open and close a file on behalf of the caller. A
registration is recognized as path mode when its metadata matches:

```text
<access>[,<flag>...]:<path>
```

Supported access values are `ro` and `rw`. Supported flags are `direct`,
`sync`, `noatime`, and `create`.

Examples:

```text
ro:/var/cache/model.bin
rw,direct:/var/cache/model.bin
rw,create:/tmp/output.bin
```

When the client requests path mode, the metadata must match this grammar exactly.
A non-matching string is handled as fd mode, in which case the client must provide
an already-open file descriptor as the device ID.

The client must assign a unique device ID to every active path-mode registration.
Path-mode backends reject an ID that is already in use. The client may reuse the ID
after the corresponding registration has been deregistered.

## Transfer behavior

The client must ensure that every local and remote transfer descriptor resolves to
a registered region with the same memory type and device ID. The requested address
range must fit inside that registration.

The client may post a transfer request again only after it reaches a terminal state.
The client must not post the same request while it is active; the binding reports
this through its normal error mechanism.

After a transfer error, the client must not assume that the destination is unchanged
or contains a valid partial result. NIXL does not roll back writes that already
completed. The client must treat the destination as invalid and repeat the complete
logical transfer from a known state.

The client must not release or reuse transfer buffers while the request is still in
progress. It must wait until the request reports success or an error.

## Deregistration behavior

Deregistration matches the same identity tuple used for registration. The client
must use the same memory type, device ID, address or offset, and length that it used
for registration. Registration metadata is ignored during this lookup, so the
client does not need to repeat it.

Within one backend, NIXL checks that the complete descriptor list is present before
removing any entry from that list. Across multiple backends, cleanup is best-effort;
an error from one backend can leave other backends already deregistered.

Each successful registration consumes one registration lifetime. If a backend
accepted the same identity more than once, the client must deregister it the same
number of times. After all matching registrations are removed, the client may reuse
caller-selected device IDs.
