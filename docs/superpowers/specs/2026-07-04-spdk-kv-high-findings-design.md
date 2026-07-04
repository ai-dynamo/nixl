# SPDK KV Core Contract Hardening Design

## Goal

Resolve the high-priority review findings in the interface-only SPDK KV core while preserving the existing `nixlSpdkKvEngine(init_params)` construction path for downstream implementations.

## Construction and initialization

`nixlSpdkKvEngine` will expose two protected constructors:

- The existing constructor accepts only `nixlBackendInitParams`. It preserves the deferred initialization model for derived classes that implement `openDevice()` and call `initDevice()`.
- A new constructor additionally accepts `std::unique_ptr<iSpdkKvDevice>`. It immediately installs the device and computes the effective key length, avoiding the deferred virtual-open protocol when a derived class can create its device before base construction.

Both paths share one private initialization routine so null-device handling, `initErr`, and key-limit calculation cannot diverge. `initDevice()` remains available for compatibility and delegates to the same routine.

## Value-size contract

All value-size fields in `iSpdkKvDevice` will use `size_t`: `maxValueLen()`, DMA allocation size, Store length, Retrieve buffer length, and Retrieve result length. The common engine therefore never narrows a NIXL descriptor length.

A concrete adapter whose native API accepts `uint32_t`, including `kv_host_shim`, is responsible for rejecting lengths above `UINT32_MAX` before narrowing. A nonzero `maxValueLen()` is an advertised device limit; zero means that the device reports no additional limit, not that the native transport can represent arbitrary sizes.

## Concurrency contract

`nixlSpdkKvEngine` will own a mutable mutex protecting its single `iSpdkKvDevice`. The shared data-plane implementation must hold this mutex for a complete device transaction, including DMA allocation, staging, Store/Retrieve/Exist, and cleanup. This provides correctness for a concrete device backed by one non-thread-safe SPDK qpair.

The initial policy deliberately serializes device transactions. A future multi-queue implementation may replace this policy behind the same interface without changing callers.

## Documentation and verification

The SPDK KV README and header documentation will describe both construction paths, the native-size validation responsibility, and serialized access. Its examples will show constructor injection as the preferred path and deferred `initDevice()` as the compatibility path.

Because this branch is interface-only, verification will consist of header compile checks, static contract checks, formatting/diff checks available in the workspace, and review of the public examples against the new signatures.
