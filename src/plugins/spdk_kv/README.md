<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# SPDK NVMe Key-Value plugin interface (`spdk_kv`)

This directory defines the **shared, backend-agnostic interfaces** for NIXL
backends that speak the generic NVMe Key-Value (KV) protocol over SPDK. It is an
**interface-only** contribution: it contains the abstract contracts, not the
concrete implementations. The concrete backends (RADOS_NKV, CSAL_NKV) are
delivered in separate backend PRs that *inherit* these interfaces.

## What lives here

| File | Kind | Purpose |
| --- | --- | --- |
| `ispdk_kv_device.h` | interface | `iSpdkKvDevice`: an abstract interface over the `kv_host_shim` NVMe-KV protocol (Store / Retrieve / Exist). |
| `spdk_kv_engine.h` | abstract base | `nixlSpdkKvEngine`: the shared NIXL backend base that drives the data plane against `iSpdkKvDevice`. Declares the South-Bound API (no bodies); the shared `.cpp` is provided by a backend PR. |

The concrete classes shown in the diagram below (`nixlKvHostShimDevice`,
`nixlFakeKvDevice`, `nixlRadosNkvEngine`, `nixlCsalNkvEngine`) are **not** part of
this interface PR. They are described here as worked examples so the design is
clear to anyone implementing a backend against these interfaces.

## Class hierarchy

```
nixlBackendEngine
├── nixlSpdkKvEngine (abstract; shared NVMe-KV data plane)
│   ├── nixlRadosNkvEngine (device: RADOS target)
│   └── nixlCsalNkvEngine  (device: CSAL target)
└── nixlRedisKVEngine (standalone; implemented directly)

iSpdkKvDevice (interface; used by nixlSpdkKvEngine)
├── nixlKvHostShimDevice (real: wraps kv_host_shim C API)
└── nixlFakeKvDevice     (test double, no SPDK/hardware needed)
```

There are two independent axes:

- **The engine axis** (`nixlBackendEngine` → `nixlSpdkKvEngine` → concrete
  backends): what NIXL calls into. Only the SPDK-family backends share
  `nixlSpdkKvEngine`; unrelated backends such as `nixlRedisKVEngine` inherit
  `nixlBackendEngine` directly, with no shared intermediate layer.
- **The device axis** (`iSpdkKvDevice` → concrete devices): the seam the shared
  engine talks to, so the transport (real SPDK vs. an in-memory fake) is swappable
  without touching the data plane.

## Design rationale

`nixlSpdkKvEngine` implements `getSupportedMems`, `registerMem`, `deregisterMem`,
`queryMem`, `prepXfer`, `postXfer`, `checkXfer`, and `releaseReqH` **once**,
against `iSpdkKvDevice`.

**RADOS_NKV and CSAL_NKV** share a base class (`nixlSpdkKvEngine`) by
**inheritance, not composition**. Both talk to a device through the same generic
`kv_host_shim` NVMe-KV protocol, and that protocol is already backend-agnostic.
Since the two concrete backends need the exact same data-plane algorithm:

```
validate → derive key → DMA-stage → store / retrieve / exist
```

a plain abstract base class is the simplest way to share it: `nixlSpdkKvEngine`
implements that algorithm once. Concrete engines inject an opened device and
override `deriveKey()`. The original deferred construction path remains
available for implementations that override `openDevice()` and call
`initDevice()` after their derived state is initialized.

Why split the **device** out as its own interface as well? So the entire engine
data plane can be unit-tested with `nixlFakeKvDevice` (an in-memory double) with
no SPDK or hardware, while production uses `nixlKvHostShimDevice`.

## The device interface: `iSpdkKvDevice`

`iSpdkKvDevice` captures the generic NVMe-KV protocol as a small, purely abstract
C++ interface. It is deliberately **semantic**: operations return `spdk_kv_status_t`
(`OK` / `NOT_FOUND` / `BUFFER_TOO_SMALL` / `ERROR`), never a raw device/NVMe status
code, so the shared engine never branches on SPDK-specific values.

```cpp
class iSpdkKvDevice {
public:
    virtual ~iSpdkKvDevice() = default;

    virtual uint32_t maxKeyLen() const = 0;
    virtual size_t maxValueLen() const = 0;

    virtual void *dmaAlloc(size_t len) = 0;
    virtual void  dmaFree(void *buf) = 0;

    virtual spdk_kv_status_t store(const void *key, uint8_t key_len,
                              const void *value, size_t value_len) = 0;
    virtual spdk_kv_status_t retrieve(const void *key, uint8_t key_len, void *value,
                                 size_t buf_len, size_t *value_len_out) = 0;
    virtual spdk_kv_status_t exist(const void *key, uint8_t key_len) = 0;
};
```

The interface carries **no** `kv_host_shim` / SPDK / NVMe detail. All of that
belongs to the concrete implementations below.

## Worked example 1 — the real device: `nixlKvHostShimDevice`

`nixlKvHostShimDevice` is the production `iSpdkKvDevice`, wrapping the
`kv_host_shim` C API. This is the class that answers *"where do the
`kv_host_shim` attributes go?"* — **they live here, never in the interface.**

Illustrative shape (delivered by the backend PR, not this PR):

```cpp
// Opaque handle; the real type lives in kv_host_shim.h and is visible ONLY in
// kv_host_shim_device.cpp, so no SPDK include leaks past this device.
struct kv_host_shim;

// SPDK-free open configuration a backend fills from its custom params.
struct KvHostShimConfig {
    std::string name = "nixl_spdk_kv";
    std::string vfu_addr;   // VFIOUSER transport dir for the SPDK KV target
    uint32_t    nsid = 0;   // 0 = auto-select first KV namespace
    bool        init_env = false;
};

class nixlKvHostShimDevice : public iSpdkKvDevice {
public:
    static std::unique_ptr<nixlKvHostShimDevice>
    open(const KvHostShimConfig &cfg, std::string &err);

    ~nixlKvHostShimDevice() override;               // kv_host_shim_close(shim_)

    uint32_t maxKeyLen() const override;            // kv_host_shim_max_key_len
    size_t maxValueLen() const override;            // kv_host_shim_max_value_len
    void *dmaAlloc(size_t len) override;            // kv_host_shim_dma_alloc
    void  dmaFree(void *buf) override;              // kv_host_shim_dma_free

    spdk_kv_status_t store(...) override;               // kv_host_shim_store
    spdk_kv_status_t retrieve(...) override;            // kv_host_shim_retrieve
    spdk_kv_status_t exist(const void *key, uint8_t key_len) override;  // ..._exist

private:
    explicit nixlKvHostShimDevice(kv_host_shim *shim) : shim_(shim) {}
    kv_host_shim *shim_ = nullptr;   // the sole kv_host_shim-specific attribute
};
```

Key points this example makes concrete:

- The opaque `kv_host_shim *shim_` handle and the `KvHostShimConfig` open
  attributes are **members of this class**, not the interface.
- The `.cpp` is the **only** translation unit that `#include`s `kv_host_shim.h`
  (and thus the SPDK headers), keeping the SPDK dependency isolated.
- The NVMe status-code → `spdk_kv_status_t` mapping (e.g. `0x85` → `BUFFER_TOO_SMALL`,
  `0x87` → `NOT_FOUND`) is done **inside** this class's `.cpp`, so it never
  reaches the shared engine.
- The abstract interface uses `size_t` for NIXL value lengths. Because the shim
  accepts `uint32_t`, this adapter rejects values larger than `UINT32_MAX`
  before narrowing. A zero `maxValueLen()` means no additional advertised
  device limit; it does not bypass the shim's representable-size check.

## Worked example 2 — the test double: `nixlFakeKvDevice`

Because the engine only depends on `iSpdkKvDevice`, a trivial in-memory
implementation makes the whole data plane testable with no SPDK or hardware:

```cpp
class nixlFakeKvDevice : public iSpdkKvDevice {
    std::map<std::vector<uint8_t>, std::vector<uint8_t>> store_;
    // dmaAlloc/dmaFree → malloc/free; store/retrieve/exist → map operations,
    // returning spdk_kv_status_t (BUFFER_TOO_SMALL/NOT_FOUND) to mirror the real device.
};
```

## Construction and device ownership

Constructor injection is the preferred path. It makes device ownership explicit
and initializes the effective key length during base construction:

```cpp
class nixlRadosNkvEngine : public nixlSpdkKvEngine {
    static std::unique_ptr<iSpdkKvDevice>
    makeDevice(const nixlBackendInitParams *init_params) {
        KvHostShimConfig cfg;
        // ... fill cfg from init_params (vfu_addr/nsid/init_env) ...
        std::string err;
        return nixlKvHostShimDevice::open(cfg, err);
    }

public:
    explicit nixlRadosNkvEngine(const nixlBackendInitParams *init_params)
        : nixlSpdkKvEngine(init_params, makeDevice(init_params)) {}

protected:
    bool deriveKey(const std::string &token_seq, uint8_t key_len,
                   std::vector<uint8_t> &out) const override;
};
```

For compatibility, a backend may keep deferred initialization. It must call
`initDevice()` from its derived constructor; `openDevice()` cannot be dispatched
correctly from the base constructor:

```cpp
nixlCsalNkvEngine::nixlCsalNkvEngine(const nixlBackendInitParams *init_params)
    : nixlSpdkKvEngine(init_params) {
    initDevice();
}

std::unique_ptr<iSpdkKvDevice>
nixlCsalNkvEngine::openDevice(std::string &err) const {
    KvHostShimConfig cfg;
    // ... fill cfg for the CSAL target ...
    return nixlKvHostShimDevice::open(cfg, err);
}
```

Everything else — registration, query, and the Store/Retrieve transfer path — is
inherited from `nixlSpdkKvEngine` and runs against whichever `iSpdkKvDevice` the
backend opened.

## Value sizes and concurrency

The shared interface uses `size_t` end to end, so a NIXL descriptor length is
never silently narrowed by the common engine. A concrete adapter validates the
length against its native API and the nonzero `maxValueLen()` advertised by the
device before submitting an operation.

Each `nixlSpdkKvEngine` owns one device and serializes a complete transaction
(DMA allocation, staging copy, Store/Retrieve/Exist, and free) with its device
mutex. This gives single-qpair SPDK implementations a safe default; the device
interface itself does not promise independent thread safety.
