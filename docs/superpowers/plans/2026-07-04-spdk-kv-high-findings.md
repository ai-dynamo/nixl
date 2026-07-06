# SPDK KV High Findings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the interface-only SPDK KV core with optional constructor injection, size-safe device operations, serialized device access, and accurate documentation.

**Architecture:** Keep the deferred `initDevice()` path for compatibility and add a constructor that receives an already-open device. Use `size_t` across the abstract value API and place a mutable mutex in the shared engine so its future data-plane implementation can serialize complete device transactions.

**Tech Stack:** C++20, Meson, GoogleTest unit-source aggregation, Markdown.

---

## Task 1: Add a compile-time interface contract test

**Files:**
- Create: `test/gtest/unit/spdk_kv/interface_contract.cpp`
- Create: `test/gtest/unit/spdk_kv/meson.build`
- Modify: `test/gtest/unit/meson.build`

- [ ] **Step 1: Write the failing contract test**

Create a concrete inspection subclass that publicly inherits the protected constructors, implements the two customization hooks, and defines a `const` method that locks `deviceMutex_`. Add static assertions requiring both constructor signatures and `size_t` value-operation signatures.

```cpp
using max_value_len_signature_t = size_t (iSpdkKvDevice::*)() const;
using store_signature_t =
    spdk_kv_status_t (iSpdkKvDevice::*)(const void *, uint8_t, const void *, size_t);
using retrieve_signature_t = spdk_kv_status_t (iSpdkKvDevice::*)(
    const void *, uint8_t, void *, size_t, size_t *);

static_assert(std::is_same_v<decltype(&iSpdkKvDevice::maxValueLen), max_value_len_signature_t>);
static_assert(std::is_same_v<decltype(&iSpdkKvDevice::store), store_signature_t>);
static_assert(std::is_same_v<decltype(&iSpdkKvDevice::retrieve), retrieve_signature_t>);
static_assert(std::is_constructible_v<InspectableSpdkKvEngine,
                                      const nixlBackendInitParams *>);
static_assert(std::is_constructible_v<InspectableSpdkKvEngine,
                                      const nixlBackendInitParams *,
                                      std::unique_ptr<iSpdkKvDevice>>);
```

- [ ] **Step 2: Run the direct compile and verify RED**

Run:

```bash
c++ -std=c++20 -fsyntax-only -Isrc/api/cpp -Isrc/core -Isrc/core/telemetry \
  -Isrc/infra -Isrc/utils -Isrc -Isubprojects/abseil-cpp-20250814.1 \
  -Isrc/plugins/spdk_kv test/gtest/unit/spdk_kv/interface_contract.cpp
```

Expected: compilation fails because value signatures still use `uint32_t`, the injecting constructor is absent, and `deviceMutex_` is absent.

- [ ] **Step 3: Wire the source into the unit-test aggregate**

Create `spdk_kv_interface_unit_test_dep` with `interface_contract.cpp` as its source and add it to `unit_test_deps` unconditionally from `test/gtest/unit/meson.build`.

## Task 2: Harden the core headers

**Files:**
- Modify: `src/plugins/spdk_kv/ispdk_kv_device.h`
- Modify: `src/plugins/spdk_kv/spdk_kv_engine.h`
- Test: `test/gtest/unit/spdk_kv/interface_contract.cpp`

- [ ] **Step 1: Make abstract value lengths size-safe**

Change `maxValueLen()` to return `size_t`; change Store and Retrieve value lengths and `value_len_out` to `size_t`. Document that adapters for narrower native APIs must range-check before narrowing.

- [ ] **Step 2: Add the injecting constructor and shared initialization declaration**

Keep the existing constructor and add:

```cpp
nixlSpdkKvEngine(const nixlBackendInitParams *init_params,
                 std::unique_ptr<iSpdkKvDevice> device);
```

Add a private `setDevice(std::unique_ptr<iSpdkKvDevice>)` declaration used by both constructor initialization and `initDevice()` implementations.

- [ ] **Step 3: Add the serialization primitive**

Add `<mutex>` and:

```cpp
mutable std::mutex deviceMutex_;
```

Document that the shared implementation holds it across a complete allocation/staging/device-operation/free transaction.

- [ ] **Step 4: Run the direct compile and verify GREEN**

Run the command from Task 1 Step 2. Expected: exit 0.

## Task 3: Update user-facing documentation

**Files:**
- Modify: `src/plugins/spdk_kv/README.md`
- Modify: `src/plugins/spdk_kv/ispdk_kv_device.h`
- Modify: `src/plugins/spdk_kv/spdk_kv_engine.h`

- [ ] **Step 1: Update API examples and construction guidance**

Show constructor injection as the preferred path, and show the existing constructor plus `initDevice()` as the compatibility path. Update example value signatures to `size_t`.

- [ ] **Step 2: Document size and concurrency guarantees**

State that native adapters validate before narrowing and that one engine serializes complete operations against its device.

- [ ] **Step 3: Remove contradictory DMA wording**

Describe the engine as independent of SPDK allocation details while still aware of an abstract transfer-capable buffer contract.

## Task 4: Verify and commit

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run contract and header compilation checks**

Run the Task 1 direct compile command and standalone compilation for both headers. Expected: exit 0.

- [ ] **Step 2: Run repository checks available in the workspace**

Run `git diff --check`. If the configured Meson unit build is usable, build its unit target; otherwise report the concrete dependency/configuration blocker.

- [ ] **Step 3: Inspect scope**

Run `git diff --stat`, `git diff`, and `git status --short`; confirm unrelated untracked dependency files were not changed.

- [ ] **Step 4: Commit the implementation**

```bash
git add src/plugins/spdk_kv test/gtest/unit/spdk_kv test/gtest/unit/meson.build \
  docs/superpowers/plans/2026-07-04-spdk-kv-high-findings.md
git commit -m "plugins: harden SPDK KV core contracts"
```
