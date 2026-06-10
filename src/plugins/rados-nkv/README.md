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

# NIXL RADOS NVMe-KV Plugin (`RADOS_NKV`)

This backend maps NIXL transfers onto the **NVMe Key-Value (KV) command set**
through an in-process SPDK KV host shim (`kv_host_shim.{h,c}`). It lets a NIXL
agent Store/Retrieve values keyed by an NVMe KV key, and probe key existence,
over the SPDK VFIOUSER transport.

The KV namespace on the SPDK target is **backend-agnostic**: the same plugin and
tests run unchanged against an in-memory `kvdev` or a **librados-backed** `kvdev`
(real Ceph/RADOS), where each value lands as an object in a Ceph pool/namespace.
The plugin name `rados-nkv` reflects this RADOS-over-NVMe-KV use case.

## Operation and memory-type mapping

The plugin mirrors the OBJ plugin's memory-type shape:

| NIXL operation        | Local mem  | Remote mem | NVMe KV command |
| --------------------- | ---------- | ---------- | --------------- |
| `NIXL_WRITE`          | `DRAM_SEG` | `OBJ_SEG`  | KV **Store**    |
| `NIXL_READ`           | `DRAM_SEG` | `OBJ_SEG`  | KV **Retrieve** |
| `queryMem` (lookup)   | —          | `OBJ_SEG`  | KV **Exist**    |

The remote `OBJ_SEG` descriptor carries the NVMe KV key in its `metaInfo` blob
(the same channel the OBJ plugin uses for an object key). `queryMem` maps to the
NVMe KV Exist command, building a cache hit/miss mask without transferring any
value data (the llm-d lookup path).

### Key handling

The token sequence in `metaInfo` may be any non-empty length. The backend hashes
it (128-bit FNV-1a) into a fixed-length NVMe KV key of `min(16, kvkml)` bytes,
where `kvkml` is the max key length advertised by the namespace (falling back to
16 when the namespace reports no limit). Callers therefore pass a token sequence,
not a raw NVMe KV key; an empty `metaInfo` is **rejected**
(`NIXL_ERR_INVALID_PARAM`).

## Dependencies

- An **SPDK** build tree providing `libspdk_nvme` and the related static
  archives, plus the `kv_host_shim.{h,c}` host shim sources.
- **DPDK** and **isa-l** static archives (built as part of SPDK).
- **libvfio-user** (SPDK VFIOUSER transport).
- System libraries pulled in by the SPDK/DPDK static archives (e.g. `numa`,
  `uuid`, `ssl`, `crypto`, `aio`).
- Optional: **librados/Ceph** when running against a librados-backed KV
  namespace (`librados`, `librbd`).

## Build Instructions

The plugin is gated behind the SPDK locations and is skipped automatically when
the SPDK shim or `libspdk_nvme.a` are not found.

```bash
# Build with the plugin enabled (adjust the SPDK paths for your tree)
meson setup build \
    -Dspdk_root=/path/to/spdk \
    -Dspdk_kv_shim_dir=/path/to/spdk/test/nvmf/kv_shim
ninja -C build

# Build the standalone round-trip test executable as well
meson configure build -Drados_nkv_build_test=true
ninja -C build
```

Relevant Meson options (see `meson_options.txt`):

| Option                  | Default                         | Description                                       |
| ----------------------- | ------------------------------- | ------------------------------------------------- |
| `spdk_root`             | `/mnt/spdk`                     | Path to the SPDK build tree to link against.      |
| `spdk_kv_shim_dir`      | `/mnt/spdk/test/nvmf/kv_shim`   | Directory containing `kv_host_shim.{h,c}`.        |
| `rados_nkv_build_test`  | `false`                         | Build the direct-engine round-trip test binary.   |

To build **without** the plugin, omit it from `-Denable_plugins` (or add it to
`-Ddisable_plugins`).

## API Reference

`nixlRadosNkvEngine` (in `rados_nkv_backend.h`) implements the NIXL South Bound
API for a storage-style backend (`supportsLocal() == true`,
`supportsRemote() == false`, `supportsNotif() == false`).

- `getSupportedMems()` → `{DRAM_SEG, OBJ_SEG}`
- `registerMem()` — records the NVMe KV key for `OBJ_SEG` descriptors.
- `queryMem()` — KV Exist; present ⇒ engaged response, absent ⇒ `std::nullopt`,
  transport error ⇒ `NIXL_ERR_BACKEND` (never masked as a miss).
- `prepXfer()` / `postXfer()` / `checkXfer()` / `releaseReqH()` — issue the KV
  Store/Retrieve synchronously through the shim and report completion inline.

### Custom backend parameters (`nixl_b_params_t`)

| Parameter   | Required | Default | Meaning                                                              |
| ----------- | -------- | ------- | -------------------------------------------------------------------- |
| `vfu_addr`  | yes      | —       | VFIOUSER transport directory for the SPDK KV target. Aliases: `socket`, `vfio_user_path`, `device`. |
| `nsid`      | no       | `0`     | NVMe namespace id; `0`/unset auto-selects the first KV namespace.    |
| `init_env`  | no       | `false` | When `false`, the host/agent owns the SPDK env (multiple engines per process). When `true`, the shim brings up its own no-hugepage SPDK env (single instance; for standalone tests). |

## Example Usage

```cpp
nixl_b_params_t params;
params["vfu_addr"] = "/path/to/vfio-user/domain/muser0/0";

nixlBackendInitParams init{};
init.localAgent  = "my_agent";
init.type        = "RADOS_NKV";
init.customParams = &params;

nixlRadosNkvEngine eng(&init);
// register a DRAM source and an OBJ_SEG descriptor whose metaInfo holds the
// token sequence (the engine hashes it into the KV key), then prepXfer/postXfer
// with NIXL_WRITE (Store) or NIXL_READ (Retrieve). See
// rados_nkv_roundtrip_test.cpp for a full example.
```

## Testing

- **Unit (no SPDK)**: `test/gtest/unit/rados-nkv/test_rados_nkv_key.cpp` covers
  the key derivation (`radosNkvDeriveKey`) — known-answer vector, determinism,
  distinct/arbitrary-length inputs, truncation, and empty/zero rejection. It is
  SPDK-free and runs as part of the standard `ninja -C build test` unit suite.
- **End-to-end (needs SPDK)**: two bring-up scripts stand up an SPDK `nvmf_tgt`
  KV namespace over VFIOUSER and run the round-trip test against it:
  - `run_roundtrip.sh` — in-memory KV namespace (`kvdev_mem`).
  - `run_roundtrip_rados.sh` — librados-backed KV namespace (real Ceph/RADOS);
    also asserts the stored value lands as an object in the Ceph pool/namespace.

  Both require a built SPDK tree and `-Drados_nkv_build_test=true`. `SPDK_ROOT`
  and the Ceph config/keyring/`rados` paths are overridable via environment.
