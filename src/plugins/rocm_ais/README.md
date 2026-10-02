<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# NIXL AIS Plugin

This plugin uses AMD Infinity Storage (AIS) hipFile APIs as an I/O backend for
NIXL. It is the ROCm counterpart to the CUDA
[GDS plugin](../cuda_gds/README.md).

The directory hosts the AIS family of backends. An abstract base engine
(`nixlAisEngine`, in `ais_backend.h`/`.cpp`) owns everything independent of how
I/O is submitted: the hipFile driver lifecycle, memory and file registration,
`queryMem`, and the `prepXfer` preamble that validates a request and translates
descriptors into logical `aisXferReq` entries. Concrete engines supply only the
submission mechanism.

Today the only backend name is:

- `AIS_MT`: multi-threaded transfers via TaskFlow (`nixlAisMtEngine`). TaskFlow
  issues one blocking `hipFileRead` or `hipFileWrite` per prepared request,
  mirroring the NVIDIA `GDS_MT` backend.

A batch engine mirroring GDS's cuFile batch path slots in alongside `AIS_MT`
once hipFile's batch API is adopted; `finalizePrep` is the hook it needs.

## Dependencies

- ROCm 7.14 or later with the HIP runtime (`libamdhip64`). 7.14 is the first
  ROCm release to ship hipFile. HIP calls are issued against device memory, so
  HIP is required even for the registration-only paths.
- hipFile (`libhipfile.so` and `hipfile.h`), which ships inside the ROCm
  install.
- [TaskFlow](https://github.com/taskflow/taskflow), provided by the Meson
  subproject, for the `AIS_MT` entry point.

Both HIP and hipFile are looked for under `rocm_path`. The plugin is skipped
silently when HIP or hipFile is not found, so a ROCm-less tree configures
unchanged.

## Build Instructions

HIP is detected independently of CUDA, so `AIS_MT` can be built alongside the
CUDA plugins when ROCm and hipFile are both present. ROCm is requested by
setting `-Drocm_path`.

```bash
$ meson setup build -Drocm_path=/opt/rocm
$ meson compile -C build
```

Relevant Meson options (see `meson_options.txt`):

| Option | Default | Description |
| --- | --- | --- |
| `rocm_path` | `''` | ROCm install prefix used for HIP and hipFile detection. Empty means ROCm is not used. |
| `disable_rocm_ais_backend` | `false` | Disable the AIS hipFile backend entirely. |

`AIS_MT` also participates in the generic `enable_plugins` / `disable_plugins` /
`static_plugins` lists. Requesting it explicitly while it is disabled, or while
HIP, hipFile or TaskFlow is missing, is a configure-time error rather than a
silent skip.

## API Reference

### Backend name

`AIS_MT` — pass to `nixlAgent::createBackend` or the C API equivalent.

### Backend parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `thread_count` | `max(1, hardware concurrency / 2)` | Number of persistent TaskFlow workers. The plugin advertises the computed value as a decimal string. Zero falls back to the default. |

### Supported memory types

`DRAM_SEG`, `VRAM_SEG`, and `FILE_SEG`, shared with the other local file
backends via `FileEngineBase`. See the note on host DRAM under
[Limitations](#limitations).

The backend is local-only: it supports local transfers, and does not support
remote transfers or notifications. Exactly one side of a transfer must be the
file side; memory-to-memory and file-to-file requests are rejected.

### File registration

`FILE_SEG` accepts either fd-in-`devId` (fd-mode) or a `"<modes>:<path>"`
string in `metaInfo` (path-mode, backend owns the open/close); see
[`src/utils/file/README.md`](../../utils/file/README.md#path-mode-file-registration).
Repeated registrations of the same open fd share one refcounted hipFile handle.
Path-mode requires a unique `devId` per registration, since in path-mode the
`devId` is a caller-chosen key with no relation to any fd.

## Limitations

- **Host DRAM buffer registration.** `hipFileBufRegister` rejects host memory
  with error 5013 (`hipFileHipMemoryTypeInvalid`); current ROCm hipFile
  supports device memory only, so VRAM is the supported path today. Setting
  `HIPFILE_ALLOW_COMPAT_MODE=true` is the escape hatch: registration then logs a
  warning and continues instead of failing, though the subsequent transfer may
  still fail on stacks without host-buffer support.
- **Filesystem support.** hipFile refuses filesystems it does not recognise as
  AIS-capable. Set `HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true` to allow them, for
  example when running the unit test against a scratch directory that is not on
  AIS-backed storage.
- **Batch I/O.** Only the multi-threaded submission strategy exists today; see
  the note on the future batch engine above.

## Example Usage

```cpp
nixlAgentConfig cfg(true);
nixlAgent agent("AisMtAgent", cfg);

nixl_b_params_t params;
params["thread_count"] = "8";

nixlBackendH *ais_mt;
nixl_status_t ret = agent.createBackend("AIS_MT", params, ais_mt);
```

A fuller end-to-end example, covering VRAM and DRAM buffers, path-mode
registration, and read/write verification, lives in
[`test/unit/plugins/ais_mt/nixl_ais_mt_test.cpp`](../../../test/unit/plugins/ais_mt/nixl_ais_mt_test.cpp).
The path-mode smoke target can be overridden with the
`NIXL_AIS_MT_PATH_MODE_FILE` environment variable; it must name a regular file
on a writable filesystem, never a raw block device.

For the containerised ROCm build and an interactive `AIS_MT` nixlbench run on an
AMD GPU, see [`docs/rocm-ci.md`](../../../docs/rocm-ci.md).
