---
title: POSIX
description: Standard POSIX I/O backend for DRAM-to-file transfers.
---

## Overview

The POSIX backend provides asynchronous file I/O for DRAM-to-file transfers. Unless an I/O mechanism is selected explicitly, it uses the first implementation available at build time in this order: Linux AIO, io_uring, then POSIX AIO.

| Property | Value |
|----------|-------|
| **Transfer Type** | DRAM ↔ File |
| **Protocol** | io_uring / linux_aio / posix_aio |
| **Best For** | CPU memory to local filesystem transfers |

## Installation

The POSIX backend is built by default with no required external dependencies.

### Optional Dependencies

For enhanced I/O performance, install one or both of these libraries:

**Linux AIO (libaio):**

```bash
# Ubuntu/Debian
sudo apt-get install libaio-dev

# RHEL/CentOS/Fedora
sudo dnf install libaio-devel
```

**io_uring (liburing):**

```bash
# Ubuntu/Debian
sudo apt install liburing-dev

# RHEL/CentOS/Fedora
sudo dnf install liburing-devel
```

<Note>
When running in Docker, io_uring syscalls are blocked by default. You need to create a custom seccomp profile that allows `io_uring_setup`, `io_uring_enter`, `io_uring_register`, and `io_uring_sync`. See the [POSIX plug-in README](https://github.com/ai-dynamo/nixl/blob/main/src/plugins/posix/README.md) for Docker configuration details.
</Note>

## Configuration

The POSIX backend has no backend-specific environment variables. Configure it with string key-value parameters when creating the backend.

### Backend Parameters

| Parameter | Accepted value | Default | Description |
|---|---|---|---|
| `use_aio` | `true` or `false` | `false` | Select Linux AIO (`libaio`). |
| `use_uring` | `true` or `false` | `false` | Select io_uring (`liburing`). |
| `use_posix_aio` | `true` or `false` | `false` | Select POSIX AIO. |
| `ios_pool_size` | Integer from 64 to 65,536 | `65536` | Number of reusable I/O entries allocated by the backend. |
| `kernel_queue_size` | Integer from 16 to 1,024 | `256` | Maximum number of operations in the kernel submission queue. |

Set only one of the three `use_*` parameters. If more than one is `true`, the backend selects `use_aio` first, then `use_uring`, then `use_posix_aio`. If none is set, it selects the first implementation compiled into NIXL using the same order.

<Warning>
An explicit selection does not fall back to another implementation. Backend creation fails if the selected mechanism was not included in the NIXL build or cannot initialize in the current environment.
</Warning>

```cpp title="POSIX backend using io_uring"
nixl_b_params_t params = {
    {"use_uring", "true"},
    {"ios_pool_size", "8192"},
    {"kernel_queue_size", "256"}
};

agent.createBackend("POSIX", params);
```

When using io_uring in a container, the container security policy must permit the io_uring system calls described above.

## When to Use

- **DRAM-to-file on local filesystems** -- Standard file I/O for reading and writing host memory to local storage.
- **Environments without GPU or GDS** -- Use POSIX when GPUDirect Storage is not available or not needed.
- **Fallback for any DRAM-to-file transfer** -- A reliable default when no specialized storage backend is required.
