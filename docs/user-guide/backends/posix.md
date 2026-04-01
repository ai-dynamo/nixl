---
title: POSIX
description: Standard POSIX I/O backend for DRAM-to-file transfers.
---

## Overview

The POSIX backend provides standard POSIX I/O for DRAM-to-file transfers. It automatically selects the best available asynchronous I/O mechanism in the following preference order: io_uring, linux_aio, or posix_aio.

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

The POSIX backend has no backend-specific environment variables or build options.

## When to Use

- **DRAM-to-file on local filesystems** -- Standard file I/O for reading and writing host memory to local storage.
- **Environments without GPU or GDS** -- Use POSIX when GPUDirect Storage is not available or not needed.
- **Fallback for any DRAM-to-file transfer** -- A reliable default when no specialized storage backend is required.
