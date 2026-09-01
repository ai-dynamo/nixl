<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NIXL POSIX Plugin

This backend provides POSIX-compliant I/O operations using either Linux AIO (libaio) by default
Optionally POSIX plugin can also use liburing.

## File registration

`FILE_SEG` descriptors accept either fd-in-`devId` (fd-mode) or a
`"<modes>:<path>"` string in `metaInfo` (path-mode, backend owns the
open/close); see [`src/utils/file/README.md`](../../utils/file/README.md#path-mode-file-registration).

## Transfer submission and completion

Submission is batched: `postXfer` enqueues every descriptor of the request but
issues at most `MAX_IO_SUBMIT_BATCH_SIZE` (64) I/Os to the kernel. The remaining
I/Os are issued by later calls to `checkXfer`, which polls the I/O queue and
submits the next batch before reaping completions. This is the same in all three
queue implementations (Linux AIO, io_uring, POSIX AIO).

Two consequences for callers:

* **A request with more than 64 descriptors is not fully submitted until it has
  been polled.** `getXferStatus` must be called repeatedly while the request
  reports `NIXL_IN_PROG`, stopping only when it reaches completion or returns an
  error; a caller that posts and then stops polling (for example one waiting on
  an external event instead) leaves the remaining I/Os unissued and the transfer
  never completes. Polling is required for progress, not only to observe it.
* **Completion is bounded by the number of polls, not only by device speed.**
  `postXfer` issues the first batch, so a request of N descriptors needs
  `ceil(N / 64) - 1` further polls to finish submitting, plus at least one more
  to observe completion. Any sleep the caller inserts between polls is therefore
  multiplied by that count. Measured on a local NVMe device (256 KiB pages,
  writes, 5 ms between status checks, median of 9 runs): a 64-descriptor request
  completed in about 13 ms after 2 polls, and a 256-descriptor request in about
  37 ms after 5 polls. Poll without sleeping, or scale the interval with the
  descriptor count, when latency matters.

The I/O queue is shared by all requests on a POSIX backend instance, so polling
any one transfer handle also advances the in-flight I/Os of the others.

`MAX_IO_SUBMIT_BATCH_SIZE` is a compile-time constant. The surrounding queue
sizes are tunable through the backend parameters `ios_pool_size` (default 65536)
and `kernel_queue_size` (default 256).

## Dependencies
To enable Linux AIO support, you need to install the libaio package:

```bash
# Ubuntu/Debian
sudo apt-get install libaio-dev

# RHEL/CentOS/Fedora
sudo dnf install libaio-devel
```

### liburing

liburing support is enabled automatically via the Meson wrap under `subprojects/liburing.wrap` (pinned to WrapDB `liburing_2.14-1`). `meson setup` builds it from source when a system `liburing` is not found via pkg-config, so no manual install is required.

To use liburing with POSIX plugin use params["use_uring"] = "true"

# Running liburing with Docker
Docker by default blocks io_uring syscalls to the host system. These need to be explicitly enabled when running NIXL agents that use the posix plugin in Docker.

## Create a seccomp json file

```bash
$> wget https://github.com/moby/moby/blob/master/profiles/seccomp/default.json

# Add the following to the section, syscalls:names in default.json
# "io_uring_setup",
# "io_uring_enter",
# "io_uring_register",
# "io_uring_sync"

# Run docker with the new seccomp json file

$> docker run --security-opt seccomp=default.json -it --runtime=runc ... <imageid>
```
