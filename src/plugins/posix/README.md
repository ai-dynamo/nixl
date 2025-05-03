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

## Dependencies
To enable Linux AIO support, you need to install the libaio package:

```bash
# Ubuntu/Debian
sudo apt-get install libaio-dev

# RHEL/CentOS/Fedora
sudo dnf install libaio-devel
```


To enable liburing support you need to install the liburing package:

```bash
# Ubuntu/Debian
sudo apt install liburing-dev
# RHEL/CentOS/Fedora
sudo dnf install liburing-devel
```

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