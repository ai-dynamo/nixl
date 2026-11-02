<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# CUObject Client Config for S3 Accelerated

This configuration applies to S3 Accelerated engines that use the CUObject Client
library. Specify each client IP address associated with an RDMA device in the
`properties.rdma_dev_addr_list` property of the JSON configuration file.

## Startup latency workaround

To reduce S3 RDMA backend initialization time, set `execution.parallel_io` to
`false` and `execution.max_io_threads` to `0` in your application's JSON
configuration. This is a temporary workaround for affected cuObject versions.
Preserve your existing RDMA addresses and other settings when applying the example
below.

```json
{
    "execution": {
        "parallel_io" : false,
        "max_io_threads" : 0
    },

    "properties": {
        "allow_compat_mode": true,
        "use_pci_p2pdma": true,
        "rdma_peer_type": "dmabuf",
        "rdma_dev_addr_list": ["10.0.1.2", "10.0.2.2"]
    }
}
```

Export the file's absolute path before launching your NIXL application:

```bash
export CUFILE_ENV_PATH_JSON=/path/to/cufile.json
```

Use an application-specific file instead of changing `/etc/cufile.json`. If the
same process also uses cuFile or the NIXL GDS backend, these settings affect its
file I/O as well. NVIDIA's [GDS release notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html#known-limitations)
also list cuFile Stream APIs as unsupported with these settings.
