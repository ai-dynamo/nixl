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

# NIXL GDS Plugin

This plugin uses NVIDIA GPUDirect storage cuFile APIs as an I/O backend for
NIXL.

[NVIDIA GDS](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)<br />
[CUDA GDS Install and Setup](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### cufile.json configuration
This section provides the recommended configuration for the cufile.json file when using the NIXL GDS plugin in compatibility mode.
The configuration ensures that sufficient POSIX buffers are available. Only the properties that need to be overridden are included in this cufile.json configuration.
The cufile.json must be explicitly exported for applications to use this configuration using
```
export CUFILE_ENV_PATH_JSON="/path/to/cufile.json"
```

### Detailed cufile.json Options:

```
{
    // NOTE : Application can override custom configuration via export CUFILE_ENV_PATH_JSON=<filepath>
    // e.g : export CUFILE_ENV_PATH_JSON="/home/<xxx>/cufile.json"

            "properties": {
                            // allow compat mode, this will enable use of cuFile posix read/writes
                            "allow_compat_mode": true,
                            // max IO chunk size (parameter should be multiples of 64K) used by cuFileRead/Write internally per IO request
                            "max_direct_io_size_kb" : 16384,
                            // device memory size (parameter should be 4K aligned) for reserving bounce buffers for the entire GPU
                            "max_device_cache_size_kb" : 2097152,
                            // Note: ensure (max_device_cache_size_kb / per_buffer_cache_size_kb) >= io_batchsize
                            "per_buffer_cache_size_kb": 16384,
                            // limit on maximum device memory size (parameter should be 4K aligned) that can be pinned for a given process
                            "max_device_pinned_mem_size_kb" : 33554432,
                            // per-io bounce-buffer size (parameter should be multiples of 64K) ranging from 1024kb to 16384kb
                            "per_buffer_cache_size_kb" : 16384,

                            // posix bounce buffer pool size allocations
                            "posix_pool_slab_size_kb" : [ 4, 1024, 16384],
                            // posix bounce buffer pool max counts
                            "posix_pool_slab_count": [512, 512, 512]
            }
}
```
