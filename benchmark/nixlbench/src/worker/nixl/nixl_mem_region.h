/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_MEM_REGION_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_MEM_REGION_H

#include <cstdlib>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>
#include <unistd.h>
#include <nixl.h>
#include "utils/utils.h"

#define CHECK_NIXL_ERROR(result, message)                                                     \
    do {                                                                                      \
        const nixl_status_t _r = (result);                                                    \
        if (0 != _r) {                                                                        \
            std::cerr << "NIXL: " << message << " (" << nixlEnumStrings::statusStr(_r) << ")" \
                      << std::endl;                                                           \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

// Convert vector of xferBenchIOV to nixl_reg_dlist_t
inline void
iovListToNixlRegDlist(const std::vector<xferBenchIOV> &iov_list, nixl_reg_dlist_t &dlist) {
    nixlBlobDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        desc.metaInfo = iov.metaInfo;
        dlist.addDesc(desc);
    }
}

// RAII wrapper around a backing file descriptor: closes the fd on destruction.
struct xferFileState {
    int fd = -1;
    uint64_t file_size = 0;
    uint64_t offset = 0;

    xferFileState() = default;

    xferFileState(int fd, uint64_t file_size, uint64_t offset) noexcept
        : fd(fd),
          file_size(file_size),
          offset(offset) {}

    ~xferFileState() {
        closeFd();
    }

    xferFileState(xferFileState &&o) noexcept
        : fd(std::exchange(o.fd, -1)),
          file_size(o.file_size),
          offset(o.offset) {}

    xferFileState &
    operator=(xferFileState &&o) noexcept {
        if (this != &o) {
            closeFd();
            fd = std::exchange(o.fd, -1);
            file_size = o.file_size;
            offset = o.offset;
        }
        return *this;
    }

    xferFileState(const xferFileState &) = delete;
    xferFileState &
    operator=(const xferFileState &) = delete;

private:
    void
    closeFd() noexcept {
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

// RAII wrapper around a NIXL memory registration: deregisters the memory and
// runs the per-IOV cleanup on destruction.
class NixlMemRegion {
    nixlAgent *agent_ = nullptr;
    nixlBackendH *backend_ = nullptr;
    nixl_mem_t seg_type_ = DRAM_SEG;
    std::vector<xferBenchIOV> iovs_;
    std::function<void(xferBenchIOV &)> cleanup_;
    nixl_opt_args_t cached_opt_args_;

public:
    NixlMemRegion() = default;
    NixlMemRegion(nixlAgent *agent,
                  nixlBackendH *backend,
                  nixl_mem_t seg_type,
                  std::vector<xferBenchIOV> iovs,
                  std::function<void(xferBenchIOV &)> cleanup = nullptr);
    ~NixlMemRegion();
    NixlMemRegion(NixlMemRegion &&o) noexcept;
    NixlMemRegion &
    operator=(NixlMemRegion &&o) noexcept;
    NixlMemRegion(const NixlMemRegion &) = delete;
    NixlMemRegion &
    operator=(const NixlMemRegion &) = delete;

    const std::vector<xferBenchIOV> &
    iovs() const {
        return iovs_;
    }

    std::vector<xferBenchIOV> &
    iovs() {
        return iovs_;
    }

private:
    void
    release();
};

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_MEM_REGION_H
