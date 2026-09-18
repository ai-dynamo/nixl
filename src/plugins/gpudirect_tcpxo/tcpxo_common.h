/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#ifndef GPUDIRECT_TCPXO_TCPXO_COMMON_H_
#define GPUDIRECT_TCPXO_TCPXO_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>

#include <limits>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "nixl_types.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define ASSIGN_OR_RETURN_NIXL(lvalue, statusor)                                            \
    {                                                                                      \
        auto status_or_lvalue_ = statusor;                                                 \
        if (!status_or_lvalue_.ok()) {                                                     \
            return AbslStatusToNixlStatus(status_or_lvalue_.status(), __FILE__, __LINE__); \
        }                                                                                  \
    }                                                                                      \
    lvalue = std::move(*statusor);

#define RETURN_IF_ERROR_NIXL(status_func)                              \
    {                                                                  \
        const auto status = status_func;                               \
        if (!status.ok()) {                                            \
            return AbslStatusToNixlStatus(status, __FILE__, __LINE__); \
        }                                                              \
    }

namespace tcpxo {

inline constexpr size_t kDxsAddrMaxLen = 40;
inline constexpr size_t kMaxGpuDevices = 8;
inline constexpr size_t kPciAddrLen = 16;

// The maximum number of fastrak flows, basically channels of communication in a peer-peer
// connection
//
inline constexpr int64_t kFastrakMaxNumFlowsPerDxsConn = 8;
inline constexpr uint8_t kFastrakInvalidIdx = std::numeric_limits<uint8_t>::max();

inline constexpr size_t kMaxNetIfs = kMaxGpuDevices + 1;
inline constexpr size_t kMaxNetIfNameLen = 16;

/* Common socket address storage structure for IPv4/IPv6 */
union SocketAddress {
    struct sockaddr sa;
    struct sockaddr_in sin;
    struct sockaddr_in6 sin6;
};

struct GpuDev {
#ifdef HAVE_CUDA
    CUdevice dev = CU_DEVICE_INVALID;
    CUcontext ctx;
#endif
    int freq;
    std::string pci_addr;
};

nixl_status_t
AbslStatusToNixlStatus(const absl::Status &, const char *file, int line);

} // namespace tcpxo

#endif // GPUDIRECT_TCPXO_TCPXO_COMMON_H_
