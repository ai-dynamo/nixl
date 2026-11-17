/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_VALKEY_VALKEY_CLIENT_H
#define NIXL_SRC_PLUGINS_VALKEY_VALKEY_CLIENT_H

#include "nixl_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Everything that talks to GLIDE's C FFI and the valkey-large-object commands.
// The engine never includes glide_ffi.h.

struct nixlValkeyConfig {
    std::string host = "127.0.0.1";
    int port = 6379;
    std::string tls = "off"; // on: verify the server certificate; insecure: do not
    bool clusterMode = false;
    std::string username;
    std::string password;
    std::string rdmaProvider = "efa-direct"; // efa-direct, or tcp (libfabric emulation)
    std::string rdmaInterface; // fabric domain, if the host has several
    std::string keyPrefix;
    int timeoutMs = 30000;
    int transferTimeoutMs = 0; // cancel an RDMA transfer after this; 0 = never
    int numWorkers = 8;
    bool writeChecksum = false; // server verifies a CRC-32c of each write

    static nixlValkeyConfig
    fromBackendParams(const nixl_b_params_t *params);
};

namespace glide {
struct RdmaRegion; // host memory registered with one client
}

namespace nixl::valkey {

// GLIDE built with rdma and libfabric loadable.
bool
rdmaUsable();

constexpr long long BLOB_MISSING = -1;
constexpr long long BLOB_ERROR = -2;

// One blocking GLIDE client: a RESP connection and, with rdma, a fabric
// session. Runs one command or transfer at a time.
class client {
public:
    static client *
    open(const nixlValkeyConfig &config, bool rdma); // null on failure

    // Release or drop every region of this client first.
    void
    close();

    // BLOB.INFO key LEN for every key in one pipeline: size, BLOB_MISSING (also
    // for a key of another type) or BLOB_ERROR.
    std::vector<long long>
    blobLens(const std::vector<std::string> &keys);

    long long
    blobLen(const std::string &key);

    glide::RdmaRegion *
    registerHost(uintptr_t addr, size_t len); // null on failure

    // BLOB.RDMA_SET/GET of len bytes at offset of region. Blocks until the
    // server replies or closeRegions is called from another thread. closed: the
    // server may still reach the memory, so drop the client (closeWithRegions).
    nixl_status_t
    transfer(bool write,
             const std::string &key,
             glide::RdmaRegion *region,
             size_t offset,
             size_t len,
             bool checksum,
             bool &closed);

    // Cancel the transfers in flight on this client: they return with closed
    // set. Safe from another thread; the client takes no transfer afterwards.
    void
    closeRegions();

private:
    explicit client(const void *handle) : handle_(handle) {}

    const void *handle_;
};

// Deregister an idle region.
void
releaseRegion(glide::RdmaRegion *region);

// Free the regions, close the client and wait up to timeout_ms for GLIDE to let
// go of memory parked by a transfer that ended early.
void
closeWithRegions(client *c, const std::vector<glide::RdmaRegion *> &regions, int timeout_ms);

} // namespace nixl::valkey

#endif // NIXL_SRC_PLUGINS_VALKEY_VALKEY_CLIENT_H
