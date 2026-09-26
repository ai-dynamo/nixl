/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
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

#include "worker/nixl/nixl_worker_odm.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <strings.h>

namespace {

xferBenchOdm::State *g_consistency_state = nullptr;

static constexpr int kStagingDevId = 0;
static constexpr size_t kSeedChunk = 0x400000ULL;

void
addXferDesc(nixl_xfer_dlist_t &dlist, uint64_t addr, size_t len, int dev_id) {
    nixlBasicDesc desc;
    desc.addr = addr;
    desc.len = len;
    desc.devId = dev_id;
    dlist.addDesc(desc);
}

bool
runSyncXfer(nixlAgent *agent,
            nixlBackendH *backend,
            const std::string &target,
            nixl_xfer_op_t op,
            uint64_t local_addr,
            size_t len,
            int local_dev_id,
            const xferBenchIOV &remote_iov) {
    nixl_xfer_dlist_t local_desc(DRAM_SEG);
    nixl_xfer_dlist_t remote_desc(DRAM_SEG);
    addXferDesc(local_desc, local_addr, len, local_dev_id);
    addXferDesc(remote_desc, remote_iov.addr, remote_iov.len, remote_iov.devId);

    nixl_opt_args_t params;
    params.backends.push_back(backend);

    nixlXferReqH *req = nullptr;
    nixl_status_t rc = agent->createXferReq(op, local_desc, remote_desc, target, req, &params);
    if (rc != NIXL_SUCCESS) {
        std::cerr << "ODM: createXferReq failed: " << nixlEnumStrings::statusStr(rc) << std::endl;
        return false;
    }

    rc = agent->postXferReq(req);
    if (rc != NIXL_SUCCESS && rc != NIXL_IN_PROG) {
        std::cerr << "ODM: postXferReq failed: " << nixlEnumStrings::statusStr(rc) << std::endl;
        agent->releaseXferReq(req);
        return false;
    }

    while (true) {
        rc = agent->getXferStatus(req);
        if (rc == NIXL_IN_PROG) {
            continue;
        }
        break;
    }

    agent->releaseXferReq(req);
    if (rc != NIXL_SUCCESS) {
        std::cerr << "ODM: transfer failed: " << nixlEnumStrings::statusStr(rc) << std::endl;
        return false;
    }
    return true;
}

bool
registerStaging(void *host, size_t len, nixlAgent *agent, nixlBackendH *backend) {
    std::vector<xferBenchIOV> staging_iov = {
        {reinterpret_cast<uint64_t>(host), len, kStagingDevId}};
    nixl_reg_dlist_t reg = iovListToNixlRegDlist(staging_iov, DRAM_SEG);
    nixl_opt_args_t opt_args;
    opt_args.backends.push_back(backend);
    const nixl_status_t rc = agent->registerMem(reg, &opt_args);
    if (rc != NIXL_SUCCESS) {
        std::cerr << "ODM: staging registerMem failed: " << nixlEnumStrings::statusStr(rc)
                  << std::endl;
        return false;
    }
    return true;
}

void
deregisterStaging(void *host, size_t len, nixlAgent *agent, nixlBackendH *backend) {
    std::vector<xferBenchIOV> staging_iov = {
        {reinterpret_cast<uint64_t>(host), len, kStagingDevId}};
    nixl_reg_dlist_t reg = iovListToNixlRegDlist(staging_iov, DRAM_SEG);
    nixl_opt_args_t opt_args;
    opt_args.backends.push_back(backend);
    agent->deregisterMem(reg, &opt_args);
}

} // namespace

namespace xferBenchOdm {

void
configureBackend(const std::vector<std::string> &devices,
                 State &state,
                 nixl_b_params_t &backend_params) {
    const std::string odm_device = (devices.empty() || devices[0] == "all") ? "odm0" : devices[0];
    backend_params["dmadev"] = odm_device;
    state.device_path_ = (odm_device[0] == '/') ? odm_device : ("/dev/" + odm_device);
    backend_params["odm_qid"] = std::to_string(xferBenchConfig::odm_qid_start);
    backend_params["odm_qid_start"] = std::to_string(xferBenchConfig::odm_qid_start);
    backend_params["odm_qid_end"] = std::to_string(xferBenchConfig::odm_qid_end);
    backend_params["num_threads"] = std::to_string(xferBenchConfig::num_threads);
    bool use_io_uring = xferBenchConfig::odm_use_io_uring;
    if (!use_io_uring) {
        if (const char *uring_env = getenv("ODM_USE_IO_URING")) {
            if (uring_env[0] == '1' || strcasecmp(uring_env, "true") == 0 ||
                strcasecmp(uring_env, "yes") == 0) {
                use_io_uring = true;
            }
        }
    }
    if (use_io_uring) {
        backend_params["odm_use_io_uring"] = "1";
    }
    std::cout << "MARVELL_ODM backend: dma_device=" << odm_device
              << " qid=" << xferBenchConfig::odm_qid_start
              << " qid_range=" << xferBenchConfig::odm_qid_start << ".."
              << xferBenchConfig::odm_qid_end << " threads=" << xferBenchConfig::num_threads
              << " io_uring=" << (use_io_uring ? "on" : "off")
              << " engine=ODM/dma-buf (both directions)"
              << " device_iova=auto (plugin GET_IOVA)" << std::endl;
}

void
State::bindNixl(nixlAgent *agent, nixlBackendH *backend, const std::string &target) {
    agent_ = agent;
    backend_ = backend;
    target_ = target;
}

void
setConsistencyState(State *state) {
    g_consistency_state = state;
}

bool
fetchWriteBufferForConsistency(const xferBenchIOV &iov, void **addr_out, bool *allocated_out) {
    if (!g_consistency_state) {
        return false;
    }
    return g_consistency_state->fetchWriteBuffer(iov, addr_out, allocated_out);
}

bool
State::fetchWriteBuffer(const xferBenchIOV &iov, void **addr_out, bool *allocated_out) {
    *addr_out = nullptr;
    *allocated_out = false;

    if (agent_ == nullptr || backend_ == nullptr || target_.empty()) {
        std::cerr << "ODM: consistency: NIXL context not bound" << std::endl;
        return false;
    }

    void *host = nullptr;
    if (posix_memalign(&host, xferBenchConfig::page_size, iov.len) != 0) {
        std::cerr << "ODM: consistency: host buffer alloc failed" << std::endl;
        return false;
    }

    if (!registerStaging(host, iov.len, agent_, backend_)) {
        free(host);
        return false;
    }

    const bool ok = runSyncXfer(agent_,
                                backend_,
                                target_,
                                NIXL_READ,
                                reinterpret_cast<uint64_t>(host),
                                iov.len,
                                kStagingDevId,
                                iov);
    deregisterStaging(host, iov.len, agent_, backend_);

    if (!ok) {
        std::cerr << "ODM: consistency: NIXL READ from device offset 0x" << std::hex << iov.addr
                  << std::dec << " failed" << std::endl;
        free(host);
        return false;
    }

    *addr_out = host;
    *allocated_out = true;
    return true;
}

bool
State::seedRegisteredBuffers(const std::vector<NixlMemRegion> &remote_regs,
                             size_t total_size,
                             uint8_t pattern) {
    if (agent_ == nullptr || backend_ == nullptr || target_.empty()) {
        std::cerr << "ODM: seed: NIXL context not bound" << std::endl;
        return false;
    }

    size_t seeded = 0;
    for (const auto &reg : remote_regs) {
        for (const auto &iov : reg.iovs()) {
            size_t offset = 0;
            while (offset < iov.len && seeded < total_size) {
                const size_t chunk = std::min({kSeedChunk, iov.len - offset, total_size - seeded});
                void *host = nullptr;
                if (posix_memalign(&host, xferBenchConfig::page_size, chunk) != 0) {
                    std::cerr << "ODM: seed: host buffer alloc failed" << std::endl;
                    return false;
                }
                memset(host, pattern, chunk);

                if (!registerStaging(host, chunk, agent_, backend_)) {
                    free(host);
                    return false;
                }

                const xferBenchIOV remote_chunk(iov.addr + offset, chunk, iov.devId);
                const bool ok = runSyncXfer(agent_,
                                            backend_,
                                            target_,
                                            NIXL_WRITE,
                                            reinterpret_cast<uint64_t>(host),
                                            chunk,
                                            kStagingDevId,
                                            remote_chunk);
                deregisterStaging(host, chunk, agent_, backend_);
                free(host);

                if (!ok) {
                    std::cerr << "ODM: seed: NIXL WRITE to device offset 0x" << std::hex
                              << (iov.addr + offset) << std::dec << " failed" << std::endl;
                    return false;
                }

                offset += chunk;
                seeded += chunk;
            }
            if (seeded >= total_size) {
                break;
            }
        }
    }
    std::cout << "ODM: seeded " << seeded << " bytes with 0x" << std::hex
              << static_cast<unsigned>(pattern) << std::dec << " (NIXL WRITE)" << std::endl;
    return true;
}

bool
State::seedDramForRead(const std::vector<NixlMemRegion> &remote_regs, size_t total_size) {
    return seedRegisteredBuffers(remote_regs, total_size, XFERBENCH_TARGET_BUFFER_ELEMENT);
}

} // namespace xferBenchOdm
